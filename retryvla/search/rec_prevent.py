"""V2R R3 (PLAN "V2R"): verified pre-failure positives D_P on LIBERO-Recover Spatial (v2 Algorithm 1 prevention branch;
Spatial counterpart of retryvla.search.goal_prevent).

python -m retryvla.search.rec_prevent [--shard i --num_shards n --workers 12]

Pre-failure states s^- come from the SOURCE episodes of the TRAIN failure groups: LIBERO-Recovery dumps every 20 steps
(data/assets/scenes_full/assets_times/bddl_scenes_spatial/<task>/<episode>/step_XXXX; the benchmark scenes are
step_0180 / step_0200). A state is loaded exactly as recorded (noise 0, gripper as recorded, no settle steps).
Per train episode:
  1. onset scan, backward from step 160 in 20-step steps: K = 8 pi0 rollouts (<= 520 steps) per grid point;
     t_last = the latest grid point with >= 2 / 8 successes (none -> "never recoverable", dropped).
  2. s^- = t_last if >= 1 of its K rollouts failed; a- = the first H = 50 actions of the first failing rollout.
  3. candidates = 24 SmolVLA chunk samples (noise 1.0) + 8 fixed primitives, <= 1 gripper flip; screen 1 trial
     (continuation seed 0), hits get K_v - 1 = 2 more trials; a- gets K_v = 3 trials (seeds 0..2, paired).
  4. pair iff p(a+) >= 2/3 and p(a+) - p(a-) >= 1/3, <= 2 positives per s^- (highest p, then closest to a-).
Positives are re-run with recording (same seed) and written as a RecoveryWriter bank (segment 0 = a+, 1 = pi0
continuation) -> data/recovery/v2r_dp/sh<k>/bank<j>; all outcomes (a-, losers = D_F, never BC) -> pairs.jsonl;
scan -> onset.jsonl; s^- load specs -> sminus.json (for rec_phys / visual checks). Resumable per episode.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np

from retryvla import REPO_ROOT

FULL = REPO_ROOT / "data/assets/scenes_full/assets_times/bddl_scenes_spatial"
OUT = REPO_ROOT / "data/recovery/v2r_dp"
POLICY = "HuggingFaceVLA/smolvla_libero"


def flips(a):
    return int(np.sum(np.diff(np.sign(a[:, 6])) != 0))


def fix_bddl(src: Path, dst: Path) -> None:
    """The scenes_full dumps declare one object per line ("akita_black_bowl_1 - akita_black_bowl" twice), which the
    LIBERO parser maps to a single object of that type (KeyError 'akita_black_bowl_1' at build). Rewrite the
    (:objects ...) block to the benchmark-scene form "a_1 a_2 - type" (same objects, same order)."""
    lines = src.read_text().splitlines()
    out, i = [], 0
    while i < len(lines):
        out.append(lines[i])
        if lines[i].strip().startswith("(:objects"):
            i += 1
            order, by = [], {}
            while i < len(lines) and lines[i].strip() != ")":
                parts = lines[i].strip().split(" - ")
                if len(parts) == 2:
                    ty = parts[1].strip()
                    if ty not in by:
                        order.append(ty); by[ty] = []
                    by[ty] += parts[0].split()
                i += 1
            ind = "    "
            out += [f"{ind}{' '.join(by[ty])} - {ty}" for ty in order]
            continue
        i += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out) + "\n")


def load_kw(group: str, t: int) -> tuple[str, dict]:
    task, ep = group.split("__")
    base = FULL / task / ep / f"step_{t:04d}"
    fixed = OUT / "bddl_fix" / task / ep / f"step_{t:04d}.bddl"
    if Path(str(base) + ".bddl").exists() and not fixed.exists():
        fix_bddl(Path(str(base) + ".bddl"), fixed)
    bddl = str(fixed)
    return bddl, dict(bddl=bddl, npy=str(base) + "_sim_state.npy", gripper_close=None, noise_xy=0.0, noise_seed=0,
                      settle_steps=0, num_steps_wait=0, build_seed=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--Kv", type=int, default=3)
    ap.add_argument("--n_vla", type=int, default=24)
    ap.add_argument("--H", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=520)
    ap.add_argument("--t_start", type=int, default=160)
    ap.add_argument("--max_pos", type=int, default=2)
    ap.add_argument("--max_groups", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    from retryvla.data.recovery_writer import RecoveryWriter
    from retryvla.envs.libero_env import parse_bddl_language
    from retryvla.envs.splits import load_split
    from retryvla.envs.vec import SimPool
    from retryvla.eval.rollout import Job, run_jobs
    from retryvla.policy import SmolVLAWrapper
    from retryvla.search import primitives
    from retryvla.utils import setup_logging

    setup_logging()
    split = load_split(REPO_ROOT / "data/splits/spatial_v1.json")
    groups = sorted(split["train_groups"])
    assert not set(groups) & set(split["test_groups"])
    groups = groups[a.shard:: a.num_shards]
    if a.max_groups:
        groups = groups[: a.max_groups]
    out = Path(a.out) / f"sh{a.shard}"
    out.mkdir(parents=True, exist_ok=True)
    # resume: a group is done if its onset row exists and its positives (if any) sit in a CLOSED bank (a killed run
    # leaves its open bank unusable -> those groups are redone; analysis keeps the last row per group)
    done = set()
    closed = {int(b.name[4:]) for b in out.glob("bank*") if (b / "CLOSED").exists()}
    if (out / "onset.jsonl").exists():
        for l in open(out / "onset.jsonl"):
            if l.strip():
                r_ = json.loads(l)
                if not r_.get("written") or r_.get("bank") in closed:
                    done.add(r_["group"])
    sm_f = out / "sminus.json"
    sminus = json.loads(sm_f.read_text()) if sm_f.exists() else {}
    policy = SmolVLAWrapper(POLICY, "cuda", 10)
    k = len(list(out.glob("bank*")))
    writer = RecoveryWriter(out / f"bank{k}", overwrite=True, write_retries=8, res=256)
    t0 = time.time()
    with SimPool(a.workers, 256) as pool:
        for n, g in enumerate(groups):
            if g in done:
                continue
            seed = zlib.crc32(g.encode()) % (2**31 - 1)
            scan, t_last, fail_acts, steps = [], None, None, 0
            for t in range(a.t_start, -1, -20):
                bddl, kw = load_kw(g, t)
                if not Path(kw["npy"]).exists():
                    continue
                task = parse_bddl_language(bddl)
                jobs = [Job("load_scene", kw, task=task, max_steps=a.max_steps, record="all", record_cfg={"img_until": -1},
                            policy_seed=(seed + 104729 * t + j) % (2**31 - 1)) for j in range(a.K)]
                res = run_jobs(pool, policy, jobs, desc=f"scan_t{t}")
                steps += sum(r.steps for r in res)
                p = float(np.mean([r.success for r in res]))
                scan.append({"t": t, "p": p})
                fails = [r for r in res if not r.success]
                if p >= 2 / a.K - 1e-9:
                    t_last = t
                    fail_acts = fails[0].record["action"][: a.H].astype(np.float32) if fails else None
                    break
            row = {"group": g, "bank": k, "scan": scan, "t_last": t_last, "has_failure": fail_acts is not None, "sim_steps": steps}
            if t_last is not None and fail_acts is not None:
                bddl, kw = load_kw(g, t_last)
                task = parse_bddl_language(bddl)
                iid = f"{g}__t{t_last:03d}"
                sminus[iid] = [bddl, kw, task]
                r0 = run_jobs(pool, None, [Job("load_scene", kw, task=task, max_steps=0, return_snapshot=True)])[0]
                snap, obs0 = r0.load_out["snapshot"], r0.load_out["obs0"]
                a_minus = fail_acts
                vla = policy.sample_chunks(obs0, task, a.n_vla, noise_scale=1.0, seed=seed % 100003)[:, : a.H]
                prim, names = primitives.library(a.H, np.random.RandomState(seed % (2**31 - 1)), n_random=0)
                cands = np.clip(np.concatenate([vla, prim]), -1, 1).astype(np.float32)
                src = [f"vla_s1.0_{i}" for i in range(len(vla))] + [f"prim_{x}" for x in names]
                keep = [i for i in range(len(cands)) if flips(cands[i]) <= 1]
                cands, src = cands[keep], [src[i] for i in keep]

                def job(c, kk, rec="none"):
                    return Job("restore_snapshot", dict(bddl=bddl, snap=snap), task=task, max_steps=a.max_steps,
                               prefix=c, record=rec, policy_seed=(seed + 7919 * kk) % (2**31 - 1))
                jobs = [job(a_minus, kk) for kk in range(a.Kv)] + [job(c, 0) for c in cands]
                res = run_jobs(pool, policy, jobs, desc="dp_screen")
                steps += sum(r.steps for r in res)
                p_minus = float(np.mean([r.success for r in res[: a.Kv]]))
                scr = res[a.Kv:]
                hit = [i for i, r in enumerate(scr) if r.success]
                extra = [job(cands[i], kk) for i in hit for kk in range(1, a.Kv)]
                res2 = run_jobs(pool, policy, extra, desc="dp_verify") if extra else []
                steps += sum(r.steps for r in res2)
                p_c = {}
                for j, i in enumerate(hit):
                    rr = [scr[i]] + res2[j * (a.Kv - 1): (j + 1) * (a.Kv - 1)]
                    p_c[i] = float(np.mean([r.success for r in rr]))
                pos = [i for i, p in p_c.items() if p >= 2 / 3 - 1e-9 and p - p_minus >= 1 / 3 - 1e-9]

                def l2m(i):
                    return float(np.linalg.norm(cands[i] - a_minus[: len(cands[i])]))
                pos = sorted(pos, key=lambda i: (-p_c[i], l2m(i)))
                writer.add_instance(iid, {"scene": iid, "task": task, "arm": "dp"})
                written = []
                for i in pos:   # re-run with recording (same seed 0 as the screen); keep only re-confirmed successes
                    if len(written) >= a.max_pos:
                        break
                    rr = run_jobs(pool, policy, [job(cands[i], 0, "all")], desc="dp_record")[0]
                    steps += rr.steps
                    if rr.success:
                        writer.add_trajectory(rr.record, task, iid, {"arm": "dp", "p_plus": p_c[i], "p_minus": p_minus,
                                                                     "source": src[i], "t_minus": t_last,
                                                                     "grip_flips": flips(cands[i]), "l2_to_minus": l2m(i),
                                                                     "score": p_c[i] - p_minus})
                        written.append(src[i])
                    rr.record = None
                row.update({"iid": iid, "p_t_last": scan[-1]["p"], "p_minus": p_minus, "n_cands": len(cands),
                            "n_screen_hit": len(hit), "p_cands": {src[i]: p for i, p in p_c.items()},
                            "positives": [src[i] for i in pos], "written": written,
                            "a_minus": a_minus.round(4).tolist(),
                            "a_plus": [cands[src.index(s_)].round(4).tolist() for s_ in written],
                            "losers": [s_ for s_ in src if s_ not in written], "sim_steps": steps})
                tmp = sm_f.with_suffix(".tmp")
                tmp.write_text(json.dumps(sminus))
                tmp.replace(sm_f)
                with open(out / "pairs.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
            with open(out / "onset.jsonl", "a") as f:
                f.write(json.dumps({k_: v for k_, v in row.items() if k_ not in ("a_minus", "a_plus")}) + "\n")
            print(json.dumps({"n": n + 1, "of": len(groups), "group": g[-40:], "t_last": t_last,
                              "p_minus": row.get("p_minus"), "hits": row.get("n_screen_hit"),
                              "written": len(row.get("written", [])), "steps": steps,
                              "wall_s": round(time.time() - t0)}), flush=True)
    writer.close()
    (out / f"bank{k}" / "CLOSED").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    (out / "CLOSED").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))


if __name__ == "__main__":
    main()
