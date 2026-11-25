"""V2R R2 (PLAN "V2R"): physical-plausibility scan of verified LIBERO-Recover recovery trajectories (Spatial port of
retryvla.search.goal_phys, same metrics / reject rules; thresholds from the Spatial D0 = lerobot/libero Spatial demos).

python -m retryvla.search.rec_phys --bank data/recovery/v2r_r1 --arm wm_diverse [--workers 12]
python -m retryvla.search.rec_phys --bank data/recovery/v2r_dp --arm dp --glob "sh*/bank*"   (D_P bank, aux targets)

Per trajectory (bridge = segment 0 + pi0 continuation = segment 1): rebuild the instance's exact start snapshot
(load_scene with the official protocol seeds, eval.seed 7, exactly like compare_arms / rec_prevent), replay ALL
recorded actions open-loop (`vec.phys_scan`), read sim_geom after every step. Outputs <bank>/phys_<arm>.jsonl (root,
episode_index, instance_id, task, metrics, reject reasons, keep, d3) + <bank>/phys/<arm>/<rid>_<ep>.npz (geom [T+1, G],
success; AUX targets) + <bank>/phys_summary_<arm>.json.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from retryvla import REPO_ROOT
from retryvla.geom.features import IDX
from retryvla.search.goal_phys import d3

EVAL_SEED = 7


def demo_thresholds() -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from retryvla.data.mix import spatial_episode_ids

    eps = spatial_episode_ids("lerobot/libero", "libero_spatial")
    d = LeRobotDataset("lerobot/libero", episodes=eps)
    hf = d.hf_dataset.with_format(None).select_columns(["action", "episode_index"])[:]
    a, ep = np.asarray(hf["action"], np.float32), np.asarray(hf["episode_index"])
    D3, D1 = [], []
    for e in np.unique(ep):
        x = a[ep == e]
        D1 += list(np.linalg.norm(np.diff(x[:, :6], axis=0), axis=1))
        for s in range(0, len(x) - 50, 25):
            D3.append(d3(x[s: s + 50]))
    return {"d3_med": float(np.median(D3)), "d1_med": float(np.median(D1)), "n_demo_eps": int(len(np.unique(ep)))}


def instance_loads(extra: dict | None = None) -> dict:
    """instance_id -> (bddl, load_kwargs, task) for every train-split LIBERO-Recover instance (compare_arms ids)
    + optional extra ids (rec_prevent s^- states: {iid: (bddl, kwargs, task)})."""
    from omegaconf import OmegaConf

    from retryvla.envs.splits import load_split, scenes_for
    from retryvla.eval.eval_recover import instance_jobs

    cfg = OmegaConf.load(REPO_ROOT / "configs/env/libero_spatial.yaml")
    rc = cfg.recover
    split = load_split(REPO_ROOT / cfg.split.file)
    scenes = scenes_for(split, "train", REPO_ROOT / rc.scene_root)
    out = {}
    for j in instance_jobs(scenes, rc, EVAL_SEED, rc.instruction, part="train"):
        out[f"{j.meta['scene']}__{j.meta['gripper_init']}{j.meta['trial']}"] = (j.load_kwargs["bddl"], j.load_kwargs, j.task)
    out.update(extra or {})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--arm", default="wm_diverse")
    ap.add_argument("--glob", default=None, help="dataset roots under --bank (default s*/p*/<arm>)")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from retryvla.envs.vec import SimPool
    from retryvla.eval.rollout import Job, run_jobs

    bank = REPO_ROOT / a.bank
    thr_f = REPO_ROOT / "data/recovery/v2r_demo_thresholds.json"
    if not thr_f.exists():
        thr_f.write_text(json.dumps(demo_thresholds(), indent=1))
    thr = json.loads(thr_f.read_text())
    TH = {"pen_cm": 1.0, "obj_jump_cm": 3.0, "d3": 2 * thr["d3_med"], "seam": 3 * thr["d1_med"], "grip_flips": 1}
    extra = {}
    for smf in sorted(bank.glob("**/sminus.json")):   # rec_prevent pre-failure states (one file per shard)
        extra.update({k: (v[0], v[1], v[2]) for k, v in json.loads(smf.read_text()).items()})
    loads = instance_loads(extra)
    outp = bank / f"phys_{a.arm}.jsonl"
    done = set()
    if outp.exists():
        done = {(json.loads(l)["root"], json.loads(l)["episode_index"]) for l in open(outp)}
    (bank / "phys" / a.arm).mkdir(parents=True, exist_ok=True)
    items = []
    for root in sorted(glob.glob(str(bank / (a.glob or f"s*/p*/{a.arm}")))):
        if not (Path(root) / "retryvla/trajectories.jsonl").exists():
            continue
        rid = root.split(str(bank) + "/")[1].replace("/", "_")
        ds = LeRobotDataset("local/x", root=root)
        hf = ds.hf_dataset.with_format(None).select_columns(["action", "segment", "episode_index"])[:]
        A, S, E = (np.asarray(hf["action"], np.float32), np.asarray(hf["segment"]).reshape(-1),
                   np.asarray(hf["episode_index"]))
        for l in open(Path(root) / "retryvla/trajectories.jsonl"):
            t = json.loads(l)
            if (root, t["episode_index"]) in done:
                continue
            m = E == t["episode_index"]
            items.append((root, rid, t, A[m], S[m]))
    print(json.dumps({"todo": len(items), "thresholds": TH}), flush=True)
    with SimPool(a.workers, 128) as pool:
        for b in range(0, len(items), pool.n):
            batch = items[b: b + pool.n]
            jobs = []
            for root, rid, t, acts, seg in batch:
                bddl, kw, task = loads[t["instance_id"]]
                jobs.append(Job("load_scene", kw, task=task, max_steps=0, return_snapshot=True))
            snaps = [r.load_out["snapshot"] for r in run_jobs(pool, None, jobs, desc="snap")]
            for w, ((root, rid, t, acts, seg), sn) in enumerate(zip(batch, snaps)):
                pool.send(w, "phys_scan", loads[t["instance_id"]][0], sn, acts)
            rows = []
            for w, (root, rid, t, acts, seg) in enumerate(batch):
                r = pool.recv(w)
                g, succ = r["geom"], r["success"]
                H = int((seg == 0).sum())
                gw = g[: min(len(g), H + 21)]
                pen = float(gw[:, IDX["max_pen"]].max() * 100)
                off = bool(gw[:, IDX["off_table"]].max() > 0.5)
                pos = gw[:, IDX["obj_x"]: IDX["obj_z"] + 1]
                free = gw[1:, IDX["finger_obj_contact"]] < 0.5
                jump = float((np.linalg.norm(np.diff(pos, axis=0), axis=1)[free].max() * 100) if free.any() else 0.0)
                dd3 = d3(acts[:H])
                seam = float(np.linalg.norm(acts[H, :6] - acts[H - 1, :6])) if 0 < H < len(acts) else 0.0
                gf = int(np.sum(np.diff(np.sign(acts[:H, 6])) != 0))
                mism = not (len(succ) and bool(succ[-1]))
                rej = [k for k, bad in (("replay_mismatch", mism), ("pen", pen > TH["pen_cm"]), ("off_table", off),
                                        ("obj_jump", jump > TH["obj_jump_cm"]), ("jerk", dd3 > TH["d3"]),
                                        ("seam", seam > TH["seam"]), ("grip_flips", gf > TH["grip_flips"])) if bad]
                np.savez_compressed(bank / "phys" / a.arm / f"{rid}_{t['episode_index']}.npz", geom=g, success=succ)
                rows.append({"root": root, "episode_index": t["episode_index"], "instance_id": t["instance_id"],
                             "task": t["task"], "H": H, "T": int(len(acts)), "pen_cm": pen, "off_table": off,
                             "obj_jump_cm": jump, "d3": dd3, "seam": seam, "grip_flips": gf, "reject": rej,
                             "keep": not rej})
            with open(outp, "a") as f:
                for r_ in rows:
                    f.write(json.dumps(r_) + "\n")
            print(json.dumps({"done": b + len(batch), "of": len(items)}), flush=True)
    allr = [json.loads(l) for l in open(outp)]
    reasons, by = {}, {}
    for r_ in allr:
        by.setdefault(r_["task"], [0, 0])
        by[r_["task"]][0] += 1
        by[r_["task"]][1] += r_["keep"]
        for k in r_["reject"]:
            reasons[k] = reasons.get(k, 0) + 1
    summ = {"thresholds": TH, "demo": thr, "n": len(allr), "kept": sum(r_["keep"] for r_ in allr),
            "kept_instances": len({r_["instance_id"] for r_ in allr if r_["keep"]}),
            "instances": len({r_["instance_id"] for r_ in allr}), "reject_reasons": reasons,
            "per_task_total_kept": by, "d3_median": float(np.median([r_["d3"] for r_ in allr])) if allr else None}
    (bank / f"phys_summary_{a.arm}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
