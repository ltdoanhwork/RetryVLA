"""B3.2: RISE-inspired imagination-only policy-improvement baseline, step 1 = data generation.

python -m retryvla.baselines.rise_lite configs/exp/b32_rise_datagen.yaml [rise.proposal=pool|vla rise.max_instances=3]

Same failure states as the PoC bank (the 256 M5b pi0 failures: train scenes x object-noise trials 2..9,
`compare_arms.select_jobs` with the M5b selection keys, failure labels from the M5b pi0 pre-check, so NO new pi0
rollouts). Every state is re-created with the official restore protocol (`load_scene`, 0 policy steps).

Per failure state s_0 (all in imagination, NO simulator rollout of any candidate):
  proposal=pool : exactly the PoC wm_diverse candidate set: the same proposal pool (Proposals(n_total=256) yields
                  232 = 96 SmolVLA samples + 96 perturbations + 40 primitives; RNG replayed in the M5b shard order)
                  + 2 x 232 elite-local latent-CEM candidates = 696 WM rollouts (= the PoC search's WM budget)
  proposal=vla  : 696 SmolVLA chunk samples (noise scales 1.0 / 1.5), RISE-like "policy's own samples"
  WM v1 imagines the end state of each 50-step chunk, competence critic v2 gives V(s_hat_50), and
      A = score - V(s_0),  score = V(s_hat_50) - lambda * U_n * std(V) - beta * roughness   (the PoC objective)
  i.e. a one-step (chunk-level) advantage with the critic on the REAL failure state as the baseline.

Output: <out_root>/<proposal>/inst/<instance_id>.pt (obs_0 images + state, all candidates, A, V, U, ...) and
index.jsonl. The trainer (retryvla.baselines.sft_rise) chooses positives / weights from it.
Analysis only (NEVER used for training): candidates that coincide with a chunk the PoC verified in M5b get that
real outcome (`poc_verified`), so the precision of imagined-advantage labels can be measured at zero new sim cost.
"""

from __future__ import annotations

import glob
import json
import logging
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from retryvla import REPO_ROOT, tracking
from retryvla.config import cli_config, to_dict
from retryvla.envs.splits import assert_allowed, load_split, scenes_for
from retryvla.envs.vec import SimPool
from retryvla.eval.rollout import Job, run_jobs
from retryvla.policy import SmolVLAWrapper
from retryvla.search import arms as A
from retryvla.search.compare_arms import iid, select_jobs
from retryvla.search.run_search import Proposals
from retryvla.search.scorers import Instance
from retryvla.search.wm_search import WMCriticRanker, diverse_ranking
from retryvla.utils import setup_logging

log = logging.getLogger("rise_lite")


def save_retry(obj, path: Path, tries: int = 6) -> None:
    """/pfss writes occasionally fail: write tmp + rename, retry with back-off."""
    for k in range(tries):
        try:
            tmp = Path(str(path) + ".tmp")
            torch.save(obj, tmp)
            tmp.replace(path)
            return
        except (RuntimeError, OSError) as e:
            log.warning("save %s failed (%s), retry %d/%d", path.name, str(e)[:80], k + 1, tries)
            time.sleep(20 * (k + 1))
    raise OSError(f"could not save {path}")


def m5b_failures(pattern: str) -> tuple[set[str], set[str]]:
    files = sorted(glob.glob(str(REPO_ROOT / pattern)))
    if not files:
        raise FileNotFoundError(pattern)
    pre = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates("instance_id", keep="last")
    return set(pre.instance_id), set(pre.instance_id[~pre.pi0_success.astype(bool)])


def poc_labels(vfile: Path, cand: np.ndarray, tol: float = 1e-3) -> tuple[np.ndarray, dict]:
    """Real verified outcomes of the M5b PoC for candidates identical to a chunk it simulated (-1 = unknown)."""
    lab = -np.ones(len(cand), np.int8)
    info = {"poc_wm_diverse_match": float("nan")}
    if not vfile.exists():
        return lab, info
    d = torch.load(vfile, map_location="cpu", weights_only=False)
    act = d["actions"]
    if not torch.is_tensor(act):
        return lab, info
    act = act.numpy().astype(np.float32)
    succ = d["success"].numpy().astype(bool)
    src = list(d["source"])
    x = cand.reshape(len(cand), -1)
    for j in range(len(act)):
        dist = np.abs(x - act[j].reshape(1, -1)).max(1)
        k = int(dist.argmin())
        if dist[k] < tol:
            lab[k] = max(lab[k], int(succ[j]))
    wd = [j for j, s in enumerate(src) if s == "wm_diverse"]
    if wd:
        hit = [np.abs(x - act[j].reshape(1, -1)).max(1).min() < tol for j in wd]
        info["poc_wm_diverse_match"] = float(np.mean(hit))
    info["poc_n_verified_matched"] = int((lab >= 0).sum())
    return lab, info


def main() -> None:
    setup_logging()
    cfg = cli_config()
    rb, sc, mb = cfg.rise, cfg.search, cfg.m5b
    prop = str(rb.proposal)
    assert prop in ("pool", "vla"), prop
    cfg.tag = f"{cfg.tag}_{prop}" + (f"_n{int(rb.max_instances)}" if int(rb.get("max_instances", 0)) else "")
    split = load_split(REPO_ROOT / cfg.split.file)
    scenes = scenes_for(split, "train", REPO_ROOT / cfg.recover.scene_root)
    assert_allowed([s.tag for s in scenes], split)
    H, NP, B = int(sc.horizon), int(mb.pool_size), int(mb.budget)
    n_wm = int(rb.n_wm_calls)
    dev = torch.device(cfg.policy.device)
    wm_ck = (REPO_ROOT / mb.wm_link).read_text().strip()
    cr_ck = (REPO_ROOT / mb.critic_link).read_text().strip()
    out = REPO_ROOT / rb.out_root / prop
    (out / "inst").mkdir(parents=True, exist_ok=True)
    all_ids, fail_ids = m5b_failures(rb.precheck_glob)

    # instance order = M5b shard order (needed to replay the Proposals RNG of each shard exactly)
    shards = []
    for k in range(int(mb.num_shards)):
        mk = OmegaConf.merge(mb, {"shard": k})
        jobs = select_jobs(scenes, cfg, mk)
        shards.append([j for j in jobs if iid(j.meta) in fail_ids])
    sel = {iid(j.meta) for k in range(int(mb.num_shards)) for j in select_jobs(scenes, cfg, OmegaConf.merge(mb, {"shard": k}))}
    assert sel == all_ids, f"instance selection differs from M5b pre-check: {len(sel ^ all_ids)} ids"
    order = [(k, j) for k, js in enumerate(shards) for j in js]
    if int(rb.get("max_instances", 0)):
        order = order[: int(rb.max_instances)]
    done = {p.stem for p in (out / "inst").glob("*.pt")}
    log.info("%d M5b failures (%d selected), %d to process, %d already done; proposal=%s, %d WM calls / instance",
             len(fail_ids), len(all_ids), len(order), len([1 for _, j in order if iid(j.meta) in done]), prop, n_wm)

    with tracking.start(cfg) as run:
        run.summary.update({"wm_ckpt": wm_ck, "critic_ckpt": cr_ck, "proposal": prop, "out": str(out),
                            "sim_policy_steps": 0, "sim_candidate_rollouts": 0})
        policy = SmolVLAWrapper(cfg.policy.path, cfg.policy.device, cfg.policy.n_action_steps)
        ranker = WMCriticRanker(wm_ck, cr_ck, cfg.wm_collect.encoder, int(cfg.wm_collect.pool_grid), dev,
                                float(mb.lambda_unc), float(mb.beta_rough))
        # restore every failure state once (official protocol, zero policy steps)
        lj = []
        for k, j in order:
            jj = Job(**{**j.__dict__})
            jj.max_steps, jj.record, jj.return_snapshot = 0, "none", False
            lj.append(jj)
        with SimPool(int(cfg.eval.n_workers), int(cfg.env.resolution)) as pool:
            res = run_jobs(pool, None, lj, desc="restore")
        obs0 = {iid(r.meta): r.load_out["obs0"] for r in res}
        props = {k: Proposals(policy, sc, H, int(cfg.eval.seed) + 77 * k) for k in range(int(mb.num_shards))}
        rows, t0 = [], time.time()
        for n, (k, j) in enumerate(order):
            ti = time.time()
            i_d = iid(j.meta)
            o = obs0[i_d]
            inst = Instance(i_d, j.meta["scene"], j.load_kwargs["bddl"], j.task, {}, o, meta=dict(j.meta))
            seed = zlib.crc32(i_d.encode()) % (2**31)
            if prop == "pool":
                pool_c, src = props[k](inst, NP)  # always called (also when done) to keep the shard RNG in sync
            if i_d in done:
                continue
            z0, q0 = ranker.encode(o)
            V0 = float(ranker.score_real(z0, q0, inst.task)[0])
            if prop == "pool":
                pool_s = ranker.score(z0, q0, inst.task, pool_c)
                cand, picks, cs = A.arm_wm_diverse(ranker, z0, q0, inst.task, pool_c, pool_s, B, seed % 10007)
                src = list(src) + ["wm_cem"] * (len(cand) - len(pool_c))
            else:
                scales = list(sc.init.vla_noise_scales)
                per = n_wm // len(scales)
                cand = np.concatenate([policy.sample_chunks(o, inst.task, per, noise_scale=float(s),
                                                            seed=(seed + 100_003 * q) % (2**31 - 1))[:, :H]
                                       for q, s in enumerate(scales)])
                cand = np.clip(cand, -1, 1).astype(np.float32)
                src = [f"vla_s{s}" for s in scales for _ in range(per)]
                parts = [ranker.score(z0, q0, inst.task, cand[a: a + NP]) for a in range(0, len(cand), NP)]
                cs = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
                picks = diverse_ranking(cs["score"], cs["zfeat"], B, seed=seed % 10007)
            assert len(cand) == n_wm, (len(cand), n_wm)
            adv = (cs["score"] - V0).astype(np.float32)
            lab, info = poc_labels(REPO_ROOT / rb.poc_verif_dir / f"{i_d}.pt", cand)
            rec = {
                "instance_id": i_d, "shard": k, "task": inst.task, "scene": inst.scene_tag,
                "task_folder": j.meta["task_folder"], "image": o["image"], "image2": o["image2"],
                "state": np.asarray(o["state"], np.float32), "actions": cand.astype(np.float32),
                "adv": adv, "value": cs["value"].astype(np.float32), "unc": cs["unc"].astype(np.float32),
                "unc_n": cs["unc_n"].astype(np.float32), "rough": cs["rough"].astype(np.float32),
                "score": cs["score"].astype(np.float32), "V0": V0, "source": src,
                "poc_style_picks": np.asarray(picks, np.int64), "poc_verified_analysis_only": lab,
                "n_wm_calls": len(cand), "wm_ckpt": wm_ck, "critic_ckpt": cr_ck,
            }
            save_retry(rec, out / "inst" / f"{i_d}.pt")
            row = {"instance_id": i_d, "shard": k, "task_folder": j.meta["task_folder"], "V0": V0,
                   "n_wm_calls": len(cand), "adv_max": float(adv.max()), "adv_mean": float(adv.mean()),
                   "adv_q70": float(np.quantile(adv, 0.7)), "frac_adv_pos": float((adv > 0).mean()),
                   "value_max": float(cs["value"].max()), "t_s": time.time() - ti, **info}
            if (lab >= 0).any():
                m = lab >= 0
                row["poc_lab_n"] = int(m.sum())
                row["poc_lab_succ"] = int((lab == 1).sum())
            rows.append(row)
            with open(out / "index.jsonl", "a") as f:
                f.write(json.dumps(row, default=float) + "\n")
            pd.DataFrame(rows).to_json(run.path("per_instance.jsonl"), orient="records", lines=True)
            log.info("[%d/%d] %s | V0 %.3f  A max %.3f  q70 %.3f  frac>0 %.2f | PoC match %.2f | %.1fs (total %.0fs)",
                     n + 1, len(order), i_d[-40:], V0, row["adv_max"], row["adv_q70"], row["frac_adv_pos"],
                     info.get("poc_wm_diverse_match", float("nan")), time.time() - ti, time.time() - t0)
        d = pd.DataFrame(rows)
        if len(d):
            run.summary.update({"n_instances": int(len(d)), "wm_calls_total": int(d.n_wm_calls.sum()),
                                "V0_mean": float(d.V0.mean()), "adv_max_mean": float(d.adv_max.mean()),
                                "frac_adv_pos_mean": float(d.frac_adv_pos.mean()),
                                "poc_wm_diverse_match_mean": float(d.get("poc_wm_diverse_match", pd.Series()).mean())})
        n_done = len(list((out / "inst").glob("*.pt")))
        run.summary.update({"n_instance_files": n_done, "config": to_dict(cfg)})
        if not int(rb.get("max_instances", 0)) and n_done >= len(fail_ids):
            (out / "CLOSED").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))


if __name__ == "__main__":
    main()
