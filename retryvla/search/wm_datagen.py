"""M5: WM-guided recovery DATA GENERATION (diverse + adaptive budget) with verification logging.

python -m retryvla.search.wm_datagen configs/exp/m5_wm_datagen.yaml [m5.shard=i m5.num_shards=N]

Per instance:
  1. candidate pool (proposals) + latent CEM, all scored by WM + critic (imagination only)
  2. verification order = diverse_ranking (best per cluster of imagined end states)
  3. verify in batches of `batch` until `target_successes` verified recoveries or `max_budget` sims
  4. every verified candidate (success AND failure) is written in the C1 latent format to
     `verif_out/<instance>.pt` -> labels exactly where the WM/critic was wrong (self-improvement data)
  5. verified successes -> LeRobotDataset (`out_root`), up to `max_trajs_per_instance`
Optional `m5.random_arm`: same budget schedule, candidates drawn uniformly from the initial pool.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import torch

from retryvla import REPO_ROOT, tracking
from retryvla.config import cli_config, to_dict
from retryvla.data.recovery_writer import RecoveryWriter
from retryvla.envs.splits import assert_allowed, load_split, scenes_for
from retryvla.envs.vec import SimPool
from retryvla.eval.rollout import Job, run_jobs
from retryvla.policy import SmolVLAWrapper
from retryvla.search.cem import CEM, CEMConfig, roughness
from retryvla.search.run_search import Proposals, build_instances
from retryvla.search.wm_search import WMCriticRanker, diverse_ranking
from retryvla.utils import setup_logging
from retryvla.wm.collect import encode_record

log = logging.getLogger("wm_datagen")


def verify(pool, policy, inst, cands, H, cont, seed0, noise_scale):
    jobs = [Job("restore_snapshot", dict(bddl=inst.bddl, snap=inst.snapshot), task=inst.task, max_steps=H + cont,
                prefix=c, record="all", policy_seed=(seed0 + k) % (2**31 - 1)) for k, c in enumerate(cands)]
    return run_jobs(pool, policy, jobs, noise_scale=noise_scale, desc="verify")


def main() -> None:
    setup_logging()
    cfg = cli_config()
    if "futures" in cfg:  # Part 4 action-free imagined futures (configs/train/af_futures_*.yaml)
        return main_futures(cfg)
    m5, sc = cfg.m5, cfg.search
    split = load_split(REPO_ROOT / cfg.split.file)
    only = {l.strip() for l in open(REPO_ROOT / m5.instance_ids_file) if l.strip()}
    scenes = [s for s in scenes_for(split, "train", REPO_ROOT / cfg.recover.scene_root)
              if s.tag in {i.rsplit("__", 1)[0] for i in only}]
    assert_allowed([s.tag for s in scenes], split)
    scenes = scenes[int(m5.shard)::int(m5.num_shards)]
    cfg.tag = f"{cfg.tag}_s{m5.shard}of{m5.num_shards}"
    dev = torch.device(cfg.policy.device)
    wm_ck = (REPO_ROOT / m5.wm_link).read_text().strip()
    cr_ck = m5.critic_ckpt if m5.get("critic_ckpt") else (REPO_ROOT / m5.critic_link).read_text().strip()
    H, NP, B, MAXB, TGT = int(sc.horizon), int(m5.pool_size), int(m5.batch), int(m5.max_budget), int(m5.target_successes)
    cont, stride = int(sc.continuation_max_steps), int(cfg.wm_collect.stride)
    vdir = REPO_ROOT / m5.verif_out
    vdir.mkdir(parents=True, exist_ok=True)
    with tracking.start(cfg) as run:
        run.summary.update({"wm_ckpt": wm_ck, "critic_ckpt": cr_ck})
        policy = SmolVLAWrapper(cfg.policy.path, cfg.policy.device, cfg.policy.n_action_steps)
        ranker = WMCriticRanker(wm_ck, cr_ck, cfg.wm_collect.encoder, int(cfg.wm_collect.pool_grid), dev,
                                float(m5.lambda_unc), float(m5.beta_rough))
        writer = RecoveryWriter(REPO_ROOT / m5.out_root, overwrite=True)
        rows, t0 = [], time.time()
        cfg.search_on.precheck_rollouts = 0
        with SimPool(int(cfg.eval.n_workers), int(cfg.env.resolution)) as pool:
            insts, _ = build_instances(pool, policy, scenes, cfg, "train")
            insts = [i for i in insts if i.instance_id in only]
            proposals = Proposals(policy, sc, H, int(cfg.eval.seed) + 77 * int(m5.shard))
            for n, inst in enumerate(insts):
                rng = np.random.default_rng(abs(hash(inst.instance_id)) % (2**31))
                pool_c, src = proposals(inst, NP)
                z0, q0 = ranker.encode(inst.obs0)
                cs = ranker.score(z0, q0, inst.task, pool_c)
                cand, csrc = pool_c, list(src)
                cem = CEM(H, 7, CEMConfig(n_samples=NP, init_std=float(m5.latent_cem_std), smooth_kernel=5), seed=n + 1)
                for it in range(1, int(m5.latent_cem_iters) + 1):
                    el = np.argsort(-cs["score"])[: int(m5.latent_elites)]
                    new = cem.sample_local(cand[el], cs["score"][el], it, temperature=0.05, decay=0.6)
                    ns = ranker.score(z0, q0, inst.task, new)
                    cand = np.concatenate([cand, new])
                    cs = {k: np.concatenate([cs[k], ns[k]]) for k in cs}
                    csrc += ["wm_cem"] * len(new)
                if m5.get("random_arm", False):
                    order, arm = rng.permutation(len(pool_c))[:MAXB], "random"
                    cand, csrc, cs = pool_c, list(src), {k: v[: len(pool_c)] for k, v in cs.items()}
                elif m5.get("diverse", True):
                    order, arm = diverse_ranking(cs["score"], cs["zfeat"], MAXB, seed=n), "wm_diverse"
                else:
                    order, arm = np.argsort(-cs["score"])[:MAXB], "wm_top"
                # adaptive verification
                verified, succ_ids = [], []
                recs, used = [], 0
                while used < len(order) and len(succ_ids) < TGT:
                    idx = order[used: used + B]
                    res = verify(pool, policy, inst, cand[idx], H, cont, 1009 * n + used, float(cfg.policy.noise_scale))
                    for j, r in zip(idx, res):
                        verified.append((j, r))
                        if r.success:
                            succ_ids.append(len(verified) - 1)
                    used += len(idx)
                # 4) verification log in C1 latent format (critic self-improvement data)
                enc_rows = [encode_record(ranker.enc, r.record, H, stride) for _, r in verified]
                torch.save({
                    "instance_id": inst.instance_id, "task": inst.task, "scene": inst.scene_tag,
                    "z": torch.stack([e["z"] for e in enc_rows]), "q": torch.as_tensor(np.stack([e["q"] for e in enc_rows])),
                    "valid": torch.as_tensor(np.stack([e["valid"] for e in enc_rows])),
                    "actions": torch.as_tensor(np.stack([cand[j] for j, _ in verified])),
                    "success": torch.as_tensor([r.success for _, r in verified]),
                    "success_step": torch.as_tensor([r.success_step if r.success else -1 for _, r in verified]),
                    "prefix_success": torch.as_tensor([r.prefix_success for _, r in verified]),
                    "source": [csrc[j] for j, _ in verified],
                    "pred_value": torch.as_tensor([float(cs["value"][j]) for j, _ in verified]),
                    "arm": arm,
                }, vdir / f"{inst.instance_id}.pt")
                # 5) dataset
                writer.add_instance(inst.instance_id, {"scene": inst.scene_tag, "task": inst.task, "arm": arm})
                for i in sorted(succ_ids, key=lambda i: -cs["score"][verified[i][0]])[: int(m5.max_trajs_per_instance)]:
                    j, r = verified[i]
                    writer.add_trajectory(r.record, inst.task, inst.instance_id,
                                          {"score": float(cs["score"][j]), "source": csrc[j], "arm": arm,
                                           "pred_value": float(cs["value"][j]), "scene": inst.scene_tag})
                for _, r in verified:
                    r.record = None
                rows.append({"instance_id": inst.instance_id, "arm": arm, "budget_used": used,
                             "successes": len(succ_ids), "solved": bool(succ_ids),
                             "rough": float(roughness(np.stack([cand[j] for j, _ in verified])).mean()),
                             "n_written": min(len(succ_ids), int(m5.max_trajs_per_instance))})
                d = pd.DataFrame(rows)
                run.log({"m5/solve": d.solved.mean(), "m5/yield": d.successes.sum() / d.budget_used.sum(),
                         "m5/written": int(d.n_written.sum())}, step=n)
                log.info("[%d/%d] %s | %s: %d succ / %d sims | solved %.2f | yield %.3f | written %d | %.0fs", n + 1,
                         len(insts), inst.instance_id[-40:], arm, len(succ_ids), used, d.solved.mean(),
                         d.successes.sum() / d.budget_used.sum(), int(d.n_written.sum()), time.time() - t0)
                d.to_json(run.path("per_instance.jsonl"), orient="records", lines=True)
        writer.close()
        d = pd.DataFrame(rows)
        run.summary.update({"n_instances": len(d), "solve_rate": float(d.solved.mean()),
                            "yield": float(d.successes.sum() / max(1, d.budget_used.sum())),
                            "sims": int(d.budget_used.sum()), "n_trajectories": int(d.n_written.sum()),
                            "rough": float(d.rough.mean()), "config": to_dict(cfg)})


def pooled_future(ranker, z0, q0, cands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Imagined end state of each candidate: per-camera mean-pooled normalized DINOv2 latent [N, 2*D] + q_H [N, 8]."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=ranker.dev.type == "cuda"):
        a = torch.as_tensor(cands, device=ranker.dev)
        zH, qH, _, _, _ = ranker.wm.rollout(z0.expand(len(a), -1, -1), q0.expand(len(a), -1), a)
    zH = zH.float()
    P = zH.shape[1] // 2  # [agentview patches | wrist patches] (WMCriticRanker.encode)
    return (torch.cat([zH[:, :P].mean(1), zH[:, P:].mean(1)], 1).cpu().numpy().astype(np.float32),
            qH.float().cpu().numpy().astype(np.float32))


def main_futures(cfg) -> None:
    """Part 4 (RQ2): ACTION-FREE imagined recovery futures from TRAIN failure states (no action labels used).

    python -m retryvla.search.wm_datagen configs/train/af_futures_m5b.yaml

    Per failure instance (the 256 M5b instances, i.e. the same failure states pi1's grounded data came from):
      proposals pool (same generator as M5b) -> WM + critic scores, then three K-sized selections of imagined end states
        filtered  : arm_wm_obj(mode=comp)  = the wm_diverse objective (C_pi score + latent CEM + diverse top-K)
        goal      : arm_wm_obj(mode=goal)  = DINO-WM goal-latent distance (unfiltered by C_pi)
        random    : uniform from the proposal pool (no WM ranking at all)
      target = per-camera pooled normalized DINOv2 latent of the imagined z_H. Optional sim check (futures.verify_variants)
      records sim_checked / sim_success flags only (never used as action labels).
    Output: <out>/futures.pt (obs table of failure observations + per-future rows), written atomically."""
    import json
    import zlib

    from retryvla.eval.eval_recover import instance_jobs
    from retryvla.search import arms as A

    fu, sc = cfg.futures, cfg.search
    split = load_split(REPO_ROOT / cfg.split.file)
    ids: set[str] = set()
    for p in sorted((REPO_ROOT).glob(fu.instances_from)):
        ids |= {json.loads(l)["instance_id"] for l in open(p)}
    scenes = [s for s in scenes_for(split, "train", REPO_ROOT / cfg.recover.scene_root)
              if s.tag in {i.rsplit("__", 1)[0] for i in ids}]
    assert_allowed([s.tag for s in scenes], split)
    canon = {s.tag: s.language_canonical for s in scenes}
    jobs = instance_jobs(scenes, cfg.recover, int(cfg.eval.seed), cfg.recover.instruction, part="train")
    iid = lambda m: f"{m['scene']}__{m['gripper_init']}{m['trial']}"  # noqa: E731
    jobs = [j for j in jobs if iid(j.meta) in ids]
    if fu.get("max_instances"):
        jobs = jobs[: int(fu.max_instances)]
    for j in jobs:
        j.max_steps, j.return_snapshot = 0, True
    dev = torch.device(cfg.policy.device)
    wm_ck = (REPO_ROOT / fu.wm_link).read_text().strip()
    cr_ck = (REPO_ROOT / fu.critic_link).read_text().strip()
    H, NP, K = int(sc.horizon), int(fu.pool_size), int(fu.k)
    cont = int(sc.continuation_max_steps)
    variants = list(fu.variants)
    vv = set(fu.get("verify_variants") or [])
    goal_feats = torch.load(REPO_ROOT / fu.goal_feats, weights_only=False) if "goal" in variants else None
    out = REPO_ROOT / fu.out_root
    out.mkdir(parents=True, exist_ok=True)
    with tracking.start(cfg) as run:
        run.summary.update({"wm_ckpt": wm_ck, "critic_ckpt": cr_ck, "n_jobs": len(jobs)})
        policy = SmolVLAWrapper(cfg.policy.path, cfg.policy.device, cfg.policy.n_action_steps)
        ranker = WMCriticRanker(wm_ck, cr_ck, cfg.wm_collect.encoder, int(cfg.wm_collect.pool_grid), dev,
                                float(fu.lambda_unc), float(fu.beta_rough))
        obs_rows, fut_rows, t0 = [], [], time.time()
        with SimPool(int(cfg.eval.n_workers), int(cfg.env.resolution)) as pool:
            loaded = run_jobs(pool, None, jobs, desc="load")
            proposals = Proposals(policy, sc, H, int(cfg.eval.seed) + 4242)
            for n, r in enumerate(loaded):
                m = r.meta
                inst_id, obs0 = iid(m), r.load_out["obs0"]
                task = next(j.task for j in jobs if iid(j.meta) == inst_id)
                bddl = next(j.load_kwargs["bddl"] for j in jobs if iid(j.meta) == inst_id)
                seed = zlib.crc32(inst_id.encode()) % (2**31)
                rng = np.random.default_rng(seed)
                pool_c, _ = proposals(type("I", (), {"obs0": obs0, "task": task})(), NP)
                z0, q0 = ranker.encode(obs0)
                pool_s = ranker.score(z0, q0, task, pool_c)
                v0 = float(ranker.score_real(z0, q0, task)[0])
                oi = len(obs_rows)
                obs_rows.append({"instance_id": inst_id, "scene": m["scene"], "task": task, "image": obs0["image"],
                                 "image2": obs0["image2"], "state": np.asarray(obs0["state"], np.float32),
                                 "z0_pool": torch.cat([z0[0, : z0.shape[1] // 2].mean(0),
                                                       z0[0, z0.shape[1] // 2:].mean(0)]).float().cpu().numpy(),
                                 "v0": v0})
                for var in variants:
                    if var == "filtered":
                        cand, idx, cs = A.arm_wm_obj(ranker, z0, q0, task, pool_c, K, seed % 10007, "comp")
                    elif var == "goal":
                        goal = ranker.norm.z(goal_feats[canon[m["scene"]]].to(dev).float())
                        cand, idx, cs = A.arm_wm_obj(ranker, z0, q0, task, pool_c, K, seed % 10007, "goal", goal=goal)
                    elif var == "random":
                        cand, idx, cs = pool_c, rng.choice(len(pool_c), size=K, replace=False), pool_s
                    else:
                        raise ValueError(var)
                    idx = np.asarray(idx)[:K]
                    zf, qf = pooled_future(ranker, z0, q0, cand[idx])
                    ok = [None] * len(idx)
                    if var in vv:
                        vj = [Job("restore_snapshot", dict(bddl=bddl, snap=r.load_out["snapshot"]), task=task,
                                  max_steps=H + cont, prefix=cand[i], record="none",
                                  policy_seed=(seed + 31 * k) % (2**31 - 1)) for k, i in enumerate(idx)]
                        ok = [bool(x.success) for x in run_jobs(pool, policy, vj, noise_scale=float(cfg.policy.noise_scale),
                                                                desc=f"verify_{var}")]
                    for rank, (i, s_ok) in enumerate(zip(idx, ok)):
                        fut_rows.append({"obs": oi, "variant": var, "rank": rank, "future_latent": zf[rank],
                                         "future_q": qf[rank], "cpi": float(cs["value"][i]), "score": float(cs["score"][i]),
                                         "unc": float(cs["unc"][i]), "sim_checked": s_ok is not None,
                                         "sim_success": bool(s_ok) if s_ok is not None else False,
                                         "actions": cand[i].astype(np.float16)})
                r.load_out = {}
                d = pd.DataFrame([{k: v for k, v in f.items() if k in ("variant", "cpi", "sim_checked", "sim_success")}
                                  for f in fut_rows])
                g = d.groupby("variant")
                msg = " | ".join(f"{v}: cpi {x.cpi.mean():.3f} succ {x[x.sim_checked].sim_success.mean() if x.sim_checked.any() else float('nan'):.3f}"
                                 for v, x in g)
                log.info("[%d/%d] %s | %s | %.0fs", n + 1, len(loaded), inst_id[-40:], msg, time.time() - t0)
                run.log({f"af/{v}_cpi": float(x.cpi.mean()) for v, x in g}, step=n)
        obj = {"obs": {k: ([o[k] for o in obs_rows] if k in ("instance_id", "scene", "task") else
                           torch.as_tensor(np.stack([o[k] for o in obs_rows]))) for k in obs_rows[0]},
               "futures": {k: ([f[k] for f in fut_rows] if k == "variant" else torch.as_tensor(np.stack([f[k] for f in fut_rows])))
                           for k in fut_rows[0]},
               "wm_ckpt": wm_ck, "critic_ckpt": cr_ck, "config": to_dict(cfg)}
        tmp = out / "futures.pt.tmp"
        torch.save(obj, tmp)
        tmp.replace(out / "futures.pt")
        d = pd.DataFrame([{k: v for k, v in f.items() if k in ("variant", "cpi", "sim_checked", "sim_success")} for f in fut_rows])
        summ = {v: {"n": int(len(x)), "cpi_mean": float(x.cpi.mean()),
                    "sim_success": float(x[x.sim_checked].sim_success.mean()) if x.sim_checked.any() else None}
                for v, x in d.groupby("variant")}
        run.summary.update({"n_obs": len(obs_rows), "n_futures": len(fut_rows), "by_variant": summ, "out": str(out)})
        log.info("wrote %s: %d obs, %s", out / "futures.pt", len(obs_rows), summ)


if __name__ == "__main__":
    main()
