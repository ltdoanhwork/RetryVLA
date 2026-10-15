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


if __name__ == "__main__":
    main()
