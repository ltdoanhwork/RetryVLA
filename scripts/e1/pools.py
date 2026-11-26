#!/usr/bin/env python
"""E1 step P: one shared 256-candidate proposal pool per held-out instance (PLAN E1 D2).

python -m scripts.e1.pools [--workers 8]

Restores the exact failure snapshot of every held-out verif_m5b instance (compare_arms precheck path, 0 policy
steps), draws SmolVLA `Proposals` (same config as M5b / RQ3, seed crc32(instance)), encodes obs0 with the WM v1
ranker (DINOv2, WM v1 normaliser = every E1 WM's normaliser) and writes data/wm/e1/pools/<instance>.pt with
pool_c [256, 50, 7], src, z0 / q0 (normalised), task, bddl, snapshot. Sanity: z0 vs the stored verif_m5b frame-0
latent (same renderer) is logged.
"""

from __future__ import annotations

import argparse
import json
import logging
import zlib

import numpy as np
import torch

from retryvla import REPO_ROOT
from retryvla.config import load_config
from retryvla.envs.splits import assert_allowed, load_split, scenes_for
from retryvla.envs.vec import SimPool
from retryvla.eval.rollout import Job, run_jobs_resilient
from retryvla.policy import SmolVLAWrapper
from retryvla.search.run_search import Proposals
from retryvla.search.scorers import Instance
from retryvla.search.wm_search import WMCriticRanker
from retryvla.utils import setup_logging

log = logging.getLogger("e1_pools")
CFG = "configs/exp/rq3_verifier_sweep.yaml"
OUT = REPO_ROOT / "data/wm/e1/pools"


def restore(pool, cfg, ids: set[str]) -> dict:
    from retryvla.search.compare_arms import iid, select_jobs

    split = load_split(REPO_ROOT / cfg.split.file)
    scenes = scenes_for(split, "train", REPO_ROOT / cfg.recover.scene_root)
    assert_allowed([s.tag for s in scenes], split)
    mb = cfg.m5b
    jobs = []
    for sh in range(int(mb.num_shards)):
        mb.shard = sh
        jobs += select_jobs(scenes, cfg, mb)
    todo = []
    for j in jobs:
        if iid(j.meta) in ids:
            jj = Job(**{**j.__dict__})
            jj.return_snapshot, jj.max_steps = True, 0
            todo.append(jj)
    res = run_jobs_resilient(pool, None, todo, retries=2, desc="restore")
    return {iid(r.meta): (todo[k], r) for k, r in enumerate(res)}


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cfg = load_config(CFG)
    OUT.mkdir(parents=True, exist_ok=True)
    held = json.loads((REPO_ROOT / "configs/exp/e1_split.json").read_text())["heldout"]
    todo = [i for i in held if not (OUT / f"{i}.pt").exists()]
    log.info("%d / %d pools to draw", len(todo), len(held))
    if not todo:
        return
    dev = torch.device("cuda")
    rk = WMCriticRanker((REPO_ROOT / "data/wm/latest_wm.txt").read_text().strip(),
                        (REPO_ROOT / "data/wm/latest_critic_v2.txt").read_text().strip(),
                        cfg.wm_collect.encoder, int(cfg.wm_collect.pool_grid), dev, 0.5, 1.0)
    policy = SmolVLAWrapper(cfg.policy.path, cfg.policy.device, cfg.policy.n_action_steps)
    H, NP = int(cfg.search.horizon), int(cfg.m5b.pool_size)
    with SimPool(a.workers, int(cfg.env.resolution)) as pool:
        got = restore(pool, cfg, set(todo))
    log.info("restored %d / %d", len(got), len(todo))
    for n, i in enumerate(todo):
        if i not in got:
            log.warning("no snapshot for %s", i); continue
        job, r = got[i]
        obs0 = r.load_out["obs0"]
        inst = Instance(i, r.meta["scene"], job.load_kwargs["bddl"], job.task, r.load_out["snapshot"], obs0,
                        meta=dict(r.meta))
        seed = zlib.crc32(i.encode()) % (2**31)
        pool_c, src = Proposals(policy, cfg.search, H, seed)(inst, NP)
        z0, q0 = rk.encode(obs0)
        v = torch.load(REPO_ROOT / "data/wm/verif_m5b" / f"{i}.pt", weights_only=False)
        zs = rk.norm.z(v["z"][0, 0].float().to(dev))[None]
        dz = float(((z0 - zs) ** 2).mean())
        torch.save({"instance_id": i, "task": job.task, "bddl": job.load_kwargs["bddl"], "snapshot": inst.snapshot,
                    "pool_c": pool_c.astype(np.float32), "src": list(src), "z0": z0.half().cpu(), "q0": q0.cpu(),
                    "z0_check_mse": dz}, OUT / f"{i}.pt")
        log.info("[%d/%d] %s pool %d, z0 vs stored frame-0 mse %.4f", n + 1, len(todo), i[-40:], len(pool_c), dz)


if __name__ == "__main__":
    main()
