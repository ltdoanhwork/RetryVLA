"""Back on Track data: on-policy rollouts of the FROZEN policy on its deployment distribution (train split only),
episode outcome labels, latents every `stride` env steps (for the liveness value) and executed actions (for the
planner's PCA action basis).

Spatial / Object (LIBERO-Recover, official protocol: close / open gripper trials, +-2 cm object noise, 520 steps):
  python -m retryvla.baselines.bot.collect --suite spatial --policy <ckpt> --out data/bot/sp_pi0 --trials 4 --seed 11
Goal (2610.05151 protocol, train-split demo inits, <= 300 steps, pi0 3B, replan 5):
  python -m retryvla.baselines.bot.collect --suite goal --policy runs/goal_rsi/m0_s7/ckpt_04000 --out data/bot/goal_m0 \
      --policy_seeds 8 9
Writes <out>/part_XXX.pt with {"episodes": [{zp [F,8,384] fp16 raw pooled DINO tokens, q [F,8], t [F], actions [T,7],
success, steps, task, scene, trial}]} (quota-safe tmp + rename, resumable by part) and <out>/meta.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from retryvla import REPO_ROOT
from retryvla.baselines.bot.liveness import pool_tokens
from retryvla.envs.vec import SimPool
from retryvla.eval.rollout import Job, run_jobs_resilient
from retryvla.utils import setup_logging
from retryvla.wm.collect import save_atomic
from retryvla.wm.encoder_dino import DinoEncoder

log = logging.getLogger("bot.collect")
SUITES = {"spatial": ("data/splits/spatial_v1.json", "data/assets/scenes/scenes/selected_bddl_scenes_spatial", 256),
          "object": ("data/splits/object_v1.json", "data/assets/scenes/scenes/selected_bddl_scenes_object", 256),
          "goal": (None, None, 128)}


def recover_jobs(suite: str, split_part: str, trials: int, seed: int, max_scenes: int = 0) -> list[Job]:
    from omegaconf import OmegaConf

    from retryvla.envs.splits import assert_allowed, load_split, scenes_for
    from retryvla.eval.eval_recover import instance_jobs

    split_f, root, _ = SUITES[suite]
    split = load_split(REPO_ROOT / split_f)
    scenes = scenes_for(split, split_part, REPO_ROOT / root)
    assert_allowed([s.tag for s in scenes], split)          # train / pilot only
    if max_scenes:
        scenes = scenes[:max_scenes]
    rc = OmegaConf.load(REPO_ROOT / "configs/env/libero_spatial.yaml").recover
    tr = (["close", "open"] * trials)[:trials]
    return instance_jobs(scenes, rc, seed, "bddl", part=split_part, trials=tr)


def encode(enc: DinoEncoder, rec: dict, stride: int, success: bool, steps: int, full: bool = False) -> dict:
    """Frames at t = 0, stride, 2 stride, ... (< steps) plus the terminal observation (goal state if success)."""
    img_t = [int(t) for t in rec["img_t"]]
    A, W, Q, T = [], [], [], []
    for j, t in enumerate(img_t):
        if t % stride == 0 and t < steps:
            A.append(rec["image"][j]); W.append(rec["image2"][j]); Q.append(rec["state"][t]); T.append(t)
    A.append(rec["final_image"]); W.append(rec["final_image2"]); Q.append(rec["final_state"]); T.append(steps)
    dev = enc.mean.device
    zs, zf = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        for i in range(0, len(A), 64):
            za = enc.encode_images(torch.as_tensor(np.stack(A[i:i + 64]), device=dev))
            zw = enc.encode_images(torch.as_tensor(np.stack(W[i:i + 64]), device=dev))
            zz = torch.cat([za, zw], 1).float()
            zs.append(pool_tokens(zz).half().cpu())
            if full:
                zf.append(zz.half().cpu())
    extra = {"z": torch.cat(zf)} if full else {}
    return {**extra, "zp": torch.cat(zs), "q": np.stack(Q).astype(np.float32), "t": np.asarray(T, np.int64),
            "actions": rec["action"].astype(np.float32)}


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=list(SUITES))
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split_part", default="train")
    ap.add_argument("--trials", type=int, default=4, help="trials per LIBERO-Recover scene")
    ap.add_argument("--seed", type=int, default=11, help="protocol seed (test evals use 7)")
    ap.add_argument("--policy_seeds", type=int, nargs="*", default=[8, 9], help="goal: one rollout per init per seed")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--part_size", type=int, default=48)
    ap.add_argument("--max_jobs", type=int, default=0)
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=0, help="override episode budget (smoke tests)")
    ap.add_argument("--full_tokens", action="store_true", help="also store the 6x6 tokens (WM fine-tuning)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    res_px = SUITES[a.suite][2]
    if a.suite == "goal":
        from retryvla.eval.eval_goal import scene_jobs
        from retryvla.policy.pi0_wrapper import Pi0Wrapper

        jobs = []
        for s in a.policy_seeds:
            js = scene_jobs("goal_rsi_v1", a.split_part, 300, seed=s)
            for j in js:
                j.meta["policy_seed"] = s
            jobs += js
        policy = Pi0Wrapper(a.policy, a.device, n_action_steps=5)
    else:
        from retryvla.policy import SmolVLAWrapper

        jobs = recover_jobs(a.suite, a.split_part, a.trials, a.seed, a.max_scenes)
        policy = SmolVLAWrapper(a.policy, a.device, 10)
    if a.max_jobs:
        jobs = jobs[: a.max_jobs]
    for j in jobs:
        j.record, j.record_cfg = "all", {"stride": a.stride}
        if a.max_steps:
            j.max_steps = a.max_steps
    enc = DinoEncoder("facebook/dinov2-small", pool_grid=6, device=a.device)
    (out / "meta.json").write_text(json.dumps({"suite": a.suite, "policy": a.policy, "n_jobs": len(jobs),
                                               "stride": a.stride, "seed": a.seed, "trials": a.trials,
                                               "policy_seeds": a.policy_seeds, "split_part": a.split_part}, indent=1))
    t0 = time.time()
    with SimPool(a.workers, res_px) as pool:
        for p0 in range(0, len(jobs), a.part_size):
            fn = out / f"part_{p0 // a.part_size:03d}.pt"
            if fn.exists():
                continue
            sub = jobs[p0: p0 + a.part_size]
            res = run_jobs_resilient(pool, policy, sub, retries=2, desc=f"bot_collect {fn.name}")
            eps = []
            for r in res:
                e = encode(enc, r.record, a.stride, bool(r.success), int(r.steps), a.full_tokens)
                e.update({"success": bool(r.success), "steps": int(r.steps), "task": sub[r.job_id].task,
                          "scene": r.meta.get("scene"), "trial": r.meta.get("trial", r.meta.get("policy_seed")),
                          "gripper_init": r.meta.get("gripper_init")})
                eps.append(e)
                r.record = None
            save_atomic({"episodes": eps}, fn)
            log.info("%s: %d episodes, SR %.3f | %.0fs", fn.name, len(eps), np.mean([e["success"] for e in eps]),
                     time.time() - t0)
    print("BOT_COLLECT_DONE", flush=True)


if __name__ == "__main__":
    main()
