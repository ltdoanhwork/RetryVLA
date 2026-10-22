"""M2 data collection: (failure state, candidate recovery segment, outcome) for the latent WM + critic.

python -m retryvla.wm.collect configs/exp/m2_collect.yaml [wm_collect.shard=i wm_collect.num_shards=N]

For every TRAIN-split failure instance whose failure episode is NOT in the M3 hard-instance set
(so search targets stay unseen by the WM/critic):
  restore snapshot -> execute candidate segment (H=50) -> frozen pi0 continues -> outcome.
Candidates come from the same proposal distribution as search (SmolVLA samples, smooth perturbations,
primitives). Images are recorded only every `stride` env steps up to t=H and encoded on the fly
with frozen DINOv2 -> one .pt per instance with
  z [K, F, P, D] fp16 (F = H/stride + 1 latent frames, P patch tokens of both cams), valid [K, F],
  q [K, F, 8], actions [K, H, 7], success/success_step/prefix_success [K], source, task.
"""

from __future__ import annotations

import logging
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from retryvla import REPO_ROOT, tracking
from retryvla.config import cli_config, to_dict
from retryvla.envs.splits import assert_allowed, load_split, scenes_for
from retryvla.envs.vec import SimPool
from retryvla.eval.rollout import Job, run_jobs
from retryvla.policy import SmolVLAWrapper
from retryvla.search.run_search import Proposals, build_instances
from retryvla.utils import setup_logging
from retryvla.wm.encoder_dino import DinoEncoder

log = logging.getLogger("wm.collect")


def group_of(tag: str) -> str:
    p = tag.split("__")
    return f"{p[0]}__{p[1]}"


def encode_record(enc: DinoEncoder, rec: dict, horizon: int, stride: int) -> dict:
    """Latent frames at t = 0, stride, ..., horizon (or until the episode ended)."""
    F = horizon // stride + 1
    img_t = list(rec["img_t"])
    frames_a, frames_w, qs, valid = [], [], [], []
    for f in range(F):
        t = f * stride
        if t in img_t:
            j = img_t.index(t)
            frames_a.append(rec["image"][j]); frames_w.append(rec["image2"][j]); qs.append(rec["state"][t])
            valid.append(True)
        elif t >= len(rec["action"]) and "final_image" in rec and (not valid or valid[-1]):
            # episode ended before t (success during the segment): use the terminal observation once
            frames_a.append(rec["final_image"]); frames_w.append(rec["final_image2"]); qs.append(rec["final_state"])
            valid.append(True)
            break
        else:
            break
    dev = enc.mean.device
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        za = enc.encode_images(torch.as_tensor(np.stack(frames_a), device=dev))
        zw = enc.encode_images(torch.as_tensor(np.stack(frames_w), device=dev))
    z = torch.cat([za, zw], 1).half().cpu()
    n = len(valid)
    zf = torch.zeros((F, *z.shape[1:]), dtype=torch.float16)
    zf[:n] = z
    qf = np.zeros((F, 8), np.float32)
    qf[:n] = np.stack(qs)
    v = np.zeros(F, bool)
    v[:n] = True
    return {"z": zf, "q": qf, "valid": v}


def main() -> None:
    setup_logging()
    cfg = cli_config()
    wc = cfg.wm_collect
    split = load_split(REPO_ROOT / cfg.split.file)
    scenes = scenes_for(split, wc.split, REPO_ROOT / cfg.recover.scene_root)
    assert_allowed([s.tag for s in scenes], split)
    if wc.get("exclude_instances_file"):
        ex = {group_of(l.strip()) for l in open(REPO_ROOT / wc.exclude_instances_file) if l.strip()}
        scenes = [s for s in scenes if s.group not in ex]
    scenes = scenes[int(wc.shard)::int(wc.num_shards)]
    cfg.tag = f"{cfg.tag}_s{wc.shard}of{wc.num_shards}"
    out = REPO_ROOT / wc.out_root
    out.mkdir(parents=True, exist_ok=True)
    H, stride, K = int(cfg.search.horizon), int(wc.stride), int(wc.n_candidates)
    log.info("collect: %d scenes, %d candidates/instance, H=%d stride=%d -> %s", len(scenes), K, H, stride, out)
    with tracking.start(cfg) as run:
        policy = SmolVLAWrapper(cfg.policy.path, cfg.policy.device, cfg.policy.n_action_steps)
        enc = DinoEncoder(wc.encoder, pool_grid=int(wc.pool_grid), device=cfg.policy.device)
        cfg.search_on.precheck_rollouts = 0
        proposals = Proposals(policy, cfg.search, H, int(cfg.eval.seed) + 1000 * int(wc.shard))
        t0, n_done, n_succ, n_traj = time.time(), 0, 0, 0
        with SimPool(int(cfg.eval.n_workers), int(cfg.env.resolution)) as pool:
            insts, _ = build_instances(pool, policy, scenes, cfg, wc.split)
            for inst in insts:
                fn = out / f"{inst.instance_id}.pt"
                if fn.exists():
                    continue
                if str(wc.get("proposal_mode", "default")) == "wide":   # primitive-free broad exploration data
                    from retryvla.search.arms import wide_actions
                    cands = wide_actions(np.random.default_rng(zlib.crc32(inst.instance_id.encode())), K, H)
                    src = ["wide"] * K
                else:
                    cands, src = proposals(inst, K)
                jobs = [Job("restore_snapshot", dict(bddl=inst.bddl, snap=inst.snapshot), task=inst.task,
                            max_steps=H + int(cfg.search.continuation_max_steps), prefix=c, record="all",
                            record_cfg={"stride": stride, "img_until": H}, policy_seed=k + 7919 * n_done)
                        for k, c in enumerate(cands)]
                res = run_jobs(pool, policy, jobs, desc="collect")
                enc_rows = [encode_record(enc, r.record, H, stride) for r in res]
                torch.save({
                    "instance_id": inst.instance_id, "task": inst.task, "scene": inst.scene_tag,
                    "z": torch.stack([e["z"] for e in enc_rows]),
                    "q": torch.as_tensor(np.stack([e["q"] for e in enc_rows])),
                    "valid": torch.as_tensor(np.stack([e["valid"] for e in enc_rows])),
                    "actions": torch.as_tensor(cands),
                    "success": torch.as_tensor([r.success for r in res]),
                    "success_step": torch.as_tensor([r.success_step if r.success else -1 for r in res]),
                    "prefix_success": torch.as_tensor([r.prefix_success for r in res]),
                    "source": src, "obs0_state": torch.as_tensor(inst.obs0["state"]),
                }, fn)
                n_done += 1
                n_traj += len(res)
                n_succ += int(sum(r.success for r in res))
                run.log({"collect/instances": n_done, "collect/success_rate": n_succ / max(1, n_traj)}, step=n_done)
                log.info("[%d/%d] %s: %d/%d success | %.0fs", n_done, len(insts), inst.instance_id[-40:],
                         sum(r.success for r in res), len(res), time.time() - t0)
        run.summary.update({"n_instances": n_done, "n_trajectories": n_traj,
                            "success_rate": n_succ / max(1, n_traj), "config": to_dict(cfg)})


if __name__ == "__main__":
    main()
