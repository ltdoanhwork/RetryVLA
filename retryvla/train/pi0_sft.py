"""pi0 LoRA SFT for the 2610.05151 reverse-direction track (their recipe; see retryvla/policy/pi0_lora.py for D2).

torchrun --nproc_per_node 2 -m retryvla.train.pi0_sft --out runs/<id> --steps 5000 --save_every 1000 \
    [--init <ckpt>] [--recovery data/...] [--warmup 1000 --decay_steps 30000] [--seed 7] [--bs 48]

- Data: D0 = data/goal_rsi/d0_train (300 train demos replayed in our sim, 128 px). Windows = every frame as anchor,
  50-step action chunk, repeated-tail padding at the episode end (lerobot clamping = openpi / their baseline).
- --recovery: extra LeRobotDatasets pooled with D0 and shuffled uniformly (their mix: no oversampling), optional
  --recovery_segments to keep only the recovery segment (segment 0) as anchors.
- Optimiser = theirs: AdamW betas (0.9, 0.95), eps 1e-8, wd 1e-10, clip 1, peak 2.5e-5, linear warmup then cosine to
  2.5e-6 over decay_steps (openpi CosineDecaySchedule). M0: warmup 1000 / decay 30000 (openpi defaults, the paper
  gives none for M0); continuation: warmup 200 / decay 2000 (paper). No EMA. Normalisation stats from D0 only.
- Global batch --bs split over ranks; bf16 autocast; gradient checkpointing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from retryvla import REPO_ROOT
from retryvla.policy import pi0_lora as P

D0 = "data/goal_rsi/d0_train"


def lr_at(step, peak, warmup, decay_steps, decay_lr):
    if step < warmup:
        return peak * (step + 1) / (warmup + 1)
    prog = min(1.0, (step - warmup) / max(1, decay_steps - warmup))
    return decay_lr + (peak - decay_lr) * 0.5 * (1 + math.cos(math.pi * prog))


class Pool(torch.utils.data.Dataset):
    """Concatenation of LeRobotDatasets; optional anchor filter on the `segment` column for recovery sets."""

    def __init__(self, roots, chunk, seg_filter=None):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.ds, self.idx = [], []
        for k, r in enumerate(roots):
            meta_fps = json.loads((Path(r) / "meta/info.json").read_text())["fps"]
            d = LeRobotDataset(f"local/{Path(r).name}", root=r,
                               delta_timestamps={"action": [i / meta_fps for i in range(chunk)]})
            keep = np.arange(len(d))
            if seg_filter is not None and k > 0 and "segment" in d.features:
                seg = np.asarray(d.hf_dataset.with_format(None).select_columns(["segment"])["segment"]).reshape(-1)
                keep = np.nonzero(np.isin(seg, seg_filter))[0]
            self.ds.append(d)
            self.idx += [(len(self.ds) - 1, int(i)) for i in keep]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        k, j = self.idx[i]
        it = self.ds[k][j]
        return {"observation.images.image": it["observation.images.image"],
                "observation.images.image2": it["observation.images.image2"],
                "observation.state": it["observation.state"], "action": it["action"], "task": it["task"],
                "src": k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--recovery", nargs="*", default=[])
    ap.add_argument("--recovery_segments", nargs="*", type=int, default=None)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--peak", type=float, default=2.5e-5)
    ap.add_argument("--decay_lr", type=float, default=2.5e-6)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--decay_steps", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(a.seed + rank)
    out = REPO_ROOT / a.out
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(a), indent=1))

    policy, cfg, info = P.load_policy(a.init, device=str(dev), for_training=True)
    policy.config.gradient_checkpointing = True
    if hasattr(policy.model, "gradient_checkpointing_enable"):
        policy.model.gradient_checkpointing_enable()
    if rank == 0:
        print("params", info, flush=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors

    stats = LeRobotDatasetMetadata("local/d0_train", root=REPO_ROOT / D0).stats
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats,
                                         preprocessor_overrides={"device_processor": {"device": str(dev)}})
    data = Pool([REPO_ROOT / D0] + [REPO_ROOT / r for r in a.recovery], cfg.chunk_size, a.recovery_segments)
    n_src = np.bincount([k for k, _ in data.idx])
    if rank == 0:
        print("anchors per source", n_src.tolist(), flush=True)
    g = torch.Generator().manual_seed(a.seed * 1000 + rank)
    sampler = torch.utils.data.RandomSampler(data, replacement=True, num_samples=a.steps * a.bs // world + 1,
                                             generator=g)
    dl = torch.utils.data.DataLoader(data, batch_size=a.bs // world, sampler=sampler, num_workers=a.workers,
                                     drop_last=True, persistent_workers=True, prefetch_factor=4)
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.peak, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(policy, device_ids=[dev.index], find_unused_parameters=True)
    else:
        model = policy
    meta = {"base": P.BASE, "parent": a.init, "args": vars(a), "params": info, "anchors_per_source": n_src.tolist()}
    t0, it = time.time(), iter(dl)
    policy.train()
    for step in range(a.steps):
        b = next(it)
        b.pop("src")
        b = pre(b)
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, a.peak, a.warmup, a.decay_steps, a.decay_lr)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if rank == 0 and (step % 50 == 0 or step == a.steps - 1):
            el = time.time() - t0
            print(f"step {step} loss {loss.item():.4f} gn {gn.item():.2f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"{el / (step + 1):.2f}s/it eta {el / (step + 1) * (a.steps - step - 1) / 60:.0f} min", flush=True)
            with open(out / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"step": step, "loss": loss.item(), "gn": gn.item()}) + "\n")
        if rank == 0 and ((step + 1) % a.save_every == 0 or step + 1 == a.steps):
            P.save_ckpt(policy, out / f"ckpt_{step + 1:05d}", meta | {"step": step + 1}, pre, post)
            print("saved", step + 1, flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        (out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))


if __name__ == "__main__":
    main()
