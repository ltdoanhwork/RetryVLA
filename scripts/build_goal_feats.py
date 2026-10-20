#!/usr/bin/env python
"""Goal latents for the goal-progress baseline (DINO-WM style objective): for each LIBERO-Spatial task, the
DINOv2 latents (same encoder / pooling as the WM, RAW = un-normalized) of the last frames of every successful demo.

python scripts/build_goal_feats.py  ->  data/wm/goal_feats.pt  {canonical task language: [G, P, D] float16}
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--out", default="data/wm/goal_feats.pt")
    args = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from retryvla.data.mix import patch_lerobot_index_mapping, spatial_episode_ids
    from retryvla.wm.encoder_dino import DinoEncoder

    patch_lerobot_index_mapping()
    eps = spatial_episode_ids("lerobot/libero", args.suite)
    ds = LeRobotDataset("lerobot/libero", episodes=eps)
    enc = DinoEncoder("facebook/dinov2-small", pool_grid=6, device="cuda")
    by_task = defaultdict(list)
    # last frame of each episode: walk the global index and keep the frame before an episode change
    ep_col = np.asarray(ds.hf_dataset.with_format(None).select_columns(["episode_index"])["episode_index"])
    last = np.nonzero(np.r_[ep_col[1:] != ep_col[:-1], True])[0]
    for k in last:
        it = ds[int(k)]
        a = (it["observation.images.image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        w = (it["observation.images.image2"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            za = enc.encode_images(torch.as_tensor(a[None], device="cuda"))
            zw = enc.encode_images(torch.as_tensor(w[None], device="cuda"))
        by_task[it["task"].strip().lower()].append(torch.cat([za, zw], 1)[0].half().cpu())
    out = {t: torch.stack(v) for t, v in by_task.items()}
    for t, v in out.items():
        print(len(v), t)
    torch.save(out, REPO / args.out)


if __name__ == "__main__":
    main()
