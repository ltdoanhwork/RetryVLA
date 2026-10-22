#!/usr/bin/env python
"""P2: DINOv2 latents of the nominal LIBERO-Spatial demos (lerobot/libero), for the policy-agnostic value critics.

Same encoder / pooling / camera order as the WM data (retryvla.wm.collect: agent cam then wrist cam, 6x6 pooled
DINOv2-small tokens, RAW = un-normalized; the WM normalizer is applied at training time), same 8-d state.
Frames every `--stride` dataset steps (dataset step == one env step: SFT trains SmolVLA's 50-step chunks on it)
plus the last frame of every episode.

python scripts/build_demo_latents.py [--stride 5] [--max_episodes N]  ->  data/wm/demo_latents_spatial.pt
  {"episodes": [{"episode": e, "task": <bddl instruction used by LIBERO-Recover / search>, "canonical": ...,
                 "T": T, "t": [F], "z": [F, 72, 384] fp16, "q": [F, 8], "actions": [T, 7]}], "stride": s}
The task string is mapped canonical -> bddl instruction (the string the critics see at search time, inst.task);
the one nominal task with no LIBERO-Recover scene keeps its canonical string.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def canon_to_bddl() -> dict[str, str]:
    from retryvla.envs.splits import load_split, scenes_for

    sp = load_split(REPO / "data/splits/spatial_v1.json")
    m = defaultdict(set)
    for part in ("train", "test"):  # language strings only (no scene content is used)
        for s in scenes_for(sp, part, REPO / "data/assets/scenes/scenes/selected_bddl_scenes_spatial"):
            m[s.language_canonical.strip().lower()].add(s.instruction("bddl"))
    assert all(len(v) == 1 for v in m.values()), m
    return {k: next(iter(v)) for k, v in m.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max_episodes", type=int, default=0)
    ap.add_argument("--out", default="data/wm/demo_latents_spatial.pt")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    import pandas as pd
    from huggingface_hub import snapshot_download
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from retryvla.data.mix import patch_lerobot_index_mapping, spatial_episode_ids
    from retryvla.wm.encoder_dino import DinoEncoder

    patch_lerobot_index_mapping()
    c2b = canon_to_bddl()
    eps = spatial_episode_ids("lerobot/libero", args.suite)
    if args.max_episodes:
        eps = eps[: args.max_episodes]
    root = Path(snapshot_download("lerobot/libero", repo_type="dataset", allow_patterns=["meta/*", "data/*"]))
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    tname = {int(r.task_index): str(lang) for lang, r in tasks.iterrows()}
    fr = pd.concat([pd.read_parquet(p, columns=["episode_index", "frame_index", "index", "task_index", "action",
                                                "observation.state"])
                    for p in sorted((root / "data").rglob("*.parquet"))])
    fr = fr[fr.episode_index.isin(set(eps))].sort_values("index")
    ds = LeRobotDataset("lerobot/libero", episodes=eps)
    # dataset-relative position of every frame (ds is ordered like the global index restricted to eps)
    pos = {int(a): i for i, a in enumerate(fr["index"].to_numpy())}
    plan = []  # (episode, t, rel index)
    meta = {}
    for e, g in fr.groupby("episode_index"):
        T = len(g)
        ts = sorted(set(range(0, T, args.stride)) | {T - 1})
        idx = g["index"].to_numpy()
        meta[int(e)] = {"T": T, "actions": torch.as_tensor(np.stack(g["action"].to_numpy()), dtype=torch.float32),
                        "q_all": np.stack(g["observation.state"].to_numpy()).astype(np.float32),
                        "canonical": tname[int(g.task_index.iloc[0])].strip().lower(), "t": ts}
        plan += [(int(e), t, pos[int(idx[t])], int(idx[t])) for t in ts]
    print(f"{len(eps)} episodes, {len(plan)} frames to encode", flush=True)

    class Sub(torch.utils.data.Dataset):
        def __len__(self):
            return len(plan)

        def __getitem__(self, i):
            it = ds[plan[i][2]]
            assert int(it["index"]) == plan[i][3], (int(it["index"]), plan[i])  # frame alignment
            f = lambda k: (it[k].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)  # noqa: E731
            return i, f("observation.images.image"), f("observation.images.image2")

    dl = torch.utils.data.DataLoader(Sub(), batch_size=64, num_workers=args.workers, shuffle=False)
    enc = DinoEncoder("facebook/dinov2-small", pool_grid=6, device="cuda")
    Z = [None] * len(plan)
    t0 = time.time()
    for bi, (ii, a, w) in enumerate(dl):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            za = enc.encode_images(a.cuda())
            zw = enc.encode_images(w.cuda())
        z = torch.cat([za, zw], 1).half().cpu()
        for k, i in enumerate(ii.tolist()):
            Z[i] = z[k]
        if bi % 20 == 0:
            print(f"  {min(len(plan), (bi + 1) * 64)}/{len(plan)} frames {time.time() - t0:.0f}s", flush=True)
    by = defaultdict(list)
    for (e, t, _, _), z in zip(plan, Z):
        by[e].append(z)
    out = []
    for e, m in meta.items():
        out.append({"episode": e, "task": c2b.get(m["canonical"], m["canonical"]), "canonical": m["canonical"],
                    "T": m["T"], "t": torch.as_tensor(m["t"]), "z": torch.stack(by[e]),
                    "q": torch.as_tensor(m["q_all"][m["t"]]), "actions": m["actions"]})
    dst = REPO / args.out
    torch.save({"episodes": out, "stride": args.stride, "suite": args.suite}, str(dst) + ".tmp")
    Path(str(dst) + ".tmp").replace(dst)
    print(f"saved {dst}: {len(out)} episodes, {sum(len(o['t']) for o in out)} frames, "
          f"tasks {sorted(set(o['task'] for o in out))}", flush=True)


if __name__ == "__main__":
    main()
