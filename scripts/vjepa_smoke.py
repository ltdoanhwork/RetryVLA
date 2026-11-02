#!/usr/bin/env python
"""Smoke / cost test of the V-JEPA 2-AC adapter: encode replayed frames, zero-shot rollout, memory + speed.
python -m scripts.vjepa_smoke [--frames data/foundation/frames/rq3]"""

from __future__ import annotations

import argparse
import glob
import time

import cv2
import numpy as np
import torch

from retryvla import REPO_ROOT
from retryvla.wm.vjepa2_ac import VJEncoder, VJWorldModel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/foundation/frames/rq3")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    f = sorted(glob.glob(str(REPO_ROOT / a.frames / "*.pt")))[0]
    d = torch.load(f, weights_only=False)
    dev = torch.device("cuda")
    t = time.time()
    enc = VJEncoder(dev)
    print(f"encoder loaded {time.time() - t:.1f}s, mem {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)
    imgs = np.stack([cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[..., ::-1] for b in d["agent"][0]])
    t = time.time()
    z = enc.encode_images(imgs)
    torch.cuda.synchronize()
    print(f"encoded {len(imgs)} frames {time.time() - t:.2f}s -> {tuple(z.shape)} mean {z.mean():.3f} std {z.std():.3f}")
    t = time.time()
    for _ in range(3):
        enc.encode_images(np.repeat(imgs[:1], 32, 0))
    torch.cuda.synchronize()
    print(f"encoder throughput {96 / (time.time() - t):.1f} frames/s")
    st = d["state"][0].float()
    qm, qs = st.mean(0), st.std(0).clamp_min(1e-3) * 0 + 1
    wm = VJWorldModel(qm * 0, qs).to(dev).eval()
    print(f"predictor loaded, mem {torch.cuda.memory_allocated() / 2**30:.2f} GB", flush=True)
    acts = d["actions"][: a.n].float().to(dev)
    z0 = z[:1].expand(len(acts), -1, -1)
    q0 = st[:1].to(dev).expand(len(acts), -1)
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        zH, qH, _, zs, _ = wm.rollout(z0, q0, acts)
    torch.cuda.synchronize()
    dt = time.time() - t
    v = d["valid"][0]
    zt = z[1:]
    e = [float(((zs[h - 1][0].float() - zt[h - 1]) ** 2).mean()) for h in range(1, 11) if v[h]]
    c = [float(((z[0] - zt[h - 1]) ** 2).mean()) for h in range(1, 11) if v[h]]
    print(f"rollout {len(acts)} cands x 10 steps: {dt:.2f}s ({len(acts) / dt:.1f} cand/s), peak "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    print("zero-shot cand-0 MSE per h:", np.round(e, 3), "copy:", np.round(c, 3))


if __name__ == "__main__":
    main()
