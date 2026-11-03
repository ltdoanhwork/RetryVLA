#!/usr/bin/env python
"""Encode replayed bridge frames (scripts/replay_bridges.py --frames_out) with the frozen V-JEPA 2 ViT-g encoder
-> pooled latents (retryvla.wm.vjepa2_ac.VJEncoder: 8x8 x 1408, fp16) in the LatentSet file format
(z [K, F, 64, 1408], q [K, F, 8], valid [K, F], actions, success, prefix_success, source, instance_id, task),
so retryvla.wm.data.load_latents / the critic code can read them.

python -m scripts.vjepa_encode --src data/foundation/frames/collect_v1 --out data/foundation/vj_latents/collect_v1 [--ends]

--ends: keep only frame 0 + the last valid frame (F = 2; critic / eval sets). Frame 0 (the failure state) is encoded
once per instance (identical across candidates); invalid frames (after an early success) are zero, valid False.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from retryvla import REPO_ROOT
from retryvla.utils import setup_logging
from retryvla.wm.vjepa2_ac import VJEncoder

log = logging.getLogger("vjepa_encode")


def dec(b: bytes) -> np.ndarray:
    return np.ascontiguousarray(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[..., ::-1])


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ends", action="store_true")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--expect", type=int, default=0, help="keep polling --src until this many outputs exist (runs "
                    "concurrently with the replay that is still writing --src)")
    a = ap.parse_args()
    src, out = REPO_ROOT / a.src, REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    enc = VJEncoder("cuda", bs=a.bs)
    t0, n_done = time.time(), 0
    while True:
        todo = [f for f in sorted(glob.glob(str(src / "*.pt"))) if not (out / Path(f).name).exists()]
        if not todo:
            if a.expect and len(list(out.glob("*.pt"))) < a.expect:
                time.sleep(60)
                continue
            break
        for f in todo:
            d = torch.load(f, weights_only=False)
            K = len(d["agent"])
            v = d["valid"].bool()
            st = d["state"].float()
            F_ = v.shape[1]
            lv = v.long().cumsum(1).argmax(1)
            stored_ends = bool(d.get("frame_ends"))
            if a.ends:
                sel = [[0, int(lv[k])] for k in range(K)]
            else:
                assert not stored_ends, "frames were stored with --frame_ends; use --ends"
                sel = [list(range(F_)) for _ in range(K)]
            Fo = 2 if a.ends else F_
            z = torch.zeros((K, Fo, 64, 1408), dtype=torch.float16)
            q = torch.zeros((K, Fo, 8))
            vo = torch.zeros((K, Fo), dtype=torch.bool)
            z0 = enc.encode_images(dec(d["agent"][0][0])[None]).half().cpu()[0]
            imgs, slots = [], []
            for k in range(K):
                for j, t in enumerate(sel[k]):
                    if not bool(v[k, t]):
                        continue
                    q[k, j], vo[k, j] = st[k, t], True
                    if t == 0:
                        z[k, j] = z0
                        continue
                    src_i = (1 if stored_ends else t)
                    imgs.append(dec(d["agent"][k][src_i])); slots.append((k, j))
            for i in range(0, len(imgs), 256):
                zz = enc.encode_images(np.stack(imgs[i: i + 256])).half().cpu()
                for (k, j), x in zip(slots[i: i + 256], zz):
                    z[k, j] = x
            o = {"instance_id": d["instance_id"], "task": d["task"], "z": z, "q": q, "valid": vo,
                 "actions": d["actions"].float(), "success": d["success"], "prefix_success": d["prefix_success"],
                 "source": d["source"], "frames": "ends" if a.ends else f"stride{d['frame_stride']}",
                 "last_valid": lv.clone()}   # index of the last valid frame on the ORIGINAL 5-step grid
            tmp = out / (Path(f).name + ".tmp")
            torch.save(o, tmp)
            os.replace(tmp, out / Path(f).name)
            n_done += 1
            log.info("[%d] %s: %d cands, %d frames encoded | %.0fs", n_done, d["instance_id"][-40:], K, len(imgs) + 1,
                     time.time() - t0)
    log.info("done: %d files -> %s", n_done, out)


if __name__ == "__main__":
    main()
