#!/usr/bin/env python
"""WM over-optimism, step 2: gallery cache (CPU). For the gallery instances (verif_m5b, frames re-rendered by
scripts.replay_bridges --frames_out, bit-exact open-loop bridges, NVIDIA EGL like the stored latents) build

  zt   [K, 11, 72, 384] fp16  normalized TRUE latents (stored at verification time)
  zi   [K, 10, 72, 384] fp16  normalized WM v1 imagined latents (open loop from the true frame 0)
  c_wm / c_true [K], ci_h / ct_h [K, 10]  C^pi v2 on imagined / true latents (end = last valid frame)
  img_a / img_w [K, 11, 128, 128, 3] uint8 agent / wrist frames (t = 0, 5, ..., 50; invalid = repeat)
  img_a256 / img_w256 for frame 0 and the end frame
  sym [K, 11, S] privileged symbolic state (data/critic3d/verif_m5b rich records), source, success, actions

python -m scripts.wm_optimism.gallery_cache --frames runs/_analysis/wm_optimism/gallery_frames
"""

from __future__ import annotations

import argparse
import glob
import io
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from retryvla import REPO_ROOT
from retryvla.search.wm_search import WMCriticRanker
from retryvla.utils import setup_logging

log = logging.getLogger("wm_opt_gallery_cache")


def dec(b: bytes, size: int | None = None) -> np.ndarray:
    im = Image.open(io.BytesIO(b)).convert("RGB")
    if size:
        im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im)


@torch.no_grad()
def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="runs/_analysis/wm_optimism/gallery_frames")
    ap.add_argument("--out", default="runs/_analysis/wm_optimism/cache/gallery.pt")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device("cpu")
    rk = WMCriticRanker((REPO_ROOT / "data/wm/latest_wm.txt").read_text().strip(),
                        (REPO_ROOT / "data/wm/latest_critic_v2.txt").read_text().strip(),
                        "facebook/dinov2-small", 6, dev, 0.5, 1.0)
    out = {}
    for ff in sorted(glob.glob(str(REPO_ROOT / a.frames / "*.pt"))):
        stem = Path(ff).stem
        fr = torch.load(ff, weights_only=False)
        d = torch.load(REPO_ROOT / "data/wm/verif_m5b" / f"{stem}.pt", weights_only=False)
        rich = REPO_ROOT / "data/critic3d/verif_m5b" / f"{stem}.pt"
        sym = torch.load(rich, weights_only=False)["sym"] if rich.exists() else None
        K = len(d["z"])
        zn = rk.norm.z(d["z"].float())
        qn = rk.norm.q(d["q"].float())
        valid = d["valid"].bool()
        lv = valid.long().cumsum(1).argmax(1)
        tid = rk.vocab.get(d["task"], 0)
        ZI, QI = [], []
        for i in range(0, K, 32):
            j = slice(i, min(K, i + 32))
            _, _, _, zs, qs = rk.wm.rollout(zn[j, 0], qn[j, 0], d["actions"][j].float())
            ZI.append(torch.stack(zs, 1)); QI.append(torch.stack(qs, 1))
        zi, qi = torch.cat(ZI), torch.cat(QI)
        ar = torch.arange(K)
        hi = (lv - 1).clamp_min(0)
        ti = torch.full((K,), tid, dtype=torch.long)
        c_wm = torch.sigmoid(rk.critic.logits(zi[ar, hi], qi[ar, hi], ti))
        c_true = torch.sigmoid(rk.critic.logits(zn[ar, lv], qn[ar, lv], ti))
        ci = torch.sigmoid(rk.critic.logits(zi.flatten(0, 1), qi.flatten(0, 1), ti.repeat_interleave(10))).view(K, 10)
        ct = torch.sigmoid(rk.critic.logits(zn[:, 1:].flatten(0, 1), qn[:, 1:].flatten(0, 1),
                                            ti.repeat_interleave(10))).view(K, 10)
        F = 11
        img_a = np.zeros((K, F, 128, 128, 3), np.uint8)
        img_w = np.zeros_like(img_a)
        a256 = np.zeros((K, 2, 256, 256, 3), np.uint8)
        w256 = np.zeros_like(a256)
        for k in range(K):
            fa, fw = fr["agent"][k], fr["wrist"][k]
            for f in range(F):
                img_a[k, f] = dec(fa[f], 128); img_w[k, f] = dec(fw[f], 128)
            a256[k, 0], a256[k, 1] = dec(fa[0]), dec(fa[int(lv[k])])
            w256[k, 0], w256[k, 1] = dec(fw[0]), dec(fw[int(lv[k])])
        # renderer check: replayed frames must match the stored trajectory (valid mask + success)
        vm = (fr["valid"].bool() == valid).float().mean().item()
        out[stem] = {"instance_id": d["instance_id"], "task": d["task"], "source": list(d["source"]),
                     "success": d["success"].clone(), "actions": d["actions"].clone(), "lv": lv,
                     "zt": zn.half(), "zi": zi.half(), "qt": qn, "qi": qi, "c_wm": c_wm, "c_true": c_true,
                     "ci_h": ci, "ct_h": ct, "img_a": torch.from_numpy(img_a), "img_w": torch.from_numpy(img_w),
                     "a256": torch.from_numpy(a256), "w256": torch.from_numpy(w256), "sym": sym,
                     "valid_match": vm}
        log.info("%s: K %d, valid match %.3f, mean c_wm %.3f c_true %.3f SR %.3f", stem[-60:], K, vm,
                 float(c_wm.mean()), float(c_true.mean()), float(d["success"].float().mean()))
    p = REPO_ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, p)
    log.info("wrote %d instances -> %s", len(out), p)


if __name__ == "__main__":
    main()
