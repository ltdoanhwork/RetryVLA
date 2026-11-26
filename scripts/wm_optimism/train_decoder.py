#!/usr/bin/env python
"""WM over-optimism, step 3 (visual aid only): latent -> pixel decoder for WM v1's DINOv2 tokens.

python -m scripts.wm_optimism.train_decoder [--device cuda|cpu] [--steps 6000]

A small decoder maps ONE camera's 36 normalized DINOv2-small tokens (6x6 pooled grid, 384-d, exactly the WM v1
state) to a 128 px RGB frame. Trained on REAL (frame, latent) pairs of the gallery instances (all re-rendered
verified bridges, every 5 env steps, agent + wrist camera, shared weights + camera embedding); the gallery cases'
own trajectories and 10 % of the other candidates are held out. At test time it decodes (a) the TRUE end latent
(= decoder fidelity) and (b) the WM-IMAGINED end latent (= the WM's "dream"). The decoder is a reconstruction aid:
what it shows is what a decoder trained on real frames reads out of the latent, not what the WM "renders".
Output: runs/_analysis/wm_optimism/cache/decoder.pt (+ decoder_log.json)
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from retryvla import REPO_ROOT
from retryvla.utils import setup_logging
from scripts.wm_optimism.common import select_cases

log = logging.getLogger("wm_opt_decoder")


class TokDecoder(nn.Module):
    def __init__(self, d_in: int = 384, w: int = 256, depth: int = 3):
        super().__init__()
        self.inp = nn.Linear(d_in, w)
        self.pos = nn.Parameter(torch.zeros(1, 36, w))
        self.cam = nn.Embedding(2, w)
        layer = nn.TransformerEncoderLayer(w, 8, 4 * w, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth)
        ch = [w, 192, 128, 96, 64]

        def up(ci, co):
            return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(ci, co, 3, padding=1),
                                 nn.GroupNorm(8, co), nn.SiLU(), nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co),
                                 nn.SiLU())
        self.ups = nn.Sequential(*[up(ch[i], ch[i + 1]) for i in range(4)])        # 8 -> 128
        self.out = nn.Conv2d(ch[-1], 3, 3, padding=1)

    def forward(self, tok: torch.Tensor, cam: torch.Tensor) -> torch.Tensor:
        x = self.inp(tok) + self.pos + self.cam(cam)[:, None]
        x = self.tf(x)
        b, p, w = x.shape
        x = x.transpose(1, 2).reshape(b, w, 6, 6)
        x = Fn.interpolate(x, size=(8, 8), mode="bilinear", align_corners=False)
        return torch.sigmoid(self.out(self.ups(x)))


def build_pairs(cache: dict, held: set):
    Z, I, C, split = [], [], [], []
    rng = np.random.default_rng(0)
    for stem, d in cache.items():
        K = len(d["success"])
        for k in range(K):
            val = (stem, k) in held or rng.random() < 0.10
            for f in range(int(d["lv"][k]) + 1):
                for cam, key in ((0, "img_a"), (1, "img_w")):
                    Z.append(d["zt"][k, f, cam * 36:(cam + 1) * 36])
                    I.append(d[key][k, f])
                    C.append(cam)
                    split.append(val)
    return torch.stack(Z), torch.stack(I), torch.as_tensor(C), torch.as_tensor(split)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/_analysis/wm_optimism/cache/gallery.pt")
    ap.add_argument("--out", default="runs/_analysis/wm_optimism/cache/decoder.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    cache = torch.load(REPO_ROOT / a.cache, weights_only=False)
    held = {(c["stem"], c["k"]) for c in select_cases(cache)}
    Z, I, C, S = build_pairs(cache, held)
    log.info("pairs: %d train / %d val (held-out gallery cases: %d)", int((~S).sum()), int(S.sum()), len(held))
    dev = torch.device(a.device)
    tr, va = torch.nonzero(~S).squeeze(1), torch.nonzero(S).squeeze(1)
    m = TokDecoder().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=4e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=4e-4, total_steps=a.steps, pct_start=0.05)
    hist, t0 = [], time.time()

    def batch(idx):
        x = Z[idx].float().to(dev)
        y = I[idx].to(dev).permute(0, 3, 1, 2).float().div(255)
        return x, C[idx].to(dev), y

    for step in range(1, a.steps + 1):
        idx = tr[torch.randint(len(tr), (a.bs,))]
        x, c, y = batch(idx)
        x = x + 0.05 * torch.randn_like(x)                     # small latent noise: robustness to imagined latents
        p = m(x, c)
        loss = (p - y).abs().mean() + 0.5 * ((Fn.avg_pool2d(p, 4) - Fn.avg_pool2d(y, 4)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if step % 250 == 0 or step == a.steps:
            m.eval()
            with torch.no_grad():
                vi = va[torch.randperm(len(va))[:512]]
                vl = []
                for j in range(0, len(vi), 128):
                    x, c, y = batch(vi[j:j + 128])
                    vl.append(float((m(x, c) - y).abs().mean()))
            m.train()
            hist.append({"step": step, "train_l1": float(loss), "val_l1": float(np.mean(vl)),
                         "s": round(time.time() - t0)})
            log.info("%s", hist[-1])
    torch.save({"model": m.state_dict(), "held": sorted(held), "hist": hist}, REPO_ROOT / a.out)
    (REPO_ROOT / a.out).with_name("decoder_log.json").write_text(json.dumps(hist, indent=1))
    log.info("saved %s", a.out)


if __name__ == "__main__":
    main()
