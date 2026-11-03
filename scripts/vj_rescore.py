#!/usr/bin/env python
"""Offline paired comparison, V-JEPA 2-AC verifier vs WM v1 verifier on the SAME sim-verified candidates (0 new sim).

python -m scripts.vj_rescore --sets data/foundation/vj_latents/verif_m5b:verif_m5b data/foundation/vj_latents/rq3:true_end_rq3
    [--out runs/_analysis/wm_vs_true/rows_vj.parquet]

Input = scripts/vjepa_encode.py --ends outputs (frame 0 + true end frame, V-JEPA pooled latents) of re-simulated
bridges. Per candidate: C^pi_VJ on the V-JEPA-imagined end latent (a_vj; fine-tuned AC predictor, open loop from the
failure frame at the same horizon as the true end) and on the TRUE end latent (b_vj). Rows carry (dir, instance_id,
arm, rank) so scripts/analyze_wm_true.py can join them with the DINO rows (a, b) of the same candidates.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd
import torch

from retryvla import REPO_ROOT
from retryvla.utils import setup_logging
from retryvla.wm.vjepa2_ac import VJRanker

log = logging.getLogger("vj_rescore")


@torch.no_grad()
def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True, help="latent_dir:dir_label (dir_label = rows.parquet 'dir')")
    ap.add_argument("--wm_link", default="data/foundation/latest_vj_wm.txt")
    ap.add_argument("--critic_link", default="data/foundation/latest_vj_critic.txt")
    ap.add_argument("--out", default="runs/_analysis/wm_vs_true/rows_vj.parquet")
    a = ap.parse_args()
    dev = torch.device("cuda")

    class _NoEnc:  # latents are precomputed: no ViT-g needed here
        pass

    rk = VJRanker((REPO_ROOT / a.wm_link).read_text().strip(), (REPO_ROOT / a.critic_link).read_text().strip(), dev,
                  encoder=_NoEnc())
    rows = []
    for spec in a.sets:
        root, label = spec.split(":")
        fs = sorted(glob.glob(str(REPO_ROOT / root / "*.pt")))
        for n, f in enumerate(fs):
            d = torch.load(f, weights_only=False)
            z, q, v = d["z"], d["q"].float(), d["valid"].bool()
            K = len(z)
            tid = rk.vocab.get(d["task"], 0)
            lv_s = v.long().cumsum(1).argmax(1)
            lv = d["last_valid"]
            ar = torch.arange(K)
            zt, qt = z[ar, lv_s].to(dev).float(), rk.norm.q(q[ar, lv_s].to(dev))
            z0, q0 = z[:, 0].to(dev).float(), rk.norm.q(q[:, 0].to(dev))
            ZI, QI = [], []
            for i in range(0, K, 64):
                j = slice(i, min(K, i + 64))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, _, _, zs, qs = rk.wm.rollout(z0[j], q0[j], d["actions"][j].to(dev).float())
                Zs, Qs = torch.stack(zs, 1).float(), torch.stack(qs, 1).float()
                hi = (lv[j] - 1).clamp_min(0).to(dev)
                aa = torch.arange(len(hi), device=dev)
                ZI.append(Zs[aa, hi]); QI.append(Qs[aa, hi])
            zi, qi = torch.cat(ZI), torch.cat(QI)
            ca = rk._values(zi, qi, tid).cpu().numpy()
            cb = rk._values(zt, qt, tid).cpu().numpy()
            mse = ((zi - zt) ** 2).mean((1, 2)).cpu().numpy()
            src = list(d["source"])
            rank, seen = [], {}
            for s in src:
                rank.append(seen.get(s, 0)); seen[s] = seen.get(s, 0) + 1
            for k in range(K):
                rows.append({"dir": label, "instance_id": d["instance_id"], "arm": src[k], "rank": rank[k],
                             "success": bool(d["success"][k]), "c_vj_wm": float(ca[k]), "c_vj_true": float(cb[k]),
                             "vj_mse_end": float(mse[k])})
            if n % 32 == 0:
                log.info("%s %d/%d, %d rows", label, n + 1, len(fs), len(rows))
    out = REPO_ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    pd.DataFrame(rows).to_parquet(tmp)
    os.replace(tmp, out)
    log.info("wrote %d rows -> %s", len(rows), out)


if __name__ == "__main__":
    main()
