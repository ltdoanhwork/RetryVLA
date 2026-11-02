#!/usr/bin/env python
"""Step 1 of the foundation-WM question: is the verifier's over-optimism the WM's fault or the critic's?

python -m scripts.diag_wm_true [--dirs verif_m5b] [--replay_dirs data/wm/true_end_rq3] [--out runs/_analysis/wm_vs_true]

For the SAME sim-verified candidates (real pi0-continuation outcome known) compute C^pi v2 on
  (a) the WM v1-imagined end latent  (open-loop rollout of the 50-step bridge from the failure-state latent), and
  (b) the TRUE end latent             (DINOv2 encoding of the real rendered frames + real robot state after the
                                       bridge was executed in sim; stored in the full verif logs, or re-simulated by
                                       scripts/replay_bridges.py for the slim RQ3-sweep logs),
plus the WM's latent prediction error vs horizon (normalized-token MSE of the open-loop rollout vs the true frame
every 5 env steps; copy-last-frame baseline). One row per candidate -> rows.parquet (no latents stored).
The imagined state is taken at the SAME horizon as the true end latent (bridges that succeed early end early),
exactly like retryvla.critic.train_critic.end_latents.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from retryvla import REPO_ROOT
from retryvla.search.wm_search import WMCriticRanker
from retryvla.utils import setup_logging

log = logging.getLogger("diag_wm_true")


@torch.no_grad()
def rows_for_file(rk: WMCriticRanker, d: dict, dname: str, bs: int = 64) -> list[dict]:
    dev = rk.dev
    z, q, valid = d["z"], d["q"].float(), d["valid"].bool()
    acts = d["actions"]
    if isinstance(acts, list) or acts.shape[1] != 50 or z.shape[1] != 11:
        return []
    K = len(z)
    tid = rk.vocab.get(d["task"], 0)
    lv = valid.long().cumsum(1).argmax(1)                      # last valid frame
    src = list(d["source"])
    rank, seen = [], {}
    for s in src:
        rank.append(seen.get(s, 0)); seen[s] = seen.get(s, 0) + 1
    out = []
    for i in range(0, K, bs):
        j = slice(i, min(K, i + bs))
        zn = rk.norm.z(z[j].to(dev).float())                   # [B, F, P, D]
        qn = rk.norm.q(q[j].to(dev))
        n = zn.shape[0]
        ar = torch.arange(n, device=dev)
        lvj = lv[j].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            zH, qH, _, zs, qs = rk.wm.rollout(zn[:, 0], qn[:, 0], acts[j].to(dev).float())
        Zs, Qs = torch.stack(zs, 1).float(), torch.stack(qs, 1).float()   # [B, 10, P, D]
        hi = (lvj - 1).clamp_min(0)
        zi, qi = Zs[ar, hi], Qs[ar, hi]
        zt, qt = zn[ar, lvj], qn[ar, lvj]
        ti = torch.full((n,), tid, device=dev, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            c_wm = torch.sigmoid(rk.critic.logits(zi, qi, ti)).float()
            c_true = torch.sigmoid(rk.critic.logits(zt, qt, ti)).float()
            c_wm_q = torch.sigmoid(rk.critic.logits(zi, qt, ti)).float()   # imagined image tokens + TRUE state
            c_wm_z = torch.sigmoid(rk.critic.logits(zt, qi, ti)).float()   # TRUE image tokens + imagined state
        vf = valid[j].to(dev)
        e_z = ((Zs - zn[:, 1:]) ** 2).mean((2, 3))             # [B, 10]
        e_c = ((zn[:, :1] - zn[:, 1:]) ** 2).mean((2, 3))
        e_q = ((Qs - qn[:, 1:]) ** 2).mean(2)
        e_end = ((zi - zt) ** 2).mean((1, 2))
        cos_end = torch.nn.functional.cosine_similarity(zi.mean(1), zt.mean(1), dim=-1)
        for b in range(n):
            k = i + b
            r = {"dir": dname, "instance_id": d["instance_id"], "task": d["task"], "arm": src[k], "rank": rank[k],
                 "success": bool(d["success"][k]), "prefix_success": bool(d["prefix_success"][k]),
                 "pred_logged": float(d["pred_value"][k]), "last_valid": int(lvj[b]),
                 "c_wm": float(c_wm[b]), "c_true": float(c_true[b]), "c_wm_trueq": float(c_wm_q[b]),
                 "c_truez_wmq": float(c_wm_z[b]), "mse_end": float(e_end[b]), "cos_end": float(cos_end[b]),
                 "pick_budget": int(d["pick_budget"][k]) if "pick_budget" in d else -1}
            for h in range(10):
                ok = bool(vf[b, h + 1])
                r[f"ez{h + 1}"] = float(e_z[b, h]) if ok else np.nan
                r[f"ec{h + 1}"] = float(e_c[b, h]) if ok else np.nan
                r[f"eq{h + 1}"] = float(e_q[b, h]) if ok else np.nan
            out.append(r)
    return out


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=["verif_m5b"], help="full verif logs under data/wm (z stored)")
    ap.add_argument("--replay_dirs", nargs="*", default=[], help="scripts/replay_bridges.py outputs (repo-relative)")
    ap.add_argument("--out", default="runs/_analysis/wm_vs_true/rows.parquet")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_files", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(a.device)
    rk = WMCriticRanker((REPO_ROOT / "data/wm/latest_wm.txt").read_text().strip(),
                        (REPO_ROOT / "data/wm/latest_critic_v2.txt").read_text().strip(),
                        "facebook/dinov2-small", 6, dev, 0.5, 1.0)
    rows = []
    dirs = [(x, REPO_ROOT / "data/wm" / x) for x in a.dirs] + [(Path(x).name, REPO_ROOT / x) for x in a.replay_dirs]
    for name, dd in dirs:
        fs = sorted(glob.glob(str(dd / "*.pt")))[: a.max_files or None]
        for n, f in enumerate(fs):
            d = torch.load(f, weights_only=False)  # no mmap: page-fault reads on /pfss ran at ~0.4 MB/s
            if d.get("slimmed"):
                continue
            rows += rows_for_file(rk, d, name)
            if n % 32 == 0:
                log.info("%s %d/%d files, %d rows", name, n + 1, len(fs), len(rows))
    out = REPO_ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    pd.DataFrame(rows).to_parquet(tmp)
    os.replace(tmp, out)
    log.info("wrote %d rows -> %s", len(rows), out)


if __name__ == "__main__":
    main()
