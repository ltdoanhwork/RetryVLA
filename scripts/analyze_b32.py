#!/usr/bin/env python
"""B3.2 RISE-inspired baseline: analysis.  python scripts/analyze_b32.py -> runs/_analysis/b32/summary.md

1. Policy comparisons on the Spatial test (dev) split, unit = test scene (stats_cluster.compare: scene x seed
   bootstrap CI + paired sign-flip p): each B3.2 arm vs pi1 (1 seed, and 3 seeds), vs demo-only SFT, vs pi1-nominal.
2. Nominal SR of every arm.
3. Imagined-label quality (analysis only, never used in training): candidates that coincide with chunks the PoC
   verified in M5b carry the real outcome -> success rate of imagined positives vs negatives, AUC of A.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from stats_cluster import compare, eps  # noqa: E402

OUT = REPO / "runs/_analysis/b32"
ARMS = {"B32 cond (pool)": "B32_rise_cond_pool", "B32 AWR (pool)": "B32_rise_awr_pool",
        "B32 cond (VLA samples)": "B32_rise_cond_vla"}
PI1 = ["M5_m5c_sft_wm_diverse_rr_recover_test", "M8_seed_pi1_s1_recover_test", "M8_seed_pi1_s2_recover_test"]


def nominal(tag: str):
    f = sorted(glob.glob(str(REPO / "runs" / f"*_{tag}" / "summary.json")))
    if not f:
        return None
    return json.load(open(f[-1]))["results"].get("success_rate")


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = s.argsort().argsort() + 1.0
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def main() -> None:
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    L = ["# B3.2 RISE-inspired imagination-only baseline", "",
         "Unit = test scene (56 x 10 trials); CI = scene (x seed) bootstrap; p = paired sign-flip on scenes.", "",
         "| comparison | seeds A / B | diff | 95% CI | p |", "|---|---|---|---|---|"]
    refs = [("pi1 (seed 0)", PI1[:1]), ("pi1 (3 seeds)", PI1), ("demo-only SFT", ["M9_sft_demo_only_recover_test"])]
    for name, tag in ARMS.items():
        A = eps(f"{tag}_recover_test")
        if A is None:
            L.append(f"| {name} | missing | | | |")
            continue
        L.append(f"| {name}: RSR {np.mean(list(A.values())):.3f} | | | | |")
        for rn, rt in refs:
            Bm = [e for e in (eps(t) for t in rt) if e]
            if not Bm:
                continue
            d, ci, p, _, sa, sb = compare([A], Bm, rng)
            L.append(f"| {name} vs {rn} | {sa} / {sb} | {d:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {p:.2g} |")
    L += ["", "| policy | nominal SR |", "|---|---|"]
    for name, tag in {**ARMS, "pi1": "M5_m5c_sft_wm_diverse_rr", "pi1-nominal": "M1_pi1nom", "demo-only": "M9_sft_demo_only"}.items():
        L.append(f"| {name} | {nominal(tag + '_nominal')} |")

    L += ["", "## Imagined labels vs real outcomes (PoC-verified chunks; analysis only)", "",
          "| data | matched chunks | real SR all | real SR imagined-positive | real SR imagined-negative | AUC(A) |",
          "|---|---|---|---|---|---|"]
    for prop in ("pool", "vla"):
        A_, Y, P_, N_ = [], [], [], []
        for f in sorted(glob.glob(str(REPO / f"data/recovery/b32_rise/{prop}/inst/*.pt"))):
            d = torch.load(f, map_location="cpu", weights_only=False)
            lab, adv = np.asarray(d["poc_verified_analysis_only"]), np.asarray(d["adv"])
            m = lab >= 0
            if not m.any():
                continue
            hi = np.quantile(adv, 0.7)
            pos = (adv >= hi) & (adv > 0)
            A_ += list(adv[m]); Y += list(lab[m] == 1)
            P_ += list(lab[m & pos] == 1); N_ += list(lab[m & ~pos] == 1)
        if Y:
            L.append(f"| {prop} | {len(Y)} | {np.mean(Y):.3f} | {np.mean(P_) if P_ else float('nan'):.3f} (n={len(P_)}) | "
                     f"{np.mean(N_) if N_ else float('nan'):.3f} (n={len(N_)}) | {auc(A_, Y):.3f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
