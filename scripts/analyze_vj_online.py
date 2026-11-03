#!/usr/bin/env python
"""Online paired comparison: V-JEPA 2-AC verifier (configs/exp/vj_rq3_sweep.yaml) vs WM v1 verifier
(configs/exp/rq3_verifier_sweep.yaml), same M5b train instances (shards 0-1), same pools / continuation seeds.
python scripts/analyze_vj_online.py -> runs/_analysis/wm_vs_true/online_vj.md"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_verifier import ece, mcnemar, signflip  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def load(pat):
    d = pd.concat([pd.read_json(f, lines=True) for f in sorted(glob.glob(str(REPO / pat)))])
    return d.drop_duplicates(["instance_id", "arm"], keep="last")


vj, wm = load("runs/*_RQ3_vj_sweep_s*/per_instance.jsonl"), load("runs/*_RQ3_verif_sweep_s*/per_instance.jsonl")
common = sorted(set(vj.instance_id) & set(wm.instance_id))
md = [f"# Online: V-JEPA 2-AC vs WM v1 verifier, {len(common)} paired train instances", ""]
md += ["| arm | verifier | solved | hits / sim | mean C^pi of picks | FP rate | ECE |", "|---|---|---|---|---|---|---|"]
pairs = {}
for arm in ("wm_unc_l0", "wm_bud_l0"):
    for name, d in (("WM v1", wm), ("V-JEPA", vj)):
        g = d[(d.arm == arm) & d.instance_id.isin(common)].set_index("instance_id").loc[common]
        y = np.concatenate(g.order_success.to_list()).astype(bool)
        p = np.concatenate(g.order_pred.to_list())
        pos = p > 0.5
        md.append(f"| {arm} | {name} | {g.solved.mean():.3f} | {y.mean():.3f} | {p.mean():.3f} | "
                  f"{(~y[pos]).mean() if pos.any() else np.nan:.3f} | {ece(p, y):.3f} |")
        pairs[(arm, name)] = g
for arm in ("wm_unc_l0", "wm_bud_l0"):
    a, b = pairs[(arm, "V-JEPA")], pairs[(arm, "WM v1")]
    oa, ob, p = mcnemar(a.solved.to_numpy(), b.solved.to_numpy())
    dh = (a.hits - b.hits).to_numpy()
    md.append(f"\n{arm}: V-JEPA vs WM v1 solved only-VJ {oa} / only-WM {ob}, McNemar p {p:.3g}; "
              f"d hits/inst {dh.mean():+.2f}, sign-flip p {signflip(dh):.3g}")
md += ["", "## Exploitation curve (wm_bud_l0: top-4 by J per imagination budget N)", "",
       "| N | WM v1 mean C^pi | WM v1 real p@4 | WM v1 FP | V-JEPA mean C^pi | V-JEPA real p@4 | V-JEPA FP |", "|---|---|---|---|---|---|---|"]
for N in (8, 32, 128, 512):
    cells = []
    for name in ("WM v1", "V-JEPA"):
        g = pairs[("wm_bud_l0", name)]
        y = np.concatenate(g.order_success.to_list()).astype(bool)
        p = np.concatenate(g.order_pred.to_list())
        bud = np.concatenate(g.order_budget.to_list())
        m = bud == N
        pos = p[m] > 0.5
        cells += [f"{p[m].mean():.3f}", f"{y[m].mean():.3f}", f"{(~y[m][pos]).mean() if pos.any() else np.nan:.3f}"]
    md.append(f"| {N} | " + " | ".join(cells) + " |")
out = REPO / "runs/_analysis/wm_vs_true/online_vj.md"
out.write_text("\n".join(md) + "\n")
print("\n".join(md))
