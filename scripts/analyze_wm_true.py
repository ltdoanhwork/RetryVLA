#!/usr/bin/env python
"""Step 1 report: WM-imagined vs TRUE end-state C^pi on the same sim-verified candidates.

python scripts/analyze_wm_true.py [--rows runs/_analysis/wm_vs_true/rows.parquet] [--out runs/_analysis/wm_vs_true]

(a) = C^pi v2 on the WM v1-imagined end latent, (b) = C^pi v2 on the TRUE end latent (real frames + real state),
y = real pi0-continuation outcome. Gap decomposition: mean(a) - SR = [mean(a) - mean(b)] (WM) + [mean(b) - SR] (critic).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_verifier import auc, ece  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def cal(p, y) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, bool)
    pos = p > 0.5
    return {"mean": float(p.mean()), "fp": float((~y[pos]).mean()) if pos.any() else np.nan, "pos": int(pos.sum()),
            "auc": auc(p, y), "ece": ece(p, y), "prec_pos": float(y[pos].mean()) if pos.any() else np.nan}


def p_at(d: pd.DataFrame, col: str, ks=(1, 4)) -> dict:
    out = {}
    for k in ks:
        v = [g.nlargest(k, col).success.mean() for _, g in d.groupby("instance_id") if len(g) >= k]
        out[k] = float(np.mean(v)) if v else np.nan
    return out


def arm_table(d: pd.DataFrame, key: str) -> list[dict]:
    rows = []
    for name, g in d.groupby(key, sort=False):
        a, b, y = g.c_wm.to_numpy(), g.c_true.to_numpy(), g.success.to_numpy()
        ca, cb = cal(a, y), cal(b, y)
        pa, pb = p_at(g, "c_wm"), p_at(g, "c_true")
        rows.append({key: name, "n": len(g), "inst": g.instance_id.nunique(), "sr": float(y.mean()),
                     **{f"a_{k}": v for k, v in ca.items()}, **{f"b_{k}": v for k, v in cb.items()},
                     "a_p1": pa[1], "a_p4": pa[4], "b_p1": pb[1], "b_p4": pb[4],
                     "gap_wm": float(a.mean() - b.mean()), "gap_critic": float(b.mean() - y.mean()),
                     "c_wm_trueq": float(g.c_wm_trueq.mean()), "c_truez_wmq": float(g.c_truez_wmq.mean()),
                     "mse_end": float(g.mse_end.mean())})
    return rows


def md_arm(rows: list[dict], key: str) -> list[str]:
    L = [f"| {key} | n | inst | real SR | mean C(a) | mean C(b) | gap WM (a-b) | gap critic (b-SR) | FP(a) | FP(b) | "
         "AUC(a) | AUC(b) | ECE(a) | ECE(b) | p@1 a / b | p@4 a / b |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r[key]} | {r['n']} | {r['inst']} | {r['sr']:.3f} | {r['a_mean']:.3f} | {r['b_mean']:.3f} | "
                 f"{r['gap_wm']:+.3f} | {r['gap_critic']:+.3f} | {r['a_fp']:.3f} | {r['b_fp']:.3f} | {r['a_auc']:.3f} | "
                 f"{r['b_auc']:.3f} | {r['a_ece']:.3f} | {r['b_ece']:.3f} | {r['a_p1']:.3f} / {r['b_p1']:.3f} | "
                 f"{r['a_p4']:.3f} / {r['b_p4']:.3f} |")
    return L


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="runs/_analysis/wm_vs_true/rows.parquet")
    ap.add_argument("--out", default="runs/_analysis/wm_vs_true")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(REPO / a.rows)
    d["group"] = np.where(d.arm.str.startswith("wm_bud"), d.arm + "@N" + d.pick_budget.astype(str), d.arm)
    md = ["# Step 1: is the verifier's over-optimism the WM's or the critic's? (WM v1 + C^pi v2)", "",
          "Same sim-verified candidates. (a) C^pi on the WM-imagined end latent; (b) C^pi on the TRUE end latent "
          "(DINOv2 of the real frames + real robot state after the bridge); y = real pi0 continuation outcome. "
          "FP = P(fail | C > 0.5); p@k = success of the per-instance top-k when ranked by (a) or by (b) within the "
          "same candidate set. c_wm_trueq / c_truez_wmq = critic on (imagined tokens, true state) / (true tokens, "
          "imagined state). Sources: verif_m5b (256 train instances x 64, latents stored at verification time, "
          "NVIDIA render) and true_end_rq3 (RQ3 sweep picks re-simulated, Mesa render; see control).", ""]
    js = {}
    for src, g in d.groupby("dir", sort=False):
        rows = arm_table(g, "group")
        js[src] = rows
        md += [f"## {src}: per arm (budget = imagined candidates behind the pick)", ""] + md_arm(rows, "group") + [""]
        allr = arm_table(g.assign(all="all"), "all")
        md += [f"pooled ({src}):", ""] + md_arm(allr, "all") + [""]
    # selection-free ranking power: within each instance, ALL candidates of all arms, ranked by (a) vs (b)
    md += ["## Ranking power on all candidates of an instance (selection mixes arms)", "",
           "| source | inst | mean SR | p@1 by (a) | p@1 by (b) | p@4 by (a) | p@4 by (b) | AUC(a) | AUC(b) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for src, g in d.groupby("dir", sort=False):
        pa, pb = p_at(g, "c_wm"), p_at(g, "c_true")
        md.append(f"| {src} | {g.instance_id.nunique()} | {g.success.mean():.3f} | {pa[1]:.3f} | {pb[1]:.3f} | "
                  f"{pa[4]:.3f} | {pb[4]:.3f} | {auc(g.c_wm, g.success):.3f} | {auc(g.c_true, g.success):.3f} |")
    md.append("")
    # WM latent error vs horizon (normalized DINO token MSE; copy = copy-the-failure-frame baseline)
    md += ["## WM v1 open-loop latent error vs horizon (normalized-token MSE; ratio = WM / copy-frame-0)", "",
           "| group | " + " | ".join(f"h{h} ({5 * h} st)" for h in (1, 2, 4, 6, 8, 10)) + " |",
           "|---|" + "---|" * 6]
    for (src, grp), g in d.groupby(["dir", "group"], sort=False):
        cells = []
        for h in (1, 2, 4, 6, 8, 10):
            e, c = g[f"ez{h}"].mean(), g[f"ec{h}"].mean()
            cells.append(f"{e:.3f} ({e / c:.2f})")
        md.append(f"| {src}:{grp} | " + " | ".join(cells) + " |")
    md.append("")
    # predicted positives: is the false positive explained by the WM error?
    md += ["## Among WM-predicted positives (C(a) > 0.5): what does the true end state say?", "",
           "| source | n pos(a) | SR | mean C(b) | share C(b) > 0.5 | SR if C(b) > 0.5 | SR if C(b) <= 0.5 | "
           "AUC(-mse_end -> success) |", "|---|---|---|---|---|---|---|---|"]
    for src, g in d.groupby("dir", sort=False):
        p = g[g.c_wm > 0.5]
        if not len(p):
            continue
        hb = p.c_true > 0.5
        md.append(f"| {src} | {len(p)} | {p.success.mean():.3f} | {p.c_true.mean():.3f} | {hb.mean():.3f} | "
                  f"{p.success[hb].mean():.3f} | {p.success[~hb].mean():.3f} | {auc(-p.mse_end, p.success):.3f} |")
    md.append("")
    # renderer control: replayed (Mesa) vs stored (NVIDIA) latents of the same verif_m5b candidates
    ctrl = REPO / "runs/_analysis/wm_vs_true/control_rows.parquet"
    if ctrl.exists():
        c = pd.read_parquet(ctrl)
        m = c.merge(d[d.dir == "verif_m5b"], on=["instance_id", "arm", "rank"], suffixes=("_mesa", ""))
        md += ["## Renderer control (same verif_m5b bridges re-simulated on the CPU shell, Mesa EGL + fp32 DINO)", "",
               f"n {len(m)}: mean |C(b)_mesa - C(b)_stored| {np.abs(m.c_true_mesa - m.c_true).mean():.4f}, "
               f"max {np.abs(m.c_true_mesa - m.c_true).max():.4f}; mean C(b) {m.c_true_mesa.mean():.3f} vs "
               f"{m.c_true.mean():.3f}; mean |C(a) diff| {np.abs(m.c_wm_mesa - m.c_wm).mean():.4f}.", ""]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    json.dump(js, open(out / "summary.json", "w"), indent=1, default=float)
    print("\n".join(md))


if __name__ == "__main__":
    main()
