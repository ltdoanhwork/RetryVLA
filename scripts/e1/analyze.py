#!/usr/bin/env python
"""E1 analysis (PLAN E1 + WM_3D_FOUNDATION.md sec. 4 decision rules), CPU.

python -m scripts.e1.analyze [--out runs/_analysis/e1]

Inputs: data/wm/e1/picks/<tag>.pt (imagination search), data/wm/e1/true/<tag>.pt (bridge replay),
runs/e1/wm/<tag>/summary.json (val open-loop ratios). Tags: WMv1 (shipped, reference) and <arm>_s<seed>.
Per (tag, instance, set): gap = mean(C_imag - C_true), Cb = mean C_true over the set's picks (C^pi v2; *_b = CRIT-V
base_s0). Comparisons X vs control (A2 / A2x3 vs A1, A1 / A3 vs A0, A4 vs A2, everything vs A0): per seed index
d_s, seed mean, instance-cluster bootstrap (1,000) of the seed-mean difference. Decision at set bud512.
Also: fixed-candidate gaps (verif_m5b random vs WM v1-picked candidates of the held-out instances), per family,
per camera token error, RQ3 / collect-group strata.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from retryvla import REPO_ROOT
from scripts.wm_optimism.cache_stats import action_family

ARMS = ["A0", "A1", "A2", "A2x3", "A3", "A4"]
COMPS = [("A2", "A1"), ("A2x3", "A1"), ("A1", "A0"), ("A3", "A0"), ("A4", "A2"), ("A4", "A0"), ("A2", "A0"),
         ("A2x3", "A0")]
SETS = ["bud32", "bud128", "bud512", "unc"]
GRASPED = 17


def rows_for(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    P = torch.load(REPO_ROOT / f"data/wm/e1/picks/{tag}.pt", weights_only=False)
    T = torch.load(REPO_ROOT / f"data/wm/e1/true/{tag}.pt", weights_only=False)
    rows, frows = [], []
    for iid, p in P.items():
        if iid not in T:
            continue
        t = T[iid]
        fam, flips, rough, _ = action_family(p["actions"].numpy())
        zm = np.nonzero(p["z_end_mask"])[0]
        zpos = {int(k): j for j, k in enumerate(zm)}
        for j, k in enumerate(t["idx"]):
            k = int(k)
            r = {"tag": tag, "instance_id": iid, "set": p["set"][k], "k": k, "fam": fam[k], "flips": int(flips[k]),
                 "c_imag": float(p["c_imag"][k]), "c_true": float(t["c_true"][j]),
                 "c_imag_b": float(p["c_imag_base"][k]), "c_true_b": float(t["c_true_base"][j]),
                 "prefix_success": bool(t["prefix_success"][j]), "grasped_end": float(t["sym_end"][j][GRASPED])}
            if k in zpos:
                zi = p["z_end"][zpos[k]].float()
                zt = t["z_end"][j].float()
                e = ((zi - zt) ** 2).mean(-1)
                r["err_agent"], r["err_wrist"] = float(e[:36].mean()), float(e[36:].mean())
            rows.append(r)
        f = p["fixed"]
        for k in range(len(f["source"])):
            frows.append({"tag": tag, "instance_id": iid, "source": f["source"][k], "success": bool(f["success"][k]),
                          "c_imag": float(f["c_imag"][k]), "c_true": float(f["c_true"][k]),
                          "c_imag_b": float(f["c_imag_base"][k]), "c_true_b": float(f["c_true_base"][k])})
    return pd.DataFrame(rows), pd.DataFrame(frows)


def inst_table(df: pd.DataFrame, s: str, crit: str = "") -> pd.DataFrame:
    d = df[df["set"] == s].copy()
    d["gap"] = d[f"c_imag{crit}"] - d[f"c_true{crit}"]
    d["cb"] = d[f"c_true{crit}"]
    return d.groupby(["tag", "instance_id"])[["gap", "cb"]].mean().reset_index()


def compare(it: pd.DataFrame, x: str, c: str, seeds, metric: str, B: int = 1000, rng_seed: int = 0) -> dict:
    """seed-paired difference x_s - c_s per instance; seed mean; instance-cluster bootstrap."""
    piv = it.pivot(index="instance_id", columns="tag", values=metric)
    cols = [(f"{x}_s{s}", f"{c}_s{s}") for s in seeds if f"{x}_s{s}" in piv and f"{c}_s{s}" in piv]
    if not cols:
        return {}
    D = np.stack([piv[a].values - piv[b].values for a, b in cols], 1)   # [I, S]
    D = D[~np.isnan(D).any(1)]
    per_seed = D.mean(0)
    m = float(D.mean())
    rng = np.random.default_rng(rng_seed)
    bs = np.array([D[rng.integers(0, len(D), len(D))].mean() for _ in range(B)])
    return {"d": m, "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)),
            "per_seed": [float(v) for v in per_seed], "n_inst": int(len(D)), "n_seeds": len(cols)}


def val_ratio(tag: str) -> float:
    p = REPO_ROOT / f"runs/e1/wm/{tag}/summary.json"
    if not p.exists():
        return float("nan")
    he = json.loads(p.read_text())["horizon_errors"]
    return float(he["10"]["ratio"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/_analysis/e1")
    a = ap.parse_args()
    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    tags = sorted({Path(f).stem for f in glob.glob(str(REPO_ROOT / "data/wm/e1/true/*.pt"))})
    parts = [rows_for(t) for t in tags]
    df = pd.concat([p[0] for p in parts], ignore_index=True)
    fx = pd.concat([p[1] for p in parts], ignore_index=True)
    df.to_parquet(out / "rows.parquet"); fx.to_parquet(out / "fixed.parquet")
    df["arm"] = df.tag.str.replace(r"_s\d+$", "", regex=True)
    fx["arm"] = fx.tag.str.replace(r"_s\d+$", "", regex=True)
    split = json.loads((REPO_ROOT / "configs/exp/e1_split.json").read_text())
    rq3 = {Path(f).stem for f in glob.glob(str(REPO_ROOT / "data/wm/true_end_rq3/*.pt"))}
    cgrp = {"__".join(Path(f).stem.split("__")[:2]) for f in glob.glob(str(REPO_ROOT / "data/wm/collect_v1/*.pt"))}
    seeds = [1, 2, 3]
    L = []
    L.append("# E1: WM-DAgger x 3D/contact aux heads, self-exploitation on 75 group-disjoint held-out instances\n")
    # 1. per arm summary (seed mean), every set
    L.append("## 1. Own-pick gap and true quality (C^pi v2), seed mean over available seeds\n")
    L.append("| arm | seeds | val h10 ratio | " + " | ".join(f"gap {s} | Cb {s}" for s in SETS) + " |")
    L.append("|---|---|---|" + "---|---|" * len(SETS))
    summ = {}
    for arm in ["WMv1"] + ARMS:
        tg = sorted(set(df.tag[df.arm == arm]))
        if not tg:
            continue
        cells = []
        for s in SETS:
            it = inst_table(df[df.tag.isin(tg)], s)
            g = it.groupby("tag")[["gap", "cb"]].mean()
            cells += [f"{g.gap.mean():+.3f}", f"{g.cb.mean():.3f}"]
            summ.setdefault(arm, {})[s] = {"gap": float(g.gap.mean()), "cb": float(g.cb.mean())}
        vr = np.nanmean([val_ratio(t) for t in tg]) if arm != "WMv1" else 0.356
        summ[arm]["val_h10"] = float(vr)
        L.append(f"| {arm} | {len(tg)} | {vr:.3f} | " + " | ".join(cells) + " |")
    # 2. pre-registered comparisons at bud512 (+ the other sets, descriptive)
    L.append("\n## 2. Paired comparisons (seed-paired, instance-cluster bootstrap 95 % CI)\n")
    L.append("| X vs control | set | d gap_own [CI] | per-seed d gap | d Cb_own [CI] | d gap (base critic) [CI] | d val h10 (rel) |")
    L.append("|---|---|---|---|---|---|---|")
    dec = {}
    for x, c in COMPS:
        for s in ["bud512", "bud128", "unc"]:
            it = inst_table(df, s)
            itb = inst_table(df, s, "_b")
            rg = compare(it, x, c, seeds, "gap")
            if not rg:
                continue
            rc = compare(it, x, c, seeds, "cb")
            rb = compare(itb, x, c, seeds, "gap")
            vx = np.nanmean([val_ratio(f"{x}_s{k}") for k in seeds]); vc = np.nanmean([val_ratio(f"{c}_s{k}") for k in seeds])
            L.append(f"| {x} vs {c} | {s} | {rg['d']:+.3f} [{rg['lo']:+.3f}, {rg['hi']:+.3f}] | "
                     f"{', '.join(f'{v:+.3f}' for v in rg['per_seed'])} | {rc['d']:+.3f} [{rc['lo']:+.3f}, {rc['hi']:+.3f}] | "
                     f"{rb['d']:+.3f} [{rb['lo']:+.3f}, {rb['hi']:+.3f}] | {(vx - vc) / vc:+.1%} |")
            if s == "bud512":
                ok_gap = rg["d"] <= -0.10 and rg["hi"] < 0 and all(v < 0 for v in rg["per_seed"]) and rg["n_seeds"] == 3
                ok_cb = rc["d"] >= 0.03 and rc["lo"] > 0
                ok_val = (vx - vc) / vc <= 0.05 if np.isfinite(vx) and np.isfinite(vc) else False
                verdict = ("PASS" if ok_gap and ok_cb and ok_val else
                           "PROMISING" if rg["d"] <= -0.05 and rg["hi"] < 0 else
                           "NULL" if abs(rg["d"]) < 0.05 else "OTHER")
                dec[f"{x}_vs_{c}"] = {"gap": rg, "cb": rc, "gap_base": rb, "val_rel": float((vx - vc) / vc),
                                      "verdict": verdict}
    L.append("\nPre-registered verdicts at bud512: " + ", ".join(f"{k} **{v['verdict']}**" for k, v in dec.items()))
    close = all(k in dec and abs(dec[k]["gap"]["d"]) < 0.05 for k in ("A2_vs_A1", "A3_vs_A0"))
    L.append(f"\nCLOSE rule (|d gap| < 0.05 for A2 vs A1 and A3 vs A0): {close}")
    # 3. fixed candidates (independence effect + sanity)
    L.append("\n## 3. Fixed candidates of the held-out instances (verif_m5b: 16 random + 48 picked by WM v1), gap = C_imag - C_true\n")
    L.append("| arm | random gap | random |gap|<0.05 | WM v1-picked gap | WM v1-picked FP (C_imag > 0.5, fail) |")
    L.append("|---|---|---|---|---|")
    for arm in ["WMv1"] + ARMS:
        d = fx[fx.arm == arm]
        if not len(d):
            continue
        g = d.c_imag - d.c_true
        rnd = d.source == "random"
        pk = ~rnd
        fp = ((d.c_imag > 0.5) & ~d.success)[pk].sum() / max(1, (d.c_imag > 0.5)[pk].sum())
        L.append(f"| {arm} | {g[rnd].mean():+.3f} | {(g[rnd].abs() < 0.05).mean():.2f} | {g[pk].mean():+.3f} | {fp:.3f} |")
    # 4. per family / camera / strata at bud512 (seed-pooled, descriptive)
    d5 = df[df.set == "bud512"].copy()
    d5["gap"] = d5.c_imag - d5.c_true
    L.append("\n## 4. bud512 picks by proposal family (heuristic), seed-pooled: gap (n)\n")
    fams = ["perturb", "pi0_s1.5", "prim+cem", "pi0_s1.0", "prim"]
    L.append("| arm | " + " | ".join(fams) + " | grasped at end (true) | agent tok err | wrist tok err |")
    L.append("|---|" + "---|" * (len(fams) + 3))
    for arm in ["WMv1"] + ARMS:
        d = d5[d5.arm == arm]
        if not len(d):
            continue
        cells = [f"{d.gap[d.fam == f].mean():+.2f} ({int((d.fam == f).sum())})" for f in fams]
        L.append(f"| {arm} | " + " | ".join(cells) + f" | {d.grasped_end.mean():.3f} | {d.err_agent.mean():.3f} | {d.err_wrist.mean():.3f} |")
    L.append("\n## 5. Strata (bud512 gap, seed mean): RQ3 instances / held-out groups also in collect_v1\n")
    L.append("| arm | RQ3 (n) | not RQ3 | group in collect_v1 | group not in collect_v1 |")
    L.append("|---|---|---|---|---|")
    d5["rq3"] = d5.instance_id.isin(rq3)
    d5["cg"] = d5.instance_id.map(lambda i: "__".join(i.split("__")[:2]) in cgrp)
    for arm in ["WMv1"] + ARMS:
        d = d5[d5.arm == arm]
        if not len(d):
            continue
        L.append(f"| {arm} | {d.gap[d.rq3].mean():+.3f} ({d.instance_id[d.rq3].nunique()}) | {d.gap[~d.rq3].mean():+.3f} | "
                 f"{d.gap[d.cg].mean():+.3f} ({d.instance_id[d.cg].nunique()}) | {d.gap[~d.cg].mean():+.3f} |")
    (out / "e1.md").write_text("\n".join(L) + "\n")
    (out / "decision.json").write_text(json.dumps({"summary": summ, "decisions": dec, "close": bool(close)}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
