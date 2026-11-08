#!/usr/bin/env python
"""RGR sweep table: python scripts/analyze_rgr.py [--root data/recovery/rgr_sweep] -> runs/_analysis/rgr/rgr.md

Per arm (paired instances = those every arm finished): solved, hits / verification, hits per 1k sim steps
(verification + segment execution), FP (imagined C^pi >= 0.5 but sim fails), mean imagined C^pi vs real SR, TRUE
geometric validity of the verified chains (measurement), #bridges per verified success, sim / imagination cost,
re-plan rate, observed vs imagined C^pi of s_1 (recursive arms); McNemar (solved) + sign-flip (hits) vs `single`;
solved per official level (L1 / L2 / L3; train has no L4)."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[1]


def signflip(d: np.ndarray, n: int = 20000, seed: int = 0) -> float:
    d = d[d != 0]
    if not len(d):
        return 1.0
    rng = np.random.default_rng(seed)
    s = rng.choice([-1, 1], size=(n, len(d))) @ np.abs(d)
    return float((np.abs(s) >= abs(d.sum())).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/recovery/rgr_sweep")
    ap.add_argument("--ref", default="single")
    ap.add_argument("--out", default="runs/_analysis/rgr")
    a = ap.parse_args()
    rows = [json.loads(l) for f in glob.glob(str(REPO / a.root / "**/rows.jsonl"), recursive=True)
            for l in open(f) if l.strip()]
    d = pd.DataFrame(rows).drop_duplicates(["instance_id", "arm"], keep="last")
    lv = yaml.safe_load(open(REPO / "configs/env/levels_spatial.yaml"))["levels"]
    d["level"] = d.scene.map(lambda s: lv.get(s, {}).get("level", "unk"))
    d["level_src"] = d.scene.map(lambda s: lv.get(s, {}).get("src", "none"))
    arms = list(dict.fromkeys(d.arm))
    n_arm = d.groupby("instance_id").arm.nunique()
    keep = n_arm[n_arm == len(arms)].index
    d = d[d.instance_id.isin(keep)]
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    L = [f"# RGR sweep ({len(keep)} paired train instances, arms: {', '.join(arms)})", ""]
    L += ["| arm | solved | hits / verif | hits / 1k sim | FP (pred>=0.5) | mean pred vs SR | true-valid chains "
          "(all / succ) | bridges / succ | sim steps (verif + exec) | imagined | replan | C(s1) hat vs obs | "
          "vs single: McNemar n01/n10 p | hits diff p |", "|" + "---|" * 14]
    ref = d[d.arm == a.ref].set_index("instance_id")
    lev_rows = []
    for arm in arms:
        x = d[d.arm == arm].set_index("instance_id")
        succ = np.concatenate([np.array(v, bool) for v in x.order_success]) if len(x) else np.zeros(0, bool)
        pred = np.concatenate([np.array(v, float) for v in x.order_pred])
        ln = np.concatenate([np.array(v, int) for v in x.order_len])
        tv = np.concatenate([np.array([np.nan if t is None else float(t) for t in v]) for v in x.order_true_valid])
        sim = (x.sim_steps_verify + x.sim_steps_exec).sum()
        pos = pred >= 0.5
        fp = float((~succ[pos]).mean()) if pos.any() else float("nan")
        rp = (x.n_replan.sum() / max(1, x.n_exec.sum())) if "n_replan" in x and x.n_replan.notna().any() else float("nan")
        v1h = [u for v in x.get("v1_hat", pd.Series(dtype=object)).dropna() for u in v]
        v1r = [u for v in x.get("v1_real", pd.Series(dtype=object)).dropna() for u in v]
        cmp_ = "-"
        hp = "-"
        if arm != a.ref:
            j = x.index.intersection(ref.index)
            s1, s0 = x.loc[j, "solved"].astype(bool), ref.loc[j, "solved"].astype(bool)
            n01, n10 = int((s1 & ~s0).sum()), int((~s1 & s0).sum())
            p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            cmp_ = f"{n01}/{n10} p {p:.3f}"
            hd = (x.loc[j, "hits"] - ref.loc[j, "hits"]).to_numpy()
            hp = f"{hd.mean():+.2f}, p {signflip(hd):.3f}"
        L.append(f"| {arm} | {x.solved.mean():.3f} | {succ.mean():.3f} | {1000 * succ.sum() / max(1, sim):.2f} | "
                 f"{fp:.3f} | {pred.mean():.3f} / {succ.mean():.3f} | {np.nanmean(tv):.3f} / "
                 f"{np.nanmean(tv[succ]) if succ.any() else float('nan'):.3f} | "
                 f"{ln[succ].mean() if succ.any() else float('nan'):.2f} | {sim / len(x):.0f} "
                 f"({x.sim_steps_verify.mean():.0f} + {x.sim_steps_exec.mean():.0f}) | {x.n_imag.mean():.0f} | "
                 f"{rp:.2f} | {np.mean(v1h) if v1h else float('nan'):.2f} / {np.mean(v1r) if v1r else float('nan'):.2f} | "
                 f"{cmp_} | {hp} |")
        for lev, g in x.groupby("level"):
            lev_rows.append({"arm": arm, "level": lev, "n": len(g), "solved": g.solved.mean(),
                             "hits_per_verif": g.hits.sum() / max(1, g.budget.sum())})
    L += ["", "Per official level (official + inherited labels; train has no L4):", ""]
    lt = pd.DataFrame(lev_rows)
    if len(lt):
        piv = lt.pivot(index="arm", columns="level", values="solved").reindex(arms)
        ns = lt.groupby("level").n.first()
        L.append("| arm | " + " | ".join(f"{c} solved (n={ns[c]})" for c in piv.columns) + " |")
        L.append("|" + "---|" * (len(piv.columns) + 1))
        for arm, r in piv.iterrows():
            L.append(f"| {arm} | " + " | ".join(f"{v:.3f}" for v in r.values) + " |")
        # L3 paired test vs ref
        L += ["", "L3 paired vs single (McNemar on solved):", ""]
        for arm in arms:
            if arm == a.ref:
                continue
            x = d[(d.arm == arm) & (d.level == "L3")].set_index("instance_id")
            r0 = d[(d.arm == a.ref) & (d.level == "L3")].set_index("instance_id")
            j = x.index.intersection(r0.index)
            s1, s0 = x.loc[j, "solved"].astype(bool), r0.loc[j, "solved"].astype(bool)
            n01, n10 = int((s1 & ~s0).sum()), int((~s1 & s0).sum())
            p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            L.append(f"- {arm}: {s1.mean():.3f} vs {s0.mean():.3f} (n {len(j)}), n01/n10 {n01}/{n10}, p {p:.3f}")
    (out / "rgr.md").write_text("\n".join(L) + "\n")
    d.to_parquet(out / "rows.parquet")
    # pre-registered distill rule (2026-11-08, before any result): best non-oracle arm by solved-rate gain over
    # `single`; distill only if n01 > n10 with McNemar p < 0.10 and its L3 solved rate is not below single's
    dec = {"arm": None, "candidates": {}}
    r0 = d[d.arm == a.ref].set_index("instance_id")
    for arm in arms:
        if arm == a.ref or "oracle" in arm:
            continue
        x = d[d.arm == arm].set_index("instance_id")
        j = x.index.intersection(r0.index)
        s1, s0 = x.loc[j, "solved"].astype(bool), r0.loc[j, "solved"].astype(bool)
        n01, n10 = int((s1 & ~s0).sum()), int((~s1 & s0).sum())
        p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        l3 = x.loc[j][x.loc[j, "level"] == "L3"].solved.mean() - r0.loc[j][r0.loc[j, "level"] == "L3"].solved.mean()
        dec["candidates"][arm] = {"d_solved": float(s1.mean() - s0.mean()), "n01": n01, "n10": n10, "p": float(p),
                                  "d_l3": float(l3), "ok": bool(n01 > n10 and p < 0.10 and l3 >= 0)}
    ok = {k: v for k, v in dec["candidates"].items() if v["ok"]}
    if ok:
        dec["arm"] = max(ok, key=lambda k: ok[k]["d_solved"])
    dec["n_instances"] = int(len(keep))
    (out / "decision.json").write_text(json.dumps(dec, indent=1))
    print("\n".join(L))
    print("DECISION", json.dumps(dec))


if __name__ == "__main__":
    main()
