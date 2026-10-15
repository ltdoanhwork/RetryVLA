#!/usr/bin/env python
"""Aggregate M5b (paired search-arm comparison) across shards/parts and test the differences.

python scripts/analyze_m5b.py [--out runs/_analysis/m5b]

Per arm: solve rate (>=1 verified success / 16 sims), hit rate (successes / sims), with instance-level
bootstrap 95% CIs. Paired tests on the SAME instances: exact McNemar on `solved`, sign-flip permutation
test on per-instance hits. Data-distribution metrics of the verified successes (jerk, roughness,
distance to SmolVLA's manifold, pairwise spread, strategy cells), task coverage, critic calibration.
"""

from __future__ import annotations

import argparse
import glob
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def load() -> pd.DataFrame:
    fs = sorted(glob.glob(str(REPO / "runs" / "*_M5_m5b_compare_arms_s*" / "per_instance.jsonl")))
    rows = [json.loads(l) for f in fs for l in open(f)]
    d = pd.DataFrame(rows)
    if d.empty:
        raise SystemExit("no M5b results yet")
    return d.drop_duplicates(["instance_id", "arm"], keep="last")


def boot_ci(x: np.ndarray, f=np.mean, n=4000, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    s = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n01, n10, p


def signflip(diff: np.ndarray, n=20000, seed=0) -> float:
    rng = np.random.default_rng(seed)
    obs = abs(diff.mean())
    s = rng.choice([-1, 1], size=(n, len(diff))) * diff[None]
    return float((np.abs(s.mean(1)) >= obs - 1e-12).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/_analysis/m5b")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    d = load()
    arms = [x for x in ["random", "wm_diverse", "wm_noise", "wm_qd"] if x in set(d.arm)]
    full = d.groupby("instance_id").arm.nunique()
    d = d[d.instance_id.isin(full[full == len(arms)].index)]  # only instances where every arm finished
    P = {k: d.pivot(index="instance_id", columns="arm", values=k)[arms] for k in ("solved", "hits", "budget")}
    N = len(P["solved"])
    res = {"n_instances": N, "arms": {}}
    lines = [f"# M5b paired search-arm comparison (N = {N} pi0-failure instances, 16 sims / arm)", "",
             "| arm | solve rate [95% CI] | hit rate [95% CI] | trajectories | jerk (succ) | manifold dist (succ) | "
             "pairwise (succ) | cells / solved inst | critic V succ / fail |", "|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        s = P["solved"][arm].astype(bool).values
        h = (P["hits"][arm] / P["budget"][arm]).values
        da = d[d.arm == arm]
        sc = da[da.hits > 0]
        r = {"solve": float(s.mean()), "solve_ci": boot_ci(s.astype(float)), "hit": float(h.mean()),
             "hit_ci": boot_ci(h), "n_traj": int(da.n_written.sum()),
             "succ_jerk": float(sc.succ_jerk.mean()), "succ_rough": float(sc.succ_rough.mean()),
             "succ_manifold": float(sc.succ_manifold.mean()), "succ_pairwise": float(sc.succ_pairwise.mean()),
             "succ_cells": float(sc.succ_cells.mean()), "all_manifold": float(da.all_manifold.mean()),
             "pred_succ": float(da.pred_succ.mean()), "pred_fail": float(da.pred_fail.mean()),
             "by_task_solve": da.groupby("task_folder").solved.mean().round(3).to_dict()}
        res["arms"][arm] = r
        lines.append(f"| {arm} | {r['solve']:.3f} [{r['solve_ci'][0]:.2f}, {r['solve_ci'][1]:.2f}] | "
                     f"{r['hit']:.3f} [{r['hit_ci'][0]:.3f}, {r['hit_ci'][1]:.3f}] | {r['n_traj']} | "
                     f"{r['succ_jerk']:.3f} | {r['succ_manifold']:.3f} | {r['succ_pairwise']:.3f} | "
                     f"{r['succ_cells']:.2f} | {r['pred_succ']:.2f} / {r['pred_fail']:.2f} |")
    lines += ["", "## Paired tests (same instances)", "",
              "| A vs B | solved: only A / only B, McNemar p | hit diff A-B [95% CI], sign-flip p |", "|---|---|---|"]
    res["paired"] = {}
    for ref in ("random", "wm_diverse"):
        for arm in arms:
            if arm == ref or ref not in arms:
                continue
            sa, sb = P["solved"][arm].astype(bool).values, P["solved"][ref].astype(bool).values
            n01, n10, p = mcnemar(sb, sa)
            dh = (P["hits"][arm] / P["budget"][arm] - P["hits"][ref] / P["budget"][ref]).values
            ci, pp = boot_ci(dh), signflip(dh)
            res["paired"][f"{arm}_vs_{ref}"] = {"only_a": n01, "only_b": n10, "mcnemar_p": p, "hit_diff": float(dh.mean()),
                                               "hit_diff_ci": ci, "signflip_p": pp}
            lines.append(f"| {arm} vs {ref} | {n01} / {n10}, p = {p:.3g} | {dh.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}], "
                         f"p = {pp:.3g} |")
    tasks = sorted(d.task_folder.unique())
    lines += ["", "## Solve rate by task", "", "| task | n | " + " | ".join(arms) + " |",
              "|---|---|" + "---|" * len(arms)]
    for t in tasks:
        dt = d[d.task_folder == t]
        lines.append(f"| {t.replace('pick_up_the_black_bowl_', '')[:60]} | {dt.instance_id.nunique()} | " +
                     " | ".join(f"{dt[dt.arm == x].solved.mean():.2f}" for x in arms) + " |")
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
