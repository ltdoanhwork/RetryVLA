#!/usr/bin/env python
"""Re-do the statistics with the right independent units (review 2026-10-20).

python scripts/stats_cluster.py  ->  runs/_analysis/stats/summary.md

- Policy comparisons on the Spatial dev split: the 10 trials of a scene are NOT independent -> cluster (scene)
  bootstrap of the paired difference; with several training seeds, resample scenes AND seeds (two-level).
  Also a paired sign-flip permutation test on per-scene mean differences (unit = scene, n = 56).
- Search comparisons: unit = failure instance (one per row) -> exact McNemar on 'solved' + bootstrap CI of the diff.
- Top-score vs other subset rules: composition of the selected training data (unique failures, tasks, length, score).
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "runs/_analysis/stats"
B = 10000


def eps(tag):
    f = sorted(glob.glob(str(REPO / "runs" / f"*_{tag}")))
    if not f:
        return None
    return {(r["scene"], r["gripper_init"], r["trial"]): float(r["success"]) for r in map(json.loads, open(f[-1] + "/episodes.jsonl"))}


def per_scene(e):
    d = defaultdict(list)
    for k, v in e.items():
        d[k[0]].append(v)
    return {s: np.mean(v) for s, v in d.items()}


def compare(A: list, Bm: list, rng):
    """A, Bm: lists (one per training seed) of episode dicts. Returns diff, CI (scene x seed bootstrap), perm p."""
    scenes = sorted(per_scene(A[0]))
    SA = np.array([[per_scene(a)[s] for s in scenes] for a in A])   # [seedsA, scenes]
    SB = np.array([[per_scene(b)[s] for s in scenes] for b in Bm])
    diff = SA.mean() - SB.mean()
    bs = []
    for _ in range(B):
        sc = rng.integers(0, len(scenes), len(scenes))
        ia = rng.integers(0, len(A), len(A))
        ib = rng.integers(0, len(Bm), len(Bm))
        bs.append(SA[ia][:, sc].mean() - SB[ib][:, sc].mean())
    d_s = SA.mean(0) - SB.mean(0)                                    # per-scene paired difference
    flips = rng.choice([-1, 1], size=(B, len(d_s)))
    p = float((np.abs((flips * d_s).mean(1)) >= abs(d_s.mean()) - 1e-12).mean())
    return diff, np.percentile(bs, [2.5, 97.5]), p, len(scenes), len(A), len(Bm)


def mcnemar(a, b):
    n01, n10 = int((a & ~b).sum()), int((b & ~a).sum())
    n = n01 + n10
    return n01, n10, (min(1.0, 2 * sum(comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n) if n else 1.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    L = ["# Statistics with cluster-aware units", "",
         "Policy comparisons: unit = test scene (56; 10 trials each, averaged), seeds resampled when available. "
         "CI = two-level bootstrap (scenes x training seeds), p = paired sign-flip test on per-scene differences.", "",
         "| comparison | seeds A / B | diff | 95% CI | p (scene-level) |", "|---|---|---|---|---|"]
    pi1 = ["M5_m5c_sft_wm_diverse_rr_recover_test", "M8_seed_pi1_s1_recover_test", "M8_seed_pi1_s2_recover_test"]
    base = ["M7_sft_base_recover_test", "M8_seed_base_s1_recover_test", "M8_seed_base_s2_recover_test"]
    weak = ["M7_sft_weak_recover_test", "M8_seed_weak_s1_recover_test", "M8_seed_weak_s2_recover_test"]
    comp = ["M9_sft_comp_recover_test"] + [f"M9_sft_comp_seed{s}_recover_test" for s in (1, 2)]
    goal = ["M9_sft_goal_recover_test"] + [f"M9_sft_goal_seed{s}_recover_test" for s in (1, 2)]
    rows = [
        ("competence-data SFT vs goal-data SFT (66 traj)", comp, goal),
        ("pi1 (WM data) vs random-data SFT (208)", pi1, ["M5_m5b_sft_random_recover_test"]),
        ("M5: WM r1 vs random (241)", ["M5_sft_wm_r1_recover_test"], ["M5_sft_random_matched_recover_test"]),
        ("pi1 vs demo-only SFT (same init + steps)", pi1, ["M9_sft_demo_only_recover_test"]),
        ("pi1 + best-of-8 vs pi1", ["M9_bon8_pi1_recover_test"], pi1[:1]),
        ("pi0 + best-of-8 vs pi0", ["M9_bon8_pi0_recover_test"], ["M0_recover_bddl"]),
        ("M7 base vs pi1 (extra training)", base, pi1),
        ("M7 weak vs M7 base", weak, base),
        ("selection: round-robin vs top-score", ["M5_m5c_sft_wm_diverse_rr_recover_test"], ["M5_m5b_sft_wm_diverse_recover_test"]),
        ("selection: random vs top-score", ["M9_sft_sel_random_recover_test"], ["M5_m5b_sft_wm_diverse_recover_test"]),
        ("selection: round-robin vs random", ["M5_m5c_sft_wm_diverse_rr_recover_test"], ["M9_sft_sel_random_recover_test"]),
    ]
    test_scenes = None
    for name, ta, tb in rows:
        A = [e for e in (eps(t) for t in ta) if e]
        Bm = [e for e in (eps(t) for t in tb) if e]
        if not A or not Bm:
            L.append(f"| {name} | missing | | | |")
            continue
        if "M0_recover_bddl" in tb:   # pi0 eval covered pilot + test: keep the test scenes only
            keys = set(A[0])
            Bm = [{k: v for k, v in b.items() if k in keys} for b in Bm]
        d, ci, p, ns, sa, sb = compare(A, Bm, rng)
        L.append(f"| {name} | {sa} / {sb} | {d:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {p:.2g} |")

    # search comparisons (unit = failure instance)
    import pandas as pd

    def load(pat):
        return pd.DataFrame([json.loads(l) for f in glob.glob(str(REPO / "runs" / pat)) for l in open(f)]).drop_duplicates(
            ["instance_id", "arm"], keep="last")
    m9, m5 = load("*_M9_mech_search_s*/per_instance.jsonl"), load("*_M5_m5b_compare_arms_s*/per_instance.jsonl")
    d = pd.concat([m5[m5.instance_id.isin(set(m9.instance_id))], m9])
    full = d.groupby("instance_id").arm.nunique()
    d = d[d.instance_id.isin(full[full == full.max()].index)]
    P = d.pivot(index="instance_id", columns="arm", values="solved").astype(bool)
    L += ["", f"Search comparisons: unit = failure instance (N = {len(P)}), exact McNemar on solved, bootstrap CI.", "",
          "| A vs B | solved A | solved B | only A / only B | diff [95% CI] | p |", "|---|---|---|---|---|---|"]
    for a, b in [("wm_diverse", "wm_goal"), ("wm_diverse", "random"), ("wm_diverse", "sim_prefix_critic"),
                 ("wm_diverse", "wm_top"), ("wm_diverse", "random_diverse"), ("wm_goal_comp", "wm_diverse")]:
        x, y = P[a].values, P[b].values
        n01, n10, p = mcnemar(x, y)
        bs = [(x[i] .mean() - y[i].mean()) for i in (rng.integers(0, len(x), len(x)) for _ in range(B))]
        ci = np.percentile(bs, [2.5, 97.5])
        L.append(f"| {a} vs {b} | {x.mean():.3f} | {y.mean():.3f} | {n01} / {n10} | {x.mean() - y.mean():+.3f} "
                 f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {p:.2g} |")
    # non-inferiority framing for WM vs sim-prefix
    x, y = P["wm_diverse"].values, P["sim_prefix_critic"].values
    bs = [(x[i].mean() - y[i].mean()) for i in (rng.integers(0, len(x), len(x)) for _ in range(B))]
    L += ["", f"WM vs real-dynamics prefix: one-sided 95% lower bound of (WM - sim_prefix) = {np.percentile(bs, 5):+.3f} "
          "(non-inferior at margin m if this bound > -m)."]

    # composition of selected training data under each subset rule
    from retryvla.data.mix import expand_roots, recovery_episode_subsets
    roots = expand_roots(["data/recovery/m5b/s*/p*/wm_diverse"])
    rows_all = {r: [json.loads(l) for l in open(Path(r) / "retryvla/trajectories.jsonl")] for r in roots}
    L += ["", "## Composition of the 208 selected trajectories (same verified bank, 423 trajectories)", "",
          "| rule | test RSR | unique failures | tasks covered | drawer traj | mean recovery len | mean critic score | mean total len |",
          "|---|---|---|---|---|---|---|---|"]
    rsr = {"score": "M5_m5b_sft_wm_diverse_recover_test", "random": "M9_sft_sel_random_recover_test",
           "roundrobin": "M5_m5c_sft_wm_diverse_rr_recover_test", "task_balanced": "M9_sft_sel_task_balanced_recover_test"}
    for mode in ["score", "random", "roundrobin", "task_balanced"]:
        subs = recovery_episode_subsets(roots, 208, mode=mode)
        sel = [t for r, s in zip(roots, subs) for t in rows_all[r] if t["episode_index"] in set(s)]
        task = lambda t: t["instance_id"].split("__")[0].replace("pick_up_the_black_bowl_", "").split("_and_place")[0]
        e = eps(rsr[mode])
        L.append(f"| {mode} | {np.mean(list(e.values())) if e else float('nan'):.3f} | {len({t['instance_id'] for t in sel})} | "
                 f"{len({task(t) for t in sel})} | {sum('top_drawer' in t['instance_id'] for t in sel)} | "
                 f"{np.mean([t['recovery_len'] for t in sel]):.1f} | {np.mean([t.get('score', 0) for t in sel]):.3f} | "
                 f"{np.mean([t['length'] for t in sel]):.1f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
