#!/usr/bin/env python
"""Compact baseline summary (Spatial dev test split) + B2 SFT paired tests.

python scripts/analyze_baselines.py [--out runs/_analysis/baselines]

Unit = test scene (56 x 10 trials averaged); several training seeds -> scene x seed bootstrap (stats_cluster.compare,
10k resamples, paired sign-flip p on per-scene differences). pi0 = test part of M0_recover_bddl.
Nominal = LIBERO-Spatial 10 tasks x 50 inits; dNSR CI = 10-task (x seed) bootstrap (analyze_b32.nom_delta).
Rows whose run dir is missing are printed as 'pending'. Sim budgets: runs/_analysis/b33/budget.json (PoC accounting).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from analyze_b32 import nom_delta, nom_tasks  # noqa: E402
from stats_cluster import compare, eps  # noqa: E402

PI1 = ["M5_m5c_sft_wm_diverse_rr_recover_test", "M8_seed_pi1_s1_recover_test", "M8_seed_pi1_s2_recover_test"]
PI1_NOM = ["M5_m5c_sft_wm_diverse_rr_nominal", "M8_seed_pi1_s1_nominal", "M8_seed_pi1_s2_nominal"]
COMP3 = ["M9_sft_comp_recover_test", "M9_sft_comp_seed1_recover_test", "M9_sft_comp_seed2_recover_test"]
GOAL3 = ["M9_sft_goal_recover_test", "M9_sft_goal_seed1_recover_test", "M9_sft_goal_seed2_recover_test"]

# method, recover tags (seeds), nominal tags, extra test-time cost, sim budget (transitions; PoC accounting)
ROWS = [
    ("pi0 (SmolVLA base)", ["@pi0"], ["M0_baseline_spatial_nominal"], "none", "0"),
    ("pi1-nominal (demo SFT, = init of all SFT rows)", ["M1_pi1nom_recover_test"], ["M1_pi1nom_nominal"], "none", "0"),
    ("demo-only (+6k steps, same init / steps as pi1)", ["M9_sft_demo_only_recover_test"],
     ["M9_sft_demo_only_nominal"], "none", "0"),
    ("random-data SFT (208 random-search recoveries)", ["M5_m5b_sft_random_recover_test"],
     ["M5_m5b_sft_random_nominal"], "none", "1.40M (256 failures: verify 1.26M + precheck 0.13M)"),
    ("goal-data SFT (66 WM latent-goal recoveries, 3 seeds)", GOAL3, [], "none",
     "2.09M (WM data 1.39M + ~0.70M verify / precheck, 127 failures; no critic)"),
    ("progress-data SFT (60 WM + B3-progress recoveries)", ["B2_sft_progress_recover_test"], [], "none",
     "2.09M (WM data 1.39M + ~0.70M verify / precheck, 127 failures; critic: demos, 0 sim)"),
    ("competence SFT, count-matched n=60 (M9 comp recipe)", ["B2_sft_comp_n60_recover_test"], [], "none",
     "2.90M (WM 1.39M + critic 0.82M + ~0.70M verify / precheck, 127 failures)"),
    ("competence SFT n=66 (M9 comp, 3 seeds)", COMP3, [], "none",
     "2.90M (WM 1.39M + critic 0.82M + ~0.70M verify / precheck, 127 failures)"),
    ("**RetryVLA pi1 (208 verified WM recoveries, 3 seeds)**", PI1, PI1_NOM, "none",
     "3.47M (PoC: WM 1.39M + critic 0.82M + verify 1.26M + precheck 0.13M, 256 failures)"),
    ("online best-of-8 on pi1 (WM + critic every replan)", ["M9_bon8_pi1_recover_test"], [],
     "8 VLA samples + 8 WM rollouts + critic per replan", "3.47M (pi1) + 0 (reuses WM / critic)"),
    ("online best-of-8 on pi0", ["M9_bon8_pi0_recover_test"], [], "8 VLA samples + 8 WM rollouts + critic per replan",
     "2.21M (WM / critic data)"),
    ("RISE-inspired cond (imagined advantage, pool)", ["B32_rise_cond_pool_recover_test"],
     ["B32_rise_cond_pool_nominal"], "none (instruction suffix)", "2.21M (WM / critic data, 0 verify)"),
    ("RISE-inspired AWR (imagined advantage, pool)", ["B32_rise_awr_pool_recover_test"],
     ["B32_rise_awr_pool_nominal"], "none", "2.21M (WM / critic data, 0 verify)"),
    ("RISE-inspired cond (imagined advantage, VLA samples)", ["B32_rise_cond_vla_recover_test"],
     ["B32_rise_cond_vla_nominal"], "none (instruction suffix)", "2.21M (WM / critic data, 0 verify)"),
    ("residual RL p25 (PPO bridge on frozen pi0)", ["B33_b33_rl_sparse_p25_recover_test"],
     ["B33_b33_rl_sparse_p25_nominal"], "DINOv2-small + MLP, first 50 steps", "0.87M (25% of PoC)"),
    ("residual RL full (PPO bridge on frozen pi0)", ["B33_b33_rl_sparse_full_recover_test"],
     ["B33_b33_rl_sparse_full_nominal"], "DINOv2-small + MLP, first 50 steps", "3.47M (= PoC)"),
]


def _complete(tag, n_expected):
    """True if the newest run dir of `tag` has a finished episodes.jsonl (>= n_expected rows); running evals ->
    pending instead of a partial number (eps() / nom_tasks() read the newest dir)."""
    import glob
    f = sorted(glob.glob(str(REPO / "runs" / f"*_{tag}")))
    if not f or not Path(f[-1], "episodes.jsonl").exists():
        return False
    return sum(1 for _ in open(Path(f[-1], "episodes.jsonl"))) >= n_expected


def load_rec(tags, test_keys):
    out = []
    for t in tags:
        if t == "@pi0":
            e = eps("M0_recover_bddl")
            e = {k: v for k, v in e.items() if k in test_keys} if e else None
        else:
            e = eps(t) if _complete(t, 560) else None
        if e:
            out.append(e)
    return out


def nom_done(tags):
    tags = [t for t in tags if _complete(t, 500)]
    return nom_tasks(tags) if tags else None


def fmt(d, ci, p):
    ps = "< 1e-4" if p == 0 else f"{p:.2g}"
    return f"{d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}], p {ps}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/_analysis/baselines")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pi1 = load_rec(PI1, None)
    keys = set(pi1[0])
    pi1nom_rec = load_rec(["M1_pi1nom_recover_test"], keys)
    pi1_n = nom_tasks(PI1_NOM)
    pi1nom_n = nom_tasks(["M1_pi1nom_nominal"])
    L = ["# Baseline summary (Spatial test = dev split; 56 scenes x 10 trials)", "",
         "| method | seeds | test RSR | nominal | dRSR vs pi1 (3 seeds) [95% CI], p | dNSR vs pi1-nominal [95% CI] (eps 2 pt) | "
         "extra test-time cost | sim budget (transitions) |", "|---|---|---|---|---|---|---|---|"]
    js = []
    for name, rt, nt, cost, bud in ROWS:
        A = load_rec(rt, keys)
        if not A:
            nn = nom_done(nt) if nt else None
            L.append(f"| {name} | - | pending | {'' if nn is None else f'{nn.mean():.3f}'} | | | {cost} | {bud} |")
            continue
        rs = [np.mean(list(e.values())) for e in A]
        rsr = f"{np.mean(rs):.3f}" + (f" ± {np.std(rs, ddof=1):.3f}" if len(rs) > 1 else "")
        if rt == PI1:
            dv = "(reference)"
        elif rt == ["M9_bon8_pi1_recover_test"]:  # best-of-8 runs ON pi1 seed 0 -> paired vs that seed only
            d, ci, p, *_ = compare(A, pi1[:1], rng)
            dv = fmt(d, ci, p) + " (vs pi1 seed 0, its base)"
        else:
            d, ci, p, *_ = compare(A, pi1, rng)
            dv = fmt(d, ci, p)
        nn = nom_done(nt) if nt else None
        nom = f"{nn.mean():.3f}" if nn is not None else "not run"
        if nn is not None and nt != ["M1_pi1nom_nominal"]:
            dn, lo, hi = nom_delta(nn, pi1nom_n, rng)
            dns = f"{dn:+.3f} [{lo:+.3f}, {hi:+.3f}]" + (" **< -eps**" if dn < -0.02 - 1e-9 else "")
        else:
            dns = "-" if nn is None else "(reference)"
        L.append(f"| {name} | {len(A)} | {rsr} | {nom} | {dv} | {dns} | {cost} | {bud} |")
        js.append({"method": name, "seeds": len(A), "rsr": rs, "nominal": None if nn is None else float(nn.mean())})
    # B2 SFT paired tests (item 3)
    L += ["", "## B2 SFT: progress-data vs count-matched competence (scene-level paired)", "",
          "| comparison | seeds A / B | diff [95% CI], p (sign-flip) | scenes A worse / better / tied |",
          "|---|---|---|---|"]
    prog = load_rec(["B2_sft_progress_recover_test"], keys)
    c60 = load_rec(["B2_sft_comp_n60_recover_test"], keys)
    comp3 = load_rec(COMP3, keys)
    goal3 = load_rec(GOAL3, keys)
    pi0 = load_rec(["@pi0"], keys)
    from stats_cluster import per_scene  # noqa: E402

    def wbt(A, Bm):
        sa = np.mean([[per_scene(x)[s] for s in sorted(per_scene(A[0]))] for x in A], 0)
        sb = np.mean([[per_scene(x)[s] for s in sorted(per_scene(A[0]))] for x in Bm], 0)
        return f"{int((sa < sb).sum())} / {int((sa > sb).sum())} / {int((sa == sb).sum())}"

    for nm, A, Bm in [("progress SFT - competence n=60", prog, c60),
                      ("progress SFT - competence n=66 (3 seeds)", prog, comp3),
                      ("progress SFT - pi1-nominal", prog, pi1nom_rec),
                      ("progress SFT - goal SFT (3 seeds)", prog, goal3),
                      ("progress SFT - pi0", prog, pi0),
                      ("competence n=60 - pi1-nominal", c60, pi1nom_rec),
                      ("competence n=60 - competence n=66 (3 seeds)", c60, comp3),
                      ("competence n=60 - goal SFT (3 seeds)", c60, goal3)]:
        if not A or not Bm:
            L.append(f"| {nm} | | pending | |")
            continue
        d, ci, p, ns, sa, sb = compare(A, Bm, rng)
        L.append(f"| {nm} | {sa} / {sb} | {fmt(d, ci, p)} | {wbt(A, Bm)} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    (out / "rows.json").write_text(json.dumps(js, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
