#!/usr/bin/env python
"""Figures for the M5 slides -> slides/assets/: m5b_search.png, m5_sft.png, m5_task_rsr.png, m5_strategies.png"""

from __future__ import annotations

import glob
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
AS = REPO / "slides/assets"
for f in Path("/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/tools/fonts/liberation-fonts-ttf-2.1.5").glob("LiberationSerif-*.ttf"):
    fm.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"], "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False})
GREY, AMBER, BLUE, TEAL, INK, NAVY = "#9AA5B1", "#E8962E", "#1F8FE5", "#2A9D8F", "#23303B", "#5A6E8C"
ARMCOL = {"random": GREY, "wm_diverse": AMBER, "wm_noise": BLUE, "wm_qd": TEAL}
ARMLAB = {"random": "random", "wm_diverse": "WM diverse", "wm_noise": "WM noise-CEM", "wm_qd": "WM quality-div."}


def episodes(tag):
    d = sorted(glob.glob(str(REPO / "runs" / f"*_{tag}")))[-1]
    return [json.loads(l) for l in open(d + "/episodes.jsonl")]


def summary(tag, key):
    d = sorted(glob.glob(str(REPO / "runs" / f"*_{tag}")))[-1]
    r = json.load(open(d + "/summary.json"))
    return r.get("results", r)[key]


def fig_search():
    s = json.load(open(REPO / "runs/_analysis/m5b/summary.json"))
    arms = ["random", "wm_diverse", "wm_noise", "wm_qd"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for k, (key, ci, title) in enumerate([("solve", "solve_ci", "Failures with ≥1 verified recovery\n(16 simulator rollouts per arm)"),
                                          ("hit", "hit_ci", "Verified successes per simulator rollout")]):
        v = [s["arms"][a][key] for a in arms]
        lo = [s["arms"][a][key] - s["arms"][a][ci][0] for a in arms]
        hi = [s["arms"][a][ci][1] - s["arms"][a][key] for a in arms]
        b = ax[k].bar(range(4), v, yerr=[lo, hi], capsize=5, color=[ARMCOL[a] for a in arms], width=0.65)
        for i, x in enumerate(v):
            ax[k].text(i, x + hi[i] + 0.01, f"{x:.2f}" if key == "solve" else f"{x:.3f}", ha="center", fontsize=13)
        ax[k].set_xticks(range(4), [ARMLAB[a] for a in arms])
        ax[k].set_title(title, fontsize=14)
    ax[0].set_ylim(0, 0.7)
    ax[1].set_ylim(0, 0.23)
    fig.text(0.5, -0.02, f"N = {s['n_instances']} paired pi0-failure instances (same instances, same continuation seeds, 95% bootstrap CI). "
             "WM diverse vs random: McNemar p = 4e-9", ha="center", fontsize=11, color="#6B7785")
    plt.tight_layout()
    plt.savefig(AS / "m5b_search.png", dpi=170, bbox_inches="tight")
    plt.close()


MODELS = [
    ("π0 (SmolVLA base)", "M0_recover_bddl", "M0_baseline_spatial_nominal", GREY),
    ("π1-nominal (demos only)", "M1_pi1nom_recover_test", "M1_pi1nom_nominal", NAVY),
    ("+208 random-search recoveries", "M5_m5b_sft_random_recover_test", "M5_m5b_sft_random_nominal", "#B9A07A"),
    ("+241 random (M5)", "M5_sft_random_matched_recover_test", "M5_sft_random_matched_nominal", "#C9B48E"),
    ("+241 WM (M5 round 1)", "M5_sft_wm_r1_recover_test", "M5_sft_wm_r1_nominal", "#F0B260"),
    ("+208 WM quality-div.", "M5_m5c_sft_wm_qd_rr_recover_test", "M5_m5c_sft_wm_qd_rr_nominal", TEAL),
    ("+208 WM diverse", "M5_m5c_sft_wm_diverse_rr_recover_test", "M5_m5c_sft_wm_diverse_rr_nominal", AMBER),
]


def fig_sft():
    rows = []
    for name, rt, nt, col in MODELS:
        try:
            r = summary(rt, "rsr")
        except Exception:
            continue
        n = None
        if nt:
            try:
                n = summary(nt, "success_rate")
            except Exception:
                n = None
        rows.append((name, r, n, col))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[3] for r in rows], height=0.62)
    for i, (name, r, n, _) in enumerate(rows):
        ax.text(r + 0.006, i, f"{r:.3f}" + (f"   (nominal {n:.2f})" if n is not None else ""), va="center", fontsize=13)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("Recovery success rate on held-out TEST failures (560 episodes, no search at test time)")
    plt.tight_layout()
    plt.savefig(AS / "m5_sft.png", dpi=170)
    plt.close()


def fig_task():
    tags = [("+208 random", "M5_m5b_sft_random_recover_test", GREY),
            ("+208 WM diverse, top-score subset (bug)", "M5_m5b_sft_wm_diverse_recover_test", "#F3C98B"),
            ("+208 WM diverse, round-robin subset", "M5_m5c_sft_wm_diverse_rr_recover_test", AMBER)]
    per = {}
    for name, t, _ in tags:
        c = defaultdict(list)
        for r in episodes(t):
            c[r["task_folder"].replace("pick_up_the_black_bowl_", "").split("_and_place")[0].replace("_", " ")].append(r["success"])
        per[name] = {k: np.mean(v) for k, v in c.items()}
        n_by = {k: len(v) for k, v in c.items()}
    tasks = sorted(n_by, key=lambda k: -n_by[k])
    fig, ax = plt.subplots(figsize=(12, 4.6))
    w = 0.27
    for i, (name, _, col) in enumerate(tags):
        ax.bar(np.arange(len(tasks)) + (i - 1) * w, [per[name][t] for t in tasks], w, color=col, label=name)
    SHORT = {"in the top drawer of the wooden cabinet": "drawer", "on the ramekin": "on ramekin",
             "between the plate and the ramekin": "between plate\n& ramekin", "on the wooden cabinet": "on cabinet",
             "on the stove": "on stove", "next to the cookie box": "next to\ncookie box", "next to the plate": "next to\nplate",
             "on the cookie box": "on cookie box", "next to the ramekin": "next to\nramekin"}
    ax.set_xticks(np.arange(len(tasks)), [f"{SHORT.get(t, t)}\n(n={n_by[t]})" for t in tasks], fontsize=11)
    ax.set_ylabel("test RSR")
    ax.legend(frameon=False, fontsize=11)
    ax.set_title("Per-task test RSR: the top-score cut dropped drawer data; round-robin restores it", fontsize=14)
    plt.tight_layout()
    plt.savefig(AS / "m5_task_rsr.png", dpi=170)
    plt.close()


def main():
    fig_search()
    fig_sft()
    fig_task()
    shutil.copy(REPO / "runs/_analysis/viz/strategies/strategies.png", AS / "m5_strategies.png")
    for k, src in [("m5_e2e_test_keys.png", "e2e/test_in_the_top_drawer_of_the_wooden_ca__episode_040_step_0200__close1/keyframes.png")]:
        shutil.copy(REPO / "runs/_analysis/viz" / src, AS / k)
    print("ok")


if __name__ == "__main__":
    main()
