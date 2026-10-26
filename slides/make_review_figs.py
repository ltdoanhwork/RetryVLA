#!/usr/bin/env python
"""Figures for the 2026-10-26 progress-review slides -> slides/assets/rev_*.png.

All numbers are copied from experiments/RESULTS.md, runs/_analysis/*/summary.md and the run summaries
(B2 success SFT, P3 r0.3 / r0.7 nominal) -- nothing is recomputed here.
Run: python slides/make_review_figs.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
AS = HERE / "assets"
REPO = HERE.parent
for f in Path("/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/tools/fonts/liberation-fonts-ttf-2.1.5").glob("LiberationSerif-*.ttf"):
    fm.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"], "font.size": 14,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 15})
GREY, AMBER, BLUE, TEAL, INK, RED, MUTED = "#9AA5B1", "#E8962E", "#1F8FE5", "#2A9D8F", "#23303B", "#C8553D", "#6B7785"
NOM_FLOOR = 0.796 - 0.02  # pi1-nominal - eps


def save(fig, name):
    fig.savefig(AS / name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", AS / name)


def fig_baselines():
    # (label, RSR, sd or None, nominal or None, colour)
    rows = [
        ("pi0 (SmolVLA base)", 0.118, None, 0.656, GREY),
        ("pi1-nominal (demo SFT, init)", 0.138, None, 0.796, GREY),
        ("demo-only (+6k steps)", 0.127, None, 0.820, GREY),
        ("RISE-inspired cond (pool)", 0.068, None, 0.718, BLUE),
        ("RISE-inspired AWR (pool)", 0.082, None, 0.776, BLUE),
        ("RISE-inspired cond (VLA samples)", 0.113, None, 0.770, BLUE),
        ("goal-data SFT (3 seeds)", 0.187, 0.002, 0.753, GREY),
        ("progress-data SFT", 0.177, None, 0.774, GREY),
        ("random-data SFT", 0.252, None, 0.752, GREY),
        ("residual RL (full budget)", 0.295, None, 0.022, RED),
        ("success-data SFT (B4 critic)", 0.427, None, 0.796, AMBER),
        ("competence SFT n=60", 0.430, None, 0.798, AMBER),
        ("competence SFT n=66 (3 seeds)", 0.417, 0.011, 0.784, AMBER),
        ("RetryVLA pi1 (3 seeds)", 0.418, 0.021, 0.785, AMBER),
    ]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(11, 7.4))
    y = np.arange(n)[::-1]
    h = 0.38
    for yi, (lab, r, sd, nom, col) in zip(y, rows):
        ax.barh(yi + h / 2, r, h, color=col, xerr=sd, capsize=3 if sd else 0)
        ax.text(r + (sd or 0) + 0.008, yi + h / 2, f"{r:.3f}", va="center", fontsize=13, color=INK)
        ax.barh(yi - h / 2, nom, h, facecolor="white", edgecolor="#5A6470", hatch="///", linewidth=0.8)
        ax.text(nom + 0.008, yi - h / 2, f"{nom:.3f}", va="center", fontsize=12, color=MUTED)
    ax.axvline(NOM_FLOOR, ls="--", color=TEAL, lw=1.4)
    ax.text(NOM_FLOOR + 0.005, n - 0.35, "nominal floor 0.776\n(pi1-nominal - 2 pt)", fontsize=12, color=TEAL, va="top")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=14)
    ax.set_xlim(0, 0.92)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("success rate   (filled = test RSR, 56 scenes x 10;  hatched = nominal LIBERO-Spatial, 10 x 50)", fontsize=13)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=AMBER, label="competence-searched + sim-verified data"),
                       Patch(color=GREY, label="other data / no recovery data"),
                       Patch(color=BLUE, label="imagination-only (no verification)"),
                       Patch(color=RED, label="residual RL (extra test-time module)"),
                       Patch(facecolor="white", edgecolor="#5A6470", hatch="///", label="nominal SR")],
              loc="upper center", bbox_to_anchor=(0.42, -0.09), ncol=3, fontsize=12, frameon=False)
    ax.grid(axis="x", color="#E3E7EB", lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "rev_baselines.png")


def fig_forest():
    # (label, diff, lo, hi, colour)
    rows = [
        ("competence - goal SFT (3 v 3 seeds)", 0.230, 0.161, 0.298, AMBER),
        ("competence n=60 - progress SFT", 0.254, 0.173, 0.334, AMBER),
        ("pi1 - demo-only (same init + steps)", 0.292, 0.206, 0.383, AMBER),
        ("pi1 - random-data SFT (208)", 0.167, 0.103, 0.236, AMBER),
        ("pi1 + best-of-8 online - pi1", 0.027, -0.016, 0.070, GREY),
        ("pi0 + best-of-8 online - pi0", 0.000, -0.021, 0.021, GREY),
        ("residual RL (full) - pi1", -0.124, -0.199, -0.054, RED),
        ("RISE-inspired cond (VLA) - pi1", -0.306, -0.386, -0.227, BLUE),
        ("search: WM - real-dynamics prefix", 0.033, -0.049, 0.115, GREY),
        ("Object: competence - demo-only", 0.181, 0.092, 0.277, TEAL),
        ("Object: competence - goal", 0.112, -0.004, 0.227, TEAL),
        ("Object: competence - random", -0.008, -0.104, 0.088, TEAL),
    ]
    fig, ax = plt.subplots(figsize=(10.0, 6.4))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, d, lo, hi, col) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=col, lw=3, solid_capstyle="round")
        ax.plot(d, yi, "o", color=col, ms=9, mec="white", mew=1.2)
        ax.text(0.43, yi, f"{d:+.3f}  [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=14, color=INK)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=16)
    ax.set_xlim(-0.42, 0.42)
    ax.set_xlabel("difference in success rate, 95% cluster-bootstrap CI\n(unit = test scene or failure instance; seeds resampled when available)", fontsize=13)
    ax.grid(axis="x", color="#E3E7EB", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Key paired comparisons (Spatial dev unless 'Object')", fontsize=15, loc="left")
    save(fig, "rev_forest.png")


def fig_q1():
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})
    arms = [("goal\n(latent)", 0.240, GREY), ("progress\ncritic", 0.213, GREY), ("random", 0.339, GREY),
            ("B4-\nnoPi0", 0.450, BLUE), ("C^pi v2\n(PoC)", 0.480, AMBER), ("Cpi-pool", 0.550, AMBER), ("B4\nsuccess", 0.559, AMBER)]
    x = np.arange(len(arms))
    ax[0].bar(x, [a[1] for a in arms], color=[a[2] for a in arms], width=0.68)
    for i, a in enumerate(arms):
        ax[0].text(i, a[1] + 0.012, f"{a[1]:.3f}", ha="center", fontsize=13)
    ax[0].set_xticks(x, [a[0] for a in arms], fontsize=13)
    ax[0].set_ylim(0, 0.66)
    ax[0].set_ylabel("failures solved (>= 1 of 16 verified)")
    ax[0].set_title("Search: only the critic changes (~127 paired failures)", fontsize=15)
    ax[0].text(0.0, -0.30, "amber = trained on pi0 continuation labels;  blue = same critic without pi0 labels",
               transform=ax[0].transAxes, fontsize=12, color=MUTED)
    sft = [("pi1-\nnominal", 0.138, None, GREY), ("goal\n(3 seeds)", 0.187, 0.002, GREY), ("progress", 0.177, None, GREY),
           ("compet.\nn=60", 0.430, None, AMBER), ("compet.\nn=66 (3 s.)", 0.417, 0.011, AMBER), ("B4\nsuccess", 0.427, None, AMBER)]
    x = np.arange(len(sft))
    ax[1].bar(x, [s[1] for s in sft], color=[s[3] for s in sft], width=0.68)
    for i, s in enumerate(sft):
        if s[2]:
            ax[1].errorbar(i, s[1], yerr=s[2], color=INK, capsize=4)
    for i, s in enumerate(sft):
        ax[1].text(i, s[1] + (s[2] or 0) + 0.012, f"{s[1]:.3f}", ha="center", fontsize=13)
    ax[1].set_xticks(x, [s[0] for s in sft], fontsize=13)
    ax[1].set_ylim(0, 0.52)
    ax[1].set_ylabel("test RSR after SFT (56 x 10)")
    ax[1].set_title("Distillation: same recipe, data from each search", fontsize=15)
    plt.tight_layout()
    save(fig, "rev_q1.png")


def fig_q2():
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1, 1.1]})
    pts = [("random", 5.0, 0.311, GREY), ("random diverse", 5.3, 0.369, GREY), ("WM goal latent", 5.4, 0.230, GREY),
           ("WM + C^pi (diverse)", 5.0, 0.475, AMBER), ("real-dynamics prefix\n+ C^pi", 16.5, 0.443, BLUE)]
    for lab, xx, yy, col in pts:
        ax[0].scatter(xx, yy, s=160, color=col, zorder=3, edgecolor="white")
        dx, ha = (0.5, "left")
        dy = {"random": -0.025, "random diverse": 0.012, "WM goal latent": -0.02, "WM + C^pi (diverse)": 0.03}.get(lab, 0.0)
        ax[0].text(xx + dx - (1.2 if lab.startswith("WM +") else 0), yy + dy, lab, fontsize=13, ha=ha, va="center")
    ax[0].annotate("", xy=(5.6, 0.468), xytext=(15.9, 0.443), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    ax[0].text(11.0, 0.385, "same solve rate at ~1/3 of the\nsim transitions: +0.033 [-0.049, +0.115]", fontsize=12, color=MUTED)
    ax[0].set_xlim(3, 22)
    ax[0].set_ylim(0.15, 0.58)
    ax[0].set_xlabel("simulator transitions per failure (thousands)")
    ax[0].set_ylabel("failures solved (16 verified sims)")
    ax[0].set_title("M9: WM vs real dynamics (122 failures)", fontsize=15)
    arms = [("random", 0.288, (0.212, 0.365), GREY), ("wm_plan\n(no library)", 0.373, (0.288, 0.458), BLUE),
            ("wm_plan_v2\n(retrained WM)", 0.398, (0.305, 0.483), BLUE), ("wm_diverse\n(library)", 0.458, (0.364, 0.551), AMBER)]
    x = np.arange(len(arms))
    ax[1].bar(x, [a[1] for a in arms], yerr=[[a[1] - a[2][0] for a in arms], [a[2][1] - a[1] for a in arms]], capsize=5,
              color=[a[3] for a in arms], width=0.66)
    for i, a in enumerate(arms):
        ax[1].text(i + 0.36, a[1], f"{a[1]:.3f}", ha="left", va="center", fontsize=13)
    ax[1].set_xticks(x, [a[0] for a in arms], fontsize=13)
    ax[1].set_ylim(0, 0.62)
    ax[1].set_ylabel("failures solved [95% CI]")
    ax[1].set_title("M11: can the WM plan without the library? (118)", fontsize=15)
    plt.tight_layout()
    save(fig, "rev_q2.png")


def fig_p3():
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.axvspan(NOM_FLOOR, 0.84, color=TEAL, alpha=0.08, lw=0)
    ax.axvline(NOM_FLOOR, ls="--", color=TEAL, lw=1.3)
    ax.text(NOM_FLOOR + 0.002, 0.20, "nominal floor 0.776\n(eps = 2 pt)", fontsize=12, color=TEAL)
    pts = [("r 0.1", 0.788, 0.323), ("r 0.3", 0.814, 0.375), ("r 0.7", 0.766, 0.436)]
    xs = [0.788, 0.814, 0.785, 0.766]
    ys = [0.323, 0.375, 0.418, 0.436]
    ax.plot(xs, ys, color=GREY, lw=1.2, zorder=1)
    lab_pos = {"r 0.1": (0.791, 0.318, "left"), "r 0.3": (0.817, 0.370, "left"), "r 0.7": (0.763, 0.452, "right")}
    for lab, xx, yy in pts:
        ok = xx >= NOM_FLOOR
        ax.scatter(xx, yy, s=130, color=TEAL if ok else RED, zorder=3, edgecolor="white")
        tx, ty, ha = lab_pos[lab]
        ax.text(tx, ty, lab + ("" if ok else "\n(fails margin)"), fontsize=13, va="center", ha=ha,
                color=INK if ok else RED)
    ax.errorbar(0.785, 0.418, yerr=0.021, fmt="o", ms=12, color=AMBER, capsize=4, zorder=4, mec="white")
    ax.text(0.787, 0.392, "r 0.5 = pi1 (3 seeds)", fontsize=13, ha="left", va="top")
    ax.scatter(0.806, 0.429, s=150, marker="D", color=BLUE, zorder=3, edgecolor="white")
    ax.text(0.809, 0.437, "r 0.5 + L_preserve", fontsize=13, va="bottom")
    ax.scatter(0.796, 0.138, s=110, marker="s", color=GREY, zorder=3)
    ax.text(0.799, 0.138, "pi1-nominal (init)", fontsize=12, va="center", color=MUTED)
    ax.set_xlim(0.735, 0.84)
    ax.set_ylim(0.10, 0.48)
    ax.set_xlabel("nominal SR (LIBERO-Spatial, 500 episodes)")
    ax.set_ylabel("test RSR (56 x 10)")
    ax.set_title("Recovery ratio sweep, seed 0 (pi1: 3 seeds)", fontsize=15)
    ax.grid(color="#E3E7EB", lw=0.6)
    ax.set_axisbelow(True)
    save(fig, "rev_p3_pareto.png")


def fig_object():
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"width_ratios": [1.5, 1]})
    pol = [("pi0", 0.219, 0.886, GREY), ("pi1-\nnominal", 0.200, 0.944, GREY), ("demo-\nonly", 0.173, 0.932, GREY),
           ("+123 goal", 0.242, 0.906, GREY), ("+123\nrandom", 0.362, 0.946, BLUE), ("+123\ncompetence", 0.354, 0.942, AMBER)]
    x = np.arange(len(pol))
    ax[0].bar(x, [p[1] for p in pol], color=[p[3] for p in pol], width=0.66)
    for i, p in enumerate(pol):
        ax[0].text(i, p[1] + 0.01, f"{p[1]:.3f}", ha="center", fontsize=13)
        ax[0].text(i, 0.475, f"nom {p[2]:.3f}", ha="center", fontsize=12, color=MUTED)
    ax[0].set_xticks(x, [p[0] for p in pol], fontsize=13)
    ax[0].set_ylim(0, 0.5)
    ax[0].set_ylabel("held-out test RSR (26 scenes x 10)")
    ax[0].set_title("Distillation on LIBERO-Object (recipe frozen, 1 seed)", fontsize=15)
    se = [("WM goal", 0.227, GREY), ("random", 0.405, BLUE), ("WM + C^pi", 0.462, AMBER)]
    x = np.arange(len(se))
    ax[1].bar(x, [s[1] for s in se], color=[s[2] for s in se], width=0.6)
    for i, s in enumerate(se):
        ax[1].text(i, s[1] + 0.01, f"{s[1]:.3f}", ha="center", fontsize=13)
    ax[1].set_xticks(x, [s[0] for s in se], fontsize=13)
    ax[1].set_ylim(0, 0.56)
    ax[1].set_ylabel("failures solved (16 sims)")
    ax[1].set_title("Search (247 pi0 failures)", fontsize=15)
    plt.tight_layout()
    save(fig, "rev_object.png")


def copy_related():
    src = REPO / "docs/related_work/figs"
    for f in ["repovla", "recovla", "racer", "dream2fix", "dinowm", "tdmpc2", "rise_overview", "wmpo"]:
        shutil.copy(src / f"{f}.png", AS / f"rw_{f}.png")
    print("copied related-work figs")


if __name__ == "__main__":
    fig_baselines()
    fig_forest()
    fig_q1()
    fig_q2()
    fig_p3()
    fig_object()
    copy_related()
