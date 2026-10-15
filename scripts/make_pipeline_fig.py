#!/usr/bin/env python
"""Overall RetryVLA pipeline diagram -> slides/assets/pipeline_overview.png"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
for f in Path("/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/tools/fonts/liberation-fonts-ttf-2.1.5").glob("LiberationSerif-*.ttf"):
    fm.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"]})

INK, MUTED, AMBER, BLUE, TEAL, RED = "#23303B", "#6B7785", "#E8962E", "#1F8FE5", "#2A9D8F", "#C8553D"
TINT, AMB_T, BLU_T, TEA_T = "#F2F4F6", "#FCEFDC", "#E3F1FC", "#E0F2EF"

fig, ax = plt.subplots(figsize=(16.6, 9))
ax.set_xlim(0, 16.6); ax.set_ylim(0, 9); ax.axis("off")


def box(x, y, w, h, title, body, fc, ec=None, tc=INK, bc=None, fs=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec or fc, lw=1.5))
    ax.text(x + 0.18, y + h - 0.22, title, fontsize=fs, fontweight="bold", color=tc, va="top")
    if body:
        ax.text(x + 0.18, y + h - 0.62, body, fontsize=10.5, color=bc or "#3C4A57", va="top", linespacing=1.35)


def arrow(x1, y1, x2, y2, col=MUTED, txt=None, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, color=col, lw=1.8,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))
    if txt:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.14, txt, fontsize=9.5, color=col, ha="center", style="italic")


ax.text(0.3, 8.45, "find recoveries from real failure states  →  verify  →  distill into the VLA  →  recover without search",
        fontsize=12.5, color=MUTED, style="italic")

# Row 1: data
box(0.3, 5.9, 2.7, 1.9, "1  Failure states", "LIBERO-Recover Spatial\n168 scenes · exact snapshot\nsplit by failure episode\ntrain / pilot / test", TINT)
box(0.3, 3.55, 2.7, 1.9, "pi0 = SmolVLA", "HuggingFaceVLA/\nsmolvla_libero\nnominal 0.66 · RSR 0.12\n(frozen during search)", BLU_T)

# Search block
ax.add_patch(FancyBboxPatch((3.5, 1.2), 7.3, 6.75, boxstyle="round,pad=0.02,rounding_size=0.15", fc="white", ec=AMBER, lw=2.2, ls="--"))
ax.text(3.7, 7.7, "2  Recovery search (offline, train/pilot only)", fontsize=14, fontweight="bold", color=AMBER)
box(3.8, 5.95, 3.2, 1.45, "Propose candidates", "SmolVLA samples (noise)\nsmooth perturbations · primitives\n50-step open-loop segments", AMB_T, fs=12)
box(7.35, 5.95, 3.2, 1.45, "Refine (CEM)", "sample around elite candidates\nkeep strategies separate\n(elite-local, not averaged)", AMB_T, fs=12)
# two scorers
box(3.8, 3.35, 3.2, 2.2, "Scorer A: simulator (M1)", "restore snapshot\nrun segment + pi0 continues\nsuccess? (exact, expensive)\n≤256 rollouts / failure", TINT, fs=12)
box(7.35, 3.35, 3.2, 2.2, "Scorer B: world model (M3)", "DINOv2 latent z0, q0\nWM imagines z50, q50\ncritic V = P(pi0 finishes)\n−uncertainty −jerk · ~800 cands/s", AMB_T, ec=AMBER, fs=12)
ax.text(7.0, 4.45, "or", fontsize=12, color=MUTED, ha="center", style="italic")
box(3.8, 1.45, 6.75, 1.5, "3  Verify in simulator", "top-k candidates (WM: diverse top-16, adaptive budget)\nrestore → run segment → SmolVLA continues → success?\nkeep verified successes · log every verification", TEA_T, fs=12)
arrow(5.4, 5.95, 5.4, 5.55, AMBER); arrow(8.95, 5.95, 8.95, 5.55, AMBER)
arrow(7.0, 6.65, 7.35, 6.65, AMBER)
arrow(8.95, 3.35, 8.95, 2.95, TEAL); arrow(5.4, 3.35, 5.4, 2.95, TEAL)
arrow(10.0, 5.6, 10.0, 6.0, AMBER, rad=0.0)
ax.text(10.25, 5.75, "elites", fontsize=9.5, color=AMBER, style="italic")

# inputs into search
arrow(3.0, 6.85, 3.8, 6.85, INK, "failure obs")
arrow(3.0, 4.5, 3.8, 4.5, BLUE, "continuation")

# WM training (M2) below left
box(0.3, 0.35, 2.95, 2.55, "M2  Train WM + critic", "C1: 144 train failures × 32\ncandidates, sim outcomes\nWM: latent dynamics (ensemble)\ncritic: real + imagined states\n(hard targets excluded)", BLU_T, fs=12)
arrow(3.25, 2.3, 7.35, 3.75, BLUE, rad=-0.2, ls="--")
ax.text(4.75, 3.12, "trained WM + critic", fontsize=9.5, color=BLUE, style="italic")
ax.text(4.3, 2.95, "", fontsize=9)
arrow(5.0, 1.45, 3.25, 0.9, TEAL, rad=-0.1, ls="--")
ax.text(4.15, 0.75, "new labels → critic v2 (M5)", fontsize=9.5, color=TEAL, style="italic", ha="center")

# Distill + evaluate
box(11.3, 5.6, 4.4, 2.2, "4  Recovery dataset", "verified trajectories as LeRobotDataset\nsegment = searched recovery\n+ SmolVLA continuation\n(images, state, action per step)", TEA_T)
box(11.3, 3.0, 4.4, 1.9, "5  Distill → pi1", "SFT SmolVLA (init pi1-nominal)\n50% LIBERO demos + 50% recoveries\nflow-matching loss", BLU_T)
box(11.3, 0.6, 4.4, 1.75, "6  Evaluate (no search)", "56 held-out test scenes × 10 trials\nRSR 0.12 (pi0) → 0.29 (pi1)\n+ nominal success", "#23303B", tc="white", bc="#DDE3E8")
arrow(10.55, 2.2, 11.3, 6.4, TEAL, rad=-0.2)
arrow(13.5, 5.6, 13.5, 4.9, TEAL); arrow(13.5, 3.0, 13.5, 2.35, BLUE)
arrow(15.7, 1.45, 15.7, 7.75, RED, rad=0.3, ls="--")
ax.text(15.7, 8.05, "loop (M6): π1 → mine its remaining failures → search again", fontsize=10, color=RED, ha="right", style="italic")

plt.savefig(REPO / "slides/assets/pipeline_overview.png", dpi=150, bbox_inches="tight", facecolor="white")
print("saved")
