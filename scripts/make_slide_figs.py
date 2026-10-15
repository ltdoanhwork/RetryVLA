#!/usr/bin/env python
"""Figures for the RetryVLA progress deck -> slides/assets/*.png (consistent palette, no chartjunk)."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "slides" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

DEV = "cpu"
INK, MUTED, AMBER, BLUE, TEAL, RED, GRID = "#23303B", "#6B7785", "#E8962E", "#1F8FE5", "#2A9D8F", "#C8553D", "#E3E7EB"


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False})
    return plt


def fig_dino_tokens():
    """Agent-view frame + its DINOv2 patch tokens (PCA -> RGB), full 16x16 and the pooled 6x6 the WM uses."""
    from sklearn.decomposition import PCA

    from retryvla.wm.encoder_dino import DinoEncoder
    from retryvla.envs.libero_env import LiberoSim, discover_scenes
    from retryvla.eval.eval_recover import noise_seed

    plt = style()
    sc = next(s for s in discover_scenes(REPO / "data/assets/scenes/scenes/selected_bddl_scenes_spatial")
              if "top_drawer" in s.tag and "episode_031__step_0180" in s.tag)
    sim = LiberoSim()
    o = sim.load_scene(sc.bddl, sc.npy, gripper_close=False, noise_xy=0.02, noise_seed=noise_seed(7, sc.tag, 1),
                       settle_steps=20, build_seed=7)["obs"]
    sim.close()
    enc = DinoEncoder("facebook/dinov2-small", pool_grid=None, device=DEV)
    with torch.no_grad():
        t16 = enc.encode_images(torch.as_tensor(o["image"], device=DEV)[None])[0].float().cpu().numpy()  # 256x384
    t6 = torch.nn.functional.adaptive_avg_pool2d(torch.as_tensor(t16).T.reshape(384, 16, 16)[None], 6)[0]
    t6 = t6.reshape(384, 36).T.numpy()
    pca = PCA(3).fit(t16)

    def rgb(t, g):
        x = pca.transform(t)
        x = (x - x.min(0)) / (x.max(0) - x.min(0) + 1e-6)
        return x.reshape(g, g, 3)

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for a in ax:
        a.grid(False); a.set_xticks([]); a.set_yticks([])
    ax[0].imshow(o["image"]); ax[0].set_title("Camera (256×256)")
    ax[1].imshow(rgb(t16, 16), interpolation="nearest"); ax[1].set_title("DINOv2 patch tokens 16×16")
    ax[2].imshow(rgb(t6, 6), interpolation="nearest"); ax[2].set_title("Pooled 6×6 = WM input")
    plt.tight_layout()
    plt.savefig(OUT / "dino_tokens.png", dpi=170, transparent=False, facecolor="white")
    plt.close(fig)
    import imageio
    imageio.imwrite(OUT / "drawer_failure_frame.png", o["image"])
    imageio.imwrite(OUT / "drawer_failure_wrist.png", o["image2"])


def fig_wm_imagination():
    """WM-imagined vs real end-effector trajectory for verified candidates (held-out instance)."""
    from retryvla.critic.train_critic import load_wm
    from retryvla.wm.data import load_latents

    plt = style()
    wm_ck = (REPO / "data/wm/latest_wm.txt").read_text().strip()
    wm, norm, ck = load_wm(wm_ck, torch.device(DEV))
    val = set(ck["val_instances"])
    ls = load_latents(["data/wm/collect_v1"])
    idx = [i for i, s in enumerate(ls.inst) if s in val and bool(ls.valid[i].all())]
    rng = np.random.default_rng(3)
    pick = rng.choice(idx, 4, replace=False)
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.3), sharey=True)
    for a, i in zip(ax, pick):
        z0 = norm.z(ls.z[i:i + 1, 0].to(DEV)); q0 = norm.q(ls.q[i:i + 1, 0].to(DEV))
        with torch.no_grad():
            _, _, _, zs, qs = wm.rollout(z0, q0, ls.actions[i:i + 1].to(DEV))
        qp = torch.cat([q0] + [q for q in qs]).cpu() * norm.q_std + norm.q_mean
        qr = ls.q[i]
        t = np.arange(11) * 5
        a.plot(t, qr[:, 2], color=INK, lw=2.2, label="real (simulator)")
        a.plot(t, qp[:, 2].numpy(), color=AMBER, lw=2.2, ls="--", marker="o", ms=3.5, label="WM imagination")
        a.set_title(("success" if bool(ls.success[i]) else "failure") + " after continuation", fontsize=10,
                    color=TEAL if bool(ls.success[i]) else RED)
        a.set_xlabel("env step")
    ax[0].set_ylabel("gripper height z (m)")
    ax[0].legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "wm_imagination.png", dpi=170, facecolor="white")
    plt.close(fig)


def fig_wm_horizon():
    plt = style()
    h = [5, 10, 20, 30, 40, 50]
    r = [0.629, 0.500, 0.396, 0.361, 0.359, 0.356]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.axhline(1.0, color=MUTED, ls=":", lw=1.5)
    ax.text(50, 1.03, "copy-last-frame baseline", color=MUTED, ha="right", fontsize=9)
    ax.plot(h, r, color=AMBER, lw=2.5, marker="o")
    for x, y in zip(h, r):
        ax.text(x, y + 0.05, f"{y:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_ylim(0, 1.15); ax.set_xlabel("open-loop horizon (env steps)"); ax.set_ylabel("latent MSE / baseline")
    ax.set_title("WM prediction error on held-out failures")
    plt.tight_layout(); plt.savefig(OUT / "wm_horizon.png", dpi=170, facecolor="white"); plt.close(fig)


def fig_critic_scores():
    """Critic V on verified candidates (M5 round-1 verifications so far): success vs failure."""
    plt = style()
    fs = glob.glob(str(REPO / "data/wm/verif_r1/*.pt"))
    if not fs:
        return
    pv, y = [], []
    for f in fs:
        d = torch.load(f, weights_only=False)
        pv += d["pred_value"].tolist(); y += d["success"].tolist()
    pv, y = np.array(pv), np.array(y, bool)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    bins = np.linspace(0, 1, 21)
    ax.hist(pv[~y], bins, color=GRID, edgecolor=MUTED, label=f"verified failure (n={(~y).sum()})")
    ax.hist(pv[y], bins, color=TEAL, alpha=0.9, label=f"verified success (n={y.sum()})")
    ax.set_xlabel("critic V (predicted P success) of chosen candidates"); ax.set_ylabel("count")
    ax.set_title("Critic is over-confident on its picks"); ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / "critic_scores.png", dpi=170, facecolor="white"); plt.close(fig)


def fig_split():
    plt = style()
    from retryvla.envs.libero_env import discover_scenes
    from retryvla.envs.splits import load_split

    sp = load_split(REPO / "data/splits/spatial_v1.json")
    sc = {s.tag: s.task_folder.replace("pick_up_the_black_bowl_", "").split("_and_place")[0].replace("_", " ")
          for s in discover_scenes(REPO / "data/assets/scenes/scenes/selected_bddl_scenes_spatial")}
    tasks = sorted(set(sc.values()), key=lambda t: -sum(v == t for v in sc.values()))
    pil, tr, te = (np.array([sum(sc[x] == t for x in sp[p]) for t in tasks]) for p in ("pilot", "train", "test"))
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    y = np.arange(len(tasks))
    ax.barh(y, pil, color=AMBER, label="pilot (search)")
    ax.barh(y, tr - pil, left=pil, color="#F3C98B", label="train (rest)")
    ax.barh(y, te, left=tr, color=INK, label="test (never searched / trained)")
    ax.set_yticks(y, [t[:34] for t in tasks]); ax.invert_yaxis(); ax.set_xlabel("failure scenes")
    ax.legend(fontsize=9, loc="lower right"); ax.grid(axis="y", visible=False)
    ax.set_title("Scenes per task and split")
    plt.tight_layout(); plt.savefig(OUT / "split.png", dpi=170, facecolor="white"); plt.close(fig)


def copy_existing():
    import shutil

    viz = REPO / "runs/_analysis/viz"
    for src, dst in [
        ("0_nominal_libero_demo/strip.png", "strip_demo.png"),
        ("1_trivial_open_gripper/strip.png", "strip_trivial.png"),
        ("2_primitive_then_smolvla/strip.png", "strip_primitive.png"),
        ("3_cem_hard_instance/strip.png", "strip_cem_bad.png"),
        ("3_cem_hard_instance/actions.png", "actions_cem_bad.png"),
        ("4_vla_sample_recovery/strip.png", "strip_vla_sample.png"),
        ("c1_between_the_plate_and_the_rame_k11/strip.png", "strip_c1_success.png"),
        ("c1_between_the_plate_and_the_rame_k0/strip.png", "strip_c1_fail.png"),
        ("c4_wm_in_the_top_drawer_of_t_episode_031_step_0180_open1/strip.png", "strip_wm_drawer.png"),
        ("c4_wm_in_the_top_drawer_of_t_episode_031_step_0180_open1/actions.png", "actions_wm_drawer.png"),
        ("compare_drawer_ep031/grid.png", "compare_grid.png"),
        ("c1_summary.png", "c1_summary.png"),
    ]:
        if (viz / src).exists():
            shutil.copy(viz / src, OUT / dst)
    g = sorted(glob.glob(str(REPO / "runs/*verify_scenes/failure_states_grid.png")))
    if g:
        shutil.copy(g[-1], OUT / "failure_grid.png")
    for f in ["m1_search_overview.png", "m1_policy_results.png"]:
        shutil.copy(REPO / "experiments/figs" / f, OUT / f)


if __name__ == "__main__":
    copy_existing()
    fig_split()
    fig_wm_horizon()
    fig_critic_scores()
    fig_dino_tokens()
    fig_wm_imagination()
    print(sorted(p.name for p in OUT.glob("*.png")))
