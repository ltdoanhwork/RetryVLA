#!/usr/bin/env python
"""Phase-1 retrospective analysis of the M5b verification logs (no new simulation).

python scripts/analyze_retro.py   ->  runs/_analysis/retro/{summary.md, budget_curve.png, calibration.png, dG_dV.png}

From data/wm/verif_m5b/*.pt (per instance: 4 arms x 16 verified candidates, in each arm's verification order, with
REAL latents at t = 0, 5, ..., 50, the critic's predicted value of the IMAGINED end state, and the verified label):
  1. budget curve: solve rate / hits per sim when an arm only verifies its first k candidates (k = 1..16)
  2. critic calibration (imagined-end-state V vs verified success): AUC, ECE, reliability; success by rank
  3. Delta G vs Delta V on REAL end states: G = -min latent distance to successful-demo goal frames (DINO-WM style),
     V = competence critic. Count verified recoveries with dG < 0 but dV > 0 (moved AWAY from the goal, yet restored
     the policy's competence) -- illustrative only.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "runs/_analysis/retro"
ARMS = ["random", "wm_diverse", "wm_noise", "wm_qd"]


def auc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or (~y).all():
        return float("nan")
    r = s.argsort().argsort() + 1
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def ece(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e, rel = 0.0, []
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (p >= lo) & (p < lo + 1 / bins)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
            rel.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return float(e), rel


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from retryvla.critic.train_critic import load_wm
    from retryvla.critic.value_model import RecoveryValue
    from retryvla.envs.libero_env import discover_scenes

    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, norm, _ = load_wm((REPO / "data/wm/latest_wm.txt").read_text().strip(), dev)
    ck = torch.load((REPO / "data/wm/latest_critic_v2.txt").read_text().strip(), map_location="cpu", weights_only=False)
    c = ck["cfg"]
    critic = RecoveryValue(dim=ck["dim"], n_patch=ck["n_patch"], n_tasks=max(16, len(ck["vocab"])), width=int(c["width"]),
                           depth=int(c["depth"]), dropout=0.0).to(dev).eval()
    critic.load_state_dict(ck["model"])
    goals = torch.load(REPO / "data/wm/goal_feats.pt", weights_only=False)
    canon = {s.tag: s.language_canonical for s in discover_scenes(REPO / "data/assets/scenes/scenes/selected_bddl_scenes_spatial")}

    seq = defaultdict(list)          # arm -> list of per-instance success sequences (verification order)
    pred = defaultdict(list)         # arm -> [(pred V imagined, success, rank)]
    dgdv = []                        # (arm, success, dG, dV, task)
    files = sorted(glob.glob(str(REPO / "data/wm/verif_m5b/*.pt")))
    for f in files:
        d = torch.load(f, weights_only=False, map_location="cpu")
        src, y, pv = list(d["source"]), d["success"].numpy().astype(bool), d["pred_value"].numpy()
        task_id = critic_vocab = ck["vocab"].get(d["task"], 0)
        z, q, valid = d["z"].float(), d["q"].float(), d["valid"].numpy().astype(bool)
        last = valid.sum(1) - 1                              # last valid frame (t=50, or terminal if success earlier)
        idx = torch.as_tensor(last)
        zH = z[torch.arange(len(z)), idx]
        qH = q[torch.arange(len(q)), idx]
        with torch.no_grad():
            Z0, Q0 = norm.z(z[:, 0].to(dev)), norm.q(q[:, 0].to(dev))
            ZH, QH = norm.z(zH.to(dev)), norm.q(qH.to(dev))
            tid = torch.full((len(z),), task_id, device=dev, dtype=torch.long)
            V0 = torch.sigmoid(critic.logits(Z0, Q0, tid)).cpu().numpy()
            VH = torch.sigmoid(critic.logits(ZH, QH, tid)).cpu().numpy()
            g = norm.z(goals[canon[d["scene"]]].to(dev).float())
            G = lambda Z: (-torch.stack([((Z - gg[None]) ** 2).mean((1, 2)) for gg in g]).min(0).values).cpu().numpy()
            G0, GH = G(Z0), G(ZH)
        task = d["scene"].split("__")[0].replace("pick_up_the_black_bowl_", "").split("_and_place")[0]
        for a in ARMS:
            m = np.array([s == a for s in src])
            if not m.any():
                continue
            seq[a].append(y[m])
            for r, (p_, yy) in enumerate(zip(pv[m], y[m])):
                pred[a].append((float(p_), bool(yy), r))
            for yy, dg, dv in zip(y[m], (GH - G0)[m], (VH - V0)[m]):
                dgdv.append((a, bool(yy), float(dg), float(dv), task))
    n_inst = len(files)

    # 1) budget curve
    K = list(range(1, 17))
    curve = {a: {"solve": [float(np.mean([s[:k].any() for s in seq[a]])) for k in K],
                 "hit": [float(np.mean([s[:k].mean() for s in seq[a]])) for k in K]} for a in ARMS}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, col in zip(ARMS, ["#9AA5B1", "#E8962E", "#1F8FE5", "#2A9D8F"]):
        ax[0].plot(K, curve[a]["solve"], "-o", ms=3, color=col, label=a)
        ax[1].plot(K, curve[a]["hit"], "-o", ms=3, color=col, label=a)
    ax[0].set_xlabel("simulator verifications per failure (k)"); ax[0].set_ylabel("failures with >=1 verified recovery")
    ax[1].set_xlabel("k"); ax[1].set_ylabel("verified successes per verification")
    ax[0].set_xscale("log", base=2); ax[1].set_xscale("log", base=2); ax[0].legend(frameon=False)
    ax[0].set_title(f"Budget curve (prefix of each arm's order, N={n_inst})"); plt.tight_layout()
    plt.savefig(OUT / "budget_curve.png", dpi=150); plt.close()

    # 2) calibration (WM arms: V of the IMAGINED end state; random arm also scored by the same critic)
    lines = [f"# Retrospective analysis of M5b verification logs (N = {n_inst} instances)", "",
             "## 1. Budget curve (first k verifications of each arm's order)", "",
             "| arm | k=1 | k=2 | k=4 | k=8 | k=16 | hits/sim k=4 | k=16 |", "|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        s, h = curve[a]["solve"], curve[a]["hit"]
        lines.append(f"| {a} | {s[0]:.2f} | {s[1]:.2f} | {s[3]:.2f} | {s[7]:.2f} | {s[15]:.2f} | {h[3]:.3f} | {h[15]:.3f} |")
    lines += ["", "Caveat: for WM arms, k<16 truncates the k=16 diverse list (not a re-run with budget k).", "",
              "## 2. Critic calibration (predicted V of the imagined end state vs verified label)", "",
              "| arm | n | AUC | ECE | mean pred | success rate | success rank 1-4 | rank 13-16 |", "|---|---|---|---|---|---|---|---|"]
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    allp = [x for a in ARMS for x in pred[a]]
    for a, rows in list(pred.items()) + [("all", allp)]:
        p_, y_, r_ = np.array([x[0] for x in rows]), np.array([x[1] for x in rows]), np.array([x[2] for x in rows])
        e, rel = ece(y_, p_)
        lines.append(f"| {a} | {len(rows)} | {auc(y_, p_):.3f} | {e:.3f} | {p_.mean():.3f} | {y_.mean():.3f} | "
                     f"{y_[r_ < 4].mean():.3f} | {y_[r_ >= 12].mean():.3f} |")
        if a == "all":
            ax.plot([x[0] for x in rel], [x[1] for x in rel], "-o", color="#E8962E", label="critic v2")
    ax.plot([0, 1], [0, 1], "--", color="grey"); ax.set_xlabel("predicted V (imagined)"); ax.set_ylabel("verified success")
    ax.set_title("Reliability (all verified candidates)"); plt.tight_layout(); plt.savefig(OUT / "calibration.png", dpi=150); plt.close()

    # 3) dG vs dV on real end states
    D = np.array([(x[2], x[3]) for x in dgdv]); Y = np.array([x[1] for x in dgdv])
    away = (D[:, 0] < 0) & (D[:, 1] > 0)
    lines += ["", "## 3. Goal progress vs competence on REAL end states (all verified candidates)", "",
              f"- verified recoveries: {int(Y.sum())}; of these with dG < 0 and dV > 0 (moved away from the goal yet "
              f"raised competence): **{int((Y & away).sum())} ({(Y & away).sum() / max(1, Y.sum()):.0%})**",
              f"- success rate when dG < 0 & dV > 0: {Y[away].mean():.3f} (n={int(away.sum())}); "
              f"dG > 0 & dV < 0: {Y[(D[:, 0] > 0) & (D[:, 1] < 0)].mean():.3f} (n={int(((D[:, 0] > 0) & (D[:, 1] < 0)).sum())})",
              f"- AUC for predicting verified success from REAL end state: G(s_H) dG {auc(Y, D[:, 0]):.3f}, dV {auc(Y, D[:, 1]):.3f}",
              "", "| task | recoveries | % with dG<0, dV>0 |", "|---|---|---|"]
    by_t = defaultdict(lambda: [0, 0])
    for (a, yy, dg, dv, t) in dgdv:
        if yy:
            by_t[t][0] += 1
            by_t[t][1] += int(dg < 0 and dv > 0)
    for t, (n_, k_) in sorted(by_t.items(), key=lambda x: -x[1][0]):
        lines.append(f"| {t} | {n_} | {k_ / n_:.0%} |")
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.scatter(D[~Y, 0], D[~Y, 1], s=3, c="#C9CED4", label="verified failure")
    ax.scatter(D[Y, 0], D[Y, 1], s=6, c="#E8962E", label="verified recovery")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("ΔG (goal-latent progress, real)"); ax.set_ylabel("ΔV (competence, real)"); ax.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "dG_dV.png", dpi=150); plt.close()
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    json.dump(curve, open(OUT / "budget_curve.json", "w"))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
