"""Train the Back on Track liveness ensemble + PCA action basis for one (policy, suite).

python -m retryvla.baselines.bot.train --data data/bot/sp_pi0 --wm_link data/wm/latest_wm.txt --out data/bot/sp_pi0/bot.pt \
    --horizon 20 [--play_roots data/wm/collect_v1]

- value: retryvla.baselines.bot.liveness (two-stage discounted liveness, gamma 0.993 per env step, 5 members)
- PCA basis (2610.19579 Eq. 4): empirical covariance of flattened action chunks [H x 6] (gripper excluded) of the
  rollouts (+ optional play data = our WM's own training actions, their "play data" role), top-k eigenvectors that
  explain 98 % of the variance.
- val: 15 % of scenes held out (episode-level) -> success / failure metrics of 2605.11479 + frame AUC (diagnostic).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import zlib

import numpy as np
import torch

from retryvla import REPO_ROOT
from retryvla.baselines.bot.liveness import load_frames, train_liveness
from retryvla.critic.train_critic import load_wm
from retryvla.utils import setup_logging

log = logging.getLogger("bot.train")


def goal_demo_windows(H: int, split_name: str = "goal_rsi_v1", part: str = "train") -> list:
    """LIBERO-Goal teleop demos of the TRAIN split (their play data: LIBERO-Goal demonstrations), H-step windows."""
    import h5py

    sp = json.loads((REPO_ROOT / f"data/splits/{split_name}.json").read_text())
    W = []
    for v in sp["tasks"].values():
        with h5py.File(v["hdf5"]) as h:
            for d in v[part]:
                a = np.asarray(h[f"data/demo_{d}"]["actions"])[:, :6]
                for s in range(0, len(a) - H + 1, 5):
                    W.append(a[s:s + H].reshape(-1))
    return W


def action_windows(roots, H: int, play_roots=(), max_play: int = 20000, seed: int = 0,
                   goal_demos: bool = False) -> np.ndarray:
    W = []
    for r in roots:
        for f in sorted(glob.glob(f"{REPO_ROOT / r}/part_*.pt")):
            for e in torch.load(f, weights_only=False)["episodes"]:
                a = np.asarray(e["actions"])[:, :6]
                for s in range(0, len(a) - H + 1, 5):
                    W.append(a[s:s + H].reshape(-1))
    n_roll = len(W)
    if goal_demos:
        W += goal_demo_windows(H)
    rng = np.random.default_rng(seed)
    P = []
    for r in play_roots:
        for f in sorted(glob.glob(f"{REPO_ROOT / r}/*.pt")):
            a = torch.load(f, weights_only=False)["actions"].numpy()[:, :, :6]      # [K, 50, 6]
            for s in range(0, a.shape[1] - H + 1, 10):
                P.append(a[:, s:s + H].reshape(a.shape[0], -1))
    if P:
        P = np.concatenate(P)
        P = P[rng.choice(len(P), min(max_play, len(P)), replace=False)]
        W = np.concatenate([np.stack(W), P])
    else:
        W = np.stack(W)
    log.info("PCA windows: %d rollout + %d play (H=%d)", n_roll, len(W) - n_roll, H)
    return W.astype(np.float64)


def pca_basis(W: np.ndarray, var: float = 0.98):
    C = np.cov(W, rowvar=False)
    lam, V = np.linalg.eigh(C)
    o = np.argsort(-lam)
    lam, V = np.clip(lam[o], 0, None), V[:, o]
    k = int(np.searchsorted(np.cumsum(lam) / lam.sum(), var) + 1)
    return V[:, :k].astype(np.float32), lam[:k].astype(np.float32), k


def main():
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--wm_link", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizon", type=int, required=True, help="planner horizon H (env steps)")
    ap.add_argument("--play_roots", nargs="*", default=[])
    ap.add_argument("--play_goal_demos", action="store_true", help="play data = LIBERO-Goal train-split demos")
    ap.add_argument("--gamma", type=float, default=0.993)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--steps1", type=int, default=8000)
    ap.add_argument("--steps2", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps_ep", type=int, default=520, help="N_s of the failure metric")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm_ck = (REPO_ROOT / a.wm_link).read_text().strip()
    _, norm, _ = load_wm(wm_ck, "cpu")
    fs = load_frames([str(REPO_ROOT / d) for d in a.data], a.gamma)
    scenes = sorted(set(fs.ep_scene))
    val_sc = {s for s in scenes if zlib.crc32(f"bot:{s}".encode()) % 100 < 15}
    val_eps = {i for i, s in enumerate(fs.ep_scene) if s in val_sc}
    log.info("liveness data: %d episodes (SR %.3f), %d frames, val %d episodes", len(fs.ep_scene),
             fs.ep_success.mean(), len(fs.ep), len(val_eps))
    ens, m = train_liveness(fs, norm, dev, a.gamma, a.stride, n_members=a.members, lr=a.lr, steps1=a.steps1,
                            steps2=a.steps2, val_eps=val_eps, seed=a.seed)
    log.info("liveness metrics: %s", json.dumps(m))
    from retryvla.baselines.bot.liveness import train_progress

    prog, pm = train_progress(fs, norm, dev, val_eps=val_eps, seed=a.seed)
    m.update(pm)
    log.info("progress head: %s", json.dumps(pm))
    W = action_windows(a.data, a.horizon, a.play_roots, goal_demos=a.play_goal_demos)
    V, lam, k = pca_basis(W)
    log.info("PCA basis: k=%d of %d dims explain 98%% (H=%d, D=6)", k, W.shape[1], a.horizon)
    out = REPO_ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ensemble": ens.cpu().state_dict(), "progress": prog.state_dict(), "n_members": a.members, "n_tok": fs.x_zp.shape[1],
                "dim": fs.x_zp.shape[2], "vocab": fs.vocab, "gamma": a.gamma, "stride": a.stride,
                "wm_ckpt": wm_ck, "pca_V": V, "pca_lam": lam, "pca_k": k, "horizon": a.horizon,
                "metrics": m, "data": a.data, "play_roots": a.play_roots, "play_goal_demos": a.play_goal_demos,
                "n_episodes": len(fs.ep_scene), "sr": float(fs.ep_success.mean())}, out)
    (out.with_suffix(".json")).write_text(json.dumps({"metrics": m, "pca_k": k, "n_episodes": len(fs.ep_scene),
                                                     "sr": float(fs.ep_success.mean()), "data": a.data}, indent=1))
    print("BOT_TRAIN_DONE", out, flush=True)


if __name__ == "__main__":
    main()
