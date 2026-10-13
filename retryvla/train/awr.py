"""M5 skeleton: advantage-weighted recovery imitation.

L = E_tau[ w(tau) * L_action(tau) ],   w(tau) = exp(clip(A(tau), -c, c) / beta)

A(tau) = relative recovery advantage of a *verified* trajectory vs. the other branches of the same
failure instance (score - mean branch score), optionally calibrated with critic values. Weights are
normalized to mean 1 over the batch. Failed branches are not imitated (they feed the critic /
pairwise ranking instead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def trajectory_advantages(traj: pd.DataFrame, branches: pd.DataFrame) -> np.ndarray:
    """traj rows: verified trajectories (instance_id, score). branches: all branches of the search."""
    base = branches.groupby("instance_id")["score"].mean()
    return (traj["score"] - traj["instance_id"].map(base)).to_numpy(dtype=float)


def awr_weights(adv: np.ndarray | torch.Tensor, beta: float = 1.0, clip: float = 2.0) -> torch.Tensor:
    a = torch.as_tensor(adv, dtype=torch.float32)
    w = torch.exp(a.clamp(-clip, clip) / beta)
    return w / w.mean().clamp_min(1e-8)


def weighted_loss(per_sample_loss: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """per_sample_loss from ``policy.forward(batch, reduction='none')``."""
    w = weights.to(per_sample_loss.device)
    return (per_sample_loss * w).sum() / w.sum().clamp_min(1e-8)
