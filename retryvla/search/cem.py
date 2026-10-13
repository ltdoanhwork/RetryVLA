"""Generic batched Cross-Entropy Method over open-loop action sequences [H, A]."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def smooth_noise(rng: np.random.RandomState, n: int, h: int, a: int, kernel: int) -> np.ndarray:
    """Temporally correlated N(0,1)-ish noise (box filter, renormalized)."""
    z = rng.randn(n, h + kernel - 1, a)
    if kernel > 1:
        c = np.cumsum(z, axis=1)
        c = np.concatenate([np.zeros((n, 1, a)), c], axis=1)
        z = (c[:, kernel:] - c[:, :-kernel]) / np.sqrt(kernel)
    return z[:, :h]


@dataclass
class CEMConfig:
    n_samples: int = 64
    n_elites: int = 8
    min_std: float = 0.05
    init_std: float = 0.4
    smooth_kernel: int = 5
    alpha: float = 0.7          # weight on the new estimate when refitting
    low: float = -1.0
    high: float = 1.0
    gripper_dim: int | None = 6  # sampled gripper is snapped to {-1, +1}


class CEM:
    def __init__(self, horizon: int, action_dim: int, cfg: CEMConfig, seed: int = 0):
        self.h, self.a, self.cfg = horizon, action_dim, cfg
        self.rng = np.random.RandomState(seed)
        self.mean = np.zeros((horizon, action_dim))
        self.std = np.full((horizon, action_dim), cfg.init_std)
        self.fitted = False

    def fit_initial(self, elites: np.ndarray) -> None:
        self.mean = elites.mean(0)
        self.std = np.maximum(elites.std(0), self.cfg.min_std)
        self.std = np.maximum(self.std, 0.5 * self.cfg.init_std)  # keep exploration after init
        self.fitted = True

    def refit(self, elites: np.ndarray) -> None:
        if not self.fitted:
            return self.fit_initial(elites)
        a = self.cfg.alpha
        self.mean = a * elites.mean(0) + (1 - a) * self.mean
        self.std = np.maximum(a * elites.std(0) + (1 - a) * self.std, self.cfg.min_std)

    def sample(self, n: int | None = None) -> np.ndarray:
        n = n or self.cfg.n_samples
        z = smooth_noise(self.rng, n, self.h, self.a, self.cfg.smooth_kernel)
        x = np.clip(self.mean[None] + self.std[None] * z, self.cfg.low, self.cfg.high)
        if self.cfg.gripper_dim is not None:
            g = x[..., self.cfg.gripper_dim]
            x[..., self.cfg.gripper_dim] = np.where(g >= 0, 1.0, -1.0)
        return x.astype(np.float32)

    @staticmethod
    def select_elites(cands: np.ndarray, scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        idx = np.argsort(-scores)[:k]
        return cands[idx], idx


def roughness(actions: np.ndarray) -> np.ndarray:
    """Mean squared first difference of the arm dims, per candidate. actions [N,H,A]."""
    d = np.diff(actions[..., :6], axis=1)
    return (d ** 2).mean(axis=(1, 2))
