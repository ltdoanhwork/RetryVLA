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

    def sample_local(self, elites: np.ndarray, elite_scores: np.ndarray, iteration: int, n: int | None = None,
                     temperature: float = 0.2, decay: float = 0.6, gripper_flip_p: float = 0.05) -> np.ndarray:
        """Elite-local refinement (fixes Gaussian mode-averaging): each sample perturbs ONE elite,
        chosen with softmax(score/T); noise std decays per iteration. The elite's gripper schedule is
        kept (rarely flipped per-step-run) so a strategy's structure survives refinement."""
        n = n or self.cfg.n_samples
        s = np.asarray(elite_scores, dtype=np.float64)
        p = np.exp((s - s.max()) / max(temperature, 1e-6))
        p /= p.sum()
        pick = self.rng.choice(len(elites), size=n, p=p)
        std = max(self.cfg.init_std * (decay ** iteration), self.cfg.min_std)
        z = smooth_noise(self.rng, n, self.h, self.a, self.cfg.smooth_kernel) * std
        x = elites[pick].astype(np.float64) + z
        if self.cfg.gripper_dim is not None:
            g = self.cfg.gripper_dim
            x[..., g] = elites[pick][..., g]
            flip = self.rng.rand(n) < gripper_flip_p * self.h / 10
            if flip.any():  # flip the gripper from a random step onward for a few samples
                t0 = self.rng.randint(0, self.h, size=n)
                for i in np.nonzero(flip)[0]:
                    x[i, t0[i]:, g] *= -1
        return np.clip(x, self.cfg.low, self.cfg.high).astype(np.float32)


def roughness(actions: np.ndarray) -> np.ndarray:
    """Mean squared first difference of the arm dims, per candidate. actions [N,H,A]."""
    d = np.diff(actions[..., :6], axis=1)
    return (d ** 2).mean(axis=(1, 2))
