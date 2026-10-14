"""Recovery Value Model  V_psi(z, q, g) = P(frozen pi0 completes the task | latent state, task).

Operates on the same normalized token space as the latent WM so it can score *imagined* end states.
Task conditioning = learned embedding over the (small, known) task vocabulary.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RecoveryValue(nn.Module):
    def __init__(self, dim: int = 384, n_patch: int = 72, n_tasks: int = 16, width: int = 256, depth: int = 3,
                 heads: int = 4, state_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(dim, width)
        self.q_proj = nn.Linear(state_dim, width)
        self.task = nn.Embedding(n_tasks, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.pos = nn.Parameter(torch.zeros(1, n_patch + 3, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.trunk = nn.TransformerEncoder(layer, depth)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def logits(self, z: torch.Tensor, q: torch.Tensor, task_idx: torch.Tensor) -> torch.Tensor:
        """z [N,P,D] normalized tokens, q [N,8] normalized, task_idx [N] long."""
        n = z.shape[0]
        x = torch.cat([self.cls.expand(n, -1, -1), self.task(task_idx)[:, None], self.q_proj(q)[:, None],
                       self.in_proj(z)], 1) + self.pos
        return self.head(self.trunk(x)[:, 0]).squeeze(-1)

    def forward(self, z, q, task_idx) -> torch.Tensor:
        return torch.sigmoid(self.logits(z, q, task_idx))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def auc(p: np.ndarray, y: np.ndarray) -> float:
    p, y = np.asarray(p, float), np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))
