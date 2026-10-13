"""M2 skeleton: Recovery Value Model  V_psi(z, q, g) = P(nominal VLA succeeds | z, q, g).

Operates on the same token space as the world model so it can score *imagined* end states.
Calibration (ECE / reliability curve) must be measured on held-out simulator states.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RecoveryValue(nn.Module):
    def __init__(self, dim: int = 384, state_dim: int = 8, text_dim: int = 384, depth: int = 2, heads: int = 6):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, dim)
        self.q_proj = nn.Linear(state_dim, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(dim, 1)
        self._text_cache: dict[str, torch.Tensor] = {}
        self.text_encoder = None  # set to a callable str -> [text_dim] tensor

    def embed_text(self, tasks: list[str], device) -> torch.Tensor:
        if self.text_encoder is None:
            raise RuntimeError("RecoveryValue.text_encoder not set")
        out = []
        for t in tasks:
            if t not in self._text_cache:
                self._text_cache[t] = self.text_encoder(t).detach()
            out.append(self._text_cache[t])
        return torch.stack(out).to(device)

    def logits(self, z: torch.Tensor, q: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        n = z.shape[0]
        x = torch.cat([self.cls.expand(n, -1, -1), self.text_proj(g)[:, None], self.q_proj(q)[:, None], z], 1)
        return self.head(self.trunk(x)[:, 0]).squeeze(-1)

    def forward(self, z: torch.Tensor, q: torch.Tensor, tasks: list[str]) -> torch.Tensor:
        return torch.sigmoid(self.logits(z, q, self.embed_text(tasks, z.device)))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)
