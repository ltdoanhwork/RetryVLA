"""M2 skeleton: action-conditioned latent dynamics  h_{t+1} = F_phi(h_t, a_t).

Interface used by ``WMCriticScorer``:
    rollout(h0 [N,T,D], actions [N,H,A]) -> (z_H [N,T_vis,D], q_H [N,8], uncertainty [N] | None)
Uncertainty = disagreement across ``n_ensemble`` dynamics heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DynamicsTransformer(nn.Module):
    def __init__(self, dim: int = 384, n_tokens: int = 129, action_dim: int = 7, depth: int = 6, heads: int = 6,
                 action_repeat: int = 5, n_ensemble: int = 3, state_dim: int = 8):
        super().__init__()
        self.dim, self.n_tokens, self.k = dim, n_tokens, action_repeat
        self.action_emb = nn.Linear(action_dim * action_repeat, dim)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, depth)
        self.heads = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_ensemble)])
        self.q_head = nn.Linear(dim, state_dim)

    def step(self, h: torch.Tensor, a_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h [N,T,D], a_chunk [N,k,A] -> (mean next h [N,T,D], ensemble var [N])."""
        x = torch.cat([h, self.action_emb(a_chunk.flatten(1))[:, None]], 1) + self.pos
        y = self.trunk(x)[:, :-1]
        preds = torch.stack([h + head(y) for head in self.heads])  # residual
        return preds.mean(0), preds.var(0).mean((1, 2))

    def predict_state(self, h: torch.Tensor) -> torch.Tensor:
        return self.q_head(h[:, -1])

    def rollout(self, h0: torch.Tensor, actions: torch.Tensor):
        n, hz, a = actions.shape
        hz = hz - hz % self.k
        h, unc = h0, torch.zeros(n, device=h0.device)
        for t in range(0, hz, self.k):
            h, v = self.step(h, actions[:, t: t + self.k])
            unc = unc + v
        return h[:, :-1], self.predict_state(h), unc


def wm_loss(model: DynamicsTransformer, h: torch.Tensor, a: torch.Tensor, q: torch.Tensor,
            lam_z: float = 1.0, lam_q: float = 1.0, lam_m: float = 0.5, multistep: int = 4) -> dict:
    """h [B, S+1, T, D] encoded sequence, a [B, S*k, A], q [B, S+1, 8].
    L = lam_z ||z_hat_{t+1}-z_{t+1}||^2 + lam_q ||q_hat-q||^2 + lam_m * L_multistep (open-loop)."""
    k = model.k
    pred, _ = model.step(h[:, 0], a[:, :k])
    l1 = nn.functional.mse_loss(pred[:, :-1], h[:, 1, :-1])
    lq = nn.functional.mse_loss(model.predict_state(pred), q[:, 1])
    lm = torch.zeros((), device=h.device)
    hh = h[:, 0]
    s = min(multistep, h.shape[1] - 1)
    for t in range(s):
        hh, _ = model.step(hh, a[:, t * k:(t + 1) * k])
        lm = lm + nn.functional.mse_loss(hh[:, :-1], h[:, t + 1, :-1]) / s
    return {"loss": lam_z * l1 + lam_q * lq + lam_m * lm, "z1": l1, "q1": lq, "multistep": lm}


@torch.no_grad()
def horizon_errors(model: DynamicsTransformer, h: torch.Tensor, a: torch.Tensor,
                   horizons=(1, 4, 8, 16)) -> dict[int, float]:
    """Open-loop latent MSE at the given horizons (in model steps = action chunks of size k)."""
    out, hh, k = {}, h[:, 0], model.k
    for t in range(max(horizons)):
        if t + 1 >= h.shape[1]:
            break
        hh, _ = model.step(hh, a[:, t * k:(t + 1) * k])
        if t + 1 in horizons:
            out[t + 1] = float(nn.functional.mse_loss(hh[:, :-1], h[:, t + 1, :-1]))
    return out
