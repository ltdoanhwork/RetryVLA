"""Latent world model: h_{t+1} = h_t + F_phi(h_t, a_{t:t+k}) over frozen-DINOv2 patch tokens + robot state.

h_t = [z_t (P patch tokens of both cameras, normalized), E_q(q_t) (1 state token)]   -> [N, P+1, D]
One model step = one action chunk of k env steps (k = collection stride).
Ensemble of output heads -> disagreement used as uncertainty U_W in search.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentWM(nn.Module):
    def __init__(self, dim: int = 384, n_patch: int = 72, action_dim: int = 7, k: int = 5, width: int = 384,
                 depth: int = 6, heads: int = 6, n_ensemble: int = 4, state_dim: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim, self.P, self.k, self.A = dim, n_patch, k, action_dim
        self.in_proj = nn.Linear(dim, width)
        self.q_in = nn.Linear(state_dim, width)
        self.a_in = nn.Sequential(nn.Linear(action_dim * k, width), nn.GELU(), nn.Linear(width, width))
        self.pos = nn.Parameter(torch.zeros(1, n_patch + 2, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.trunk = nn.TransformerEncoder(layer, depth)
        self.z_heads = nn.ModuleList([nn.Linear(width, dim) for _ in range(n_ensemble)])
        self.q_heads = nn.ModuleList([nn.Linear(width, state_dim) for _ in range(n_ensemble)])
        for h in list(self.z_heads) + list(self.q_heads):  # start as identity dynamics (residual = 0)
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def step(self, z: torch.Tensor, q: torch.Tensor, a_chunk: torch.Tensor):
        """z [N,P,D], q [N,8] (normalized), a_chunk [N,k,A] -> z', q' (ensemble mean), unc [N]."""
        x = torch.cat([self.in_proj(z), self.q_in(q)[:, None], self.a_in(a_chunk.flatten(1))[:, None]], 1)
        y = self.trunk(x + self.pos)
        dz = torch.stack([h(y[:, : self.P]) for h in self.z_heads])     # [E,N,P,D]
        dq = torch.stack([h(y[:, self.P]) for h in self.q_heads])       # [E,N,8]
        unc = dz.var(0).mean((1, 2)) + dq.var(0).mean(1)
        return z + dz.mean(0), q + dq.mean(0), unc, (z[None] + dz, q[None] + dq)

    def rollout(self, z0: torch.Tensor, q0: torch.Tensor, actions: torch.Tensor):
        """Open-loop imagination. actions [N,H,A] with H a multiple of k.
        Returns final (z, q), per-step lists, and accumulated uncertainty [N]."""
        n, H, _ = actions.shape
        z, q, unc = z0, q0, torch.zeros(n, device=z0.device)
        zs, qs = [], []
        for t in range(0, H - H % self.k, self.k):
            z, q, u, _ = self.step(z, q, actions[:, t: t + self.k])
            unc = unc + u
            zs.append(z)
            qs.append(q)
        return z, q, unc, zs, qs


def wm_loss(model: LatentWM, z: torch.Tensor, q: torch.Tensor, a: torch.Tensor, m: torch.Tensor,
            lam_q: float = 1.0, lam_open: float = 1.0) -> dict:
    """z [B,F,P,D], q [B,F,8], a [B,H,A], m [B,F] valid-transition target mask (frame f usable as target).
    1-step teacher-forced loss on every valid transition (per ensemble member) + open-loop multistep loss."""
    B, F = z.shape[:2]
    k = model.k
    l1 = z.new_zeros(())
    lq = z.new_zeros(())
    lo = z.new_zeros(())
    n1 = no = 0
    zo, qo = z[:, 0], q[:, 0]
    for f in range(F - 1):
        tgt = m[:, f + 1] & m[:, f]
        if not tgt.any():
            break
        ac = a[:, f * k:(f + 1) * k]
        _, _, _, (ze, qe) = model.step(z[:, f], q[:, f], ac)           # teacher forcing (ensemble)
        E, P = ze.shape[0], ze.shape[2]
        w = tgt.float()
        l1 = l1 + (((ze - z[None, :, f + 1]) ** 2).mean(-1).sum(-1) * w[None]).sum() / (w.sum() * E * P + 1e-6)
        lq = lq + (((qe - q[None, :, f + 1]) ** 2).mean(-1) * w[None]).sum() / (w.sum() * E + 1e-6)
        zo, qo, _, _ = model.step(zo, qo, ac)                          # open loop
        lo = lo + (((zo - z[:, f + 1]) ** 2).mean(-1).sum(-1) * w).sum() / (w.sum() * P + 1e-6)
        lo = lo + lam_q * (((qo - q[:, f + 1]) ** 2).mean(-1) * w).sum() / (w.sum() + 1e-6)
        n1 += 1
        no += 1
    n1, no = max(n1, 1), max(no, 1)
    loss = l1 / n1 + lam_q * lq / n1 + lam_open * lo / no
    return {"loss": loss, "z1": (l1 / n1).detach(), "q1": (lq / n1).detach(), "open": (lo / no).detach()}


@torch.no_grad()
def horizon_errors(model: LatentWM, z: torch.Tensor, q: torch.Tensor, a: torch.Tensor, m: torch.Tensor,
                   horizons=(1, 2, 4, 6, 8, 10)) -> dict[int, dict[str, float]]:
    """Open-loop latent MSE vs. the copy-last-frame baseline at several horizons (model steps)."""
    out = {}
    k = model.k
    zo, qo = z[:, 0], q[:, 0]
    for f in range(z.shape[1] - 1):
        zo, qo, _, _ = model.step(zo, qo, a[:, f * k:(f + 1) * k])
        h = f + 1
        if h in horizons:
            v = m[:, h] & m[:, 0]
            if v.sum() == 0:
                continue
            e = ((zo - z[:, h]) ** 2).mean((1, 2))[v].mean().item()
            b = ((z[:, 0] - z[:, h]) ** 2).mean((1, 2))[v].mean().item()
            eq = ((qo - q[:, h]) ** 2).mean(1)[v].mean().item()
            out[h] = {"z_mse": e, "copy_baseline": b, "ratio": e / max(b, 1e-9), "q_mse": eq, "n": int(v.sum())}
    return out
