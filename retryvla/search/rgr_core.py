"""Recursive Policy-Conditioned Geometric Recovery (RGR): imagination scoring, backward chaining, geometry.

  B_0     = {s : C^pi(s, g) >= tau}
  B_{k+1} = B_k U {s : exists bridge b, T_hat(s, b) in B_k and GeomValid(s, b)}
Search finds a chain of short bridges b_1..b_K from the failure state through intermediate (geom-valid) states into
B_0, where the frozen VLA pi takes over. Bridge score J(b) = C^pi(s_H, g) - lam * E_geom(b) - mu * R(b) - 1[invalid]
(U_WM is not used: the WM's own ensemble heads are identical (U == 0), see NOTES RQ3; mu R = action roughness as in
every earlier arm). E_geom from the observation-only geometry head on imagined / observed latents, or from the
privileged sim state (ORACLE, upper bound). Default search paths are untouched: this module is only used by
retryvla.search.rgr_search.
"""

from __future__ import annotations

import numpy as np
import torch

from retryvla.geom.features import geom_energy
from retryvla.search.cem import CEM, CEMConfig, roughness
from retryvla.search.wm_search import diverse_ranking


@torch.no_grad()
def imagine(ranker, head, zs, qs, task: str, cands: np.ndarray, g_ref=None, bs: int = 256, keep: bool = False) -> dict:
    """Score candidate bridges from (possibly different, possibly imagined) start latents.
    zs [S,P,D] / qs [S,8] normalized start states (S == 1 -> broadcast, else S == N); cands [N,H,7].
    g_ref [S|1, G] geometry of the start state (head prediction or oracle) for E_geom (None -> no geometry).
    Returns value [N] (C^pi of the imagined end state), rough [N], zfeat [N,F], g [N,G], E [N], valid [N] and
    (keep) zH [N,P,D] fp16 / qH [N,8] on the GPU for chaining."""
    dev = ranker.dev
    tid = ranker.vocab.get(task, 0)
    V, ZF, GG, ZH, QH = [], [], [], [], []
    for i in range(0, len(cands), bs):
        a = torch.as_tensor(np.ascontiguousarray(cands[i: i + bs]), device=dev)
        n = len(a)
        z0 = zs.expand(n, -1, -1) if zs.shape[0] == 1 else zs[i: i + n]
        q0 = qs.expand(n, -1) if qs.shape[0] == 1 else qs[i: i + n]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            zH, qH, _, _, _ = ranker.wm.rollout(z0.float(), q0.float(), a)
            v = torch.sigmoid(ranker.critic.logits(zH.float(), qH.float(), torch.full((n,), tid, device=dev,
                                                                                     dtype=torch.long)))
        V.append(v.float().cpu().numpy())
        ZF.append(torch.cat([zH.float().mean(1), qH.float()], 1).cpu().numpy())
        if head is not None:
            GG.append(head.predict(zH.float(), qH.float()).cpu().numpy())
        if keep:
            ZH.append(zH.half()); QH.append(qH.float())
    out = {"value": np.concatenate(V), "rough": roughness(cands), "zfeat": np.concatenate(ZF)}
    if head is not None:
        g = np.concatenate(GG)
        ref = g if g_ref is None else np.asarray(g_ref, np.float32)
        if ref.ndim == 1 or len(ref) == 1:
            ref = np.broadcast_to(ref.reshape(1, -1), g.shape)
        E, valid, _ = geom_energy(ref, g)
        out.update(g=g, E=E.astype(np.float32), valid=valid)
    if keep:
        out["zH"], out["qH"] = torch.cat(ZH), torch.cat(QH)
    return out


def objective(cs: dict, lam: float, mu: float, geo: bool) -> np.ndarray:
    s = cs["value"] - mu * cs["rough"]
    if geo and "E" in cs:
        s = s - lam * cs["E"] - 1.0 * (~cs["valid"])
    return s


def cem_rounds(ranker, head, z0, q0, task, pool_c, cs, iters, lam, mu, geo, seed, g_ref=None, n=None,
               elites=16, std=0.25, keep=False):
    """`iters` elite-local latent-CEM rounds (exactly the wm_diverse / wm_unc settings: T 0.05, decay 0.6)."""
    cand = pool_c
    cem = CEM(pool_c.shape[1], 7, CEMConfig(n_samples=n or len(pool_c), init_std=std, smooth_kernel=5), seed=seed)
    for it in range(1, iters + 1):
        s = objective(cs, lam, mu, geo)
        el = np.argsort(-s)[:elites]
        new = cem.sample_local(cand[el], s[el], it, temperature=0.05, decay=0.6)
        ns = imagine(ranker, head, z0, q0, task, new, g_ref, keep=keep)
        cand = np.concatenate([cand, new])
        cs = {k: (torch.cat([cs[k], ns[k]]) if torch.is_tensor(cs[k]) else np.concatenate([cs[k], ns[k]])) for k in cs}
    return cand, cs


def lookahead(ranker, head, cs1, cand1, top1: np.ndarray, cont: np.ndarray, task, lam, mu, geo, gamma: float,
              tau: float):
    """Backward chaining one level deep in imagination: for each level-1 bridge in top1, imagine every continuation
    bridge in `cont` [n2,H,7] from its IMAGINED end state; J2 = objective of the 2-bridge chain end, intermediate
    state must be geom-valid (geo). Chain value Q(b1) = max(J1(b1), gamma * max_b2 J2(b1, b2)); B_0 membership of
    the imagined s_1 (C^pi >= tau) is logged. Returns Q [len(top1)], best continuation indices [len(top1), n2] sorted
    by J2, J2 matrix, value matrix."""
    n2 = len(cont)
    t1 = torch.as_tensor(np.asarray(top1), device=cs1["zH"].device, dtype=torch.long)
    zs = cs1["zH"][t1].repeat_interleave(n2, 0)
    qs = cs1["qH"][t1].repeat_interleave(n2, 0)
    g_ref = np.repeat(cs1["g"][top1], n2, 0) if geo and "g" in cs1 else None
    c2 = np.tile(cont, (len(top1), 1, 1))
    cs2 = imagine(ranker, head if geo else None, zs, qs, task, c2, g_ref)
    J2 = objective(cs2, lam, mu, geo).reshape(len(top1), n2)
    V2 = cs2["value"].reshape(len(top1), n2)
    J1 = objective({k: v[top1] for k, v in cs1.items() if not torch.is_tensor(v)}, lam, mu, geo)
    if geo and "valid" in cs1:  # an invalid intermediate state cannot be chained through
        J2 = np.where(cs1["valid"][top1][:, None], J2, -np.inf)
    Q = np.maximum(J1, gamma * J2.max(1))
    order = np.argsort(-J2, axis=1)
    return Q, order, J2, V2, cs1["value"][top1] >= tau


def select_diverse(score, zfeat, k, seed):
    return diverse_ranking(score, zfeat, k, seed=seed)


def latent_discrepancy(z_real: torch.Tensor, z_hat: torch.Tensor) -> np.ndarray:
    """Mean squared difference of normalized latents [N] (WM v1 h10 open-loop error is ~0.32 on average)."""
    return ((z_real.float() - z_hat.float()) ** 2).mean((1, 2)).cpu().numpy()
