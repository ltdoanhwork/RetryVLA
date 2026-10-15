"""Candidate-selection arms for WM-guided recovery search + data-distribution metrics (M5b).

Every arm returns (candidates [N, H, 7], picks [B] indices into them, sources, scores) and is judged by
the SAME simulator verification budget B, so arms differ only in WHICH B candidates they pay for.

  random      B uniform draws from the shared initial proposal pool (no WM)
  wm_diverse  pool + action-space latent CEM (elite-local), k-means diverse top-B  (M5 default)
  wm_noise    CEM in SmolVLA's flow-matching NOISE space (pCN moves), diverse top-B: every candidate is a
              decoded SmolVLA chunk, i.e. on the policy's own action manifold by construction
  wm_qd       MAP-Elites over interpretable strategy cells (gripper schedule x net motion), mutation of
              uniformly chosen cell elites, best-per-cell top-B: diversity of STRATEGY by design
"""

from __future__ import annotations

import numpy as np
import torch

from retryvla.search.cem import CEM, CEMConfig, roughness
from retryvla.search.wm_search import diverse_ranking

ARM_NAMES = ("random", "wm_diverse", "wm_noise", "wm_qd")


# ---------------------------------------------------------------- descriptors & metrics
def descriptor(a: np.ndarray, move_thr: float = 0.1) -> np.ndarray:
    """Strategy cell id per candidate [N] from actions [N, H, 7].
    gripper: start/end state (open/close) -> 4 classes; z: up / flat / down; xy: still or 4 directions."""
    g = a[..., 6] > 0
    k = max(1, a.shape[1] // 10)
    gcls = 2 * (g[:, :k].mean(1) > 0.5) + (g[:, -k:].mean(1) > 0.5)
    m = a[..., :3].mean(1)
    zb = np.where(m[:, 2] > move_thr, 0, np.where(m[:, 2] < -move_thr, 2, 1))
    r = np.linalg.norm(m[:, :2], axis=1)
    ang = (np.floor((np.arctan2(m[:, 1], m[:, 0]) + np.pi / 4) / (np.pi / 2)) % 4).astype(int)
    xyb = np.where(r < move_thr, 0, 1 + ang)
    return (gcls * 3 + zb) * 5 + xyb  # 4*3*5 = 60 cells


def jerk(a: np.ndarray) -> np.ndarray:
    d = np.diff(a[..., :6], n=2, axis=1)
    return np.abs(d).mean(axis=(1, 2))


def manifold_dist(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Distance of each candidate to the closest SmolVLA sample at the same failure observation
    (mean per-step L2 over arm dims). ref = reference SmolVLA chunks [R, H, 7]."""
    if len(a) == 0:
        return np.zeros(0)
    x, r = a[..., :6].reshape(len(a), -1), ref[..., :6].reshape(len(ref), -1)
    d2 = (x ** 2).sum(1)[:, None] + (r ** 2).sum(1)[None] - 2 * x @ r.T
    return np.sqrt(np.maximum(d2, 0).min(1) / a.shape[1])


def mean_pairwise(a: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    x = a[..., :6].reshape(len(a), -1)
    d = np.sqrt(np.maximum(((x[:, None] - x[None]) ** 2).sum(-1), 0) / a.shape[1])
    return float(d[np.triu_indices(len(a), 1)].mean())


def set_metrics(a: np.ndarray, ref: np.ndarray) -> dict:
    """Distribution metrics of a candidate set (e.g. the verified successes of one arm)."""
    if len(a) == 0:
        return {"n": 0, "rough": np.nan, "jerk": np.nan, "manifold": np.nan, "pairwise": np.nan, "cells": 0}
    return {"n": len(a), "rough": float(roughness(a).mean()), "jerk": float(jerk(a).mean()),
            "manifold": float(manifold_dist(a, ref).mean()), "pairwise": mean_pairwise(a),
            "cells": int(len(set(descriptor(a).tolist())))}


# ---------------------------------------------------------------- arms
def arm_random(pool_c, pool_s, B, rng):
    picks = rng.choice(len(pool_c), size=min(B, len(pool_c)), replace=False)
    return pool_c, picks, None


def arm_wm_diverse(ranker, z0, q0, task, pool_c, pool_s, B, seed, iters=2, elites=16, std=0.25):
    cand, cs = pool_c, pool_s
    cem = CEM(pool_c.shape[1], 7, CEMConfig(n_samples=len(pool_c), init_std=std, smooth_kernel=5), seed=seed)
    for it in range(1, iters + 1):
        el = np.argsort(-cs["score"])[:elites]
        new = cem.sample_local(cand[el], cs["score"][el], it, temperature=0.05, decay=0.6)
        ns = ranker.score(z0, q0, task, new)
        cand = np.concatenate([cand, new])
        cs = {k: np.concatenate([cs[k], ns[k]]) for k in cs}
    return cand, diverse_ranking(cs["score"], cs["zfeat"], B, seed=seed), cs


def arm_wm_noise(policy, ranker, obs0, z0, q0, task, H, B, seed, n=256, iters=2, scales=(1.0, 1.5),
                 betas=(0.5, 0.3), elites=16, temperature=0.05):
    """CEM over SmolVLA's initial flow-matching noise with preconditioned Crank-Nicolson moves
    eps' = sqrt(1-b^2) * eps_elite + b * N(0, I): stays N(0, s^2)-distributed, so decoded chunks remain
    ordinary SmolVLA samples, while the WM+critic score steers which ones are kept."""
    g = torch.Generator().manual_seed(int(seed))
    shape = (policy.chunk_size, policy.max_action_dim)
    per = n // len(scales)
    eps = torch.cat([torch.randn((per, *shape), generator=g) * s for s in scales])
    scl = torch.cat([torch.full((per,), float(s)) for s in scales])
    noises, acts = [eps], [np.clip(policy.chunks_from_noise(obs0, task, eps)[:, :H], -1, 1).astype(np.float32)]
    cs = ranker.score(z0, q0, task, acts[0])
    rng = np.random.default_rng(seed)
    for it in range(iters):
        s_all = cs["score"]
        el = np.argsort(-s_all)[:elites]
        p = np.exp((s_all[el] - s_all[el].max()) / temperature)
        p /= p.sum()
        par = torch.as_tensor(el[rng.choice(len(el), size=n, p=p)])
        E = torch.cat(noises)
        b = betas[min(it, len(betas) - 1)]
        new = np.sqrt(1 - b * b) * E[par] + b * torch.randn((n, *shape), generator=g) * scl[par].view(-1, 1, 1)
        a = np.clip(policy.chunks_from_noise(obs0, task, new)[:, :H], -1, 1).astype(np.float32)
        ns = ranker.score(z0, q0, task, a)
        noises.append(new); acts.append(a)
        scl = torch.cat([scl, scl[par]])
        cs = {k: np.concatenate([cs[k], ns[k]]) for k in cs}
    cand = np.concatenate(acts)
    return cand, diverse_ranking(cs["score"], cs["zfeat"], B, seed=seed), cs


def arm_wm_qd(ranker, z0, q0, task, pool_c, pool_s, B, seed, iters=2, n=256, std=0.3):
    """MAP-Elites: archive = best candidate per strategy cell; parents drawn UNIFORMLY over occupied
    cells (exploration), mutated with smooth noise (gripper schedule kept); final picks = best-per-cell,
    highest-scoring cells first."""
    cand, cs = pool_c, pool_s
    cem = CEM(pool_c.shape[1], 7, CEMConfig(n_samples=n, init_std=std, smooth_kernel=5), seed=seed + 991)

    def archive(cs, cand):
        cell = descriptor(cand)
        best = {}
        for i in np.argsort(-cs["score"]):
            best.setdefault(int(cell[i]), int(i))
        return best

    for it in range(1, iters + 1):
        arc = archive(cs, cand)
        el = np.array(list(arc.values()))
        new = cem.sample_local(cand[el], np.zeros(len(el)), it, n=n, temperature=1.0, decay=0.67)
        ns = ranker.score(z0, q0, task, new)
        cand = np.concatenate([cand, new])
        cs = {k: np.concatenate([cs[k], ns[k]]) for k in cs}
    arc = archive(cs, cand)
    picks = sorted(arc.values(), key=lambda i: -cs["score"][i])[:B]
    if len(picks) < B:  # fewer occupied cells than budget: fill by score
        rest = [i for i in np.argsort(-cs["score"]) if i not in set(picks)]
        picks += rest[: B - len(picks)]
    return cand, np.array(picks), cs
