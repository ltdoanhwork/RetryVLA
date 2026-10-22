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

from collections import defaultdict

import numpy as np
import torch

from retryvla.search.cem import CEM, CEMConfig, roughness
from retryvla.search.wm_search import diverse_ranking

ARM_NAMES = ("random", "wm_diverse", "wm_noise", "wm_qd", "wm_two_stage", "wm_goal", "wm_goal_comp", "wm_top",
             "random_diverse", "sim_prefix_critic", "wm_no_prim", "wm_plan", "wm_plan_v2")


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-8)


def objective(cs: dict, mode: str, lam: float = 0.5, beta: float = 1.0) -> np.ndarray:
    """Search objective on WM-scored candidates. comp = RetryVLA (continuation value V of the frozen policy);
    goal = latent distance of the imagined end state to successful-demo goal frames (DINO-WM style);
    goal_comp = z-scored sum. Same uncertainty / roughness penalties for all (scaled to each objective's spread)."""
    if mode == "comp":
        return cs["score"]
    g = cs["goal"]
    if mode == "goal":
        return g - lam * cs["unc_n"] * g.std() - beta * cs["rough"]
    if mode == "goal_comp":
        s = _z(cs["value"]) + _z(g)
        return s - lam * cs["unc_n"] * s.std() - beta * cs["rough"]
    raise ValueError(mode)


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
    a = a[:, : ref.shape[1]]  # multi-segment candidates: compare the first segment
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


def arm_wm_two_stage(ranker, z0, q0, task, pool_c, pool_s, B, seed, n_first=16, n_second=32):
    """Two-segment (2 x H) recoveries for multi-stage fixes (e.g. get clear, THEN re-position) before handing
    control back. First segments: diverse top-n_first of the scored pool. Second segments: drawn from the same pool
    (state-independent primitives / perturbed chunks). All n_first x n_second concatenations are imagined end to
    end (WM rollout over 2H, critic on the final latent, ensemble uncertainty over the whole rollout penalizes
    extrapolation), then diverse top-B."""
    rng = np.random.default_rng(seed)
    first = diverse_ranking(pool_s["score"], pool_s["zfeat"], n_first, seed=seed)
    second = rng.choice(len(pool_c), size=min(n_second, len(pool_c)), replace=False)
    cand = np.concatenate([np.concatenate([pool_c[i], pool_c[j]], 0)[None] for i in first for j in second]).astype(np.float32)
    cs = ranker.score(z0, q0, task, cand)
    return cand, diverse_ranking(cs["score"], cs["zfeat"], B, seed=seed), cs


def arm_wm_obj(ranker, z0, q0, task, pool_c, B, seed, mode, goal=None, diverse=True, iters=2, elites=16, std=0.25):
    """Identical to arm_wm_diverse (pool + elite-local latent CEM + diverse top-B) except the objective."""
    cs = ranker.score(z0, q0, task, pool_c, goal=goal)
    cand = pool_c
    cem = CEM(pool_c.shape[1], 7, CEMConfig(n_samples=len(pool_c), init_std=std, smooth_kernel=5), seed=seed)
    for it in range(1, iters + 1):
        s = objective(cs, mode)
        el = np.argsort(-s)[:elites]
        if mode == "comp":  # exactly the M5 setting
            new = cem.sample_local(cand[el], s[el], it, temperature=0.05, decay=0.6)
        else:  # other objectives live on other scales: z-score elites (T=0.33 ~ the comp setting on z-scale)
            new = cem.sample_local(cand[el], _z(s[el]), it, temperature=0.33, decay=0.6)
        ns = ranker.score(z0, q0, task, new, goal=goal)
        cand = np.concatenate([cand, new])
        cs = {k: np.concatenate([cs[k], ns[k]]) for k in cs}
    s = objective(cs, mode)
    cs = dict(cs, score=s)
    picks = diverse_ranking(s, cs["zfeat"], B, seed=seed) if diverse else np.argsort(-s)[:B]
    return cand, picks, cs


def arm_random_diverse(pool_c, B, rng):
    """No WM, no critic: action-space diversity only (one random candidate per strategy cell, round robin)."""
    cell = descriptor(pool_c)
    by = defaultdict(list)
    for i in rng.permutation(len(pool_c)):
        by[int(cell[i])].append(int(i))
    order = list(by)
    rng.shuffle(order)
    picks = [by[c][r] for r in range(max(len(v) for v in by.values())) for c in order if r < len(by[c])][:B]
    return np.array(picks)


def wide_actions(rng: np.random.Generator, n: int, h: int, std: float = 0.5, kernel: int = 7) -> np.ndarray:
    """Broad, primitive-free exploration sequences: temporally smooth Gaussian deltas over the WHOLE arm action range
    (pos + rot), gripper = random piecewise-constant schedule with 0-2 switches. Used (a) as extra WM training data so
    the WM is accurate outside the proposal distribution, (b) as the planner's initial population."""
    from retryvla.search.cem import smooth_noise

    rs = np.random.RandomState(int(rng.integers(1 << 30)))
    a = smooth_noise(rs, n, h, 7, kernel) * std
    a[..., 3:6] *= 0.5                                         # rotations: half the range
    g = np.where(rng.random(n) < 0.5, 1.0, -1.0)[:, None].repeat(h, 1)
    for i in range(n):
        for _ in range(int(rng.integers(0, 3))):
            t0 = int(rng.integers(1, h))
            g[i, t0:] *= -1
    a[..., 6] = g
    return np.clip(a, -1, 1).astype(np.float32)


def arm_wm_plan(ranker, z0, q0, task, H, B, seed, n=256, iters=5, elites=24, std0=0.35, decay=0.7):
    """WM as the PLANNER (no proposal library, no VLA samples, no primitives): start from broad noise, then
    elite-local CEM in imagination for `iters` rounds (each elite perturbed separately, gripper schedule kept; no
    Gaussian mode averaging), objective = competence critic - uncertainty - roughness. Final picks: diverse top-B."""
    rng = np.random.default_rng(seed)
    cand = wide_actions(rng, n, H)
    cs = ranker.score(z0, q0, task, cand)
    cem = CEM(H, 7, CEMConfig(n_samples=n, init_std=std0, smooth_kernel=5), seed=seed)
    for it in range(1, iters + 1):
        el = np.argsort(-cs["score"])[:elites]
        new = cem.sample_local(cand[el], cs["score"][el], it, temperature=0.05, decay=decay, gripper_flip_p=0.1)
        ns = ranker.score(z0, q0, task, new)
        cand = np.concatenate([cand, new])
        cs = {k: np.concatenate([cs[k], ns[k]]) for k in cs}
    return cand, diverse_ranking(cs["score"], cs["zfeat"], B, seed=seed), cs
