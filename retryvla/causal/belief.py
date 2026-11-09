"""ACR belief, trial allocation, expected value of information (EVI) and information gain.

Notation: particles h = 0..P-1 (Tier B: hypothesis particles; Tier A: causes), families f = 0..F-1,
V[h, f] = planning success probability of ONE trial of family f under h (trials assumed independent given h).
An allocation n[f] (sum = B) succeeds under h with prob 1 - prod_f (1 - V[h, f])^n[f].
"""

from __future__ import annotations

import numpy as np


def success_prob(n: np.ndarray, V: np.ndarray) -> np.ndarray:
    """[P] P(at least one success | h) for allocation n [F]."""
    return 1.0 - np.prod((1.0 - np.clip(V, 0, 1)) ** n[None, :], axis=1)


def allocate(b: np.ndarray, V: np.ndarray, B: int, cap: int | None = None) -> np.ndarray:
    """Greedy allocation of B trials maximising sum_h b(h) P(success | h, n). cap = max trials per family (the number
    of distinct variants in the outcome table). Ties -> lower family index (deterministic)."""
    F = V.shape[1]
    n = np.zeros(F, int)
    for _ in range(int(B)):
        best, bf = -1.0, -1
        for f in range(F):
            if cap is not None and n[f] >= cap:
                continue
            n[f] += 1
            u = float(b @ success_prob(n, V))
            n[f] -= 1
            if u > best + 1e-12:
                best, bf = u, f
        if bf < 0:
            break
        n[bf] += 1
    return n


def value(b_eval: np.ndarray, n: np.ndarray, V: np.ndarray) -> float:
    return float(b_eval @ success_prob(n, V))


def kernel_lik(r: np.ndarray, R: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """[P] Gaussian-kernel likelihood of the observed response r [D] under particle responses R [P, D]."""
    z = (R - r[None]) / scale[None]
    ll = -0.5 * (z * z).sum(1)
    return np.exp(ll - ll.max())


def scales(R: np.ndarray, floor: np.ndarray, frac: float = 0.25) -> np.ndarray:
    return np.maximum(floor, frac * R.std(0))


def posterior(b0: np.ndarray, lik: np.ndarray) -> np.ndarray:
    p = b0 * lik
    s = p.sum()
    return p / s if s > 0 else b0.copy()


def entropy(p: np.ndarray, groups: np.ndarray | None = None) -> float:
    """Entropy of p, optionally of its marginal over group labels (e.g. hypothesis TYPE, not magnitude)."""
    if groups is not None:
        p = np.array([p[groups == g].sum() for g in np.unique(groups)])
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def evi_particles(b0: np.ndarray, V: np.ndarray, R: np.ndarray, scale: np.ndarray, B: int, cap: int | None = None
                  ) -> tuple[float, float]:
    """(EVI, expected info gain over particles) of one probe whose response under particle h is R[h] (deterministic
    planning sim). EVI = E_h~b0 [value of the allocation chosen after seeing R[h], evaluated under h]
                       - E_h~b0 [value of the prior allocation under h]."""
    n0 = allocate(b0, V, B, cap)
    base = value(b0, n0, V)
    tot, eig = 0.0, 0.0
    H0 = entropy(b0)
    for h in range(len(b0)):
        if b0[h] <= 0:
            continue
        post = posterior(b0, kernel_lik(R[h], R, scale))
        n = allocate(post, V, B, cap)
        tot += b0[h] * float(success_prob(n, V)[h])
        eig += b0[h] * (H0 - entropy(post))
    return tot - base, eig


def evi_samples(b0: np.ndarray, V: np.ndarray, posts: np.ndarray, truth: np.ndarray, w: np.ndarray, B: int,
                cap: int | None = None) -> tuple[float, float]:
    """Tier A (classifier) version: posts [N, P] = cross-fitted classifier posteriors of N reference instances whose
    true cause is truth[N]; w [N] = weights (sum 1). Same EVI / info-gain definitions with the empirical predictive."""
    n0 = allocate(b0, V, B, cap)
    base = float(sum(w[i] * success_prob(n0, V)[truth[i]] for i in range(len(w))))
    tot, eig, H0 = 0.0, 0.0, entropy(b0)
    for i in range(len(w)):
        n = allocate(posts[i], V, B, cap)
        tot += w[i] * float(success_prob(n, V)[truth[i]])
        eig += w[i] * (H0 - entropy(posts[i]))
    return tot - base, eig


def beta_mean(s: np.ndarray, n: np.ndarray, a: float = 0.5) -> np.ndarray:
    """Smoothed success rate (Jeffreys prior)."""
    return (s + a) / (n + 2 * a)
