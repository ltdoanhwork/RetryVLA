"""ACR belief / allocator / EVI (CPU only, no sim).  python -m pytest -q tests/test_causal_belief.py"""

from __future__ import annotations

import numpy as np

from retryvla.causal import belief as Bf


def test_allocate_concentrates_and_hedges():
    V = np.array([[0.9, 0.0], [0.0, 0.9]])
    assert list(Bf.allocate(np.array([1.0, 0.0]), V, 4)) == [4, 0]
    n = Bf.allocate(np.array([0.5, 0.5]), V, 4)
    assert n.min() >= 1  # hedges under uncertainty


def test_evi_zero_when_decision_irrelevant():
    V = np.array([[0.3, 0.1], [0.3, 0.1]])          # same best family under both h
    R = np.array([[0.0], [1.0]])                    # perfectly separating probe
    evi, eig = Bf.evi_particles(np.array([0.5, 0.5]), V, R, np.array([0.1]), B=2)
    assert abs(evi) < 1e-9 and eig > 0.5            # informative but worthless for the decision


def test_evi_positive_when_relevant():
    V = np.array([[0.6, 0.0], [0.0, 0.6]])
    R = np.array([[0.0], [1.0]])
    evi, _ = Bf.evi_particles(np.array([0.5, 0.5]), V, R, np.array([0.1]), B=1)
    assert evi > 0.25
    evi_u, _ = Bf.evi_particles(np.array([0.5, 0.5]), V, np.array([[0.0], [0.0]]), np.array([0.1]), B=1)
    assert abs(evi_u) < 1e-9                        # uninformative probe


def test_cap_respected():
    V = np.array([[0.2, 0.1, 0.05]])
    assert Bf.allocate(np.array([1.0]), V, 5, cap=2).max() <= 2
