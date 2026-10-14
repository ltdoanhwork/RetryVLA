"""Scorer interface shared by the oracle simulator scorer (M1) and the learned WM+critic
scorer (M3). Swapping scorers is a config change (``search.scorer``)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Instance:
    """A failure instance: a LIBERO-Recover scene restored with a fixed protocol seed."""
    instance_id: str
    scene_tag: str
    bddl: str
    task: str
    snapshot: dict          # exact sim snapshot after the official restore protocol
    obs0: dict              # policy obs at the failure state
    meta: dict = field(default_factory=dict)


@dataclass
class ScoreOut:
    scores: np.ndarray                 # [N] higher is better
    success: np.ndarray                # [N] bool (verified in sim for oracle; predicted for WM)
    success_step: np.ndarray           # [N] int (-1 if none)
    progress: np.ndarray               # [N] dense goal-progress after the rollout
    records: list[dict | None]         # sim-recorded trajectories for successes (oracle only)
    info: dict[str, Any] = field(default_factory=dict)


class Scorer(ABC):
    @abstractmethod
    def score(self, inst: Instance, cands: np.ndarray, iteration: int) -> ScoreOut:
        ...


class WMCriticScorer(Scorer):
    """M3 scorer: imagined rollout in the latent WM + recovery critic (no simulator calls).
    S(A) = V_psi(imagined end state) - lambda * U_WM(A) - beta * roughness(A).
    Predictions only: verify the top candidates in the simulator before writing data
    (``retryvla.search.wm_search`` does this end to end)."""

    def __init__(self, ranker):
        self.ranker = ranker  # retryvla.search.wm_search.WMCriticRanker

    def score(self, inst: Instance, cands: np.ndarray, iteration: int) -> ScoreOut:
        z0, q0 = self.ranker.encode(inst.obs0)
        s = self.ranker.score(z0, q0, inst.task, cands)
        n = len(cands)
        return ScoreOut(s["score"], s["value"] > 0.5, np.full(n, -1), np.zeros(n), [None] * n,
                        {"value": s["value"], "uncertainty": s["unc"]})


def make_scorer(name: str, **kw) -> Scorer:
    if name == "oracle_sim":
        from retryvla.search.oracle_sim import OracleSimScorer

        return OracleSimScorer(**kw)
    if name == "wm_critic":
        return WMCriticScorer(**kw)
    raise ValueError(name)
