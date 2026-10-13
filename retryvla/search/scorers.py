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
    """M3: score candidates by rolling them out in the latent world model and evaluating the
    recovery value critic at the imagined end state:
        S(A) = V_psi(z_H, q_H, g) - lambda * U_W(A) - beta * C(A)
    Successes are *predicted*; they must be validated in the simulator before use (Stage 4)."""

    def __init__(self, wm, critic, encoder, uncertainty_weight: float = 0.1, smooth_cost: float = 0.02):
        self.wm, self.critic, self.encoder = wm, critic, encoder
        self.lam, self.beta = uncertainty_weight, smooth_cost

    def score(self, inst: Instance, cands: np.ndarray, iteration: int) -> ScoreOut:
        import torch

        from retryvla.search.cem import roughness

        with torch.no_grad():
            h0 = self.encoder.encode_obs(inst.obs0)                          # [1, T, D]
            a = torch.as_tensor(cands, device=h0.device, dtype=h0.dtype)      # [N, H, A]
            zH, qH, unc = self.wm.rollout(h0.expand(len(cands), *h0.shape[1:]), a)  # imagined end state
            v = self.critic(zH, qH, [inst.task] * len(cands)).float().cpu().numpy()
            u = unc.float().cpu().numpy() if unc is not None else np.zeros(len(cands))
        scores = v - self.lam * u - self.beta * roughness(cands)
        n = len(cands)
        return ScoreOut(scores, v > 0.5, np.full(n, -1), np.zeros(n), [None] * n,
                        {"value": v, "uncertainty": u})


def make_scorer(name: str, **kw) -> Scorer:
    if name == "oracle_sim":
        from retryvla.search.oracle_sim import OracleSimScorer

        return OracleSimScorer(**kw)
    if name == "wm_critic":
        return WMCriticScorer(**kw)
    raise ValueError(name)
