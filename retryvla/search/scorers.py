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


# ---------------------------------------------------------------- verifier uncertainty (RQ3, plan Part 3)
def member_rollout(wm, z0, q0, actions, boot: dict | None = None):
    """Open-loop imagination of EACH ensemble member on its own trajectory (member e always uses head e;
    LatentWM.rollout instead feeds the ensemble MEAN back each step). z0 [N,P,D], q0 [N,8], actions [N,H,A]
    (normalized latents) -> z [E,N,P,D], q [E,N,8]. Costs E trunk passes per step (batch E*N).
    boot: bootstrap heads on the frozen trunk (retryvla.wm.boot_heads) -- REQUIRED for a real ensemble: the
    WM's own heads are bit-identical in every checkpoint so far (U == 0)."""
    import torch

    n, H = actions.shape[0], actions.shape[1]
    E = int(boot["n"]) if boot is not None else len(wm.z_heads)
    z, q, a = z0.repeat(E, 1, 1), q0.repeat(E, 1), actions.repeat(E, 1, 1)
    who = torch.arange(E, device=z0.device).repeat_interleave(n)
    ar = torch.arange(E * n, device=z0.device)
    if boot is not None:
        from retryvla.wm.boot_heads import trunk_features

        zW, zb, qW, qb = boot["z_W"][who], boot["z_b"][who], boot["q_W"][who], boot["q_b"][who]
    for t in range(0, H - H % wm.k, wm.k):
        if boot is None:
            _, _, _, (ze, qe) = wm.step(z, q, a[:, t: t + wm.k])   # [E, E*N, P, D]: every head on every row
            z, q = ze[who, ar], qe[who, ar]
        else:
            y = trunk_features(wm, z, q, a[:, t: t + wm.k]).float()
            z = z.float() + torch.bmm(y[:, : wm.P], zW) + zb[:, None]
            q = q.float() + torch.bmm(y[:, wm.P][:, None], qW)[:, 0] + qb
    return z.view(E, n, *z.shape[1:]), q.view(E, n, -1)


def _mc_critic(ranker, p: float):
    """Copy of ranker.critic with dropout p active (MC dropout; C^pi v2 was trained with dropout 0.1)."""
    import copy

    import torch.nn as nn

    key = ("_mc_critic", p)
    if getattr(ranker, "_mc_cache", None) is None:
        ranker._mc_cache = {}
    if key not in ranker._mc_cache:
        c = copy.deepcopy(ranker.critic)
        for m in c.modules():
            if isinstance(m, nn.Dropout):
                m.p = p
        ranker._mc_cache[key] = c.train()
    return ranker._mc_cache[key]


def ensemble_values(ranker, z0, q0, task: str, cands: np.ndarray, bs: int = 128) -> dict[str, np.ndarray]:
    """C^pi on the ensemble-mean rollout (= ranker.score 'value') plus C^pi on each member's own rollout.
    Members = bootstrap heads on the frozen WM trunk (ranker.boot, retryvla.wm.boot_heads); member e's critic call
    additionally uses MC-dropout mask e if ranker.mc_dropout > 0 (fixed seed -> deterministic).
      u_wmb  = std_e C^pi(z_H^e)                 (WM disagreement in C^pi units, deterministic critic)
      u_crit = std_e C^pi_drop_e(z_H)            (critic MC-dropout on the mean rollout)
      u_ens  = std_e C^pi_drop_e(z_H^e)          (both; = u_wmb if mc_dropout == 0)  <- used by J
    All in the critic's own units (probability of pi-success), so J = V - lambda U_ens is scale-free.
    Returns value [N], v_members [N,E], u_ens / u_wmb / u_crit / u_wm (WM's own, == 0 for degenerate heads),
    rough [N], zfeat [N,F]."""
    import torch

    from retryvla.search.cem import roughness

    if hasattr(ranker, "ensemble_values"):  # non-DINO WMs (retryvla.wm.vjepa2_ac.VJRanker) implement their own
        return ranker.ensemble_values(z0, q0, task, cands)
    tid = ranker.vocab.get(task, 0)
    dev = ranker.dev
    boot = getattr(ranker, "boot", None)
    p_mc = float(getattr(ranker, "mc_dropout", 0.0) or 0.0)
    V, VM, VW, VC, UW, ZF = [], [], [], [], [], []
    with torch.no_grad(), torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        torch.manual_seed(12345)
        for i in range(0, len(cands), bs):
            a = torch.as_tensor(np.ascontiguousarray(cands[i: i + bs]), device=dev)
            n = len(a)
            ti = torch.full((n,), tid, device=dev, dtype=torch.long)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                zH, qH, unc, _, _ = ranker.wm.rollout(z0.expand(n, -1, -1), q0.expand(n, -1), a)
                v = torch.sigmoid(ranker.critic.logits(zH.float(), qH.float(), ti))
                ze, qe = member_rollout(ranker.wm, z0.expand(n, -1, -1), q0.expand(n, -1), a, boot)
                E = ze.shape[0]
                zf, qf, tf = ze.flatten(0, 1).float(), qe.flatten(0, 1).float(), ti.repeat(E)
                vw = torch.sigmoid(ranker.critic.logits(zf, qf, tf)).view(E, n)
                if p_mc > 0:
                    mc = _mc_critic(ranker, p_mc)
                    vm = torch.sigmoid(mc.logits(zf, qf, tf)).view(E, n)
                    vc = torch.sigmoid(mc.logits(zH.float().repeat(E, 1, 1), qH.float().repeat(E, 1), tf)).view(E, n)
                else:
                    vm, vc = vw, v[None].expand(E, n)
            V.append(v.float().cpu().numpy()); UW.append(unc.float().cpu().numpy())
            VM.append(vm.float().T.cpu().numpy()); VW.append(vw.float().T.cpu().numpy()); VC.append(vc.float().T.cpu().numpy())
            ZF.append(torch.cat([zH.float().mean(1), qH.float()], 1).cpu().numpy())
    VM, VW, VC = np.concatenate(VM), np.concatenate(VW), np.concatenate(VC)
    return {"value": np.concatenate(V), "v_members": VM, "u_ens": VM.std(1), "u_wmb": VW.std(1), "u_crit": VC.std(1),
            "v_wmb_mean": VW.mean(1), "u_wm": np.concatenate(UW), "rough": roughness(cands), "zfeat": np.concatenate(ZF)}


def unc_objective(cs: dict, lam: float, mu: float) -> np.ndarray:
    """J = C^pi - lambda * U_ens - mu * cost (cost = action roughness, the same term as beta in ranker.score)."""
    return cs["value"] - float(lam) * cs["u_ens"] - float(mu) * cs["rough"]


class EnsembleUncScorer(Scorer):
    """WM + critic scorer with the ensemble-disagreement penalty J = C^pi - lambda U_ens - mu cost.
    Predictions only (like WMCriticScorer): verify in sim before writing data."""

    def __init__(self, ranker, lam: float = 0.0, mu: float = 1.0):
        self.ranker, self.lam, self.mu = ranker, lam, mu

    def score(self, inst: Instance, cands: np.ndarray, iteration: int) -> ScoreOut:
        z0, q0 = self.ranker.encode(inst.obs0)
        cs = ensemble_values(self.ranker, z0, q0, inst.task, cands)
        n = len(cands)
        return ScoreOut(unc_objective(cs, self.lam, self.mu), cs["value"] > 0.5, np.full(n, -1), np.zeros(n),
                        [None] * n, {"value": cs["value"], "u_ens": cs["u_ens"], "u_wm": cs["u_wm"]})


def make_scorer(name: str, **kw) -> Scorer:
    if name == "oracle_sim":
        from retryvla.search.oracle_sim import OracleSimScorer

        return OracleSimScorer(**kw)
    if name == "wm_critic":
        return WMCriticScorer(**kw)
    if name == "wm_critic_unc":
        return EnsembleUncScorer(**kw)
    raise ValueError(name)
