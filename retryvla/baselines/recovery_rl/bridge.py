"""ResidualBridgePolicy: frozen base VLA + residual bridge controller behind the `predict_chunk` interface.

Drop-in for `run_jobs` / eval_recover / eval_nominal: the base predicts its chunk; for decisions inside the bridge
window (episode env step < horizon) the residual controller modifies it. After the window the base acts alone.

The episode step is decoded from the run_jobs seed convention `seed = policy_seed * 100_003 + n_calls`
(rollout.py), i.e. n_calls = seed % 100_003, step = n_calls * n_action_steps (valid for jobs without an open-loop
prefix, which is the case in eval_recover / eval_nominal). This keeps rollout.py untouched.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from retryvla.baselines.recovery_rl.policy import ActorCritic, FeatureEncoder, apply_residual, hashed_text, load_actor

SEED_MOD = 100_003  # rollout.py: seeds = policy_seed * 100_003 + n_calls


class ResidualBridge:
    """Batched decision logic shared by the PPO episode runner and the eval wrapper."""

    def __init__(self, model: ActorCritic, features: FeatureEncoder, horizon: int, n_act: int, scale: float,
                 device: str = "cuda"):
        self.model, self.features = model, features
        self.horizon, self.n_act, self.scale = int(horizon), int(n_act), float(scale)
        self.n_dec = (self.horizon + self.n_act - 1) // self.n_act
        self.device = torch.device(device)
        self._text: dict[str, np.ndarray] = {}

    def text(self, t: str) -> np.ndarray:
        if t not in self._text:
            self._text[t] = hashed_text(t)
        return self._text[t]

    @torch.no_grad()
    def inputs(self, obs: Sequence[dict], tasks: Sequence[str], chunks: np.ndarray, dec_idx: Sequence[int]):
        img = self.features(list(obs))
        st = torch.as_tensor(np.stack([o["state"] for o in obs]), device=self.device).float()
        tx = torch.as_tensor(np.stack([self.text(t) for t in tasks]), device=self.device)
        ch = torch.as_tensor(np.asarray(chunks)[:, : self.n_act], device=self.device).float()
        di = torch.as_tensor(np.asarray(dec_idx), device=self.device)
        return img, self.model.low_input(st, tx, ch, di)

    @torch.no_grad()
    def decide(self, obs, tasks, chunks: np.ndarray, dec_idx, sample: bool, gen: torch.Generator | None = None):
        img, low = self.inputs(obs, tasks, chunks, dec_idx)
        dc, dg = self.model.dist(img, low)
        if sample:
            a_c = dc.mean + dc.stddev * torch.randn(dc.mean.shape, generator=gen, device="cpu").to(self.device)
            u = torch.rand(len(obs), generator=gen, device="cpu").to(self.device)
            cdf = dg.probs.cumsum(-1)
            a_g = (u[:, None] > cdf).sum(-1).clamp(max=2)
        else:
            a_c, a_g = dc.mean, dg.probs.argmax(-1)
        logp = dc.log_prob(a_c).sum(-1) + dg.log_prob(a_g)
        v = self.model.value(img, low)
        ac, ag = a_c.cpu().numpy(), a_g.cpu().numpy()
        out = apply_residual(chunks, ac, ag, self.scale)
        info = {"img": img.cpu(), "low": low.cpu(), "a_cont": a_c.cpu(), "a_grip": a_g.cpu(), "logp": logp.cpu(),
                "value": v.cpu(), "delta_sq": (self.scale * np.clip(ac, -1, 1)) ** 2}
        return out, info


class ResidualBridgePolicy:
    """`predict_chunk` wrapper for evaluation (deterministic residual: Gaussian mean, gripper argmax)."""

    def __init__(self, base, ckpt: str, device: str = "cuda"):
        model, ck = load_actor(ckpt, device)
        rc = ck["rl_cfg"]
        feats = FeatureEncoder(rc.get("encoder", "facebook/dinov2-small"), int(rc.get("pool_grid", 2)), device)
        self.base = base
        self.bridge = ResidualBridge(model, feats, int(rc["horizon"]), base.n_action_steps, float(rc["scale"]), device)
        assert model.n_act == base.n_action_steps, "residual was trained with a different n_action_steps"
        self.calls = 0
        self.ckpt = ckpt

    @property
    def n_action_steps(self) -> int:
        return self.base.n_action_steps

    def __getattr__(self, k):  # chunk_size, action_dim, ...
        return getattr(self.base, k)

    def predict_chunk(self, obs, tasks, seeds=None, noise_scale: float = 1.0) -> np.ndarray:
        ch = self.base.predict_chunk(obs, tasks, seeds, noise_scale)
        if len(obs) == 0:
            return ch
        assert seeds is not None, "ResidualBridgePolicy needs run_jobs seeds to know the episode step"
        dec = [int(s) % SEED_MOD for s in seeds]
        rows = [i for i, d in enumerate(dec) if d * self.n_action_steps < self.bridge.horizon]
        if rows:
            mod, _ = self.bridge.decide([obs[i] for i in rows], [tasks[i] for i in rows], ch[rows],
                                        [dec[i] for i in rows], sample=False)
            ch = np.array(ch, copy=True)
            ch[rows] = mod
            self.calls += len(rows)
        return ch


def maybe_wrap(cfg, policy, run=None):
    """Backward-compatible hook for eval scripts: cfg.residual.ckpt set -> wrap the policy."""
    rc = cfg.get("residual") or {}
    ck = rc.get("ckpt") if hasattr(rc, "get") else None
    if not ck:
        return policy
    from pathlib import Path

    from retryvla import REPO_ROOT

    p = Path(str(ck))
    if not p.is_absolute():
        p = REPO_ROOT / p
    if p.suffix == ".txt":  # link file
        p = Path(p.read_text().strip())
    wrapped = ResidualBridgePolicy(policy, str(p), str(cfg.policy.device))
    if run is not None:
        run.summary["residual"] = {"ckpt": str(p), "horizon": wrapped.bridge.horizon, "scale": wrapped.bridge.scale}
    return wrapped
