"""Observation features + actor-critic of the residual bridge controller.

Input per decision (one decision = one pi0 chunk of n_act env steps inside the bridge window):
  - frozen DINOv2-small patch tokens of both cameras, avg-pooled to a 2x2 grid  -> 2 x 4 x 384 = 3072
  - 8-D proprio state (eef pos, axis-angle, gripper qpos)
  - instruction: fixed hashed bag of uni+bi-grams (256-D, no learning; paraphrase-tolerant, no extra model)
  - pi0's planned chunk for this decision (n_act x 7): the residual sees what the frozen VLA intends
  - one-hot decision index inside the bridge (0..n_dec-1)
Output (actor): 6-D Gaussian residual delta on the arm action (held constant over the chunk; executed as
scale * clip(a, -1, 1)), and a 3-way gripper head {follow pi0, force open, force close}. Critic: V(s).
Initialisation: residual mean ~0 and gripper head biased to "follow", so the untrained controller == pi0.
"""

from __future__ import annotations

import re
import zlib

import numpy as np
import torch
import torch.nn as nn

TEXT_DIM = 256


def hashed_text(task: str, dim: int = TEXT_DIM) -> np.ndarray:
    w = re.findall(r"[a-z]+", task.lower())
    grams = w + [f"{a}_{b}" for a, b in zip(w, w[1:])]
    v = np.zeros(dim, np.float32)
    for g in grams:
        h = zlib.crc32(g.encode())
        v[h % dim] += 1.0 if (h >> 16) & 1 else -1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class FeatureEncoder:
    """Frozen DINOv2 (same backbone as the RetryVLA WM, coarser pooling)."""

    def __init__(self, model_name: str = "facebook/dinov2-small", pool_grid: int = 2, device: str = "cuda"):
        from retryvla.wm.encoder_dino import DinoEncoder

        self.enc = DinoEncoder(model_name, pool_grid=pool_grid, device=device)
        self.device = torch.device(device)
        self.img_dim = 2 * pool_grid * pool_grid * self.enc.dim

    @torch.no_grad()
    def __call__(self, obs: list[dict]) -> torch.Tensor:
        a = torch.as_tensor(np.stack([o["image"] for o in obs]), device=self.device)
        w = torch.as_tensor(np.stack([o["image2"] for o in obs]), device=self.device)
        z = self.enc.encode_images(torch.cat([a, w], 0))  # [2B, P, D]
        B = len(obs)
        return torch.cat([z[:B].flatten(1), z[B:].flatten(1)], 1).float()


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


class _Trunk(nn.Module):
    def __init__(self, img_dim: int, n_act: int, n_dec: int, hidden: int):
        super().__init__()
        self.img = nn.Sequential(nn.LayerNorm(img_dim), nn.Linear(img_dim, hidden), nn.GELU())
        self.low_dim = 8 + TEXT_DIM + n_act * 7 + n_dec
        self.low = nn.Sequential(nn.LayerNorm(self.low_dim), nn.Linear(self.low_dim, hidden), nn.GELU())
        self.out_dim = 2 * hidden

    def forward(self, img, low):
        return torch.cat([self.img(img), self.low(low)], -1)


class ActorCritic(nn.Module):
    def __init__(self, img_dim: int, n_act: int = 10, n_dec: int = 5, hidden: int = 256, init_log_std: float = -0.5,
                 grip_follow_bias: float = 2.0):
        super().__init__()
        self.hparams = dict(img_dim=img_dim, n_act=n_act, n_dec=n_dec, hidden=hidden, init_log_std=init_log_std,
                            grip_follow_bias=grip_follow_bias)
        self.n_act, self.n_dec = n_act, n_dec
        self.ta, self.tc = _Trunk(img_dim, n_act, n_dec, hidden), _Trunk(img_dim, n_act, n_dec, hidden)
        self.mu = _mlp(self.ta.out_dim, hidden, 6)
        self.grip = _mlp(self.ta.out_dim, hidden, 3)
        self.v = _mlp(self.tc.out_dim, hidden, 1)
        self.log_std = nn.Parameter(torch.full((6,), float(init_log_std)))
        with torch.no_grad():
            for head in (self.mu, self.grip):
                head[-1].weight.mul_(0.01)
                head[-1].bias.zero_()
            self.grip[-1].bias[0] = grip_follow_bias

    def low_input(self, state, text, pi0_chunk, dec_idx) -> torch.Tensor:
        oh = nn.functional.one_hot(dec_idx.long().clamp(0, self.n_dec - 1), self.n_dec).float()
        return torch.cat([state, text, pi0_chunk.flatten(1), oh], -1)

    def dist(self, img, low):
        h = self.ta(img, low)
        mu = self.mu(h)
        std = self.log_std.clamp(-3.0, 0.5).exp().expand_as(mu)
        return torch.distributions.Normal(mu, std), torch.distributions.Categorical(logits=self.grip(h))

    def value(self, img, low) -> torch.Tensor:
        return self.v(self.tc(img, low)).squeeze(-1)

    def evaluate(self, img, low, a_cont, a_grip):
        dc, dg = self.dist(img, low)
        logp = dc.log_prob(a_cont).sum(-1) + dg.log_prob(a_grip)
        ent = dc.entropy().sum(-1) + dg.entropy()
        return logp, ent, self.value(img, low)


def apply_residual(chunk: np.ndarray, a_cont: np.ndarray, a_grip: np.ndarray, scale: float) -> np.ndarray:
    """chunk [B, T, 7] pi0 actions; a_cont [B, 6]; a_grip [B] in {0 follow, 1 open, 2 close}."""
    out = np.array(chunk, dtype=np.float32, copy=True)
    out[..., :6] += float(scale) * np.clip(a_cont, -1.0, 1.0)[:, None, :]
    out[a_grip == 1, :, 6] = -1.0
    out[a_grip == 2, :, 6] = 1.0
    return np.clip(out, -1.0, 1.0)


def save_ckpt(path, model: ActorCritic, extra: dict | None = None) -> None:
    torch.save({"model": model.state_dict(), "hparams": model.hparams, **(extra or {})}, path)


def load_actor(path, device="cuda") -> tuple[ActorCritic, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ActorCritic(**ck["hparams"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck
