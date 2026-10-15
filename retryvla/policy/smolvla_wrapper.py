"""SmolVLA wrapper: batched action-chunk prediction from RetryVLA obs dicts.

Bypasses lerobot's env loop but uses the checkpoint's own pre/post-processors, so the
input/output interface is identical to ``lerobot-eval --env.type=libero``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


class SmolVLAWrapper:
    def __init__(self, path: str, device: str = "cuda", n_action_steps: int | None = None):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.path = path
        self.device = torch.device(device)
        self.policy = SmolVLAPolicy.from_pretrained(path)
        self.policy.config.device = str(self.device)
        self.policy.to(self.device).eval()
        if n_action_steps is not None:
            self.policy.config.n_action_steps = int(n_action_steps)
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        cfg = self.policy.config
        self.chunk_size = cfg.chunk_size
        self.max_action_dim = cfg.max_action_dim
        self.action_dim = cfg.action_feature.shape[0]

    @property
    def n_action_steps(self) -> int:
        return int(self.policy.config.n_action_steps)

    def _batch(self, obs: Sequence[dict], tasks: Sequence[str]) -> dict:
        def img(key):
            x = torch.from_numpy(np.stack([o[key] for o in obs]))  # B,H,W,C uint8
            return x.permute(0, 3, 1, 2).float().div_(255.0)

        return {
            "observation.images.image": img("image"),
            "observation.images.image2": img("image2"),
            "observation.state": torch.from_numpy(np.stack([o["state"] for o in obs])).float(),
            "task": list(tasks),
        }

    def make_noise(self, n: int, seeds: Sequence[int] | None = None, scale: float = 1.0) -> torch.Tensor:
        shape = (self.chunk_size, self.max_action_dim)
        if seeds is None:
            z = torch.randn((n, *shape))
        else:
            z = torch.stack([torch.randn(shape, generator=torch.Generator().manual_seed(int(s))) for s in seeds])
        return (z * float(scale)).to(self.device)

    @torch.no_grad()
    def predict_chunk(self, obs: Sequence[dict], tasks: Sequence[str], seeds: Sequence[int] | None = None,
                      noise_scale: float = 1.0) -> np.ndarray:
        """Returns unnormalized actions [B, chunk_size, action_dim] (env convention)."""
        if len(obs) == 0:
            return np.zeros((0, self.chunk_size, self.action_dim), np.float32)
        batch = self.pre(self._batch(obs, tasks))
        noise = self.make_noise(len(obs), seeds, noise_scale)
        self.policy.reset()
        actions = self.policy.predict_action_chunk(batch, noise=noise)  # [B, T, A] normalized
        actions = self.post(actions)
        return actions.float().cpu().numpy()

    @torch.no_grad()
    def chunks_from_noise(self, obs: dict, task: str, noise: torch.Tensor, batch_size: int = 64) -> np.ndarray:
        """Decode given flow-matching noise tensors [n, chunk_size, max_action_dim] at ONE observation.
        Searching over `noise` keeps every candidate on SmolVLA's own action manifold."""
        out = []
        for s in range(0, len(noise), batch_size):
            nz = noise[s: s + batch_size].to(self.device, torch.float32)
            k = len(nz)
            batch = self.pre(self._batch([obs] * k, [task] * k))
            self.policy.reset()
            out.append(self.post(self.policy.predict_action_chunk(batch, noise=nz)).float().cpu().numpy())
        return np.concatenate(out, 0)

    @torch.no_grad()
    def sample_chunks(self, obs: dict, task: str, n: int, noise_scale: float = 1.0,
                      seed: int = 0, batch_size: int = 64) -> np.ndarray:
        """n independent chunk samples at a single observation -> [n, T, A]."""
        out = []
        for s in range(0, n, batch_size):
            k = min(batch_size, n - s)
            out.append(self.predict_chunk([obs] * k, [task] * k, seeds=[seed + s + j for j in range(k)],
                                          noise_scale=noise_scale))
        return np.concatenate(out, 0)
