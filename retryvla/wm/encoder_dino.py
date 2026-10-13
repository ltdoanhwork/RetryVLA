"""M2 skeleton: frozen DINOv2 patch-feature encoder + robot-state projection.

h_t = [E_DINO(o_t) (spatial patch tokens, both cameras), E_q(q_t)]

Spatial patch tokens are kept (optionally pooled to a coarse grid) instead of a single global
embedding: L3/L4 failures depend on object position / orientation / relations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DinoEncoder(nn.Module):
    def __init__(self, model_name: str = "facebook/dinov2-small", image_size: int = 224, pool_grid: int | None = 8,
                 state_dim: int = 8, state_tokens: int = 1, device: str = "cuda"):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(model_name).eval().requires_grad_(False)
        self.image_size, self.pool_grid = image_size, pool_grid
        self.dim = self.backbone.config.hidden_size
        self.state_proj = nn.Linear(state_dim, self.dim * state_tokens)
        self.state_tokens = state_tokens
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device)

    @torch.no_grad()
    def encode_images(self, img_uint8: torch.Tensor) -> torch.Tensor:
        """[B, H, W, 3] uint8 -> [B, P, D] patch tokens (pooled to pool_grid^2 if set)."""
        x = img_uint8.permute(0, 3, 1, 2).float().div(255.0)
        x = nn.functional.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        tok = self.backbone(pixel_values=x).last_hidden_state[:, 1:]  # drop CLS
        if self.pool_grid:
            b, p, d = tok.shape
            g = int(p ** 0.5)
            tok = tok.transpose(1, 2).reshape(b, d, g, g)
            tok = nn.functional.adaptive_avg_pool2d(tok, self.pool_grid).flatten(2).transpose(1, 2)
        return tok

    def encode_state(self, q: torch.Tensor) -> torch.Tensor:
        return self.state_proj(q).view(q.shape[0], self.state_tokens, self.dim)

    def encode_obs(self, obs: dict) -> torch.Tensor:
        """Single RetryVLA obs dict -> [1, T, D] tokens (agent cam, wrist cam, state)."""
        dev = self.mean.device
        a = self.encode_images(torch.as_tensor(obs["image"], device=dev)[None])
        w = self.encode_images(torch.as_tensor(obs["image2"], device=dev)[None])
        q = self.encode_state(torch.as_tensor(obs["state"], device=dev, dtype=torch.float32)[None])
        return torch.cat([a, w, q], 1)
