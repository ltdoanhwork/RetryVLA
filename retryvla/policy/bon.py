"""Online best-of-N planning baseline: at every replanning step sample N SmolVLA chunks, imagine each with the latent
WM, score with the competence critic (same penalties as search), execute the best. No distillation; costs N VLA
samples + N WM rollouts per replanning step at deployment.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


class BestOfNPolicy:
    def __init__(self, base, ranker, n: int, horizon: int = 50):
        self.base, self.ranker, self.n, self.h = base, ranker, int(n), int(horizon)
        self.calls = 0

    @property
    def n_action_steps(self) -> int:
        return self.base.n_action_steps

    def __getattr__(self, k):  # chunk_size, action_dim, ...
        return getattr(self.base, k)

    @torch.no_grad()
    def predict_chunk(self, obs: Sequence[dict], tasks: Sequence[str], seeds: Sequence[int] | None = None,
                      noise_scale: float = 1.0) -> np.ndarray:
        if len(obs) == 0:
            return self.base.predict_chunk(obs, tasks, seeds, noise_scale)
        B, N = len(obs), self.n
        sd = [None] * B if seeds is None else list(seeds)
        rep_o = [o for o in obs for _ in range(N)]
        rep_t = [t for t in tasks for _ in range(N)]
        rep_s = None if seeds is None else [int(s) * 1000 + j for s in sd for j in range(N)]
        ch = self.base.predict_chunk(rep_o, rep_t, rep_s, noise_scale)  # [B*N, T, A]
        out = []
        for b in range(B):
            c = np.clip(ch[b * N:(b + 1) * N], -1, 1).astype(np.float32)
            z0, q0 = self.ranker.encode(obs[b])
            s = self.ranker.score(z0, q0, tasks[b], c[:, : self.h])["score"]
            out.append(c[int(np.argmax(s))])
        self.calls += B
        return np.stack(out)
