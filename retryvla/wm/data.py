"""Load collected latent trajectories (retryvla.wm.collect output) for WM / critic training."""

from __future__ import annotations

import glob
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from retryvla import REPO_ROOT


@dataclass
class LatentSet:
    z: torch.Tensor          # [N, F, P, D] fp16 (raw DINO tokens)
    q: torch.Tensor          # [N, F, 8]
    valid: torch.Tensor      # [N, F] bool
    actions: torch.Tensor    # [N, H, 7]
    success: torch.Tensor    # [N] bool
    prefix_success: torch.Tensor
    inst: list[str]          # instance id per trajectory
    task: list[str]
    source: list[str]

    def __len__(self) -> int:
        return len(self.success)

    def subset(self, idx) -> "LatentSet":
        idx = torch.as_tensor(idx, dtype=torch.long)
        il = idx.tolist()
        return LatentSet(self.z[idx], self.q[idx], self.valid[idx], self.actions[idx], self.success[idx],
                         self.prefix_success[idx], [self.inst[i] for i in il], [self.task[i] for i in il],
                         [self.source[i] for i in il])

    def last_valid(self) -> torch.Tensor:
        return self.valid.long().cumsum(1).argmax(1)  # index of last valid frame


def load_latents(roots) -> LatentSet:
    files = sorted(f for r in roots for f in glob.glob(str(REPO_ROOT / r / "*.pt")))
    if not files:
        raise FileNotFoundError(f"no latent files under {roots}")
    Z, Q, V, A, S, PS, I, T, SRC = [], [], [], [], [], [], [], [], []
    for f in files:
        d = torch.load(f, weights_only=False)
        k = len(d["success"])
        Z.append(d["z"]); Q.append(d["q"]); V.append(d["valid"]); A.append(d["actions"])
        S.append(d["success"]); PS.append(d["prefix_success"])
        I += [d["instance_id"]] * k; T += [d["task"]] * k; SRC += list(d["source"])
    return LatentSet(torch.cat(Z), torch.cat(Q).float(), torch.cat(V), torch.cat(A).float(), torch.cat(S),
                     torch.cat(PS), I, T, SRC)


def group_split(ls: LatentSet, val_frac: float = 0.15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Split by failure episode (task__episode) so val failures are unseen."""
    groups = ["__".join(i.split("__")[:2]) for i in ls.inst]
    is_val = np.array([(zlib.crc32(f"{seed}:{g}".encode()) % 1000) / 1000 < val_frac for g in groups])
    return np.nonzero(~is_val)[0], np.nonzero(is_val)[0]


class Normalizer:
    """Per-dim standardization of DINO tokens and robot state (fit on training trajectories)."""

    def __init__(self, z_mean, z_std, q_mean, q_std):
        self.z_mean, self.z_std, self.q_mean, self.q_std = z_mean, z_std, q_mean, q_std

    @classmethod
    def fit(cls, ls: LatentSet) -> "Normalizer":
        m = ls.valid
        zf = ls.z[m].float()                     # [M, P, D]
        qf = ls.q[m]
        return cls(zf.mean((0, 1)), zf.std((0, 1)).clamp_min(1e-3), qf.mean(0), qf.std(0).clamp_min(1e-3))

    def z(self, z: torch.Tensor) -> torch.Tensor:
        return (z.float() - self.z_mean.to(z.device)) / self.z_std.to(z.device)

    def q(self, q: torch.Tensor) -> torch.Tensor:
        return (q.float() - self.q_mean.to(q.device)) / self.q_std.to(q.device)

    def state_dict(self) -> dict:
        return {"z_mean": self.z_mean, "z_std": self.z_std, "q_mean": self.q_mean, "q_std": self.q_std}

    @classmethod
    def from_state(cls, d: dict) -> "Normalizer":
        return cls(d["z_mean"], d["z_std"], d["q_mean"], d["q_std"])


def task_vocab(tasks) -> dict[str, int]:
    return {t: i for i, t in enumerate(sorted(set(tasks)))}


def save_ckpt(path: str | Path, **kw) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(kw, path)
