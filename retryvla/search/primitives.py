"""Hand-designed recovery primitives in LIBERO's OSC delta action space.

Action = [dx, dy, dz, droll, dpitch, dyaw, gripper] in [-1, 1]; gripper -1 open / +1 close.
With LIBERO's OSC_POSE limits, |dx|=1 is ~5 mm per step at 20 Hz. Primitives are padded
(zero arm motion, final gripper state held) to the search horizon.
"""

from __future__ import annotations

import numpy as np


def _seg(n: int, dpos=(0, 0, 0), grip: float = -1.0) -> np.ndarray:
    a = np.zeros((n, 7), np.float32)
    a[:, :3] = dpos
    a[:, 6] = grip
    return a


def _pad(x: np.ndarray, h: int) -> np.ndarray:
    if len(x) >= h:
        return x[:h]
    tail = np.zeros((h - len(x), 7), np.float32)
    tail[:, 6] = x[-1, 6] if len(x) else -1.0
    return np.concatenate([x, tail], 0)


def open_gripper(h: int) -> np.ndarray:
    return _pad(_seg(10, grip=-1.0), h)


def retreat(h: int, up: float = 0.6, steps: int = 20) -> np.ndarray:
    return _pad(np.concatenate([_seg(6, grip=-1.0), _seg(steps, (0, 0, up), -1.0)]), h)


def reposition(h: int, direction: float, dist_steps: int = 15, up: float = 0.6) -> np.ndarray:
    dx, dy = 0.6 * np.cos(direction), 0.6 * np.sin(direction)
    return _pad(np.concatenate([
        _seg(6, grip=-1.0), _seg(12, (0, 0, up), -1.0), _seg(dist_steps, (dx, dy, 0), -1.0),
    ]), h)


def regrasp(h: int, down_steps: int = 12, down: float = -0.6) -> np.ndarray:
    return _pad(np.concatenate([
        _seg(6, grip=-1.0), _seg(10, (0, 0, 0.5), -1.0), _seg(down_steps, (0, 0, down), -1.0),
        _seg(10, grip=1.0), _seg(10, (0, 0, 0.6), 1.0),
    ]), h)


def library(h: int, rng: np.random.RandomState, n_random: int = 8, include_fixed: bool = True
            ) -> tuple[np.ndarray, list[str]]:
    """Fixed primitives + randomized variants. Returns [N, h, 7] and names."""
    seqs, names = [open_gripper(h), retreat(h), retreat(h, steps=30), regrasp(h)], [
        "open", "retreat20", "retreat30", "regrasp"]
    for k in range(4):
        th = k * np.pi / 2
        seqs.append(reposition(h, th))
        names.append(f"reposition_{int(np.degrees(th))}")
    if not include_fixed:
        seqs, names = [], []
    for k in range(n_random):
        kind = rng.randint(3)
        if kind == 0:
            seqs.append(reposition(h, rng.uniform(0, 2 * np.pi), int(rng.randint(8, 25)), rng.uniform(0.3, 0.8)))
            names.append("reposition_rand")
        elif kind == 1:
            seqs.append(retreat(h, rng.uniform(0.3, 0.9), int(rng.randint(10, 35))))
            names.append("retreat_rand")
        else:
            seqs.append(regrasp(h, int(rng.randint(6, 20)), rng.uniform(-0.9, -0.3)))
            names.append("regrasp_rand")
    return np.stack(seqs).astype(np.float32), names
