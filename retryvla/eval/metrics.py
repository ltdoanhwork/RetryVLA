"""Metrics + small IO helpers shared by eval scripts."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def rate_table(rows: Iterable[dict], key: str, success_key: str = "success") -> dict[str, dict[str, Any]]:
    agg: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        agg[str(r[key])].append(bool(r[success_key]))
    out = {}
    for k, v in sorted(agg.items()):
        lo, hi = wilson(sum(v), len(v))
        out[k] = {"n": len(v), "k": int(sum(v)), "rate": float(np.mean(v)), "ci95": [lo, hi]}
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_video(frames: np.ndarray, path: str | Path, fps: int = 20) -> None:
    import imageio

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(path), [np.asarray(f) for f in frames], fps=fps)
