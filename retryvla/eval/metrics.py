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


LEVELS = ("L1", "L2", "L3", "L4")


def level_table(rows: Iterable[dict], levels: dict[str, str], scene_key: str = "scene") -> dict[str, dict[str, Any]]:
    """Per-level RSR (L1-L4 columns, always present, n=0 if empty) with levels looked up by scene tag, so old
    ``episodes.jsonl`` files (all ``level: unk``) can be re-broken-down without re-running rollouts.
    ``levels`` = ``splits.load_levels(scene_root[, official_only])``; scenes not in the map count as ``unk``."""
    rel = [{"level": levels.get(r[scene_key], "unk"), "success": r["success"]} for r in rows]
    tab = rate_table(rel, "level")
    for lv in LEVELS:
        tab.setdefault(lv, {"n": 0, "k": 0, "rate": float("nan"), "ci95": [float("nan"), float("nan")]})
    return dict(sorted(tab.items()))


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
