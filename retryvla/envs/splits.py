"""Scene-level train / pilot / test splits for LIBERO-Recover.

Leakage unit = ``Scene.group`` (task folder + source failure episode): the scenes captured at
step_0180 / step_0200 of one failed episode are near-duplicates and must land in the same split.
Stratified by task folder. ``pilot`` is a subset of ``train``. ``test`` is never touched by
search / critic / WM training / SFT (``assert_allowed``).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from retryvla.envs.libero_env import Scene, discover_scenes


def build_split(scenes: list[Scene], test_frac: float = 0.35, pilot_scenes: int = 60, seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    by_task: dict[str, list[str]] = defaultdict(list)
    for g in sorted({s.group for s in scenes}):
        by_task[g.split("__")[0]].append(g)
    test_groups, train_groups = [], []
    for task in sorted(by_task):
        gs = list(by_task[task])
        rng.shuffle(gs)
        n_test = int(round(test_frac * len(gs)))
        if len(gs) >= 2:
            n_test = min(max(n_test, 1), len(gs) - 1)
        else:
            n_test = 0  # single-episode tasks stay in train
        test_groups += gs[:n_test]
        train_groups += gs[n_test:]
    g2s: dict[str, list[str]] = defaultdict(list)
    for s in scenes:
        g2s[s.group].append(s.tag)
    train = sorted(t for g in train_groups for t in g2s[g])
    test = sorted(t for g in test_groups for t in g2s[g])
    # pilot: group-aware subset of train, round-robin over tasks for coverage
    tg = defaultdict(list)
    for g in train_groups:
        tg[g.split("__")[0]].append(g)
    for v in tg.values():
        rng.shuffle(v)
    pilot: list[str] = []
    while len(pilot) < pilot_scenes and any(tg.values()):
        for task in sorted(tg):
            if tg[task] and len(pilot) < pilot_scenes:
                pilot += g2s[tg[task].pop()]
    return {
        "version": "spatial_v1",
        "seed": seed,
        "test_frac": test_frac,
        "group_key": "task_folder__episode",
        "n_scenes": len(scenes),
        "n_groups": len(g2s),
        "train": train,
        "pilot": sorted(pilot),
        "test": test,
        "test_groups": sorted(test_groups),
        "train_groups": sorted(train_groups),
    }


def save_split(split: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(split, indent=1))


def load_split(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def split_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def scenes_for(split: dict, part: str, scene_root: str | Path) -> list[Scene]:
    tags = set(split[part])
    return [s for s in discover_scenes(scene_root) if s.tag in tags]


class TestSplitLeak(RuntimeError):
    pass


def assert_allowed(tags, split: dict) -> None:
    """Raise if any scene tag belongs to the test split (search/training code must call this)."""
    bad = sorted(set(tags) & set(split["test"]))
    if bad:
        raise TestSplitLeak(f"{len(bad)} test-split scenes requested for search/training, e.g. {bad[:3]}")
