"""Dataset mixing for recovery SFT: nominal LIBERO demos + verified recovery trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CORE_KEYS = ("observation.images.image", "observation.images.image2", "observation.state", "action",
             "action_is_pad", "task")
SEGMENT_IDS = {"recovery": 0, "continuation": 1}


def spatial_episode_ids(repo_id: str, suite: str) -> list[int]:
    """Episodes of ``repo_id`` whose task language belongs to the given LIBERO suite."""
    import pandas as pd
    from huggingface_hub import snapshot_download
    from libero.libero import benchmark

    langs = {t.language for t in benchmark.get_benchmark_dict()[suite]().tasks}
    root = Path(snapshot_download(repo_id, repo_type="dataset", allow_patterns=["meta/*", "data/*"]))
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    keep_idx = {int(r.task_index) for lang, r in tasks.iterrows() if lang in langs}
    frames = pd.concat([pd.read_parquet(p, columns=["episode_index", "task_index"])
                        for p in sorted((root / "data").rglob("*.parquet"))])
    ep_task = frames.groupby("episode_index")["task_index"].first()
    return sorted(int(e) for e, t in ep_task.items() if int(t) in keep_idx)


def recovery_episode_subset(root: str | Path, n: int | None) -> list[int] | None:
    """Top-n episodes by search score (None = all)."""
    if n is None:
        return None
    rows = [json.loads(l) for l in open(Path(root) / "retryvla" / "trajectories.jsonl")]
    rows.sort(key=lambda r: -r.get("score", 0.0))
    return sorted(r["episode_index"] for r in rows[:n])


def count_trajectories(root: str | Path) -> int:
    p = Path(root) / "retryvla" / "trajectories.jsonl"
    return sum(1 for _ in open(p)) if p.exists() else 0


class MixDataset(Dataset):
    """Virtual dataset of length ``n``: item i draws from recovery data with prob ``ratio``."""

    def __init__(self, original, recovery: list, ratio: float, n: int, segments=("recovery", "continuation"),
                 seed: int = 0):
        self.orig, self.rec, self.ratio, self.n, self.seed = original, recovery, float(ratio), int(n), seed
        allowed = {SEGMENT_IDS[s] for s in segments}
        self.rec_index: list[tuple[int, int]] = []
        for di, ds in enumerate(recovery):
            seg = np.asarray(ds.hf_dataset["segment"]).reshape(-1)
            self.rec_index += [(di, int(i)) for i in np.nonzero(np.isin(seg, list(allowed)))[0]]
        if not self.rec_index:
            self.ratio = 0.0

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng((self.seed, i))
        if rng.random() < self.ratio:
            di, j = self.rec_index[rng.integers(len(self.rec_index))]
            item, src = self.rec[di][j], 1
        else:
            item, src = self.orig[int(rng.integers(len(self.orig)))], 0
        out = {k: item[k] for k in CORE_KEYS if k in item}
        out["source"] = torch.tensor(src)
        return out
