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


def patch_lerobot_index_mapping() -> None:
    """lerobot 0.5.1 builds the abs->rel index map with ``hf_dataset["index"]``, which runs the
    torch transform and decodes every image of image-format datasets (30+ min on /pfss for our
    recovery data when an episode subset is selected). Read the raw column instead."""
    import torch
    from lerobot.datasets import dataset_reader as dr

    if getattr(dr, "_retryvla_patched", False):
        return

    def _build_index_mapping(self) -> None:
        self._absolute_to_relative_idx = None
        if self.episodes is not None and self.hf_dataset is not None:
            col = self.hf_dataset.with_format(None).select_columns(["index"])["index"]
            self._absolute_to_relative_idx = {
                (a.item() if isinstance(a, torch.Tensor) else int(a)): i for i, a in enumerate(col)
            }

    for name in dir(dr):
        cls = getattr(dr, name)
        if isinstance(cls, type) and "_build_index_mapping" in cls.__dict__:
            cls._build_index_mapping = _build_index_mapping
    dr._retryvla_patched = True


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


def expand_roots(patterns) -> list[str]:
    """Expand glob patterns in train.recovery_datasets; skip incomplete resumable parts (p<k> without CLOSED)."""
    import glob as _glob

    out = []
    for p in patterns:
        p = str(p)
        if not any(ch in p for ch in "*?["):
            out.append(p)
            continue
        for d in sorted(_glob.glob(p)):
            par = Path(d).parent
            if par.name.startswith("p") and par.name[1:].isdigit() and not (par / "CLOSED").exists():
                continue
            if (Path(d) / "retryvla" / "trajectories.jsonl").exists():
                out.append(d)
    return out


def recovery_episode_subsets(roots, n: int | None, min_len: int = 0, mode: str = "score",
                             seed: int = 0) -> list[list[int] | None]:
    """Pick n trajectories across several dataset roots (None = all).
    mode="score": global top-n by search score. Biased: the critic scores hard tasks (e.g. drawer) low, so a
      top-n cut silently drops them (M5b: 4 drawer trajectories left out of 208).
    mode="roundrobin": coverage-preserving. Instances in random order; take each instance's best trajectory,
      then each instance's 2nd best, ... until n.
    min_len drops trivial recoveries (e.g. <=15 steps: the failure state was already nearly solved)."""
    if n is None and not min_len:
        return [None] * len(roots)
    rows = []
    for k, root in enumerate(roots):
        for l in open(Path(root) / "retryvla" / "trajectories.jsonl"):
            r = json.loads(l)
            if r.get("length", 0) >= min_len:
                rows.append((r.get("score", 0.0), k, r["episode_index"], r.get("instance_id", f"{k}_{r['episode_index']}")))
    if mode == "roundrobin":
        by: dict[str, list] = {}
        for row in sorted(rows, key=lambda x: -x[0]):
            by.setdefault(row[3], []).append(row)
        ids = sorted(by)
        np.random.default_rng(seed).shuffle(ids)
        rows = [by[i][r] for r in range(max(len(v) for v in by.values())) for i in ids if r < len(by[i])]
    elif mode == "random":
        rng = np.random.default_rng(seed)
        rows = [rows[i] for i in rng.permutation(len(rows))]
    elif mode == "task_balanced":
        # round robin over TASKS (task = instance id prefix), inside a task round robin over its instances
        rng = np.random.default_rng(seed)
        by_t: dict[str, dict[str, list]] = {}
        for row in sorted(rows, key=lambda x: -x[0]):
            t_ = str(row[3]).split("__")[0]
            by_t.setdefault(t_, {}).setdefault(row[3], []).append(row)
        queues = {}
        for t_, insts in by_t.items():
            ids = sorted(insts)
            rng.shuffle(ids)
            queues[t_] = [insts[i][r] for r in range(max(len(v) for v in insts.values())) for i in ids if r < len(insts[i])]
        rows, k = [], 0
        while any(queues.values()):
            for t_ in sorted(queues):
                if queues[t_]:
                    rows.append(queues[t_].pop(0))
    elif mode == "score":
        rows.sort(key=lambda x: -x[0])
    else:
        raise ValueError(mode)
    keep = [[] for _ in roots]
    for _, k, e, _ in (rows if n is None else rows[:n]):
        keep[k].append(e)
    return [sorted(v) for v in keep]


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


def select_futures(fut: dict, variant: str, n: int | None, seed: int = 0, filters: dict | None = None) -> np.ndarray:
    """Row indices of ``futures.pt['futures']`` for one variant, round-robin over failure states (rank 0 of every
    state first, then rank 1, ...) so a count cap keeps failure-level coverage (the M9 lesson). n=None = all.
    filters: optional {"min_cpi": x, "sim": "success"|"fail"|"unchecked"} (analysis variants; default none)."""
    var = np.asarray(fut["variant"])
    rows = np.nonzero(var == variant)[0]
    f = filters or {}
    if f.get("min_cpi") is not None:
        rows = rows[fut["cpi"].numpy()[rows] >= float(f["min_cpi"])]
    if f.get("sim"):
        chk, ok = fut["sim_checked"].numpy()[rows], fut["sim_success"].numpy()[rows]
        keep = {"success": chk & ok, "fail": chk & ~ok, "unchecked": ~chk}[f["sim"]]
        rows = rows[keep]
    obs, rank = fut["obs"].numpy()[rows], fut["rank"].numpy()[rows]
    order = np.random.default_rng(seed).permutation(int(obs.max()) + 1 if len(obs) else 0)
    pos = np.empty_like(order)
    pos[order] = np.arange(len(order))
    rows = rows[np.lexsort((pos[obs], rank))]  # primary key rank, then a fixed random order of failure states
    return rows if n is None else rows[: int(n)]


class FutureDataset(Dataset):
    """Part 4 (RQ2) action-free samples: the FAILURE observation (images, state, task) of a train failure state +
    the pooled DINOv2 latent of one WM-imagined future of it (``retryvla.search.wm_datagen.main_futures``).
    No action label: ``action`` is a zero placeholder with ``action_is_pad`` all True and ``has_action`` False, so
    the sample is excluded from the flow-matching loss and only feeds L_future (retryvla/train/sft.py).
    ``center``: subtract the mean target over the selected rows (cosine loss on the residual direction)."""

    def __init__(self, path: str | Path, variant: str, n: int | None = None, chunk: int = 50, action_dim: int = 7,
                 seed: int = 0, center: bool = True, filters: dict | None = None):
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.obs, fut = d["obs"], d["futures"]
        self.rows = select_futures(fut, variant, n, seed, filters)
        self.obs_idx = fut["obs"][self.rows].long()
        tgt = fut["future_latent"][self.rows].float()
        self.target_mean = tgt.mean(0) if center and len(tgt) else torch.zeros(tgt.shape[1:])
        self.target = tgt - self.target_mean
        self.cpi = fut["cpi"][self.rows].float()
        self.sim_success = fut["sim_success"][self.rows].bool()
        self.sim_checked = fut["sim_checked"][self.rows].bool()
        self.chunk, self.adim = chunk, action_dim
        self.num_episodes = len(self.rows)        # "episodes" = imagined futures (for logging / budget)
        self.num_frames = len(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, j: int) -> dict:
        o = int(self.obs_idx[j])
        img = lambda k: self.obs[k][o].permute(2, 0, 1).float() / 255.0  # noqa: E731  HWC uint8 -> CHW [0,1]
        return {"observation.images.image": img("image"), "observation.images.image2": img("image2"),
                "observation.state": self.obs["state"][o].float(),
                "action": torch.zeros(self.chunk, self.adim), "action_is_pad": torch.ones(self.chunk, dtype=torch.bool),
                "task": self.obs["task"][o], "future_latent": self.target[j],
                "has_action": torch.tensor(False), "has_future": torch.tensor(True)}


class MixDataset(Dataset):
    """Virtual dataset of length ``n``: item i draws from recovery data with prob ``ratio``.

    Part 4 (default OFF): ``futures`` = list of FutureDataset; with prob ``future_ratio`` an item is an action-free
    imagined-future sample instead (drawn BEFORE the nominal / recovery split; the remaining items keep the
    nominal : recovery = (1 - ratio) : ratio split). Only then do items carry ``future_latent`` / ``has_action`` /
    ``has_future`` (missing = has_action True, no future target). With futures=None the random stream and the items
    are exactly the pre-Part-4 ones."""

    def __init__(self, original, recovery: list, ratio: float, n: int, segments=("recovery", "continuation"),
                 seed: int = 0, futures: list | None = None, future_ratio: float = 0.0):
        self.orig, self.rec, self.ratio, self.n, self.seed = original, recovery, float(ratio), int(n), seed
        self.fut = [f for f in (futures or []) if len(f)]
        self.fut_index = [(di, j) for di, f in enumerate(self.fut) for j in range(len(f))]
        self.future_ratio = float(future_ratio) if self.fut_index else 0.0
        self.with_future = self.future_ratio > 0
        self.future_dim = int(self.fut[0].target.shape[-1]) if self.fut else 0
        allowed = {SEGMENT_IDS[s] for s in segments}
        self.rec_index: list[tuple[int, int]] = []
        for di, ds in enumerate(recovery):
            # read ONLY the segment column (hf_dataset["segment"] would decode every image)
            seg = np.asarray(ds.hf_dataset.with_format(None).select_columns(["segment"])["segment"]).reshape(-1)
            self.rec_index += [(di, int(i)) for i in np.nonzero(np.isin(seg, list(allowed)))[0]]
        if not self.rec_index:
            self.ratio = 0.0

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng((self.seed, i))
        if self.with_future:
            return self._getitem_future(rng)
        if rng.random() < self.ratio:
            di, j = self.rec_index[rng.integers(len(self.rec_index))]
            item, src = self.rec[di][j], 1
        else:
            item, src = self.orig[int(rng.integers(len(self.orig)))], 0
        out = {k: item[k] for k in CORE_KEYS if k in item}
        # image-format datasets get an observation delta dim ([1,C,H,W]); video ones don't
        for k in ("observation.images.image", "observation.images.image2", "observation.state"):
            v = out.get(k)
            if v is not None and v.ndim == (4 if "images" in k else 2) and v.shape[0] == 1:
                out[k] = v[0]
        out["source"] = torch.tensor(src)
        return out

    def _getitem_future(self, rng) -> dict:
        """source: 0 nominal, 1 grounded recovery, 2 action-free imagined future."""
        if rng.random() < self.future_ratio:
            di, j = self.fut_index[rng.integers(len(self.fut_index))]
            out = self.fut[di][j]
            out["source"] = torch.tensor(2)
            return out
        if rng.random() < self.ratio:
            di, j = self.rec_index[rng.integers(len(self.rec_index))]
            item, src = self.rec[di][j], 1
        else:
            item, src = self.orig[int(rng.integers(len(self.orig)))], 0
        out = {k: item[k] for k in CORE_KEYS if k in item}
        for k in ("observation.images.image", "observation.images.image2", "observation.state"):
            v = out.get(k)
            if v is not None and v.ndim == (4 if "images" in k else 2) and v.shape[0] == 1:
                out[k] = v[0]
        out["future_latent"] = torch.zeros(self.future_dim)
        out["has_action"], out["has_future"] = torch.tensor(True), torch.tensor(False)
        out["source"] = torch.tensor(src)
        return out
