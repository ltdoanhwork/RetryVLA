"""Local experiment tracking: run dirs, config snapshot, git state, jsonl + TensorBoard."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from retryvla import REPO_ROOT, RUNS_DIR


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""


def file_sha(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        return "missing"
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


class Run:
    """One experiment run. Use as a context manager or call ``finish()``."""

    def __init__(self, cfg: DictConfig, run_dir: Path):
        self.cfg = cfg
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._metrics = open(self.dir / "metrics.jsonl", "a", buffering=1)
        self._tb = None
        self._t0 = time.time()
        self.summary: dict[str, Any] = {}

    @property
    def id(self) -> str:
        return self.dir.name

    @property
    def tb(self):
        if self._tb is None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(str(self.dir / "tb"))
            except Exception:  # tensorboard missing -> jsonl only
                self._tb = False
        return self._tb or None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        rec = {"time": round(time.time() - self._t0, 3), "step": step, **metrics}
        self._metrics.write(json.dumps(rec, default=float) + "\n")
        tb = self.tb
        if tb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    tb.add_scalar(k, v, global_step=step if step is not None else 0)

    def path(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def finish(self, status: str = "done", **summary: Any) -> None:
        self.summary.update(summary)
        out = {
            "run_id": self.id,
            "milestone": self.cfg.get("milestone"),
            "tag": self.cfg.get("tag"),
            "status": status,
            "git_sha": _git("rev-parse", "--short", "HEAD").strip(),
            "wall_time_s": round(time.time() - self._t0, 1),
            "results": self.summary,
        }
        (self.dir / "summary.json").write_text(json.dumps(out, indent=2, default=float))
        self._metrics.close()
        if self._tb:
            self._tb.close()

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.finish("done")
        else:
            self.finish("failed", error=f"{exc_type.__name__}: {exc}")


def start(cfg: DictConfig, run_dir: str | Path | None = None) -> Run:
    """Create ``runs/<YYYYMMDD-HHMM>_<milestone>_<tag>/`` and snapshot config + git state."""
    if run_dir is None:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = RUNS_DIR / f"{stamp}_{cfg.milestone}_{cfg.tag}"
    run_dir = Path(run_dir)
    run = Run(cfg, run_dir)
    split_file = cfg.get("split", {}).get("file") if isinstance(cfg.get("split"), DictConfig) else None
    meta = {
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "split_sha": file_sha(REPO_ROOT / split_file) if split_file else None,
    }
    OmegaConf.save(cfg, run.dir / "config.yaml")
    git_txt = (
        f"sha: {_git('rev-parse', 'HEAD').strip()}\n"
        f"branch: {_git('rev-parse', '--abbrev-ref', 'HEAD').strip()}\n"
        f"meta: {json.dumps(meta)}\n\n--- status ---\n{_git('status', '--short')}\n"
        f"--- diff ---\n{_git('diff', 'HEAD')}"
    )
    (run.dir / "git.txt").write_text(git_txt)
    run.summary["meta"] = meta
    print(f"[retryvla] run dir: {run.dir}")
    return run
