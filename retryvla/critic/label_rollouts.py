"""M2 skeleton: collect critic labels.

Labels y(s) = 1[frozen pi0 succeeds from state s], s drawn from TRAIN-split states only:
  * failure instances (M0 pre-check rollouts, ``precheck.csv`` of an M1 run),
  * intermediate states along M1 CEM branches (``branches.parquet``: replay the candidate prefix,
    snapshot after the recovery segment, label with the branch outcome), and
  * nominal demo states (positives) for calibration coverage.
Multiple pi0 rollouts per state give soft labels (MC estimate of P(success)).

Data flow: replay prefix in sim -> encode obs with ``DinoEncoder`` -> (tokens, q, task, y).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_branch_labels(dataset_root: str | Path) -> pd.DataFrame:
    """Branch-level labels already produced by M1 search (no extra simulation)."""
    df = pd.read_parquet(Path(dataset_root) / "retryvla" / "branches.parquet")
    df["actions"] = [np.frombuffer(b, dtype=np.float32).reshape(h, 7) for b, h in zip(df["actions"], df["horizon"])]
    return df


def main() -> None:
    raise NotImplementedError("M2 critic label collection is a skeleton; see module doc")


if __name__ == "__main__":
    main()
