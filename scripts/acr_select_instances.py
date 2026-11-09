"""ACR instance selection (deterministic, pre-registered in PLAN "ACR"; train split only).

python scripts/acr_select_instances.py   -> data/acr/instances.json

Source: the 127 paired RGR train instances (runs/_analysis/rgr/rows.parquet, arm `single`).
  Tier B: 3 calibration + 25 evaluation states, drawn (seed 20261109) stratified round-robin over task folders, from
          instances whose object is NOT inside the drawer (an object in a drawer cannot show push / slide effects).
          Calibration states are only used to fix the hidden-magnitude ranges; never reported.
  Tier A: 30 instances `single` left unsolved + 30 it solved (seed 20261109); the oracle cause is labelled at run
          time from the snapshot (retryvla/causal/causes.py). The two tiers may share states (independent questions).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from retryvla import REPO_ROOT

SEED = 20261109


def strat(df: pd.DataFrame, n: int, rng) -> list[str]:
    groups = {k: list(rng.permutation(g.instance_id.tolist())) for k, g in df.groupby("task_folder")}
    keys = sorted(groups)
    out = []
    while len(out) < n and any(groups.values()):
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


def main() -> None:
    d = pd.read_parquet(REPO_ROOT / "runs/_analysis/rgr/rows.parquet")
    s = d[d.arm == "single"].drop_duplicates("instance_id").sort_values("instance_id").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    nb = s[~s.task_folder.str.contains("drawer")]
    tb = strat(nb, 28, rng)
    rng = np.random.default_rng(SEED + 1)
    uns = sorted(s[~s.solved].instance_id)
    sol = sorted(s[s.solved].instance_id)
    ta = sorted(rng.choice(uns, 30, replace=False).tolist()) + sorted(rng.choice(sol, 30, replace=False).tolist())
    out = {"seed": SEED, "source": "runs/_analysis/rgr/rows.parquet (arm single, 127 train instances)",
           "tier_b_calib": tb[:3], "tier_b": tb[3:], "tier_a": ta,
           "single_solved": {i: bool(v) for i, v in zip(s.instance_id, s.solved)},
           "level": {i: str(v) for i, v in zip(s.instance_id, s.level)}}
    p = REPO_ROOT / "data/acr/instances.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(f"tier B calib {len(tb[:3])}, eval {len(tb[3:])}; tier A {len(ta)} -> {p}")
    print(pd.Series([s.set_index('instance_id').task_folder[i][12:40] for i in tb[3:]]).value_counts())


if __name__ == "__main__":
    main()
