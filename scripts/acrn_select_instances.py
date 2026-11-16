"""ACR-N instance selection (deterministic, pre-registered in PLAN "ACR-N"; TRAIN split only).

python scripts/acrn_select_instances.py   -> data/acr/instances_n.json

Source: the 127 paired RGR train instances (runs/_analysis/rgr/rows.parquet, arm `single`).
  U (primary): ALL instances `single` left unsolved (73), priority order L3 -> L2 -> L1, random within level.
  S (control): 24 `single`-solved instances, stratified round-robin over task folders.
  Collection order interleaves 3 U : 1 S, so a partial run keeps the priority and the mix.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from retryvla import REPO_ROOT

SEED = 20261116


def main() -> None:
    d = pd.read_parquet(REPO_ROOT / "runs/_analysis/rgr/rows.parquet")
    s = d[d.arm == "single"].drop_duplicates("instance_id").sort_values("instance_id").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    U = []
    for lv in ("L3", "L2", "L1"):
        ids = sorted(s[(~s.solved) & (s.level == lv)].instance_id)
        U += list(rng.permutation(ids))
    U += [i for i in sorted(s[~s.solved].instance_id) if i not in U]
    sol = s[s.solved]
    groups = {k: list(rng.permutation(sorted(g.instance_id))) for k, g in sol.groupby("task_folder")}
    S = []
    while len(S) < 24 and any(groups.values()):
        for k in sorted(groups):
            if groups[k] and len(S) < 24:
                S.append(groups[k].pop())
    order, su = [], list(S)
    for k, u in enumerate(U):
        order.append(u)
        if k % 3 == 2 and su:
            order.append(su.pop(0))
    order += su
    out = {"seed": SEED, "source": "runs/_analysis/rgr/rows.parquet (arm single, 127 train instances)",
           "U": U, "S": S, "order": [str(x) for x in order],
           "set": {**{i: "U" for i in U}, **{i: "S" for i in S}},
           "single_solved": {i: bool(v) for i, v in zip(s.instance_id, s.solved)},
           "single_hits": {i: int(v) for i, v in zip(s.instance_id, s.hits)},
           "level": {i: str(v) for i, v in zip(s.instance_id, s.level)}}
    p = REPO_ROOT / "data/acr/instances_n.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    print(f"U {len(U)} (levels {s[~s.solved].level.value_counts().to_dict()}), S {len(S)} -> {p}")


if __name__ == "__main__":
    main()
