#!/usr/bin/env python3
"""Generate configs/dag/b2x.json: P2 controls for B4 (B4-noPi0, Cpi-pool critics + their search arms), lane local:0:h.

python3 scripts/make_dag_b2x.py
Runs in parallel with DAG b2p3 (lane local:0:g) with its own runner (tmux rv_dag_b2x):
  python3 -u scripts/shells/dag_run.py configs/dag/b2x.json --slots local:0:h
Order: critic B4-noPi0 -> critic Cpi-pool -> search shards 0-3 of 8 (configs/exp/b2x_value_search.yaml, the same 122
paired M5b instances as b2_value_search).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NW = "eval.n_workers=8"


def main() -> None:
    T = [{"name": "b2x_crit_success_nopi0", "module": "retryvla.critic.train_value_baseline", "deps": [],
          "args": ["configs/exp/b2_train_success_nopi0.yaml"]},
         {"name": "b2x_crit_cpi_pool", "module": "retryvla.critic.train_value_baseline", "deps": [],
          "args": ["configs/exp/b2_train_cpi_pool.yaml"]}]
    for s in range(4):
        T.append({"name": f"b2x_search_s{s}", "module": "retryvla.search.compare_arms",
                  "deps": ["b2x_crit_success_nopi0", "b2x_crit_cpi_pool"],
                  "args": ["configs/exp/b2x_value_search.yaml", f"m5b.shard={s}", "m5b.num_shards=8", NW]})
    need = {"retryvla.search.compare_arms": 10, "retryvla.critic.train_value_baseline": 8}
    for t in T:  # shared GPU: gate on free memory + retry on CUDA OOM
        t["args"] = ["--min_free_gb", str(need[t["module"]]), t["module"], *t["args"]]
        t["module"] = "retryvla.retry_oom"
    out = REPO / "configs" / "dag" / "b2x.json"
    json.dump({"name": "b2x", "tasks": T}, open(out, "w"), indent=1)
    print(f"{out}: {len(T)} tasks")


if __name__ == "__main__":
    main()
