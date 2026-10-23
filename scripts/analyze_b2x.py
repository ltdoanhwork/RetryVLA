#!/usr/bin/env python
"""P2 controls for B4: is B4's search win 'more continuation labels' or 'policy-agnostic value'?

python scripts/analyze_b2x.py [--out runs/_analysis/b2x]

Arms (all 16 verified sims / instance, same pi0 failures, same pool / WM v1 / continuation seeds):
  M5b : random, wm_diverse (= C^pi v2)
  B2  : wm_progress (B3), wm_success (B4)                      runs/*_B2_value_search_s*
  B2x : wm_success_nopi0 (B4-noPi0), wm_cpi_pool (Cpi-pool)    runs/*_B2_valuex_search_s*
  M9  : wm_goal
Rows unioned over run dirs / resumed parts (dedup instance_id x arm, last wins). Pre-registered paired tests
(unit = failure instance, exact McNemar on `solved`, instance-bootstrap 95% CI, sign-flip on hits):
  B4 vs C^pi v2, B4-noPi0 vs C^pi v2, Cpi-pool vs C^pi v2, B4 vs Cpi-pool  (+ B4 vs B4-noPi0, B3 vs C^pi v2).
Each pair is tested on the instances present for BOTH arms ("pair n") and the table also reports the all-arm set.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_m5b import boot_ci, mcnemar, signflip  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SRC = {"M5b": ("*_M5_m5b_compare_arms_s[0-3]of8", ["random", "wm_diverse"]),
       "M9": ("*_M9_mech_search_s[0-3]of8", ["wm_goal"]),
       "B2": ("*_B2_value_search_s[0-3]of8", ["wm_progress", "wm_success"]),
       "B2x": ("*_B2_valuex_search_s[0-3]of8", ["wm_success_nopi0", "wm_cpi_pool"])}
NAME = {"random": "random", "wm_diverse": "C^pi v2", "wm_goal": "goal", "wm_progress": "B3 progress",
        "wm_success": "B4 success", "wm_success_nopi0": "B4-noPi0", "wm_cpi_pool": "Cpi-pool"}
PAIRS = [("wm_success", "wm_diverse"), ("wm_success_nopi0", "wm_diverse"), ("wm_cpi_pool", "wm_diverse"),
         ("wm_success", "wm_cpi_pool"), ("wm_success", "wm_success_nopi0"), ("wm_progress", "wm_diverse"),
         # secondary (not pre-registered): every value arm vs random and vs M9 goal
         *[(x, y) for y in ("random", "wm_goal")
           for x in ("wm_progress", "wm_success", "wm_success_nopi0", "wm_cpi_pool", "wm_diverse")]]


def load() -> pd.DataFrame:
    out = []
    for src, (pat, arms) in SRC.items():
        for f in sorted(glob.glob(str(REPO / "runs" / pat / "per_instance.jsonl"))):
            shard = int(re.search(r"_s(\d+)of8", f).group(1))
            for l in open(f):
                r = json.loads(l)
                if r["arm"] in arms:
                    out.append({**r, "src": src, "shard": shard})
    return pd.DataFrame(out).drop_duplicates(["instance_id", "arm"], keep="last")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/_analysis/b2x")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    d = load()
    arms = [x for x in NAME if x in set(d.arm)]
    inst = {x: set(d[d.arm == x].instance_id) for x in arms}
    rows = []
    for x in arms:
        g = d[d.arm == x]
        mine = sorted(inst[x] & inst.get("wm_diverse", inst[x]))
        gg = g.set_index("instance_id").loc[mine]
        ref = d[d.arm == "wm_diverse"].set_index("instance_id").loc[mine] if "wm_diverse" in inst else None
        cpi = pd.to_numeric(g.get("pred_cpi_mean"), errors="coerce") if "pred_cpi_mean" in g else None
        rows.append({"arm": x, "name": NAME[x], "n_done": len(g), "per_shard": g.groupby("shard").size().to_dict(),
                     "solved": float(g.solved.mean()), "hits_per_sim": float(g.hits.sum() / g.budget.sum()),
                     "n_paired_cpi": len(mine), "solved_on_paired": float(gg.solved.mean()) if len(mine) else np.nan,
                     "cpi_on_paired": float(ref.solved.mean()) if ref is not None and len(mine) else np.nan,
                     "pred_own_mean": float(g.pred_mean.mean()),
                     "pred_cpi_mean": float(cpi.mean()) if cpi is not None else np.nan})
    T = pd.DataFrame(rows)
    T.to_csv(out / "per_arm.csv", index=False)
    common = sorted(set.intersection(*inst.values())) if inst else []
    S = d.pivot(index="instance_id", columns="arm", values="solved")
    H = d.pivot(index="instance_id", columns="arm", values="hits")
    tests = []
    for x, y in PAIRS:
        if x not in inst or y not in inst:
            continue
        for scope, ids in (("pair", sorted(inst[x] & inst[y])), ("all_arms", common)):
            if not ids:
                continue
            sx, sy = S.loc[ids, x].astype(bool).values, S.loc[ids, y].astype(bool).values
            n01, n10, p = mcnemar(sy, sx)
            diff = sx.astype(float) - sy.astype(float)
            lo, hi = boot_ci(diff)
            hd = (H.loc[ids, x] - H.loc[ids, y]).astype(float).values
            tests.append({"arm": NAME[x], "vs": NAME[y], "scope": scope, "n": len(ids), "solved_arm": sx.mean(),
                          "solved_vs": sy.mean(), "d_solved": diff.mean(), "ci": f"[{lo:+.3f}, {hi:+.3f}]",
                          "only_arm": n01, "only_vs": n10, "mcnemar_p": p, "d_hits": hd.mean(),
                          "signflip_p_hits": signflip(hd)})
    P = pd.DataFrame(tests)
    P.to_csv(out / "paired_tests.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.drop(columns=["per_shard"]).round(3).to_string(index=False))
    print(T[["arm", "per_shard"]].to_string(index=False))
    print(f"all-arm paired set: {len(common)} instances")
    if len(P):
        print(P.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
