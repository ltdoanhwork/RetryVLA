#!/usr/bin/env python3
"""B3.3: simulator-transition budget of the RetryVLA PoC (pi1 = M5c wm_diverse round-robin), from logs.

python scripts/b33_budget.py [--out runs/_analysis/b33/budget.json]

Counts env transitions (MuJoCo steps executed by the policy / bridge, excluding the 20 gripper-settle steps of
load_scene, which every method pays identically). Per logged rollout: steps = success_step if success else the
rollout cap (H + continuation = 350 for every logged source here).
  - wm_data   : data/wm/collect_v1 (144 train instances x 32 candidates): trains WM v1 + critic v1.
  - critic_v2 : data/wm/verif_r1 (M5 round-1 verifications on 108 pilot instances): extra critic-v2 training data.
  - verify    : data/wm/verif_m5b, arm wm_diverse only (256 M5b failures x 16): the data pi1 is distilled from.
  - precheck  : one pi0 rollout per selected M5b instance (288, cap 520) to find the failures. Not logged per step
                -> estimated (failures 520 steps; pi0 successes: mean pi0 success step on test, M0 log). Shared by
                RetryVLA and the RL baseline (both start from the same failure states), reported, not matched.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]


def steps_of(d: dict, cap: int, mask=None) -> tuple[int, int]:
    s = d["success"].bool().numpy()
    st = d["success_step"].numpy()
    n = np.where(s, st, cap)
    if mask is not None:
        n = n[mask]
    return int(n.sum()), int(len(n))


def root_steps(pattern: str, cap: int, arm: str | None = None) -> dict:
    tot, n, ninst = 0, 0, 0
    for f in sorted(glob.glob(str(REPO / pattern))):
        d = torch.load(f, weights_only=False, map_location="cpu")
        mask = None
        if arm is not None:
            src = d.get("source")
            if isinstance(src, list) and d.get("arm") == "m5b_all":
                mask = np.array([x == arm for x in src])
            elif d.get("arm") not in (arm, None):
                continue
        t, k = steps_of(d, cap, mask)
        tot, n, ninst = tot + t, n + k, ninst + 1
    return {"transitions": tot, "rollouts": n, "instances": ninst, "per_instance": tot / max(1, ninst)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/_analysis/b33/budget.json")
    a = ap.parse_args()
    out = {}
    out["wm_data"] = root_steps("data/wm/collect_v1/*.pt", 350)
    out["critic_v2"] = root_steps("data/wm/verif_r1/*.pt", 350)
    out["verify"] = root_steps("data/wm/verif_m5b/*.pt", 350, arm="wm_diverse")
    # cross-check with per_instance.jsonl where sim_steps_verify exists (newer arms only)
    rows = []
    for f in sorted(glob.glob(str(REPO / "runs/*_M5_m5b_compare_arms_s*/per_instance.jsonl"))):
        rows.append(pd.read_json(f, lines=True))
    pi = pd.concat(rows) if rows else pd.DataFrame()
    out["verify"]["per_instance_jsonl_has_sim_steps"] = bool(len(pi) and "sim_steps_verify" in pi.columns)
    pre = []
    for f in sorted(glob.glob(str(REPO / "runs/*_M5_m5b_compare_arms_s*/precheck.csv"))):
        pre.append(pd.read_csv(f))
    pre = pd.concat(pre).drop_duplicates("instance_id") if pre else pd.DataFrame()
    n_sel, n_fail = len(pre), int((~pre.pi0_success.astype(bool)).sum()) if len(pre) else 0
    ep = pd.read_json(sorted(glob.glob(str(REPO / "runs/*_M0_recover_bddl/episodes.jsonl")))[0], lines=True)
    ok = ep[ep.success.astype(bool)]
    mean_succ = float(ok.steps.mean()) if len(ok) else 260.0
    out["precheck_est"] = {"transitions": int(n_fail * 520 + (n_sel - n_fail) * mean_succ), "rollouts": n_sel,
                           "instances": n_fail, "note": f"estimated: {n_fail} failures x 520 + {n_sel - n_fail} "
                                                         f"successes x {mean_succ:.0f} (M0 mean success step)"}
    poc = out["wm_data"]["transitions"] + out["critic_v2"]["transitions"] + out["verify"]["transitions"]
    out["poc_total_matched"] = poc
    out["poc_per_failure"] = poc / max(1, out["verify"]["instances"])
    out["n_failures"] = out["verify"]["instances"]
    p = REPO / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
