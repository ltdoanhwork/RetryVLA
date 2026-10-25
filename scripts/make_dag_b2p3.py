#!/usr/bin/env python3
"""Generate configs/dag/b2p3.json (P2 value-critic baselines + P3 recovery-nominal Pareto), lane local:0:g.

python3 scripts/make_dag_b2p3.py [--seeds 0 1 2]   # seeds 1-2 only APPEND P3 / P2-SFT tasks (seed 0 tasks unchanged)
Order (one lane, tasks start in list order when their deps are done; an eval directly follows its SFT):
  critics -> P2 search shards 0-3 -> P2 SFT progress (+ RSR) -> P3 L_preserve, r0.1, r0.7, r0.3 (+ RSR + nominal)
  -> P2 SFT success (+ RSR) -> extra seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NW = "eval.n_workers=8"
WR = "eval.worker_retries=2"  # survive a MuJoCo sim-worker segfault (P3 r0.3 nominal died at 316/500, 2026-10-24)
COMP = "data/recovery/m5b/s[0-3]/p*/wm_diverse"


def rec(name, dep, milestone, tag):
    return {"name": name, "module": "retryvla.eval.eval_recover", "deps": [dep],
            "args": ["configs/exp/m0_baseline_spatial.yaml", f"milestone={milestone}", f"recover_tag={tag}_recover_test",
                     f"policy.path=@glob:runs/*_{milestone}_{tag}/checkpoints/006000/pretrained_model",
                     "recover_eval.splits=[test]", NW, "eval.save_videos=none", WR]}


def nom(name, dep, milestone, tag):
    return {"name": name, "module": "retryvla.eval.eval_nominal", "deps": [dep],
            "args": ["configs/exp/m0_baseline_spatial.yaml", f"milestone={milestone}", f"nominal_tag={tag}_nominal",
                     f"policy.path=@glob:runs/*_{milestone}_{tag}/checkpoints/006000/pretrained_model", NW,
                     "eval.save_videos=none", WR]}


def p2_sft(arm, seed, deps):
    tag = f"sft_{arm}" + (f"_seed{seed}" if seed else "")
    n = f"b2_sft_{arm}_s{seed}"
    # count matched to the competence data AND capped at 66 = the M9 comp / goal SFT count, so the result pairs with
    # the existing 3-seed comp (0.417) / goal (0.187) runs; if the progress bank has < 66, append a comp run at that n.
    t = {"name": n, "module": "retryvla.train.sft", "deps": deps,
         "args": [f"configs/exp/b2_sft_{arm}.yaml", f"tag={tag}", f"train.seed={seed}",
                  f"train.max_recovery_trajectories=@ntrajmin:66:data/recovery/b2_value/s*/p*/wm_{arm},{COMP}"]}
    return [t, rec(f"b2_rec_{arm}_s{seed}", n, "B2", tag)]


def ntraj(pat):
    import glob
    return sum(1 for f in glob.glob(str(REPO / pat / "retryvla" / "trajectories.jsonl")) for _ in open(f))


def comp_matched(arm, seed):
    """Count-matched competence control for a P2 SFT whose bank gives fewer than the M9 comp count (66): the M9 comp
    recipe exactly (configs/exp/m9_sft_comp.yaml: init pi1-nominal, round-robin, M5b wm_diverse s0-3 pool) with
    max_recovery_trajectories = the SAME @ntrajmin expression as the arm's SFT, so both resolve to the same n."""
    bank = f"data/recovery/b2_value/s*/p*/wm_{arm}"
    n = min(66, ntraj(bank), ntraj(COMP))
    if n == 66:  # pairs with the existing 3-seed M9 comp runs (n = 66), no extra control needed
        return []
    tag = f"sft_comp_n{n}" + (f"_seed{seed}" if seed else "")
    name = f"b2_sft_comp_{arm}n_s{seed}"
    t = {"name": name, "module": "retryvla.train.sft", "deps": [f"b2_sft_{arm}_s{seed}"],
         "args": ["configs/exp/m9_sft_comp.yaml", "milestone=B2", f"tag={tag}", f"train.seed={seed}",
                  "train.recovery_datasets=[" + ",".join(f"data/recovery/m5b/s{s}/p*/wm_diverse" for s in range(4)) + "]",
                  f"train.max_recovery_trajectories=@ntrajmin:66:{bank},{COMP}"]}
    return [t, rec(f"b2_rec_comp_{arm}n_s{seed}", name, "B2", tag)]


def p3(point, seed):
    if point == "pres":
        cfg, extra = "configs/exp/p3_sft_preserve.yaml", []
    else:
        cfg, extra = "configs/exp/p3_sft_ratio.yaml", [f"train.recovery_ratio={point / 10:.1f}"]
    tag = f"p3_{'preserve' if point == 'pres' else f'r0{point}'}_seed{seed}"
    n = f"{tag}_sft"
    t = {"name": n, "module": "retryvla.train.sft", "deps": [],
         "args": [cfg, f"tag={tag}", f"train.seed={seed}", *extra]}
    return [t, rec(f"{tag}_rec", n, "P3", tag), nom(f"{tag}_nom", n, "P3", tag)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    T = [{"name": "b2_crit_progress", "module": "retryvla.critic.train_value_baseline", "deps": [],
          "args": ["configs/exp/b2_train_progress.yaml"]},
         {"name": "b2_crit_success", "module": "retryvla.critic.train_value_baseline", "deps": [],
          "args": ["configs/exp/b2_train_success.yaml"]}]
    search = []
    for s in range(4):
        search.append(f"b2_search_s{s}")
        T.append({"name": f"b2_search_s{s}", "module": "retryvla.search.compare_arms",
                  "deps": ["b2_crit_progress", "b2_crit_success"],
                  "args": ["configs/exp/b2_value_search.yaml", f"m5b.shard={s}", "m5b.num_shards=8", NW]})
    T += p2_sft("progress", 0, search)
    T += comp_matched("progress", 0)  # progress bank = 60 < 66 -> comp at n = 60 (added 2026-10-23)
    for p in ("pres", 1, 7, 3):
        T += p3(p, 0)
    T += p2_sft("success", 0, search)
    # nominal of the success SFT (progress / comp n60 nominals ran in configs/dag/nom_fill.json); added 2026-10-25
    T.append(nom("b2_nom_success_s0", "b2_sft_success_s0", "B2", "sft_success"))
    T += comp_matched("success", 0)  # success bank final (224) -> n = 66 = M9 comp, nothing added
    for seed in a.seeds:
        if seed == 0:
            continue
        for p in ("pres", 1, 7, 3):
            T += p3(p, seed)
        T += p2_sft("progress", seed, search)
    # shared GPU: gate on free memory + retry on CUDA OOM (retryvla/retry_oom.py) so one OOM does not stop the DAG
    need = {"retryvla.train.sft": 32, "retryvla.search.compare_arms": 10, "retryvla.eval.eval_recover": 10,
            "retryvla.eval.eval_nominal": 10, "retryvla.critic.train_value_baseline": 8}
    for t in T:
        g = need[t["module"]] + (6 if "p3_sft_preserve" in t["args"][0] else 0)
        t["args"] = ["--min_free_gb", str(g), t["module"], *t["args"]]
        t["module"] = "retryvla.retry_oom"
    out = REPO / "configs" / "dag" / "b2p3.json"
    json.dump({"name": "b2p3", "tasks": T}, open(out, "w"), indent=1)
    print(f"{out}: {len(T)} tasks")


if __name__ == "__main__":
    main()
