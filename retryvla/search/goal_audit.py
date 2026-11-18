"""2610.05151 reverse direction, RetryVLA stage 1: audit M0 on the 300 LIBERO-Goal TRAIN scenes.

python -m retryvla.search.goal_audit --ckpt <M0> --seeds 7 8 [--workers 16]

One rollout per (train scene, policy seed) with their protocol (10 settle steps, <= 300 actions, replan 5). Every
executed action is kept, so any intermediate state is reproducible bit-exactly: restore the init state, then replay
the action prefix open-loop (deterministic MuJoCo; this is also how the paper restores entries).
Seed 7 = search instances (= their audit seed); seed 8 = WM / critic training data only (disjoint rollouts; analogue of
the Spatial M5b split "WM / critic trials 0-1, search trials 2-9").
Entries (pre-registered 2026-11-18, before any audit result): for every FAILED rollout of length T, the states at
t in {T/4, T/2, 3T/4}, rounded down to a multiple of 5 (no auditor; analogue of their Coverage mode: early / mid / late).
Output: data/goal_rsi/audit_<tag>/{rollouts.jsonl, actions_s<seed>.npz, entries_s<seed>.jsonl, summary.json}.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from retryvla import REPO_ROOT
from retryvla.envs.vec import SimPool
from retryvla.eval.eval_goal import scene_jobs
from retryvla.eval.rollout import run_jobs


def entries_for(T: int) -> list[int]:
    return sorted({max(5, (T * k // 4) // 5 * 5) for k in (1, 2, 3)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 8])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="data/goal_rsi/audit_m0")
    a = ap.parse_args()
    from retryvla.policy.pi0_wrapper import Pi0Wrapper

    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    policy = Pi0Wrapper(a.ckpt, n_action_steps=5)
    summ = {"ckpt": a.ckpt, "seeds": {}}
    with SimPool(a.workers, 128) as pool:
        for seed in a.seeds:
            if (out / f"entries_s{seed}.jsonl").exists():
                continue
            jobs = scene_jobs("goal_rsi_v1", "train", 300, seed)
            for j in jobs:
                j.record = "all"
                j.record_cfg = {"stride": 100000, "img_until": 0}   # actions / states only
            t0 = time.time()
            res = run_jobs(pool, policy, jobs, desc=f"audit_s{seed}")
            acts, rows, ents = {}, [], []
            for r in res:
                m = r.meta
                key = m["scene"]
                acts[key] = r.record["action"]
                rows.append({"scene": key, "task": m["task"], "demo": m["demo"], "seed": seed,
                             "success": bool(r.success), "T": int(r.steps)})
                if not r.success:
                    for t in entries_for(int(r.steps)):
                        ents.append({"entry_id": f"{key}__s{seed}__t{t:03d}", "scene": key, "task": m["task"],
                                     "demo": m["demo"], "seed": seed, "t": t, "T": int(r.steps)})
            np.savez_compressed(out / f"actions_s{seed}.npz", **acts)
            with open(out / "rollouts.jsonl", "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            with open(out / f"entries_s{seed}.jsonl", "w") as f:
                for e in ents:
                    f.write(json.dumps(e) + "\n")
            sr = float(np.mean([r["success"] for r in rows]))
            by = {}
            for r in rows:
                by.setdefault(r["task"], []).append(r["success"])
            summ["seeds"][seed] = {"n": len(rows), "successes": int(sum(r["success"] for r in rows)), "sr": sr,
                                   "failures": int(sum(not r["success"] for r in rows)), "entries": len(ents),
                                   "by_task_fail": {k: int(len(v) - sum(v)) for k, v in by.items()},
                                   "wall_s": time.time() - t0}
            print(json.dumps({seed: summ["seeds"][seed]}), flush=True)
    old = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {"seeds": {}}
    old["seeds"].update({str(k): v for k, v in summ["seeds"].items()})
    old["ckpt"] = a.ckpt
    (out / "summary.json").write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
