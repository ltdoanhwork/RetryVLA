"""Own-failure recovery eval (RSR-style) on LIBERO-Goal audit entries (2610.05151 reverse direction, Goal campaign).

python -m retryvla.eval.eval_goal_entries --ckpt <ckpt> --entries data/goal_rsi/audit_m0_s8/entries_s8.jsonl \
    --audit_dir data/goal_rsi/audit_m0_s8 [--trials 1 --workers 8 --tag s8fixed]

Per entry (failed M0 rollout x cut t): restore the init state (their protocol, 10 settle steps), replay M0's recorded
actions[:t] open-loop (bit-exact = the failure state), then the evaluated policy acts alone for <= 300 steps
(`--cont`); success = BDDL goal. Policy seeds hashed from (entry, trial). Output <ckpt>/eval_entries_<tag>/
{episodes.jsonl, summary.json} (RSR = success rate over entries x trials, per task, per cut rule).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from pathlib import Path

import h5py
import numpy as np

from retryvla import REPO_ROOT
from retryvla.envs.vec import SimPool
from retryvla.eval.rollout import Job, run_jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--entries", required=True)
    ap.add_argument("--audit_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--cont", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from libero.libero import get_libero_path

    from retryvla.policy.pi0_wrapper import Pi0Wrapper

    out = Path(a.ckpt) / f"eval_entries_{a.tag}"
    if (out / "summary.json").exists():
        print("exists", out)
        return
    out.mkdir(parents=True, exist_ok=True)
    sp = json.loads((REPO_ROOT / "data/splits/goal_rsi_v1.json").read_text())
    ents = [json.loads(l) for l in open(REPO_ROOT / a.entries) if l.strip()]
    acts = {}
    jobs = []
    for e in ents:
        v = sp["tasks"][e["task"]]
        assert e["demo"] in v["train"]
        if e["seed"] not in acts:
            acts[e["seed"]] = np.load(REPO_ROOT / a.audit_dir / f"actions_s{e['seed']}.npz")
        with h5py.File(v["hdf5"]) as h:
            dd = h[f"data/demo_{e['demo']}"]
            init = np.asarray(dd.attrs["init_state"]) if "init_state" in dd.attrs else np.asarray(dd["states"][0])
        pre = acts[e["seed"]][e["scene"]][: e["t"]]
        bddl = os.path.join(get_libero_path("bddl_files"), "libero_goal", f"{e['task']}.bddl")
        for k in range(a.trials):
            jobs.append(Job("load_init_state", dict(bddl=bddl, state=init, num_steps_wait=10, seed=0, return_snapshot=False),
                            task=v["language"], max_steps=len(pre) + a.cont, prefix=pre,
                            policy_seed=zlib.crc32(f"{e['entry_id']}|{k}|eval".encode()) % (2**31 - 1),
                            meta={"entry_id": e["entry_id"], "task": e["task"], "rule": e.get("rule", "fixed"),
                                  "trial": k, "t": e["t"]}))
    policy = Pi0Wrapper(a.ckpt, n_action_steps=5)
    t0 = time.time()
    with SimPool(a.workers, 128) as pool:
        res = run_jobs(pool, policy, jobs, desc=f"entries_{a.tag}")
    rows = [{**r.meta, "success": bool(r.success), "steps": int(r.steps), "prefix_success": bool(r.prefix_success)}
            for r in res]
    by_task, by_rule = {}, {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r["success"])
        by_rule.setdefault(r["rule"], []).append(r["success"])
    summ = {"ckpt": a.ckpt, "entries": a.entries, "n": len(rows), "rsr": float(np.mean([r["success"] for r in rows])),
            "successes": int(sum(r["success"] for r in rows)),
            "by_task": {k: float(np.mean(v)) for k, v in by_task.items()},
            "by_rule": {k: float(np.mean(v)) for k, v in by_rule.items()}, "wall_s": time.time() - t0}
    for k in range(12):
        try:
            with open(out / "episodes.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            (out / "summary.json").write_text(json.dumps(summ, indent=1))
            break
        except OSError as e:
            print(f"write failed ({e}), retry in 5 min", flush=True)
            time.sleep(300)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
