"""2610.05151 evaluation protocol on our LIBERO-Goal split (reverse-direction track).

python -m retryvla.eval.eval_goal --ckpt <pi0 ckpt dir> --split val [--workers 10] [--out runs/<id>]

Per scene (task + demo init_state): restore the init state, 10 no-op settle steps (openpi LIBERO example; D3), then
the policy alone for <= 300 env actions, chunk 50, execute 5 then replan, policy randomness reset to seed 7 for every
trial, success = native BDDL check. 128 px renders (their training resolution; the policy pads + resizes to 224).
Writes episodes.jsonl + summary.json (successes / 100, per task).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from retryvla import REPO_ROOT
from retryvla.envs.vec import SimPool
from retryvla.eval.rollout import Job, run_jobs


def scene_jobs(split_name: str, part: str, max_steps: int = 300, seed: int = 7) -> list[Job]:
    from libero.libero import get_libero_path
    import os

    sp = json.loads((REPO_ROOT / f"data/splits/{split_name}.json").read_text())
    jobs = []
    for task, v in sp["tasks"].items():
        bddl = os.path.join(get_libero_path("bddl_files"), "libero_goal", f"{task}.bddl")
        with h5py.File(v["hdf5"]) as h:
            for d in v[part]:
                dd = h[f"data/demo_{d}"]
                init = np.asarray(dd.attrs["init_state"]) if "init_state" in dd.attrs else np.asarray(dd["states"][0])
                jobs.append(Job("load_init_state", dict(bddl=bddl, state=init, num_steps_wait=10, seed=0,
                                                        return_snapshot=False),
                                task=v["language"], max_steps=max_steps, policy_seed=seed,
                                meta={"task": task, "demo": int(d), "scene": f"{task}__demo_{d}", "part": part}))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--split_name", default="goal_rsi_v1")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--max_jobs", type=int, default=0)
    ap.add_argument("--save_actions", action="store_true",
                    help="store every executed action per episode (actions.npz) for the offline failure-event detector")
    ap.add_argument("--bot", default=None, help="Back on Track cfg as JSON (retryvla.policy.bot.BoTCfg); test-time arm")
    a = ap.parse_args()
    from retryvla.policy.pi0_wrapper import Pi0Wrapper

    out = Path(a.out) if a.out else Path(a.ckpt) / f"eval_{a.split}"
    out.mkdir(parents=True, exist_ok=True)
    jobs = scene_jobs(a.split_name, a.split, a.max_steps)
    if a.max_jobs:
        jobs = jobs[: a.max_jobs]
    if a.save_actions:
        for j in jobs:
            j.record, j.record_cfg = "all", {"stride": 100000, "img_until": 0}
    policy = Pi0Wrapper(a.ckpt, n_action_steps=5)
    bench = None
    t0 = time.time()
    with SimPool(a.workers, 128) as pool:
        if a.bot:
            import copy

            from retryvla.policy.bot import make_wrapper

            policy = make_wrapper(policy, json.loads(a.bot))
            j0 = copy.copy(jobs[0]); j0.max_steps = 0
            o0 = run_jobs(pool, None, [j0])[0].load_out["obs0"]
            bench = policy.bench(o0, jobs[0].task)
            print("BoT bench (batch 1):", json.dumps(bench), flush=True)
            policy.eps.clear()
            t0 = time.time()
        stats = {}

        def _cb(r):
            if hasattr(policy, "pop_episode_stats"):
                stats[r.job_id] = policy.pop_episode_stats(id(jobs[r.job_id]))

        res = run_jobs(pool, policy, jobs, desc=f"goal_{a.split}", on_result=_cb)
    rows = [{**r.meta, "success": bool(r.success), "steps": int(r.steps), **stats.get(r.job_id, {})} for r in res]
    if a.save_actions:
        try:
            np.savez_compressed(out / "actions.npz", **{r.meta["scene"]: r.record["action"] for r in res if r.record})
        except OSError as e:
            print(f"actions.npz write failed ({e})", flush=True)
    for k in range(12):   # /pfss quota (EDQUOT): wait it out instead of losing the 100 rollouts
        try:
            with open(out / "episodes.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            break
        except OSError as e:
            print(f"episodes write failed ({e}), retry {k + 1}/12 in 5 min", flush=True)
            time.sleep(300)
    by = {}
    for r in rows:
        by.setdefault(r["task"], []).append(r["success"])
    summ = {"ckpt": a.ckpt, "split": a.split, "n": len(rows), "successes": int(sum(r["success"] for r in rows)),
            "sr": float(np.mean([r["success"] for r in rows])), "by_task": {k: int(sum(v)) for k, v in by.items()},
            "wall_s": time.time() - t0}
    if a.bot:
        summ["bot"] = json.loads(a.bot)
        summ["bot_bench"] = bench
        if hasattr(policy, "stats"):
            summ["bot_stats"] = policy.stats()
    for k in range(12):
        try:
            (out / "summary.json").write_text(json.dumps(summ, indent=1))
            break
        except OSError as e:
            print(f"summary write failed ({e}), retry {k + 1}/12 in 5 min", flush=True)
            time.sleep(300)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
