"""Backtrack-distill data: run the Backtrack baseline (pi0 + retreat / retrace, ``policy/backtrack.py``) from the SAME
256 pi0 train-failure instances as pi1's data (M5b: train scenes x object-noise trials 2..9, pi0 failed the precheck),
``attempts`` continuation seeds per instance, and write every successful episode (backtrack segments = segment 0,
pi0 control = segment 1) as a LeRobotDataset in the standard ``RecoveryWriter`` format, so the standard SFT recipe
(``configs/exp/bt_sft_distill.yaml``: pi1-nominal init, 6k steps, 50 % recovery, 208 round-robin trajectories) can
distill it with no change to ``train/sft.py``.

python -m retryvla.baselines.backtrack_datagen configs/exp/bt_datagen.yaml [btd.shard=i btd.num_shards=N]

Output: ``<out_root>/s<shard>/p<k>/bt`` (+ ``CLOSED`` when the part finished), ``attempts.jsonl`` (one row per
episode) in the run dir. Resumable: instances listed in a CLOSED part's ``done.jsonl`` are skipped.
"""

from __future__ import annotations

import glob
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from retryvla import REPO_ROOT, tracking
from retryvla.config import cli_config, to_dict
from retryvla.data.recovery_writer import RecoveryWriter
from retryvla.envs.splits import assert_allowed, load_split, scenes_for
from retryvla.envs.vec import SimPool
from retryvla.eval.eval_recover import instance_jobs, noise_seed
from retryvla.eval.metrics import write_jsonl
from retryvla.eval.rollout import Job, run_jobs_resilient
from retryvla.policy import SmolVLAWrapper
from retryvla.policy.backtrack import BacktrackPolicy
from retryvla.utils import setup_logging

log = logging.getLogger("backtrack_datagen")


def iid(meta: dict) -> str:  # same instance id as search/compare_arms.iid
    return f"{meta['scene']}__{meta['gripper_init']}{meta['trial']}"


def pi0_failures(globs) -> set[str]:
    out = set()
    for pat in globs:
        for f in sorted(glob.glob(str(REPO_ROOT / pat))):
            d = pd.read_csv(f)
            out |= set(d.loc[~d.pi0_success.astype(bool), "instance_id"])
    return out


def main() -> None:
    setup_logging()
    cfg = cli_config()
    bd = cfg.btd
    split = load_split(REPO_ROOT / cfg.split.file)
    scenes = scenes_for(split, "train", REPO_ROOT / cfg.recover.scene_root)
    assert_allowed([s.tag for s in scenes], split)
    fails = pi0_failures(list(bd.precheck_from))
    seed = int(cfg.eval.seed)
    base = [j for j in instance_jobs(scenes, cfg.recover, seed, cfg.recover.instruction, part="train")
            if iid(j.meta) in fails]
    base = sorted(base, key=lambda j: iid(j.meta))[int(bd.shard):: int(bd.num_shards)]
    if bd.get("max_instances"):  # smoke tests
        base = base[: int(bd.max_instances)]
    out_root = REPO_ROOT / bd.out_root / f"s{int(bd.shard)}"
    done = set()
    for p in sorted(out_root.glob("p*")):
        if (p / "CLOSED").exists() and (p / "done.jsonl").exists():
            done |= {json.loads(l)["instance_id"] for l in open(p / "done.jsonl")}
    base = [j for j in base if iid(j.meta) not in done]
    k = 0
    while (out_root / f"p{k}").exists():
        k += 1
    part = out_root / f"p{k}"
    part.mkdir(parents=True, exist_ok=True)
    jobs = []
    for j in base:
        for a in range(int(bd.attempts)):
            jj = Job(**{**j.__dict__})
            # attempt 0 = the eval-protocol policy seed of this (scene, trial); others = fresh continuation seeds
            jj.policy_seed = j.policy_seed if a == 0 else noise_seed(seed + 7919 * a, j.meta["scene"], int(j.meta["trial"]))
            jj.record = "success"
            jj.meta = {**j.meta, "attempt": a}
            jobs.append(jj)
    log.info("%d pi0-failure instances (shard %s/%s, %d already done) x %d attempts = %d episodes -> %s",
             len(base), bd.shard, bd.num_shards, len(done), int(bd.attempts), len(jobs), part)

    with tracking.start(cfg) as run:
        run.summary.update({"n_instances": len(base), "attempts": int(bd.attempts), "part": str(part),
                            "backtrack": to_dict(cfg.backtrack), "config": to_dict(cfg)})
        policy = BacktrackPolicy(SmolVLAWrapper(cfg.policy.path, cfg.policy.device, cfg.policy.n_action_steps),
                                 to_dict(cfg.backtrack))
        n_act = policy.n_action_steps
        w = RecoveryWriter(part / "bt", overwrite=True, write_retries=int(bd.get("write_retries", 3)))
        rows, kept, t0 = [], Counter(), time.time()
        doneset: Counter = Counter()
        n_att = Counter(iid(j.meta) for j in jobs)
        done_f = open(part / "done.jsonl", "a")

        def on_result(r):
            m = r.meta
            i = iid(m)
            ev = policy.events(jobs[r.job_id].policy_seed)
            policy.drop(jobs[r.job_id].policy_seed)
            row = {**m, "instance_id": i, "success": r.success, "steps": r.steps, "n_bt": ev.get("n_bt", 0),
                   "triggers": ev.get("triggers", [])}
            if r.success and r.record is not None and kept[i] < int(bd.max_trajs_per_instance):
                rec = r.record
                seg = np.ones(len(rec["action"]), np.int64)
                for c in ev.get("bt_calls", []):
                    seg[c * n_act:(c + 1) * n_act] = 0
                rec["segment"] = seg
                w.add_instance(i, {"scene": m["scene"], "task": jobs[r.job_id].task, "arm": "backtrack"})
                w.add_trajectory(rec, jobs[r.job_id].task, i, {
                    "score": 1.0 - 0.001 * r.steps, "arm": "backtrack", "scene": m["scene"], "attempt": m["attempt"],
                    "n_bt": row["n_bt"], "success_step": int(r.steps)})
                kept[i] += 1
                row["kept"] = True
            r.record = None
            rows.append(row)
            doneset[i] += 1
            if doneset[i] == n_att[i]:
                done_f.write(json.dumps({"instance_id": i, "kept": kept[i]}) + "\n")
                done_f.flush()
            if len(rows) % 50 == 0:
                log.info("%d/%d episodes | success %.3f | kept %d traj over %d instances | %.0fs", len(rows), len(jobs),
                         np.mean([x["success"] for x in rows]), sum(kept.values()), len(kept), time.time() - t0)

        with SimPool(int(cfg.eval.n_workers), int(cfg.env.resolution)) as pool:
            run_jobs_resilient(pool, policy, jobs, retries=int(cfg.eval.get("worker_retries", 0) or 0),
                               noise_scale=float(cfg.policy.noise_scale), on_result=on_result, desc="bt_datagen")
        w.close()
        done_f.close()
        (part / "CLOSED").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        write_jsonl(run.path("attempts.jsonl"), rows)
        succ_inst = {x["instance_id"] for x in rows if x["success"]}
        run.summary.update({"episode_success": float(np.mean([x["success"] for x in rows])) if rows else 0.0,
                            "instances_solved": len(succ_inst), "trajectories": int(sum(kept.values())),
                            "sim_transitions": int(sum(x["steps"] for x in rows)), "bt_stats": policy.stats()})
        log.info("DONE: %d episodes, success %.3f, solved %d / %d instances, %d trajectories kept",
                 len(rows), run.summary["episode_success"], len(succ_inst), len(base), sum(kept.values()))


if __name__ == "__main__":
    main()
