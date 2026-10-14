"""Asynchronous batched rollout engine.

A ``Job`` = load a start state (nominal init / recovery scene / exact snapshot) ->
optional open-loop ``prefix`` actions (recovery segment) -> closed-loop policy until success
or ``max_steps`` total env steps. Workers simulate in parallel; whenever some workers need a
new action chunk, the main process runs one batched policy call for all of them while the
others keep stepping.
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from retryvla.envs.vec import SimPool

log = logging.getLogger(__name__)


@dataclass
class Job:
    load_cmd: str                  # load_nominal | load_scene | restore_snapshot
    load_kwargs: dict
    task: str                      # instruction given to the policy
    max_steps: int
    prefix: np.ndarray | None = None
    record: str = "none"           # none | success | all
    policy_seed: int = 0
    return_snapshot: bool = False  # return the start snapshot (load_scene only)
    record_cfg: dict | None = None  # {"stride": k, "img_segments": [0]} -> sparse image recording
    meta: dict = field(default_factory=dict)


@dataclass
class Result:
    job_id: int
    success: bool
    steps: int
    success_step: int | None
    prefix_success: bool
    progress: float
    record: dict | None
    load_out: dict
    meta: dict


class _W:
    __slots__ = ("job_id", "job", "phase", "queue", "steps", "obs", "load_out", "n_calls", "prefix_success",
                 "success")

    def __init__(self):
        self.job_id = None
        self.phase = "idle"


def run_jobs(pool: SimPool, policy, jobs: list[Job], noise_scale: float = 1.0,
             on_result: Callable[[Result], None] | None = None, desc: str = "rollouts",
             batch_wait_s: float = 0.03) -> list[Result]:
    """Run all jobs; returns results ordered by job index."""
    n_act = policy.n_action_steps if policy is not None else 1
    pending = collections.deque(range(len(jobs)))
    ws = [_W() for _ in range(pool.n)]
    results: dict[int, Result] = {}
    t0, last_log = time.time(), 0.0

    def finish(i: int, w: _W) -> None:
        keep = w.job.record == "all" or (w.job.record == "success" and w.success)
        pool.send(i, "finish", keep_record=keep)
        w.phase = "finishing"

    while pending or any(w.phase != "idle" for w in ws):
        # 1) assign new jobs
        for i, w in enumerate(ws):
            if w.phase == "idle" and pending:
                jid = pending.popleft()
                w.job_id, w.job = jid, jobs[jid]
                w.queue, w.steps, w.n_calls, w.prefix_success, w.success = [], 0, 0, False, False
                pool.send(i, "begin", w.job.load_cmd, w.job.load_kwargs, record=w.job.record != "none",
                          return_snapshot=w.job.return_snapshot, record_cfg=w.job.record_cfg)
                w.phase = "loading"

        # 2) (unused: chunks are executed as one step_seq per worker)

        # 3) one batched policy call for all workers needing a chunk; execute the n_act-step
        #    open-loop chunk in a single round trip (policy only observes at chunk boundaries)
        need = [i for i, w in enumerate(ws) if w.phase == "act"]
        if need:
            obs = [ws[i].obs for i in need]
            tasks = [ws[i].job.task for i in need]
            seeds = [ws[i].job.policy_seed * 100_003 + ws[i].n_calls for i in need]
            chunks = policy.predict_chunk(obs, tasks, seeds=seeds, noise_scale=noise_scale)
            for i, ch in zip(need, chunks):
                w = ws[i]
                w.n_calls += 1
                k = min(n_act, w.job.max_steps - w.steps)
                pool.send(i, "step_seq", ch[:k], stop_on_success=True, segment=1)
                w.phase = "stepping"

        # 4) collect replies
        busy = [i for i, w in enumerate(ws) if w.phase in ("loading", "prefix", "stepping", "finishing")]
        if not busy:
            continue
        ready = pool.ready(busy)
        # small gather window so more workers join the next batched policy call
        more = pool.ready([i for i in busy if i not in ready], timeout=batch_wait_s) if len(ready) < len(busy) else []
        for i in ready + more:
            w = ws[i]
            out = pool.recv(i)
            if w.phase == "loading":
                w.load_out = {k: v for k, v in out.items() if k != "obs"}
                w.load_out["obs0"] = out["obs"]
                w.obs = out["obs"]
                if w.job.max_steps <= 0:
                    finish(i, w)
                elif w.job.prefix is not None and len(w.job.prefix):
                    pre = np.asarray(w.job.prefix)[: w.job.max_steps]
                    pool.send(i, "step_seq", pre, stop_on_success=True, segment=0)
                    w.phase = "prefix"
                elif policy is None:
                    finish(i, w)
                else:
                    w.phase = "act"
            elif w.phase == "prefix":
                w.steps += out["n"]
                w.obs = out["obs"]
                w.prefix_success = w.success = bool(out["success"])
                if w.success or w.steps >= w.job.max_steps or policy is None:
                    finish(i, w)
                else:
                    w.phase = "act"
            elif w.phase == "stepping":
                w.steps += out["n"]
                w.obs = out["obs"]
                w.success = bool(out["success"])
                if w.success or w.steps >= w.job.max_steps:
                    finish(i, w)
                else:
                    w.phase = "act"
            elif w.phase == "finishing":
                r = Result(
                    job_id=w.job_id, success=w.success, steps=w.steps,
                    success_step=w.steps if w.success else None, prefix_success=w.prefix_success,
                    progress=float(out["progress"]), record=out["record"], load_out=w.load_out,
                    meta=w.job.meta,
                )
                results[w.job_id] = r
                if on_result is not None:
                    on_result(r)
                w.phase, w.job, w.obs, w.queue = "idle", None, None, []

        now = time.time()
        if now - last_log > 30:
            last_log = now
            done = len(results)
            sr = np.mean([r.success for r in results.values()]) if results else float("nan")
            log.info("[%s] %d/%d done | SR %.3f | %.0fs", desc, done, len(jobs), sr, now - t0)
    return [results[j] for j in range(len(jobs))]
