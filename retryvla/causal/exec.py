"""Run short per-item command sequences on a SimPool in parallel (ACR probes: restore_hidden -> probe).
items: [[(cmd, args, kwargs), ...]] -> the LAST command's result of every item, in order. No policy involved."""

from __future__ import annotations

import collections
import logging

from retryvla.envs.vec import SimPool, WorkerDied

log = logging.getLogger(__name__)


def run_seqs(pool: SimPool, items: list[list[tuple]], retries: int = 2) -> list:
    for attempt in range(retries + 1):
        try:
            return _run(pool, items)
        except WorkerDied as e:
            if attempt == retries:
                raise
            log.warning("%s; respawning pool, re-running %d command sequences", e, len(items))
            pool.respawn()
    return []


def _run(pool, items):
    pend = collections.deque(range(len(items)))
    step = [None] * pool.n
    job = [None] * pool.n
    out = {}
    while pend or any(s is not None for s in step):
        for i in range(pool.n):
            if step[i] is None and pend:
                j = pend.popleft()
                cmd, a, k = items[j][0]
                pool.send(i, cmd, *a, **k)
                step[i], job[i] = 0, j
        busy = [i for i in range(pool.n) if step[i] is not None]
        for i in pool.ready(busy):
            r = pool.recv(i)
            j = job[i]
            if step[i] + 1 < len(items[j]):
                step[i] += 1
                cmd, a, k = items[j][step[i]]
                pool.send(i, cmd, *a, **k)
            else:
                out[j] = r
                step[i] = None
    return [out[j] for j in range(len(items))]
