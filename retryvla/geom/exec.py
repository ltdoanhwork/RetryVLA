"""Parallel open-loop bridge execution on a SimPool (RGR): restore exact snapshot -> bridge segment ->
end obs + privileged geometry trace + exact end snapshot. No policy involved."""

from __future__ import annotations

import collections
import logging
import time

from retryvla.envs.vec import SimPool, WorkerDied

log = logging.getLogger(__name__)


def run_bridges(pool: SimPool, items: list[tuple], every: int = 5, snapshot: bool = True,
                retries: int = 2) -> list[dict]:
    """items: [(bddl, snap, actions [h, 7])] -> [dict(success, n, geom [F, G], snapshot, obs)] in order."""
    for attempt in range(retries + 1):
        try:
            return _run(pool, items, every, snapshot)
        except WorkerDied as e:
            if attempt == retries:
                raise
            log.warning("%s; respawning pool, re-running %d bridges", e, len(items))
            pool.respawn()
    return []


def _run(pool, items, every, snapshot):
    pend = collections.deque(range(len(items)))
    phase = [None] * pool.n
    job = [None] * pool.n
    out: dict[int, dict] = {}
    t0 = time.time()
    while pend or any(p is not None for p in phase):
        for i in range(pool.n):
            if phase[i] is None and pend:
                j = pend.popleft()
                bddl, snap, _ = items[j]
                pool.send(i, "restore_snapshot", bddl, snap)
                phase[i], job[i] = "restore", j
        busy = [i for i in range(pool.n) if phase[i] is not None]
        for i in pool.ready(busy):
            r = pool.recv(i)
            if phase[i] == "restore":
                pool.send(i, "bridge", items[job[i]][2], every=every, snapshot=snapshot)
                phase[i] = "bridge"
            else:
                out[job[i]] = r
                phase[i] = None
    log.debug("ran %d bridges in %.0fs", len(items), time.time() - t0)
    return [out[j] for j in range(len(items))]
