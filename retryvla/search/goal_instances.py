"""2610.05151 reverse direction: failure instances on LIBERO-Goal from the M0 audit (retryvla.search.goal_audit).

An entry (scene, audit seed, t) is restored by: load_init_state (their protocol: 10 settle steps) -> replay M0's
recorded actions[:t] open-loop (deterministic) -> exact snapshot + obs. The resulting `Instance` objects plug into the
existing search / collect code (restore_snapshot jobs), exactly like the Spatial LIBERO-Recover instances.
"""

from __future__ import annotations

import json
import os

import h5py
import numpy as np

from retryvla import REPO_ROOT
from retryvla.search.scorers import Instance


def load_entries(audit_dir: str, seed: int, shard: int = 0, num_shards: int = 1, max_n: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in open(REPO_ROOT / audit_dir / f"entries_s{seed}.jsonl") if l.strip()]
    rows = rows[shard::num_shards]
    return rows[:max_n] if max_n else rows


def build_goal_instances(pool, entries: list[dict], audit_dir: str, split_name: str = "goal_rsi_v1") -> list[Instance]:
    from libero.libero import get_libero_path

    sp = json.loads((REPO_ROOT / f"data/splits/{split_name}.json").read_text())
    acts = {}
    out: list[Instance | None] = [None] * len(entries)
    todo = list(range(len(entries)))
    while todo:
        batch, todo = todo[: pool.n], todo[pool.n:]
        for w, k in enumerate(batch):
            e = entries[k]
            v = sp["tasks"][e["task"]]
            assert e["demo"] in v["train"], f"entry {e['entry_id']} is not a TRAIN scene"
            if e["seed"] not in acts:
                acts[e["seed"]] = np.load(REPO_ROOT / audit_dir / f"actions_s{e['seed']}.npz")
            with h5py.File(v["hdf5"]) as h:
                dd = h[f"data/demo_{e['demo']}"]
                init = np.asarray(dd.attrs["init_state"]) if "init_state" in dd.attrs else np.asarray(dd["states"][0])
            bddl = os.path.join(get_libero_path("bddl_files"), "libero_goal", f"{e['task']}.bddl")
            e["_bddl"], e["_lang"] = bddl, v["language"]
            pool.send(w, "begin", "load_init_state", dict(bddl=bddl, state=init, num_steps_wait=10, seed=0,
                                                          return_snapshot=False))
        for w, k in enumerate(batch):
            pool.recv(w)
            pre = acts[entries[k]["seed"]][entries[k]["scene"]][: entries[k]["t"]]
            pool.send(w, "step_seq", pre, stop_on_success=False)
        obs = {}
        for w, k in enumerate(batch):
            r = pool.recv(w)
            obs[k] = r["obs"]
            pool.send(w, "snapshot")
        for w, k in enumerate(batch):
            snap = pool.recv(w)
            e = entries[k]
            out[k] = Instance(e["entry_id"], e["scene"], e["_bddl"], e["_lang"], snap, obs[k],
                              meta={"task_folder": e["task"], "gripper_init": "na", "scene": e["scene"],
                                    "t": e["t"], "T": e["T"], "seed": e["seed"], "demo": e["demo"]})
    return [x for x in out if x is not None]
