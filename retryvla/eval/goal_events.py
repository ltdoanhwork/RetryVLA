"""Failure-event detector E (RetryVLA v2 plan 6.1), FROZEN 2026-11-24 before any P2-3 result. Version: E-v1.

python -m retryvla.eval.goal_events --eval_dir <ckpt>/eval_val [--workers 12]

Offline and independent of every model: replays each val episode bit-exactly from its init state with the actions
saved by `eval_goal --save_actions`, reading PRIVILEGED sim state every 5 steps, and extracts events:
- `grasp_lost`: the first object of interest was held (finger contact and lifted >= 3 cm above its start height)
  and then loses finger contact while > 2 cm above the table, falling >= 3 cm within 10 steps.
- `displaced`: any free object moves > 3 cm in xy while not in finger contact and NOT the first object of
  interest (collateral push).
- `tipped`: the first object's up-axis z component drops below 0.8 (from >= 0.95 at start).
- `off_table`: any free object leaves the table / falls below its top.
- `stall`: eef speed < 1 mm / step for >= 40 consecutive steps (8 samples) while the task is not yet done.
Per episode: events with type + step, failure incidence (>= 1 event), first-attempt success (success without any event
before the success step), completion steps. Output <eval_dir>/events.jsonl + events_summary.json.
Thresholds are fixed here; any change = new version tag, logged in NOTES.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from retryvla import REPO_ROOT
from retryvla.envs.vec import SimPool

VERSION = "E-v1"
EVERY = 5


def detect(samples: list[dict], success_step) -> list[dict]:
    ev = []
    if not samples:
        return ev
    s0 = samples[0]
    held = False
    still = 0
    lifted_peak = -1e9
    for k, s in enumerate(samples):
        t = s["t"]
        if success_step is not None and t >= success_step:
            break
        lift = s["obj_z"] - s0["obj_z"]
        if s["contact"] and lift >= 0.03:
            held, lifted_peak = True, max(lifted_peak, s["obj_z"])
        if held and not s["contact"] and s["obj_z_tab"] > 0.02:
            nxt = [x for x in samples[k: k + 3]]
            if nxt and lifted_peak - min(x["obj_z"] for x in nxt) >= 0.03:
                ev.append({"type": "grasp_lost", "t": t})
                held = False
        if s0["obj_upz"] >= 0.95 and s["obj_upz"] < 0.8 and not any(e["type"] == "tipped" for e in ev):
            ev.append({"type": "tipped", "t": t})
        for name, xy in s["free_xy"].items():
            if name == s["obj_name"] or s["contact_any"].get(name, False):
                continue
            if np.linalg.norm(np.asarray(xy) - np.asarray(s0["free_xy"].get(name, xy))) > 0.03 and \
                    not any(e["type"] == "displaced" and e.get("obj") == name for e in ev):
                ev.append({"type": "displaced", "t": t, "obj": name})
        if s["off_table"] and not any(e["type"] == "off_table" for e in ev):
            ev.append({"type": "off_table", "t": t})
        if k > 0:
            sp = np.linalg.norm(np.asarray(s["eef"]) - np.asarray(samples[k - 1]["eef"])) / EVERY
            still = still + 1 if sp < 0.001 else 0
            if still == 8 and not any(e["type"] == "stall" for e in ev):
                ev.append({"type": "stall", "t": t})
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    from libero.libero import get_libero_path

    d = Path(a.eval_dir)
    if (d / "events_summary.json").exists():
        print("exists"); return
    eps = [json.loads(l) for l in open(d / "episodes.jsonl")]
    acts = np.load(d / "actions.npz")
    sp = json.loads((REPO_ROOT / "data/splits/goal_rsi_v1.json").read_text())
    rows = []
    with SimPool(a.workers, 128) as pool:
        todo = list(eps)
        while todo:
            batch, todo = todo[: pool.n], todo[pool.n:]
            for w, e in enumerate(batch):
                v = sp["tasks"][e["task"]]
                with h5py.File(v["hdf5"]) as h:
                    dd = h[f"data/demo_{e['demo']}"]
                    init = np.asarray(dd.attrs["init_state"]) if "init_state" in dd.attrs else np.asarray(dd["states"][0])
                bddl = os.path.join(get_libero_path("bddl_files"), "libero_goal", f"{e['task']}.bddl")
                pool.send(w, "events_scan", bddl, init, acts[e["scene"]], EVERY)
            for w, e in enumerate(batch):
                samples = pool.recv(w)
                ss = e["steps"] if e["success"] else None
                ev = detect(samples, ss)
                rows.append({"scene": e["scene"], "task": e["task"], "success": e["success"], "steps": e["steps"],
                             "events": ev, "n_events": len(ev),
                             "first_attempt_success": bool(e["success"] and not ev)})
    with open(d / "events.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    types = {}
    for r in rows:
        for e in r["events"]:
            types[e["type"]] = types.get(e["type"], 0) + 1
    succ = [r for r in rows if r["success"]]
    summ = {"version": VERSION, "n": len(rows), "failure_incidence": float(np.mean([r["n_events"] > 0 for r in rows])),
            "first_attempt_success": float(np.mean([r["first_attempt_success"] for r in rows])),
            "success": float(np.mean([r["success"] for r in rows])),
            "p_success_given_event": float(np.mean([r["success"] for r in rows if r["n_events"]])) if any(r["n_events"] for r in rows) else None,
            "completion_steps_mean": float(np.mean([r["steps"] for r in succ])) if succ else None,
            "completion_steps_median": float(np.median([r["steps"] for r in succ])) if succ else None,
            "events_by_type": types}
    (d / "events_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
