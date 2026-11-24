"""Goal FC (pre-registered PLAN 2026-11-24 "G2-phys / FC / AUX"): failure-critical segment labelling of M0's failed
audit rollouts with physical signals + the onset scan.

python -m retryvla.search.goal_fc --seed 7 [--workers 10]

For each failed audit rollout (seed s): replay it bit-exactly (init state, 10 settle steps, M0's recorded actions) with the
frozen detector E-v1 (retryvla.eval.goal_events.detect via the worker hook events_scan; privileged MuJoCo state every 5
steps: grasp_lost / displaced / tipped / off_table / stall) and read the onset scan (t_last = last recoverable 20-step
grid point, t_on = t_last + 20).
- first physical event after t_last: t_ev (None if no event).
- failure-critical segment FC = [t_last, t_end) with t_end = t_ev + 5 if t_last < t_ev <= t_on + 40 else t_on
  (the physical event confirms / sharpens the onset; events much later than the onset are consequences, not causes).
- refined s^- = t_last (unchanged); refined s^f = t_end (first state after the failure-critical segment).
- D_F (negatives) = M0's actions inside FC only (never whole episodes); also reported: events BEFORE t_last
  (detector false alarms or recoverable incidents).
Outputs data/goal_rsi/fc_s<seed>/{fc.jsonl, D_F.npz (entry -> FC actions), summary.json}.
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
from retryvla.eval.goal_events import EVERY, detect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    from libero.libero import get_libero_path

    od = REPO_ROOT / f"data/goal_rsi/onset_s{a.seed}"
    aud = REPO_ROOT / f"data/goal_rsi/audit_m0_s{a.seed}"
    out = REPO_ROOT / f"data/goal_rsi/fc_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    scan = [json.loads(l) for l in open(od / "scan.jsonl")]
    acts = np.load(aud / f"actions_s{a.seed}.npz")
    sp = json.loads((REPO_ROOT / "data/splits/goal_rsi_v1.json").read_text())
    rows, DF = [], {}
    with SimPool(a.workers, 128) as pool:
        for b in range(0, len(scan), pool.n):
            batch = scan[b: b + pool.n]
            for w, r in enumerate(batch):
                v = sp["tasks"][r["task"]]
                with h5py.File(v["hdf5"]) as h:
                    dd = h[f"data/demo_{r['demo']}"]
                    init = np.asarray(dd.attrs["init_state"]) if "init_state" in dd.attrs else np.asarray(dd["states"][0])
                bddl = os.path.join(get_libero_path("bddl_files"), "libero_goal", f"{r['task']}.bddl")
                pool.send(w, "events_scan", bddl, init, acts[r["scene"]], EVERY)
            for w, r in enumerate(batch):
                ev = detect(pool.recv(w), None)
                tl, ton = r["t_last"], r["t_on"]
                if tl is None:
                    row = {"scene": r["scene"], "task": r["task"], "T": r["T"], "never_recoverable": True, "events": ev}
                else:
                    after = [e for e in ev if e["t"] > tl]
                    t_ev = after[0]["t"] if after else None
                    t_end = t_ev + 5 if (t_ev is not None and t_ev <= ton + 40) else ton
                    t_end = int(min(max(t_end, tl + 5), r["T"]))
                    row = {"scene": r["scene"], "task": r["task"], "T": r["T"], "never_recoverable": False,
                           "t_last": tl, "t_on": ton, "t_event": t_ev, "event_type": after[0]["type"] if after else None,
                           "fc": [tl, t_end], "fc_len": t_end - tl, "s_minus": tl, "s_f": t_end,
                           "events_before_t_last": [e for e in ev if e["t"] <= tl], "events": ev,
                           "event_confirms_onset": bool(t_ev is not None and t_ev <= ton + 40)}
                    DF[r["scene"]] = acts[r["scene"]][tl:t_end]
                rows.append(row)
    with open(out / "fc.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    np.savez_compressed(out / "D_F.npz", **DF)
    ok = [r for r in rows if not r["never_recoverable"]]
    types = {}
    for r in ok:
        if r["event_type"]:
            types[r["event_type"]] = types.get(r["event_type"], 0) + 1
    summ = {"rollouts": len(rows), "labelled": len(ok),
            "event_confirms_onset": int(sum(r["event_confirms_onset"] for r in ok)),
            "first_event_type_after_t_last": types,
            "fc_len_mean": float(np.mean([r["fc_len"] for r in ok])) if ok else None,
            "rollouts_with_events_before_t_last": int(sum(bool(r["events_before_t_last"]) for r in ok)),
            "D_F_actions": int(sum(len(v) for v in DF.values()))}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
