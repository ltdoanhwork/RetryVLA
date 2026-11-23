"""CUT-1 lite (pre-registered PLAN "Goal campaign" stage 2): failure-onset detection by sim branching on M0's failed
audit rollouts (LIBERO-Goal, 2610.05151 protocol).

python -m retryvla.search.goal_onset --ckpt <M0> --seed 7 [--K 8 --need 2 --grid 20 --workers 12]

For each failed rollout (audit seed s, length T) a BACKWARD scan over the 20-step grid t = T-20, T-40, ..., 0:
restore the init state (their protocol, 10 settle steps), replay M0's recorded actions[:t] open-loop (bit-exact), and
let M0 continue K times (policy seeds hashed from (rollout, t, k); <= 300 steps after t). p_pi(t) = successes / K.
- t_last = the latest grid point with >= `need` successes of K (p_pi > 1/8 for K 8, need 2). This is the
  last-recoverable point; the scan stops there.
- t_on = t_last + 20 (first unrecoverable grid point on the monotone envelope), capped at T - 5.
- Rollouts with < `need` / K at every scanned t (down to 0): `never_recoverable`, no entry.
All rollouts advance one grid step per round, in parallel (K jobs each), so the sim pool stays full. The scan is
charged in sim steps (reported).
Outputs data/goal_rsi/onset_s<seed>/:
- scan.jsonl (per rollout: p(t), t_last, t_on, sim steps);
- entries_s<seed>.jsonl (cut rule `onset`: state at t_on) and entries_pre_s<seed>.jsonl (rule `pre_onset`: t_last);
- actions_s<seed>.npz (link to the audit's).
Entries use the goal_audit schema, so goal_instances / compare_arms / wm.collect accept them unchanged.
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
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--need", type=int, default=2)
    ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--cont", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--audit_dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_rollouts", type=int, default=0)
    a = ap.parse_args()
    from libero.libero import get_libero_path

    from retryvla.policy.pi0_wrapper import Pi0Wrapper

    aud = REPO_ROOT / (a.audit_dir or f"data/goal_rsi/audit_m0_s{a.seed}")
    out = REPO_ROOT / (a.out or f"data/goal_rsi/onset_s{a.seed}")
    out.mkdir(parents=True, exist_ok=True)
    sp = json.loads((REPO_ROOT / "data/splits/goal_rsi_v1.json").read_text())
    rows = [json.loads(l) for l in open(aud / "rollouts.jsonl") if l.strip()]
    fails = [r for r in rows if not r["success"] and r["seed"] == a.seed]
    if a.max_rollouts:
        fails = fails[: a.max_rollouts]
    acts = np.load(aud / f"actions_s{a.seed}.npz")
    link = out / f"actions_s{a.seed}.npz"
    if not link.exists():
        os.symlink(aud / f"actions_s{a.seed}.npz", link)
    done = {}
    if (out / "scan.jsonl").exists():
        done = {json.loads(l)["scene"]: json.loads(l) for l in open(out / "scan.jsonl") if l.strip()}
    st = {}
    for r in fails:
        if r["scene"] in done:
            continue
        v = sp["tasks"][r["task"]]
        assert r["demo"] in v["train"]
        with h5py.File(v["hdf5"]) as h:
            dd = h[f"data/demo_{r['demo']}"]
            init = np.asarray(dd.attrs["init_state"]) if "init_state" in dd.attrs else np.asarray(dd["states"][0])
        T = int(r["T"])
        st[r["scene"]] = {"r": r, "init": init, "T": T, "t": (T - 1) // a.grid * a.grid if T % a.grid == 0 else T // a.grid * a.grid,
                          "p": {}, "steps": 0, "bddl": os.path.join(get_libero_path("bddl_files"), "libero_goal",
                                                                     f"{r['task']}.bddl"), "lang": v["language"]}
        if st[r["scene"]]["t"] >= T:
            st[r["scene"]]["t"] -= a.grid
    policy = Pi0Wrapper(a.ckpt, n_action_steps=5)
    t0, rnd = time.time(), 0
    with SimPool(a.workers, 128) as pool:
        while st:
            jobs = []
            for key, s in st.items():
                pre = acts[key][: s["t"]]
                for k in range(a.K):
                    jobs.append(Job("load_init_state", dict(bddl=s["bddl"], state=s["init"], num_steps_wait=10, seed=0,
                                                            return_snapshot=False),
                                    task=s["lang"], max_steps=len(pre) + a.cont, prefix=pre if len(pre) else None,
                                    policy_seed=zlib.crc32(f"{key}|{s['t']}|{k}".encode()) % (2**31 - 1),
                                    meta={"scene": key}))
            res = run_jobs(pool, policy, jobs, desc=f"onset_r{rnd}")
            by = {}
            for j, r in zip(jobs, res):
                by.setdefault(j.meta["scene"], []).append(r)
            for key, rs in by.items():
                s = st[key]
                n_succ = int(sum(r.success for r in rs))
                s["p"][s["t"]] = n_succ / a.K
                s["steps"] += int(sum(r.steps for r in rs))
                fin = None
                if n_succ >= a.need:
                    fin = {"t_last": s["t"], "t_on": min(s["t"] + a.grid, s["T"] - 5), "never_recoverable": False}
                elif s["t"] - a.grid < 0:
                    fin = {"t_last": None, "t_on": None, "never_recoverable": True}
                if fin:
                    row = {"scene": key, "task": s["r"]["task"], "demo": s["r"]["demo"], "seed": a.seed, "T": s["T"],
                           "p": {str(k): v for k, v in sorted(s["p"].items())}, "sim_steps": s["steps"], **fin}
                    with open(out / "scan.jsonl", "a") as f:
                        f.write(json.dumps(row) + "\n")
                    done[key] = row
                    del st[key]
                else:
                    s["t"] -= a.grid
            rnd += 1
            print(json.dumps({"round": rnd, "active": len(st), "done": len(done), "wall_s": round(time.time() - t0)}),
                  flush=True)
    ents, pres = [], []
    for key, row in done.items():
        if row["never_recoverable"]:
            continue
        base = {"scene": key, "task": row["task"], "demo": row["demo"], "seed": a.seed, "T": row["T"]}
        ents.append({**base, "entry_id": f"{key}__s{a.seed}__on{row['t_on']:03d}", "t": row["t_on"], "rule": "onset"})
        if row["t_last"] >= 5:
            pres.append({**base, "entry_id": f"{key}__s{a.seed}__pre{row['t_last']:03d}", "t": row["t_last"],
                         "rule": "pre_onset"})
    with open(out / f"entries_s{a.seed}.jsonl", "w") as f:
        for e in ents:
            f.write(json.dumps(e) + "\n")
    with open(out / f"entries_pre_s{a.seed}.jsonl", "w") as f:
        for e in pres:
            f.write(json.dumps(e) + "\n")
    summ = {"rollouts": len(done), "never_recoverable": sum(r["never_recoverable"] for r in done.values()),
            "onset_entries": len(ents), "sim_steps": int(sum(r["sim_steps"] for r in done.values())),
            "t_on_frac_mean": float(np.mean([r["t_on"] / r["T"] for r in done.values() if r["t_on"] is not None]))
            if ents else None, "K": a.K, "need": a.need, "grid": a.grid}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
