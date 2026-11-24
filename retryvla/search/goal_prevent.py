"""Goal P2-3: verified pre-failure positives D_P at onset-adjacent states s^- (RetryVLA v2, Algorithm 1 prevention branch).

python -m retryvla.search.goal_prevent --ckpt <M0> --seed 7 [--n_vla 24 --K 3 --workers 10]

s^- = the `pre_onset` entries of the onset scan (data/goal_rsi/onset_s<seed>/entries_pre_s<seed>.jsonl): the last
20-step grid point where M0 still succeeds >= 2 / 8 continuations, just before the failure becomes unrecoverable.
Per s^- (restored bit-exactly by replaying M0's audit actions):
- a- (original): M0's own recorded next 50 actions from the audit rollout (open-loop), then M0 continues (<= 300 steps);
  K trials (continuation seeds).
- candidates a: `n_vla` pi0 chunk samples at noise 1.0 + the 8 fixed primitives, curated as in G2 (<= 1 gripper sign
  change). Each is screened with 1 trial; screen successes get K - 1 more trials.
- Pair (a+, a-) iff p(a+) >= 2/3 (K 3) and p(a+) - p(a-) >= 1/3. Up to `max_pos` positives per s^- (highest
  p, then the original-closest by L2 to a-).
- Written: D_P bank (RecoveryWriter: bridge = a+, segment 0; continuation = M0, segment 1; meta with p(a+), p(a-), source)
  -> data/recovery/goal_dp_s<seed>/; pairs.jsonl with all outcomes (including a- actions) for a later hinge L_P.
Sim steps are counted and reported (funnel: s^- -> candidates -> screened -> verified pairs).
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np

from retryvla import REPO_ROOT
from retryvla.data.recovery_writer import RecoveryWriter
from retryvla.envs.vec import SimPool
from retryvla.eval.rollout import Job, run_jobs
from retryvla.search import primitives
from retryvla.search.goal_instances import build_goal_instances


def flips(a):
    return int(np.sum(np.diff(np.sign(a[:, 6])) != 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_vla", type=int, default=24)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--H", type=int, default=50)
    ap.add_argument("--cont", type=int, default=300)
    ap.add_argument("--max_pos", type=int, default=2)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    a = ap.parse_args()
    from retryvla.policy.pi0_wrapper import Pi0Wrapper

    od = REPO_ROOT / f"data/goal_rsi/onset_s{a.seed}"
    ents = [json.loads(l) for l in open(od / f"entries_pre_s{a.seed}.jsonl") if l.strip()][a.shard::a.num_shards]
    out = REPO_ROOT / f"data/recovery/goal_dp_s{a.seed}" / f"sh{a.shard}"
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    if (out / "pairs.jsonl").exists():
        done = {json.loads(l)["entry_id"] for l in open(out / "pairs.jsonl") if l.strip()}
    ents = [e for e in ents if e["entry_id"] not in done]
    acts = np.load(od / f"actions_s{a.seed}.npz")
    policy = Pi0Wrapper(a.ckpt, n_action_steps=5)
    k = len(list(out.glob("bank*")))   # one LeRobotDataset part per (re)start; training globs bank*
    writer = RecoveryWriter(out / f"bank{k}", overwrite=True, write_retries=8, res=128)
    t0 = time.time()
    with SimPool(a.workers, 128) as pool:
        for n, e in enumerate(ents):
            inst = build_goal_instances(pool, [dict(e)], str(od.relative_to(REPO_ROOT)))[0]
            seed = zlib.crc32(e["entry_id"].encode()) % (2**31 - 1)
            # a- = M0's recorded actions from s^- (shorter than H when the audit rollout ended; then M0 itself continues
            # after the prefix in every trial, which is exactly the original behaviour)
            a_minus = acts[e["scene"]][e["t"]: e["t"] + a.H].astype(np.float32)
            vla = policy.sample_chunks(inst.obs0, inst.task, a.n_vla, noise_scale=1.0, seed=seed)[:, : a.H]
            prim, names = primitives.library(a.H, np.random.RandomState(seed), n_random=0)
            cands = np.clip(np.concatenate([vla, prim]), -1, 1).astype(np.float32)
            src = [f"vla_s1.0_{i}" for i in range(len(vla))] + [f"prim_{n_}" for n_ in names]
            keep = [i for i in range(len(cands)) if flips(cands[i]) <= 1]
            cands, src = cands[keep], [src[i] for i in keep]

            def job(c, k, rec="none"):
                return Job("restore_snapshot", dict(bddl=inst.bddl, snap=inst.snapshot), task=inst.task,
                           max_steps=len(c) + a.cont, prefix=c, record=rec, policy_seed=(seed + 7919 * k) % (2**31 - 1))
            # a- with K trials, every candidate screened once (trial seed 0 shared -> paired)
            jobs = [job(a_minus, k) for k in range(a.K)] + [job(c, 0, "all") for c in cands]
            res = run_jobs(pool, policy, jobs, desc="dp_screen")
            steps = sum(r.steps for r in res)
            p_minus = float(np.mean([r.success for r in res[: a.K]]))
            scr = res[a.K:]
            hit = [i for i, r in enumerate(scr) if r.success]
            extra = [job(cands[i], k) for i in hit for k in range(1, a.K)]
            res2 = run_jobs(pool, policy, extra, desc="dp_verify") if extra else []
            steps += sum(r.steps for r in res2)
            p_c = {}
            for j, i in enumerate(hit):
                rr = [scr[i]] + res2[j * (a.K - 1): (j + 1) * (a.K - 1)]
                p_c[i] = float(np.mean([r.success for r in rr]))
            pos = [i for i, p in p_c.items() if p >= 2 / 3 - 1e-9 and p - p_minus >= 1 / 3 - 1e-9]
            def l2m(i):
                n_ = min(len(a_minus), len(cands[i]))
                return float(np.linalg.norm(cands[i][:n_] - a_minus[:n_]))
            pos = sorted(pos, key=lambda i: (-p_c[i], l2m(i)))[: a.max_pos]
            writer.add_instance(e["entry_id"], {"scene": e["scene"], "task": e["task"], "arm": "dp"})
            for i in pos:
                writer.add_trajectory(scr[i].record, inst.task, e["entry_id"],
                                      {"arm": "dp", "p_plus": p_c[i], "p_minus": p_minus, "source": src[i],
                                       "t_minus": e["t"], "grip_flips": flips(cands[i]),
                                       "l2_to_minus": l2m(i)})
            row = {"entry_id": e["entry_id"], "task": e["task"], "t": e["t"], "p_minus": p_minus, "n_cands": len(cands),
                   "n_screen_hit": len(hit), "p_cands": {src[i]: p for i, p in p_c.items()},
                   "positives": [src[i] for i in pos], "a_minus": a_minus.round(4).tolist(),
                   "a_plus": [cands[i].round(4).tolist() for i in pos], "sim_steps": int(steps)}
            for r in scr:
                r.record = None
            with open(out / "pairs.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({"n": n + 1, "of": len(ents), "entry": e["entry_id"][-40:], "p_minus": p_minus,
                              "hits": len(hit), "pos": len(pos), "steps": int(steps),
                              "wall_s": round(time.time() - t0)}), flush=True)
    writer.close()
    (out / "CLOSED").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))


if __name__ == "__main__":
    main()
