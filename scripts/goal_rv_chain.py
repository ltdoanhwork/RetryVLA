"""Unattended chain: RetryVLA pipeline on the 2610.05151 protocol (pre-registered PLAN "2610.05151 comparison",
stage-2 bullet, 2026-11-18). Local shell 8b889199, GPUs 1-3 (never 0).

python scripts/goal_rv_chain.py [--gpus 1,2,3]

A. wait for the M0 audits (tmux rv_goal_audit_s7 / _s8): data/goal_rsi/audit_m0_s{7,8}/entries_s*.jsonl
B. WM / C^pi data on SEED-8 entries: retryvla.wm.collect configs/exp/goal_collect.yaml, 3 shards in parallel
C. Goal WM (goal_train_wm) then C^pi (goal_train_critic) on GPU 1
D. search on SEED-7 entries: compare_arms configs/exp/goal_search.yaml, arms random + wm_diverse, 3 shards
E. distillation (their recipe: M0 + 2k updates, D0 + bank pooled uniformly, warmup 200, cosine), arms
   wm (wm_diverse bank) and rand (random bank), same trajectory count (min of the two banks, round-robin over
   entries), seeds 7 / 8 / 9, one run per GPU (accum 3, global bs 48), each followed by the val eval
F. scripts/analyze_goal_rsi.py -> runs/goal_rsi/main.{json,md}
Every stage is resumable (markers / outputs checked); every job is retried once.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV = f"source {REPO}/env/activate.sh >/dev/null 2>&1; export HF_HUB_OFFLINE=1; cd {REPO}; "
LOGS = REPO / "runs/goal_rsi/logs_rv"
M0 = "runs/goal_rsi/m0_s7/ckpt_04000"


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def start(cmd: str, gpu: str, name: str) -> subprocess.Popen:
    full = f"{ENV} export CUDA_VISIBLE_DEVICES={gpu} MUJOCO_EGL_DEVICE_ID={gpu}; {cmd}"
    log("START", name, "gpu", gpu, cmd)
    return subprocess.Popen(["bash", "-lc", full], stdout=open(LOGS / f"{name}.log", "a"), stderr=subprocess.STDOUT)


def run_parallel(jobs: list[tuple[str, str, str]], tries: int = 2) -> bool:
    """jobs = [(cmd, gpu, name)]; all in parallel, failed ones retried."""
    left = list(jobs)
    for k in range(tries):
        ps = [(j, start(*j)) for j in left]
        left = []
        for j, p in ps:
            rc = p.wait()
            log("END", j[2], "rc", rc)
            if rc != 0:
                left.append(j)
        if not left:
            return True
        time.sleep(60)
    return False


def n_traj(pattern: str) -> int:
    return sum(sum(1 for _ in open(f)) for f in glob.glob(str(REPO / pattern)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="1,2,3")
    a = ap.parse_args()
    G = a.gpus.split(",")
    LOGS.mkdir(parents=True, exist_ok=True)

    # A
    while not all((REPO / f"data/goal_rsi/audit_m0_s{s}/entries_s{s}.jsonl").exists() for s in (7, 8)):
        time.sleep(60)
    for s in (7, 8):
        log("audit", s, json.loads((REPO / f"data/goal_rsi/audit_m0_s{s}/summary.json").read_text())["seeds"])

    # B
    n8 = sum(1 for _ in open(REPO / "data/goal_rsi/audit_m0_s8/entries_s8.jsonl"))
    have = len(glob.glob(str(REPO / "data/wm/goal_collect/*.pt")))
    if have < n8:
        ok = run_parallel([(f"python -m retryvla.wm.collect configs/exp/goal_collect.yaml wm_collect.shard={i} "
                            f"wm_collect.num_shards={len(G)} eval.n_workers=8", g, f"collect_s{i}")
                           for i, g in enumerate(G)])
        if not ok:
            raise SystemExit("collect failed")
    log("collect files", len(glob.glob(str(REPO / "data/wm/goal_collect/*.pt"))), "of", n8)

    # C
    if not (REPO / "data/wm/goal_latest_wm.txt").exists():
        if not run_parallel([("python -m retryvla.wm.train_wm configs/exp/goal_train_wm.yaml", G[0], "train_wm")]):
            raise SystemExit("WM training failed")
    if not (REPO / "data/wm/goal_latest_critic.txt").exists():
        if not run_parallel([("python -m retryvla.critic.train_critic configs/exp/goal_train_critic.yaml", G[0],
                              "train_critic")]):
            raise SystemExit("critic training failed")

    # D
    n7 = sum(1 for _ in open(REPO / "data/goal_rsi/audit_m0_s7/entries_s7.jsonl"))
    closed = glob.glob(str(REPO / "data/recovery/goal_search/s*/p*/CLOSED"))
    if len(closed) < len(G):
        ok = run_parallel([(f"python -m retryvla.search.compare_arms configs/exp/goal_search.yaml m5b.shard={i} "
                            f"m5b.num_shards={len(G)} eval.n_workers=8", g, f"search_s{i}") for i, g in enumerate(G)])
        if not ok:
            raise SystemExit("search failed")
    nw = n_traj("data/recovery/goal_search/s*/p*/wm_diverse/retryvla/trajectories.jsonl")
    nr = n_traj("data/recovery/goal_search/s*/p*/random/retryvla/trajectories.jsonl")
    nmatch = min(nw, nr)
    log("bank sizes wm_diverse", nw, "random", nr, "-> matched", nmatch, "entries", n7)
    (REPO / "runs/goal_rsi/bank.json").write_text(json.dumps({"wm_diverse": nw, "random": nr, "matched": nmatch,
                                                              "entries_s7": n7}, indent=1))

    # E
    runs = []
    for arm, glob_arm in (("wm", "wm_diverse"), ("rand", "random")):
        for seed in (7, 8, 9):
            out = f"runs/goal_rsi/rec_{arm}_s{seed}"
            runs.append((out, glob_arm, seed))
    todo = [r for r in runs if not (REPO / r[0] / "ckpt_02000/eval_val/summary.json").exists()]
    while todo:
        wave, todo = todo[: len(G)], todo[len(G):]
        jobs = []
        for (out, glob_arm, seed), g in zip(wave, G):
            tr = (f"torchrun --nproc_per_node 1 --master_port {29700 + int(g) * 10 + seed} -m retryvla.train.pi0_sft "
                  f"--out {out} --init {M0} --steps 2000 --save_every 2000 --bs 48 --accum 3 --warmup 200 "
                  f"--decay_steps 2000 --seed {seed} --max_traj {nmatch} "
                  f"--recovery 'data/recovery/goal_search/s*/p*/{glob_arm}'")
            ev = f"python -m retryvla.eval.eval_goal --ckpt {out}/ckpt_02000 --split val --workers 8"
            cmd = (f"( [ -f {out}/DONE ] || {tr} ) && ( [ -f {out}/ckpt_02000/eval_val/summary.json ] || {ev} )")
            jobs.append((cmd, g, Path(out).name))
        if not run_parallel(jobs):
            log("some distillation runs failed; continuing with the rest")

    # F
    if (REPO / "scripts/analyze_goal_rsi.py").exists():
        subprocess.call(["bash", "-lc", f"{ENV} python scripts/analyze_goal_rsi.py"],
                        stdout=open(LOGS / "analyze.log", "a"), stderr=subprocess.STDOUT)
    log("CHAIN DONE")


if __name__ == "__main__":
    main()
