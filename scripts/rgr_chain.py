#!/usr/bin/env python
"""RGR chain (user 2026-11-08). Agent box, CPU only, local tmux `rgr_chain`; launches on the agent-requested
user-account shell d2ecd68c (vla_RetryVLA_DoanhLT_rgr) via scripts/shells/launch_remote.sh (claims in shells.json):

  1. wait for the collect_v1 geometry labels (scripts/geom_labels.py, 144 files)
  2. rv_rgr_head : python -m retryvla.geom.head --steps 6000          (geometry head, train split only)
  3. rv_rgr_sw<k>: retry_oom rgr_search configs/exp/rgr_sweep.yaml m5b.shard=<k>, k in SHARDS, <= MAX_PAR at a time
  4. python scripts/analyze_rgr.py after every finished sweep shard (runs/_analysis/rgr/rgr.md)
watchdog: log not written for STALL_MIN min / session gone -> kill exact session, release claim, relaunch (rgr_search
resumes from done.jsonl; pools stay paired). Console: runs/_logs/rgr_chain_console.log
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xwm_chain import ENV, TMUX, rc_of, release  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SH, SH_FULL, GPU = "d2ecd68c", "d2ecd68c-9ef7-43e7-be02-31f14c60a74c", 0
SHARDS = [int(x) for x in os.environ.get("RGR_SHARDS", "0,1,2,3").split(",")]
MAX_PAR = int(os.environ.get("RGR_PAR", "2"))
WORKERS = int(os.environ.get("RGR_WORKERS", "8"))
STALL_MIN, POLL_S, MAX_TRIES = 45, 240, 4


def say(msg: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), "[rgr_chain]", msg, flush=True)


def live_sessions() -> set[str] | None:
    try:
        out = subprocess.run(["det", "shell", "open", SH_FULL, "--", TMUX, "ls"], capture_output=True, text=True,
                             timeout=180, env=ENV).stdout.replace("\r", "")
    except Exception:  # noqa: BLE001
        return None
    return {ln.split(":")[0] for ln in out.splitlines() if ln.startswith("rv_")}


def kill(sess: str) -> None:
    assert sess.startswith("rv_rgr_")
    subprocess.run(["det", "shell", "open", SH_FULL, "--", TMUX, "kill-session", "-t", f"={sess}"],
                   capture_output=True, text=True, timeout=180, env=ENV)


def launch(sess: str, args: list[str]) -> str | None:
    p = subprocess.run([str(REPO / "scripts/shells/launch_remote.sh"), SH, str(GPU), sess, *args], capture_output=True,
                       text=True, env=ENV, cwd=REPO, timeout=600)
    out = (p.stdout + p.stderr).replace("\r", "")
    m = re.search(r"LAUNCHED \S+ gpu=\d+ log=(\S+)", out)
    if not m:
        say(f"launch {sess} failed: {out.strip().splitlines()[-1:] if out.strip() else p.returncode}")
        return None
    say(f"launched {sess}: {m.group(1)}")
    return m.group(1)


def analyze() -> None:
    subprocess.run(["bash", "-c", "source env/activate.sh >/dev/null 2>&1; python scripts/analyze_rgr.py > /dev/null"],
                   cwd=REPO, timeout=1800)


def run_blocking(n: str, args: list[str]) -> int:
    """launch one task and wait for RV_DONE (watchdog: stall / session gone -> kill exact session + relaunch)."""
    for _ in range(MAX_TRIES):
        log = launch(n, args)
        if log is None:
            time.sleep(POLL_S)
            continue
        while True:
            time.sleep(POLL_S)
            rc = rc_of(log)
            if rc is not None:
                release(n)
                say(f"{n} ended rc={rc}")
                if rc == 0:
                    return 0
                break
            live = live_sessions()
            idle = (time.time() - os.path.getmtime(log)) / 60 if os.path.exists(log) else 0.0
            if idle > STALL_MIN or (live is not None and n not in live and idle > 5):
                say(f"WATCHDOG {n}: stalled / gone ({idle:.0f} min) -> kill + relaunch")
                if live and n in live:
                    kill(n)
                release(n)
                break
    return 1


def n_traj(dirs: list[str]) -> int:
    return sum(1 for d in dirs for f in glob.glob(str(REPO / d / "retryvla/trajectories.jsonl")) for _ in open(f))


def kill_shell() -> None:
    if os.environ.get("RGR_KEEP_SHELL"):
        say("RGR_KEEP_SHELL set: shell kept")
        return
    p = subprocess.run(["det", "shell", "kill", SH_FULL], capture_output=True, text=True, env=ENV, timeout=300)
    say(f"det shell kill {SH}: rc {p.returncode} {p.stdout.strip()[-200:]} {p.stderr.strip()[-200:]}")


def phase2() -> None:
    """pre-registered decision (scripts/analyze_rgr.py) -> optional matched 1-seed distill of the chosen arm vs
    `single` -> test RSR + nominal -> kill the agent-requested shell."""
    import json
    import threading

    dec = json.loads((REPO / "runs/_analysis/rgr/decision.json").read_text())
    say(f"decision: {dec}")
    arm = dec.get("arm")
    if arm:
        closed = [str(Path(p).parent.relative_to(REPO)) for p in glob.glob(str(REPO / "data/recovery/rgr_sweep/s*/p*/CLOSED"))]
        dirs = {a: [f"{c}/{a}" for c in closed] for a in (arm, "single")}
        n = min(208, n_traj(dirs[arm]), n_traj(dirs["single"]))
        say(f"distill {arm} vs single, {n} trajectories each, parts {closed}")
        cks = {}

        def sft_eval(a: str) -> None:
            tag = f"sft_rgr_{a}"
            rc = run_blocking(f"rv_rgr_sft_{a}"[:40], ["retryvla.retry_oom", "--min_free_gb", "30", "retryvla.train.sft",
                              "configs/exp/rgr_sft.yaml", f"tag={tag}", f"train.recovery_datasets=[{','.join(dirs[a])}]",
                              f"train.max_recovery_trajectories={n}"])
            if rc:
                return
            ck = sorted(glob.glob(str(REPO / f"runs/*_RGR_{tag}/checkpoints/006000/pretrained_model")), key=os.path.getmtime)[-1]
            cks[a] = ck
            for k, args in (("rec", ["retryvla.eval.eval_recover", "configs/exp/m0_baseline_spatial.yaml", "milestone=RGR",
                                     f"recover_tag={tag}_recover_test", f"policy.path={ck}", "recover_eval.splits=[test]"]),
                            ("nom", ["retryvla.eval.eval_nominal", "configs/exp/m0_baseline_spatial.yaml", "milestone=RGR",
                                     f"nominal_tag={tag}_nominal", f"policy.path={ck}"])):
                run_blocking(f"rv_rgr_{k}_{a}"[:40], args + ["eval.n_workers=8", "eval.save_videos=none",
                                                            "eval.worker_retries=2"])

        ts = [threading.Thread(target=sft_eval, args=(a,)) for a in (arm, "single")]
        for t in ts:
            t.start()
            time.sleep(120)
        for t in ts:
            t.join()
        say(f"distill done: {cks}")
    kill_shell()
    say("PHASE 2 DONE")


def main() -> None:
    tasks = {"rv_rgr_head": ["retryvla.geom.head", "--steps", "6000"]}
    for k in SHARDS:
        tasks[f"rv_rgr_sw{k}"] = ["retryvla.retry_oom", "--min_free_gb", "20", "retryvla.search.rgr_search",
                                  "configs/exp/rgr_sweep.yaml", f"m5b.shard={k}", f"eval.n_workers={WORKERS}"]
    st = {n: {"log": None, "done": False, "tries": 0} for n in tasks}
    if os.environ.get("RGR_SKIP_HEAD"):
        st["rv_rgr_head"]["done"] = True
    say(f"start: {list(tasks)} par {MAX_PAR} workers {WORKERS}")
    while True:
        live = live_sessions()
        for n, s in st.items():
            if s["done"] or s["log"] is None:
                continue
            rc = rc_of(s["log"])
            if rc is not None:
                say(f"{n} ended rc={rc}")
                release(n)
                s["log"] = None
                if rc == 0:
                    s["done"] = True
                    if n.startswith("rv_rgr_sw"):
                        analyze()
                else:
                    s["tries"] += 1
                continue
            idle = (time.time() - os.path.getmtime(s["log"])) / 60 if os.path.exists(s["log"]) else 0.0
            gone = live is not None and n not in live and idle > 5
            if idle > STALL_MIN or gone:
                say(f"WATCHDOG {n}: {'session gone' if gone else f'stalled {idle:.0f} min'} -> kill + relaunch")
                if live and n in live:
                    kill(n)
                release(n)
                s["log"] = None
                s["tries"] += 1
        labels_ok = len(glob.glob(str(REPO / "data/geom/labels_collect_v1/*.pt"))) >= 144
        running = sum(1 for s in st.values() if s["log"] is not None)
        for n, s in st.items():
            if s["done"] or s["log"] is not None or s["tries"] >= MAX_TRIES or live is None:
                continue
            if n == "rv_rgr_head":
                if not labels_ok or running >= MAX_PAR:
                    continue
            else:
                if not st["rv_rgr_head"]["done"] or running >= MAX_PAR:
                    continue
            if live and n in live:  # adopt (chain restart)
                logs = sorted(glob.glob(str(REPO / f"runs/_logs/{n}_*.log")), key=os.path.getmtime)
                s["log"] = logs[-1] if logs else None
                say(f"adopted {n}: {s['log']}")
            else:
                s["log"] = launch(n, tasks[n])
                if s["log"] is None:
                    s["tries"] += 1
            running += s["log"] is not None
        if all(s["done"] for s in st.values()):
            analyze()
            say("ALL SWEEPS DONE")
            phase2()
            return
        if all(s["done"] or s["tries"] >= MAX_TRIES for s in st.values()):
            say(f"GIVING UP: {[(n, s['tries']) for n, s in st.items() if not s['done']]}")
            return
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
