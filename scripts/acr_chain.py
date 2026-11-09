#!/usr/bin/env python
"""ACR chain (user 2026-11-09). Agent box, CPU only, local tmux `acr_chain`; launches on the agent-requested
user-account shell e26be342 (vla_RetryVLA_DoanhLT_acr) via scripts/shells/launch_remote.sh (claims in shells.json):

  1. rv_acr_B<l>   acr_search tier B (25 states), ACR_LANES lanes in parallel (state files are resumable)
  2. rv_acr_A<l>   acr_search tier A (60 states), same lanes
  3. analyze_acr.py (proprio / tactile / oracle feature sets) -> runs/_analysis/acr
  4. rv_acr_pp<T>  postprobe re-run of the evi_probe / no_probe allocations of tier T from the TRUE post-probe state
  5. analyze again; keep the shell unless ACR_KILL_SHELL=1 (follow-ups of the independent review need it)
watchdog: log not written for STALL_MIN min / session gone -> kill exact session, release claim, relaunch.
Console: runs/_logs/acr_chain_console.log
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from xwm_chain import ENV, TMUX, rc_of, release  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SH, SH_FULL, GPU = "e26be342", "e26be342-358e-4fe7-b980-37f81461b60f", 0
LANES = int(os.environ.get("ACR_LANES", "2"))
WORKERS = int(os.environ.get("ACR_WORKERS", "8"))
TIERS = os.environ.get("ACR_TIERS", "B,A").split(",")
STALL_MIN, POLL_S, MAX_TRIES = 45, 180, 4
CFG = "configs/exp/acr_poc.yaml"


def say(msg: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), "[acr_chain]", msg, flush=True)


def live_sessions() -> set[str] | None:
    try:
        out = subprocess.run(["det", "shell", "open", SH_FULL, "--", TMUX, "ls"], capture_output=True, text=True,
                             timeout=180, env=ENV).stdout.replace("\r", "")
    except Exception:  # noqa: BLE001
        return None
    return {ln.split(":")[0] for ln in out.splitlines() if ln.startswith("rv_")}


def kill(sess: str) -> None:
    assert sess.startswith("rv_acr_")
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


def run_group(tasks: dict[str, list[str]]) -> dict[str, int]:
    """Run all tasks in parallel with watchdog + retries; returns final rc per task."""
    st = {n: {"log": None, "rc": None, "tries": 0} for n in tasks}
    while any(s["rc"] is None for s in st.values()):
        live = live_sessions()
        for n, s in st.items():
            if s["rc"] is not None:
                continue
            if s["log"] is None:
                if s["tries"] >= MAX_TRIES:
                    s["rc"] = 99
                    continue
                if live is None:
                    continue
                if n in live:
                    kill(n)
                s["log"] = launch(n, tasks[n])
                s["tries"] += 1
                continue
            rc = rc_of(s["log"])
            if rc is not None:
                release(n)
                say(f"{n} ended rc={rc}")
                if rc == 0:
                    s["rc"] = 0
                else:
                    s["log"] = None
                continue
            idle = (time.time() - os.path.getmtime(s["log"])) / 60 if os.path.exists(s["log"]) else 0.0
            if idle > STALL_MIN or (live is not None and n not in live and idle > 5):
                say(f"WATCHDOG {n}: stalled / gone ({idle:.0f} min) -> kill + relaunch")
                if live and n in live:
                    kill(n)
                release(n)
                s["log"] = None
        time.sleep(POLL_S)
    return {n: s["rc"] for n, s in st.items()}


def analyze() -> None:
    for fs in ("proprio", "tactile", "oracle"):
        subprocess.run(["bash", "-c", f"source env/activate.sh >/dev/null 2>&1; python scripts/analyze_acr.py --fset {fs}"
                        " > /dev/null 2> runs/_logs/acr_analyze_err.log"], cwd=REPO, timeout=3600)
    say("analysis written: runs/_analysis/acr/acr_<fset>.md")


def main() -> None:
    say(f"start: tiers {TIERS}, {LANES} lanes x {WORKERS} workers")
    for t in TIERS:
        tasks = {f"rv_acr_{t}{l}": ["retryvla.retry_oom", "--min_free_gb", "10", "retryvla.search.acr_search", CFG,
                                    f"acr.tier={t}", f"acr.lane={l}", f"acr.num_lanes={LANES}",
                                    f"eval.n_workers={WORKERS}"] for l in range(LANES)}
        say(f"tier {t}: {run_group(tasks)}")
        analyze()
    for t in TIERS:
        dec = REPO / f"runs/_analysis/acr/decisions_{t}.json"
        if not dec.exists():
            say(f"no decisions for tier {t}")
            continue
        tasks = {f"rv_acr_pp{t}{l}": ["retryvla.retry_oom", "--min_free_gb", "10", "retryvla.search.acr_search", CFG,
                                      f"acr.tier={t}", "acr.mode=postprobe", f"acr.decisions={dec.relative_to(REPO)}",
                                      f"acr.lane={l}", f"acr.num_lanes={LANES}", f"eval.n_workers={WORKERS}"]
                 for l in range(LANES)}
        say(f"postprobe {t}: {run_group(tasks)}")
    analyze()
    if os.environ.get("ACR_KILL_SHELL"):
        p = subprocess.run(["det", "shell", "kill", SH_FULL], capture_output=True, text=True, env=ENV, timeout=300)
        say(f"det shell kill {SH}: rc {p.returncode}")
    say("ACR CHAIN DONE")


if __name__ == "__main__":
    main()
