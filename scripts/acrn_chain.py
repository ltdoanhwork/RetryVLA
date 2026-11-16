#!/usr/bin/env python
"""ACR-N chain (user pivot 2026-11-16; design PLAN "ACR-N"). Agent box, CPU only, local tmux `acrn_chain`; launches on
the two agent-requested USER-account shells (cb2b6ab1 = vla_RetryVLA_DoanhLT_acr2, f02ea5c5 = ..._acr2b) via
scripts/shells/launch_remote.sh (claims in shells.json):

  1. rv_acrn_c<l>   acr_search tier N collect, 4 lanes (2 per shell; per-state files, resumable)
  2. analyze_acrn.py --write-decisions -> runs/_analysis/acrn
  3. rv_acrn_pp<l>  postprobe: re-run the evi_probe / random_probe trial sequences from the TRUE post-probe state
  4. analyze again; det shell kill both shells unless ACRN_KEEP_SHELL=1
watchdog: log not written for STALL_MIN min / session gone -> kill exact session, release claim, relaunch.
Console: runs/_logs/acrn_chain_console.log
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
SHELLS = {"cb2b6ab1": "cb2b6ab1-0ffa-43fc-a595-dfe07138c6cc", "f02ea5c5": "f02ea5c5-ce1a-4538-8795-042482822093"}
LANES = [("cb2b6ab1", 0), ("cb2b6ab1", 0), ("f02ea5c5", 0), ("f02ea5c5", 0)]
WORKERS = int(os.environ.get("ACRN_WORKERS", "12"))
STALL_MIN, POLL_S, MAX_TRIES = 45, 180, 4
CFG = "configs/exp/acrn.yaml"


def say(msg: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), "[acrn_chain]", msg, flush=True)


def live_sessions(sh: str) -> set[str] | None:
    try:
        out = subprocess.run(["det", "shell", "open", SHELLS[sh], "--", TMUX, "ls"], capture_output=True, text=True,
                             timeout=180, env=ENV).stdout.replace("\r", "")
    except Exception:  # noqa: BLE001
        return None
    return {ln.split(":")[0] for ln in out.splitlines() if ln.startswith("rv_")}


def kill(sh: str, sess: str) -> None:
    assert sess.startswith("rv_acrn_")
    subprocess.run(["det", "shell", "open", SHELLS[sh], "--", TMUX, "kill-session", "-t", f"={sess}"],
                   capture_output=True, text=True, timeout=180, env=ENV)


def launch(sh: str, gpu: int, sess: str, args: list[str]) -> str | None:
    p = subprocess.run([str(REPO / "scripts/shells/launch_remote.sh"), sh, str(gpu), sess, *args], capture_output=True,
                       text=True, env=ENV, cwd=REPO, timeout=600)
    out = (p.stdout + p.stderr).replace("\r", "")
    m = re.search(r"LAUNCHED \S+ gpu=\d+ log=(\S+)", out)
    if not m:
        say(f"launch {sess} failed: {out.strip().splitlines()[-1:] if out.strip() else p.returncode}")
        return None
    say(f"launched {sess} on {sh}: {m.group(1)}")
    return m.group(1)


def run_group(tasks: dict[str, tuple[str, int, list[str]]]) -> dict[str, int]:
    st = {n: {"log": None, "rc": None, "tries": 0} for n in tasks}
    while any(s["rc"] is None for s in st.values()):
        live = {sh: live_sessions(sh) for sh in SHELLS}
        for n, s in st.items():
            sh, gpu, args = tasks[n]
            lv = live[sh]
            if s["rc"] is not None:
                continue
            if s["log"] is None:
                if s["tries"] >= MAX_TRIES:
                    s["rc"] = 99
                    continue
                if lv is None:
                    continue
                if n in lv:
                    kill(sh, n)
                s["log"] = launch(sh, gpu, n, args)
                s["tries"] += 1
                time.sleep(20)
                continue
            rc = rc_of(s["log"])
            if rc is not None:
                release(n)
                say(f"{n} ended rc={rc}")
                s["rc"] = 0 if rc == 0 else None
                if rc != 0:
                    s["log"] = None
                continue
            idle = (time.time() - os.path.getmtime(s["log"])) / 60 if os.path.exists(s["log"]) else 0.0
            if idle > STALL_MIN or (lv is not None and n not in lv and idle > 5):
                say(f"WATCHDOG {n}: stalled / gone ({idle:.0f} min) -> kill + relaunch")
                if lv and n in lv:
                    kill(sh, n)
                release(n)
                s["log"] = None
        time.sleep(POLL_S)
    return {n: s["rc"] for n, s in st.items()}


def analyze(extra: str = "") -> None:
    subprocess.run(["bash", "-c", "source env/activate.sh >/dev/null 2>&1; python scripts/analyze_acrn.py "
                    f"{extra} > runs/_logs/acrn_analyze.log 2>&1"], cwd=REPO, timeout=7200)
    say(f"analysis written (runs/_analysis/acrn/acrn.md) {extra}")


def main() -> None:
    nl = len(LANES)
    say(f"start: {nl} lanes x {WORKERS} workers")
    if not os.environ.get("ACRN_SKIP_COLLECT"):
        tasks = {f"rv_acrn_c{l}": (sh, g, ["retryvla.retry_oom", "--min_free_gb", "10", "retryvla.search.acr_search",
                                           CFG, f"acr.lane={l}", f"acr.num_lanes={nl}", f"eval.n_workers={WORKERS}"])
                 for l, (sh, g) in enumerate(LANES)}
        say(f"collect: {run_group(tasks)}")
    analyze("--write-decisions")
    dec = REPO / "runs/_analysis/acrn/decisions.json"
    if dec.exists():
        tasks = {f"rv_acrn_pp{l}": (sh, g, ["retryvla.retry_oom", "--min_free_gb", "10", "retryvla.search.acr_search",
                                            CFG, "acr.mode=postprobe", f"acr.decisions={dec.relative_to(REPO)}",
                                            f"acr.lane={l}", f"acr.num_lanes={nl}", f"eval.n_workers={WORKERS}"])
                 for l, (sh, g) in enumerate(LANES)}
        say(f"postprobe: {run_group(tasks)}")
    analyze()
    if not os.environ.get("ACRN_KEEP_SHELL"):
        for sh, full in SHELLS.items():
            p = subprocess.run(["det", "shell", "kill", full], capture_output=True, text=True, env=ENV, timeout=300)
            say(f"det shell kill {sh}: rc {p.returncode}")
    say("ACRN CHAIN DONE")


if __name__ == "__main__":
    main()
