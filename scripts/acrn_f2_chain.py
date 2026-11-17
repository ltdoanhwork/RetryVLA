#!/usr/bin/env python
"""ACR-N F2 chain (PLAN "ACR-N follow-ups", 2026-11-17). Local machine 8b889199 (GPUs 1-3 only; GPU0 = the user's LAWAM
server, never used). Lanes (launch_remote.sh local fast path, claims 8b889199:<gpu>):
  rv_acrnf2_<l>   acr_search tier N mode f2 (head-pose grasp_slow / hover / grasp_firm x 8), 2 lanes
  rv_acrnsf_<k>   rgr_search arm single, fresh seeds (configs/exp/acrn_single_fresh.yaml), M5b shards 0-3
watchdog: rc != 0 or log idle > STALL_MIN -> kill exact session, relaunch (both jobs resume from their per-state files).
Then scripts/analyze_acrn_f2.py. Console: runs/_logs/acrn_f2_chain_console.log
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
SH = "8b889199"
STALL_MIN, POLL_S, MAX_TRIES = 45, 120, 4
W_F2, W_SF = int(os.environ.get("F2_WORKERS", "6")), int(os.environ.get("SF_WORKERS", "3"))
TASKS = {}
for l, g in enumerate((3, 1)):
    TASKS[f"rv_acrnf2_{l}"] = (g, ["retryvla.retry_oom", "--min_free_gb", "10", "retryvla.search.acr_search",
                                   "configs/exp/acrn.yaml", "acr.mode=f2", f"acr.lane={l}", "acr.num_lanes=2",
                                   f"eval.n_workers={W_F2}"])
for k, g in zip(range(4), (2, 2, 3, 1)):
    TASKS[f"rv_acrnsf_{k}"] = (g, ["retryvla.retry_oom", "--min_free_gb", "10", "retryvla.search.rgr_search",
                                   "configs/exp/acrn_single_fresh.yaml", f"m5b.shard={k}", f"eval.n_workers={W_SF}"])


def say(msg):
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), "[acrn_f2_chain]", msg, flush=True)


def live():
    out = subprocess.run([TMUX, "ls"], capture_output=True, text=True).stdout
    return {ln.split(":")[0] for ln in out.splitlines()}


def launch(n, g, args):
    p = subprocess.run([str(REPO / "scripts/shells/launch_remote.sh"), SH, str(g), n, *args], capture_output=True,
                       text=True, env=ENV, cwd=REPO, timeout=600)
    m = re.search(r"LAUNCHED \S+ gpu=\d+ log=(\S+)", p.stdout + p.stderr)
    say(f"launch {n} gpu {g}: {m.group(1) if m else (p.stdout + p.stderr).strip()[-200:]}")
    return m.group(1) if m else None


def main():
    st = {n: {"log": None, "rc": None, "tries": 0} for n in TASKS}
    while any(s["rc"] is None for s in st.values()):
        lv = live()
        for n, s in st.items():
            if s["rc"] is not None:
                continue
            g, args = TASKS[n]
            if s["log"] is None:
                if s["tries"] >= MAX_TRIES:
                    s["rc"] = 99
                    continue
                if n in lv:
                    subprocess.run([TMUX, "kill-session", "-t", f"={n}"])
                s["log"] = launch(n, g, args)
                s["tries"] += 1
                time.sleep(30)
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
            idle = (time.time() - os.path.getmtime(s["log"])) / 60 if os.path.exists(s["log"]) else 0
            if idle > STALL_MIN or (n not in lv and idle > 5):
                say(f"WATCHDOG {n}: idle {idle:.0f} min -> relaunch")
                if n in lv:
                    subprocess.run([TMUX, "kill-session", "-t", f"={n}"])
                release(n)
                s["log"] = None
        time.sleep(POLL_S)
    say(f"all done: { {n: s['rc'] for n, s in st.items()} }")
    subprocess.run(["bash", "-c", "source env/activate.sh >/dev/null 2>&1; python scripts/analyze_acrn_f2.py "
                    "> runs/_logs/acrn_f2_analyze.log 2>&1"], cwd=REPO, timeout=3600)
    say("ACRN F2 CHAIN DONE")


if __name__ == "__main__":
    main()
