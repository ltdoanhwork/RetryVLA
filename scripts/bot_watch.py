"""Local watchdog for the Back on Track Goal chain on shell 92927808 (tmux rv_bot_watch on the agent box).

python scripts/bot_watch.py
Every 10 min: read-only probe of the remote tmux; if rv_bot_goal is gone and its newest log has no CHAIN_DONE,
relaunch it (<= 3 times; the chain is resumable via done markers). Re-asserts its claim in shells.json each poll
(another chain's finish() releases claims by shell prefix). When CHAIN_DONE: run scripts/analyze_bot_goal.py, write a
NOTES line and stop. Never kills the shell (the agent does that after the secondary queue).
"""

from __future__ import annotations

import glob
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SID = "92927808-d6f2-45a1-ba15-6eb7315d7531"
SHORT = SID[:8]
LANE = "rv_bot_goal"
SH = str(REPO / "scripts/shells")


def log(**kw):
    kw["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(json.dumps(kw), flush=True)


def det(*args, timeout=240) -> str:
    try:
        p = subprocess.run(["env", "-u", "LD_LIBRARY_PATH", "det", "shell", "open", SID, "--", *args],
                           capture_output=True, text=True, timeout=timeout, cwd="/tmp")
        return p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def claim():
    p = REPO / "scripts/shells/shells.json"
    try:
        d = json.loads(p.read_text())
        k = f"{SHORT}:0:{LANE}"
        if k not in d["claims"]:
            d["claims"][k] = {"tmux": LANE, "args": "scripts/bot_goal_chain.sh (BoT / ForkCast Goal arms)",
                              "since": time.strftime("%Y-%m-%dT%H:%M:%S")}
            p.write_text(json.dumps(d, indent=1))
    except (OSError, ValueError) as e:
        log(event="claim_failed", err=str(e)[:100])


def chain_done() -> bool:
    logs = sorted(glob.glob(str(REPO / f"runs/_logs/{LANE}_*.log")))
    return bool(logs) and "CHAIN_DONE" in Path(logs[-1]).read_text(errors="ignore")


def main():
    relaunch = 0
    while True:
        claim()
        if chain_done():
            subprocess.call(["bash", "-lc", f"cd {REPO} && source env/activate.sh >/dev/null 2>&1 && "
                             "python scripts/analyze_bot_goal.py > runs/bot_goal/analyze.log 2>&1"])
            with open(REPO / "experiments/NOTES.md", "a") as f:
                f.write(f"- {time.strftime('%Y-%m-%d %H:%M UTC')}: BoT / ForkCast Goal chain DONE on {SHORT} "
                        "(bot_watch); table runs/bot_goal/main.md (reading + RESULTS by the agent).\n")
            log(event="done")
            return
        out = det("bash", f"{SH}/probe_tmux.sh")
        tl = [l for l in out.splitlines() if l.startswith("TMUX")]
        if not tl:
            log(event="probe_failed", out=out[-200:])
        elif LANE not in tl[0].split()[1:]:
            if chain_done():
                continue
            if relaunch >= 3:
                log(event="give_up")
                return
            relaunch += 1
            out2 = det("bash", f"{SH}/tmux_start.sh", LANE, "scripts/bot_goal_chain.sh")
            log(event="relaunch", n=relaunch, out=[l for l in out2.splitlines() if "LAUNCHED" in l or "EXISTS" in l])
        else:
            log(event="poll", up=True, gpus=[l[5:] for l in out.splitlines() if l.startswith("GPUS")])
        time.sleep(600)


if __name__ == "__main__":
    main()
