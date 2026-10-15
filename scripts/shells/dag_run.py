#!/usr/bin/env python3
"""Dependency-driven job runner over a FIXED set of GPU slots (borrowed Determined shells).

python3 scripts/shells/dag_run.py <dag.json> --slots aa1cf25d:0 aa1cf25d:1 [--dry]

dag.json: {"name": "m5", "tasks": [{"name": "r1_s0", "module": "...", "args": [...], "deps": []}, ...]}
Arg tokens resolved at launch time:
  @glob:<pattern>      -> newest path matching the glob (e.g. a checkpoint dir)
  @ntraj:<glob>:<cap>  -> min(cap, total trajectories in all matching recovery datasets)
A task is done when its log has `RV_DONE rc=0`; failed on `rc!=0` (runner stops and prints FAILED).
Claims are released by SESSION name (never by slot). Stop: touch runs/_logs/<name>_STOP.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SH = REPO / "scripts" / "shells"
os.environ.pop("LD_LIBRARY_PATH", None)


def say(log: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(log, "a") as f:
        f.write(line + "\n")


def logs(name: str) -> list[str]:
    return sorted(glob.glob(str(REPO / "runs" / "_logs" / f"rv_{name}_*.log")))


def status(name: str) -> str:
    st = "todo"
    for f in logs(name):
        txt = open(f, errors="ignore").read()
        if "RV_DONE rc=0" in txt:
            return "done"
        if "RV_DONE rc=" in txt:
            st = "failed"
    return st


def claims() -> dict:
    return json.load(open(SH / "shells.json")).get("claims", {})


def running(name: str) -> bool:
    return any(v["tmux"] == f"rv_{name}" for v in claims().values())


def release(name: str) -> None:
    for k, v in claims().items():
        if v["tmux"] == f"rv_{name}":
            subprocess.run([str(SH / "remote_ctl.sh"), "stop", k], capture_output=True)


def free_slots(allowed: list[str]) -> list[str]:
    try:
        out = subprocess.run(["python3", str(SH / "find_free.py"), "--json"], capture_output=True, text=True,
                             timeout=300).stdout
        free = {f"{s['short']}:{s['gpu']}" for s in json.loads(out)}
    except Exception:
        return []
    used = set(claims())
    return [s for s in allowed if s in free and s not in used]


def resolve(arg: str) -> str:
    if "@glob:" in arg:
        pre, pat = arg.split("@glob:", 1)
        m = sorted(glob.glob(str(REPO / pat)))
        if not m:
            raise FileNotFoundError(pat)
        return pre + os.path.relpath(m[-1], REPO)
    if "@ntraj:" in arg:
        pre, rest = arg.split("@ntraj:", 1)
        pat, cap = rest.rsplit(":", 1)
        n = sum(1 for f in glob.glob(str(REPO / pat / "retryvla" / "trajectories.jsonl")) for _ in open(f))
        return pre + str(min(int(cap), n))
    return arg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dag")
    ap.add_argument("--slots", nargs="+", required=True)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--poll", type=int, default=120)
    a = ap.parse_args()
    dag = json.load(open(a.dag))
    name = dag["name"]
    log = REPO / "runs" / "_logs" / f"dag_{name}.log"
    stop = REPO / "runs" / "_logs" / f"{name}_STOP"
    stop.unlink(missing_ok=True)
    tasks = {t["name"]: t for t in dag["tasks"]}
    say(log, f"DAG {name}: {len(tasks)} tasks on {a.slots}")
    while True:
        if stop.exists():
            say(log, "stopped by STOP file")
            return
        st = {n: status(n) for n in tasks}
        for n, s in st.items():
            if s == "done" and running(n):
                release(n)
        if any(s == "failed" and not running(n) for n, s in st.items()):
            bad = [n for n, s in st.items() if s == "failed"]
            say(log, f"FAILED {bad}")
            print('AGENT_LOOP_WAKE_dag {"prompt":"DAG task failed"}', flush=True)
            sys.exit(1)
        if all(s == "done" for s in st.values()):
            say(log, "ALL DONE")
            print('AGENT_LOOP_WAKE_dag {"prompt":"DAG finished"}', flush=True)
            return
        ready = [n for n, t in tasks.items() if st[n] == "todo" and not running(n)
                 and all(st[d] == "done" for d in t.get("deps", []))]
        if ready:
            for slot in free_slots(a.slots):
                if not ready:
                    break
                n = ready.pop(0)
                t = tasks[n]
                try:
                    args = [resolve(x) for x in t["args"]]
                except Exception as e:
                    say(log, f"cannot resolve args of {n}: {e}")
                    continue
                cmd = [str(SH / "launch_remote.sh"), *slot.split(":"), n, t["module"], *args]
                if a.dry:
                    say(log, "DRY " + " ".join(cmd[1:]))
                    continue
                out = subprocess.run(cmd, capture_output=True, text=True).stdout
                ok = next((l for l in out.splitlines() if "LAUNCHED" in l or "REFUSED" in l or "fail" in l), out[-200:])
                say(log, f"launch {n} on {slot}: {ok}")
            if a.dry:
                return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
