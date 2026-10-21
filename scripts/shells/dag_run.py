#!/usr/bin/env python3
"""Dependency-driven job runner over a FIXED set of GPU slots (borrowed Determined shells).

python3 scripts/shells/dag_run.py <dag.json> --slots aa1cf25d:0 aa1cf25d:1 [--dry]

dag.json: {"name": "m5", "tasks": [{"name": "r1_s0", "module": "...", "args": [...], "deps": []}, ...]}
Arg tokens resolved at launch time:
  @glob:<pattern>      -> newest path matching the glob (e.g. a checkpoint dir)
  @ntraj:<glob>:<cap>  -> min(cap, total trajectories in all matching recovery datasets)
  @ntrajmin:<cap>:<g1>,<g2>,... -> min(cap, min over groups of trajectories in that group's datasets)
Slots: <shell8>:<gpu> (borrowed Determined shell) or local:<gpu>:<lane> (this machine; several lanes may
share one GPU, e.g. SFT + a CPU-bound eval on one 80 GB A100).
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


TMUX = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin/tmux"


def set_claim(key: str, sess: str, args: str) -> None:
    p = SH / "shells.json"
    d = json.load(open(p))
    d.setdefault("claims", {})[key] = {"tmux": sess, "args": args, "since": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(d, open(p, "w"), indent=1)


def pop_claim(key: str) -> None:
    p = SH / "shells.json"
    d = json.load(open(p))
    d.get("claims", {}).pop(key, None)
    json.dump(d, open(p, "w"), indent=1)


def release(name: str) -> None:
    for k, v in claims().items():
        if v["tmux"] == f"rv_{name}":
            if k.startswith("local:"):
                subprocess.run([TMUX, "kill-session", "-t", f"={v['tmux']}"], capture_output=True)
                pop_claim(k)
            else:
                subprocess.run([str(SH / "remote_ctl.sh"), "stop", k], capture_output=True)


def local_alive(sess: str) -> bool:
    return subprocess.run([TMUX, "has-session", "-t", f"={sess}"], capture_output=True).returncode == 0


def free_slots(allowed: list[str]) -> list[str]:
    used = claims()
    for k, v in list(used.items()):  # local lane whose tmux died without RV_DONE (e.g. killed): free it
        if k.startswith("local:") and k in allowed and not local_alive(v["tmux"]):
            pop_claim(k)
    used = set(claims())
    out = [s for s in allowed if s.startswith("local:") and s not in used]
    remote = [s for s in allowed if not s.startswith("local:")]
    if remote:
        try:
            js = subprocess.run(["python3", str(SH / "find_free.py"), "--json"], capture_output=True, text=True,
                                timeout=300).stdout
            free = {f"{s['short']}:{s['gpu']}" for s in json.loads(js)}
        except Exception:
            free = set()
        out += [s for s in remote if s in free and s not in used]
    return out


def usable(d: str) -> bool:
    """Resumable datasets live in .../p<k>/<arm>; only parts marked CLOSED are complete."""
    par = Path(d).parent
    return not (par.name.startswith("p") and par.name[1:].isdigit() and not (par / "CLOSED").exists())


def ntraj(pat: str) -> int:
    return sum(1 for d in glob.glob(str(REPO / pat)) if usable(d)
               for f in glob.glob(d + "/retryvla/trajectories.jsonl") for _ in open(f))


def resolve(arg: str) -> str:
    if "@glob:" in arg:
        pre, pat = arg.split("@glob:", 1)
        m = sorted(glob.glob(str(REPO / pat)))
        if not m:
            raise FileNotFoundError(pat)
        return pre + os.path.relpath(m[-1], REPO)
    if "@ntrajmin:" in arg:
        pre, rest = arg.split("@ntrajmin:", 1)
        cap, groups = rest.split(":", 1)
        return pre + str(min([int(cap)] + [ntraj(g) for g in groups.split(",")]))
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
    allow = json.load(open(SH / "shells.json")).get("allow_only")
    if allow:
        bad = [s for s in a.slots if not s.startswith("local:") and s.split(":")[0] not in allow]
        if bad:
            sys.exit(f"refusing slots outside shells.json:allow_only: {bad}")
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
        for k, v in list(claims().items()):  # local lanes whose tmux died (crash / kill) must not block their task
            if k.startswith("local:") and k in a.slots and not local_alive(v["tmux"]):
                pop_claim(k)
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
                if slot.startswith("local:"):
                    cmd = ["bash", str(SH / "remote_tmux.sh"), f"rv_{n}", slot.split(":")[1], t["module"], *args]
                else:
                    cmd = [str(SH / "launch_remote.sh"), *slot.split(":"), n, t["module"], *args]
                if a.dry:
                    say(log, "DRY " + " ".join(cmd[1:]))
                    continue
                out = subprocess.run(cmd, capture_output=True, text=True).stdout
                if slot.startswith("local:") and "LAUNCHED" in out:
                    set_claim(slot, f"rv_{n}", " ".join([t["module"], *args]))
                ok = next((l for l in out.splitlines() if "LAUNCHED" in l or "REFUSED" in l or "fail" in l), out[-200:])
                say(log, f"launch {n} on {slot}: {ok}")
            if a.dry:
                return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
