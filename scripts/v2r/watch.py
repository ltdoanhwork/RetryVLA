"""V2R local watchdog (agent box, tmux rv_v2r_watch; PLAN "V2R"). Every 10 min, for each V2R shell:
- shell RUNNING but no `rv_v2r_chain` session and runs/v2r/ALL_DONE absent -> start the chain (<= 6 relaunches / shell);
- chain heartbeat (runs/v2r/hb_<host>) older than 45 min -> kill the exact session and relaunch;
- shell 2 only: when it first appears, start search lanes for the not-yet-started R1 shards (search_lane_b.sh);
- shell idle (runs/v2r/idle_<host> older than 2 h, no task running there) or ALL_DONE -> det shell kill
  (unless scripts/shells/keep_<id>), claims released, shells.json updated.
Log: runs/v2r/watch.jsonl.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V = REPO / "runs/v2r"
SHELLS = {"ea7f9ee3": "ea7f9ee3-a123-4dd6-9b88-4791f00c6f48", "f72a5ddd": "f72a5ddd-a436-428b-a803-f460009c84b4"}
TM = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin/tmux"
S2 = REPO / "scripts/v2r"


def log(**kw):
    kw["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(json.dumps(kw), flush=True)
    try:
        with open(V / "watch.jsonl", "a") as f:
            f.write(json.dumps(kw) + "\n")
    except OSError:
        pass


def det(*args, timeout=240):
    try:
        p = subprocess.run(["env", "-u", "LD_LIBRARY_PATH", "det", *args], capture_output=True, text=True,
                           timeout=timeout, cwd="/tmp")
        return (p.stdout + p.stderr).replace("\r", "")
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def state(sid):
    for l in det("shell", "list").splitlines():
        if SHELLS[sid] in l:
            return l.split("|")[3].strip()
    return "GONE"


def probe(sid):
    out = det("shell", "open", SHELLS[sid], "--", "bash", str(S2 / "probe.sh"), timeout=240)
    d = {}
    for l in out.splitlines():
        for k in ("HOST", "TMUX", "GPUS", "LOAD"):
            if l.startswith(k + " "):
                d[k] = l[len(k) + 1:].strip()
    return d


def run_on(sid, script, *args):
    return det("shell", "open", SHELLS[sid], "--", "bash", str(S2 / script), *args, timeout=240)


def release(sid):
    p = REPO / "scripts/shells/shells.json"
    d = json.load(open(p))
    for k in [k for k in d.get("claims", {}) if k.startswith(sid)]:
        d["claims"].pop(k)
    d.setdefault("gone", {})[sid] = f"V2R shell killed {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} (idle / done)"
    d.get("allow_only", {}).pop(sid, None)
    tmp = p.with_name(".shells.json.v2r.tmp")
    json.dump(d, open(tmp, "w"), indent=1)
    os.replace(tmp, p)


def main():
    relaunch = {s: 0 for s in SHELLS}
    lanes_b = (V / "lanes_b_started").exists()
    killed = set()
    while True:
        for sid in SHELLS:
            if sid in killed:
                continue
            st = state(sid)
            if st != "RUNNING":
                log(sid=sid, state=st)
                continue
            pr = probe(sid)
            if "TMUX" not in pr:
                log(sid=sid, event="probe_failed")
                continue
            host = pr.get("HOST", "?")
            sess = pr["TMUX"].split()
            hb = V / f"hb_{host}"
            all_done = (V / "ALL_DONE").exists()
            if sid == "f72a5ddd" and not lanes_b and not all_done:
                out = run_on(sid, "start_lanes_b.sh")
                log(sid=sid, event="lanes_b", out=[l for l in out.splitlines() if "LAUNCHED" in l or "EXISTS" in l])
                (V / "lanes_b_started").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
                lanes_b = True
            if "rv_v2r_chain" in sess and hb.exists() and time.time() - hb.stat().st_mtime > 2700:
                log(sid=sid, event="stall_kill_chain")
                run_on(sid, "kill_chain.sh")
                sess.remove("rv_v2r_chain")
            if "rv_v2r_chain" not in sess and not all_done and relaunch[sid] < 6:
                out = run_on(sid, "start_chain.sh")
                relaunch[sid] += 1
                log(sid=sid, event="start_chain", n=relaunch[sid], out=[l for l in out.splitlines() if "LAUNCHED" in l])
            idle = V / f"idle_{host}"
            idle_long = idle.exists() and time.time() - float(idle.read_text() or 0) > 7200
            busy_other = [s for s in sess if s.startswith("rv_v2r_s") and not s.startswith("rv_v2r_smoke")]
            if (all_done or idle_long) and not busy_other and not (REPO / f"scripts/shells/keep_{sid}").exists():
                if all_done or sid == "f72a5ddd":   # the primary shell is kept until ALL_DONE
                    log(sid=sid, event="kill_shell", all_done=all_done, idle_long=idle_long)
                    run_on(sid, "kill_chain.sh")
                    det("shell", "kill", SHELLS[sid])
                    release(sid)
                    killed.add(sid)
            log(sid=sid, event="poll", host=host, sess=sess, gpus=pr.get("GPUS"), load=pr.get("LOAD"),
                hb_age=round(time.time() - hb.stat().st_mtime) if hb.exists() else None)
        for req in sorted(V.glob("register_req_*")):   # chain stage registration requests (git works here)
            st = req.name[len("register_req_"):]
            if not (V / f"register_done_{st}").exists():
                rc = subprocess.call(["bash", "-lc", f"cd {REPO} && source env/activate.sh >/dev/null 2>&1 && "
                                      f"python scripts/v2r/register_stage.py {st} >> runs/v2r/register.log 2>&1"])
                log(event="registered", stage=st, rc=rc)
        if len(killed) == len(SHELLS):
            log(event="watch_done")
            return
        time.sleep(600)


if __name__ == "__main__":
    main()
