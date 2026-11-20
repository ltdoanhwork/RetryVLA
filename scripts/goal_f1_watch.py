"""Local watcher for the 2610.05151 follow-up F1 on borrowed shell 63e0f914 (vla_roboworm_1, owner's GPUs).

python scripts/goal_f1_watch.py            (runs in local tmux rv_f1_watch on the agent box)

Every POLL_S (12 min) a READ-ONLY probe (scripts/shells/probe_owner.sh via `det shell open`):
- WAIT -> launch when, for 2 consecutive polls:
  - both GPUs have max util over 5 samples < 10 % and >= 45 GB free;
  - no owner (non-RetryVLA) GPU process uses > 1 GB;
  - no owner process uses >= 50 % CPU (3 s top sample);
  - load per core < 0.5.
  On the first launch: envcheck (OK video decode + OK sim), then register in shells.json (allow_only + share_ok,
  light-share note, claims 63e0f914:<gpu>). Then start the lanes rv_f1_g0 (wm s7-9) and rv_f1_g1 (random s7-9)
  via tmux_start.sh.
- RUN -> back-off (kill ONLY our rv_f1_* sessions with kill_rv.sh, requeue, back to WAIT) when either:
  - an owner GPU process > 1 GB appears;
  - owner CPU >= 150 % summed, on 2 consecutive polls.
  Lanes are resumable (eval-summary / DONE markers + trainer resume.pt every 250 steps). A lane that dies on its
  own is relaunched (max 3 failures per lane).
- DONE -> analyze_goal_f1.py, auto-entries in RESULTS / NOTES, claims released, commit + push (explicit paths).
- Give up if F1 was never launched within 72 h (NOTES entry + commit).
Owner processes and tmux sessions (egoprm / nativemem / robotood / anything not rv_*) are never touched.
Log: runs/goal_rsi/f1_watch.jsonl (one row per poll) + stdout.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SID = "63e0f914-dd96-4c9a-b110-7d84b8a5d075"
SHORT = SID[:8]
POLL_S = 720
GIVEUP_S = 72 * 3600
LANES = {"rv_f1_g0": (0, ["wm:7", "wm:8", "wm:9"]), "rv_f1_g1": (1, ["rand:7", "rand:8", "rand:9"])}
RUNS = [f"f1_{a}_s{s}" for a in ("wm", "rand") for s in (7, 8, 9)]
LOG = REPO / "runs/goal_rsi/f1_watch.jsonl"
SH = str(REPO / "scripts/shells")


def log(**kw):
    kw["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(json.dumps(kw), flush=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(kw) + "\n")


def det(*args, timeout=300) -> str:
    try:
        p = subprocess.run(["env", "-u", "LD_LIBRARY_PATH", "det", "shell", "open", SID, "--", *args],
                           capture_output=True, text=True, timeout=timeout, cwd="/tmp")
        return p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def probe() -> dict:
    txt = det("bash", f"{SH}/probe_owner.sh", timeout=180)
    info = {"gpus": {}, "apps": [], "procs": [], "tmux": [], "pfss": False, "raw_err": None}
    for line in txt.splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "GPU" and len(f) >= 5:
            info["gpus"][int(f[1])] = {"used": int(f[2]), "total": int(f[3]), "util": int(f[4])}
        elif f[0] == "APP" and len(f) >= 5:
            info["apps"].append({"pid": int(f[2]), "mib": int(f[3]), "ours": f[4] == "1"})
        elif f[0] == "CPU":
            info["nproc"], info["load"] = int(f[1]), float(f[2])
        elif f[0] == "PROC" and len(f) >= 4:
            info["procs"].append({"pid": int(f[1]), "cpu": float(f[2]), "ours": f[3] == "1", "cmd": " ".join(f[4:])[:60]})
        elif f[0] == "TMUX":
            info["tmux"] = f[1:]
        elif f[0] == "PFSS":
            info["pfss"] = f[1] == "ok"
    if not info["gpus"]:
        info["raw_err"] = txt.strip()[-200:]
    return info


def owner_gpu_mib(i):
    return sum(a["mib"] for a in i["apps"] if not a["ours"])


def owner_cpu(i):
    return sum(p["cpu"] for p in i["procs"] if not p["ours"])


def idle_ok(i) -> tuple[bool, str]:
    if i["raw_err"] or len(i["gpus"]) < 2 or not i["pfss"]:
        return False, f"probe failed: {i['raw_err']}"
    for k, g in i["gpus"].items():
        if g["util"] >= 10 or g["total"] - g["used"] < 45_000:
            return False, f"gpu{k} util {g['util']} free {g['total'] - g['used']}"
    if owner_gpu_mib(i) > 1000:
        return False, f"owner GPU procs {owner_gpu_mib(i)} MiB"
    heavy = [p for p in i["procs"] if not p["ours"] and p["cpu"] >= 50]
    if heavy:
        return False, f"owner heavy CPU {[(p['cmd'], p['cpu']) for p in heavy]}"
    if i.get("load", 99) / max(1, i.get("nproc", 1)) >= 0.5:
        return False, f"load {i.get('load')}"
    return True, "idle"


def complete() -> bool:
    return all((REPO / f"runs/goal_rsi/{r}/ckpt_02000/eval_val/summary.json").exists() for r in RUNS)


def lane_complete(name) -> bool:
    _, runs = LANES[name]
    return all((REPO / f"runs/goal_rsi/f1_{r.split(':')[0]}_s{r.split(':')[1]}/ckpt_02000/eval_val/summary.json").exists()
               for r in runs)


def shells_json(update):
    p = REPO / "scripts/shells/shells.json"
    d = json.loads(p.read_text())
    update(d)
    p.write_text(json.dumps(d, indent=1))


def register():
    note = ("vla_roboworm_1 (owner's 2x A100, node shared with tmux egoprm / nativemem / robotood). User 2026-11-20: "
            "F1 only when idle. LIGHT SHARE: launched by scripts/goal_f1_watch.py after 2 idle polls; back-off kills only "
            "rv_f1_* on owner ramp-up; never touch non-rv sessions / owner processes")

    def up(d):
        d.setdefault("allow_only", {})[SHORT] = note
        d.setdefault("share_ok", {})[SHORT] = True
        for n, (g, _) in LANES.items():
            d["claims"][f"{SHORT}:{g}:{n}"] = {"tmux": n, "args": f"goal_f1_lane.sh {g} {' '.join(LANES[n][1])}",
                                              "since": time.strftime("%Y-%m-%dT%H:%M:%S")}
    shells_json(up)


def release():
    shells_json(lambda d: [d["claims"].pop(k) for k in [k for k in d["claims"] if k.startswith(SHORT)]])


def commit(msg, paths):
    subprocess.call(["bash", "-lc", f"cd {REPO} && scripts/git_commit.sh {json.dumps(msg)} {' '.join(paths)} && "
                     "(env -u LD_LIBRARY_PATH git push -q origin main || (env -u LD_LIBRARY_PATH git pull -q --rebase "
                     "origin main && env -u LD_LIBRARY_PATH git push -q origin main))"])


def finish():
    subprocess.call(["bash", "-lc", f"cd {REPO} && source env/activate.sh >/dev/null 2>&1 && "
                     "python scripts/analyze_goal_f1.py > runs/goal_rsi/f1_analyze.log 2>&1"])
    md = (REPO / "runs/goal_rsi/f1.md").read_text() if (REPO / "runs/goal_rsi/f1.md").exists() else "(analysis failed)"
    ts = time.strftime("%Y-%m-%d %H:%M UTC")
    with open(REPO / "experiments/RESULTS.md", "a") as f:
        f.write(f"\n## 2610.05151 follow-up F1: share-matched distillation (AUTO-REGISTERED {ts} by goal_f1_watch; "
                "reading pending)\nPre-registration: RESULTS \"main comparison\" (2026-11-20). Shell 63e0f914 "
                "(vla_roboworm_1), lanes rv_f1_g0 / g1. Bridges only, 36 bridges round-robin (~4 % anchors), else = "
                "main comparison. `scripts/analyze_goal_f1.py` -> runs/goal_rsi/f1.{json,md}.\n\n" + md + "\n")
    with open(REPO / "experiments/NOTES.md", "a") as f:
        f.write(f"- {ts}: F1 finished on 63e0f914 (goal_f1_watch). Claims released. Poll log "
                "runs/goal_rsi/f1_watch.jsonl. Results auto-appended to RESULTS (reading pending).\n")
    release()
    commit("2610.05151 F1 share-matched distillation: results (auto-registered by goal_f1_watch) + claims released",
           ["experiments/RESULTS.md", "experiments/NOTES.md", "scripts/shells/shells.json"])


def main():
    t0 = time.time()
    launched_once, ok_streak, cpu_streak = False, 0, 0
    fails = {n: 0 for n in LANES}
    state = "WAIT"
    log(event="start", lanes=LANES, poll_s=POLL_S)
    while True:
        if complete():
            log(event="complete")
            finish()
            log(event="finished")
            return
        if not launched_once and time.time() - t0 > GIVEUP_S:
            log(event="giveup")
            with open(REPO / "experiments/NOTES.md", "a") as f:
                f.write(f"- {time.strftime('%Y-%m-%d %H:%M UTC')}: F1 watcher GAVE UP: 63e0f914 never idle for 2 "
                        "consecutive polls in 72 h (log runs/goal_rsi/f1_watch.jsonl). F1 not run.\n")
            commit("2610.05151 F1: watcher gave up after 72 h (63e0f914 never idle)", ["experiments/NOTES.md"])
            return
        i = probe()
        ours_up = [n for n in LANES if n in i["tmux"]]
        if state == "WAIT":
            ok, why = idle_ok(i)
            ok_streak = ok_streak + 1 if ok else 0
            log(event="poll", state=state, ok=ok, why=why, streak=ok_streak, gpus=i["gpus"], owner_cpu=owner_cpu(i))
            if ok_streak >= 2:
                if not launched_once:
                    env = det("bash", f"{SH}/envcheck.sh", timeout=900)
                    good = "OK video decode" in env and "OK sim" in env
                    log(event="envcheck", ok=good, out=[l for l in env.splitlines() if l.startswith("OK")])
                    if not good:
                        ok_streak = 0
                        time.sleep(POLL_S)
                        continue
                    register()
                    commit("shells: 63e0f914 (vla_roboworm_1) light-share for 2610.05151 F1 (watcher launch)",
                           ["scripts/shells/shells.json"])
                for n, (g, runs) in LANES.items():
                    if not lane_complete(n) and n not in ours_up:
                        out = det("bash", f"{SH}/tmux_start.sh", n, "scripts/goal_f1_lane.sh", str(g), *runs)
                        log(event="launch", lane=n, out=out.strip().splitlines()[-1:] if out else None)
                launched_once, state, cpu_streak = True, "RUN", 0
        else:  # RUN
            og, oc = owner_gpu_mib(i), owner_cpu(i)
            cpu_streak = cpu_streak + 1 if oc >= 150 else 0
            log(event="poll", state=state, lanes_up=ours_up, owner_gpu_mib=og, owner_cpu=oc, gpus=i["gpus"])
            if i["raw_err"]:
                pass  # probe failure: keep running, retry next poll
            elif og > 1000 or cpu_streak >= 2:
                out = det("bash", f"{SH}/kill_rv.sh", *LANES)
                log(event="backoff", owner_gpu_mib=og, owner_cpu=oc, out=out.strip().splitlines()[-2:])
                state, ok_streak = "WAIT", 0
            else:
                for n, (g, runs) in LANES.items():
                    if n not in ours_up and not lane_complete(n):
                        fails[n] += 1
                        if fails[n] <= 3:
                            out = det("bash", f"{SH}/tmux_start.sh", n, "scripts/goal_f1_lane.sh", str(g), *runs)
                            log(event="relaunch", lane=n, fails=fails[n], out=out.strip().splitlines()[-1:])
                        else:
                            log(event="lane_gave_up", lane=n)
                if all(fails[n] > 3 or lane_complete(n) for n in LANES) and not complete():
                    log(event="all_lanes_failed")
                    release()
                    return
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
