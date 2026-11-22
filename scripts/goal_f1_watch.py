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

FIX 2026-11-22 (launch -> back-off loop 2026-11-20 / 21, 6 cycles):
- Root cause: nvidia-smi inside the container lists HOST pids, so probe_owner.sh could not match them to our
  processes. Our own two pi0 jobs (2 x 30 GB) were counted as "owner GPU 60 GB" and the watcher killed them every
  poll. The SIGHUP / SIGTERM in the lane logs came from that back-off kill (kill_rv.sh + tmux kill-session), not from
  the det session closing: the lanes were alive at the next poll, 12 min after launch.
- Fix: attribute memory PER GPU.
  - other = mem_used - sum(listed in-container apps) = processes of OTHER containers (the owner's GPU0 job).
  - in-container apps are ours while one of our lanes is up on that GPU, and the owner's otherwise.
  - Back-off on GPU g when other(g) grows > 4 GB above its value at our launch, or free(g) < 6 GB. Also when owner CPU
    (container processes, pid-resolvable) is >= 600 % on 2 consecutive polls.
- Per-GPU launch: a lane rv_f1_g<g> starts on any GPU idle for 2 consecutive polls:
  - util < 10 %, free >= 45 GB, other(g) <= 1 GB, and no owner in-container app on g;
  - global: owner CPU < 400 %, load per core < 0.5.
  Lanes share one locked queue of the 6 runs (goal_f1_lane.sh), so one idle GPU runs all seeds sequentially.
  The owner-CPU gate changed from "no owner process >= 50 %" to "owner CPU < 400 % (4 of 32 cores)": the owner keeps
  one 100 % CPU search.py running for days.
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
LANES = {"rv_f1_g0": 0, "rv_f1_g1": 1}   # lane session -> GPU; lanes share one locked queue of the 6 runs
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
    info = {"gpus": {}, "apps": [], "procs": [], "tmux": [], "pfss": False, "raw_err": None, "uuid2idx": {}}
    for line in txt.splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "GPU" and len(f) >= 5:
            info["gpus"][int(f[1])] = {"used": int(f[2]), "total": int(f[3]), "util": int(f[4])}
        elif f[0] == "APP" and len(f) >= 5:
            info["apps"].append({"uuid": f[1], "pid": int(f[2]), "mib": int(f[3])})
        elif f[0] == "GUUID" and len(f) >= 3:
            info["uuid2idx"][f[2]] = int(f[1])
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


def guuid_apps(i):
    """per GPU index: sum of listed in-container app MiB."""
    out = {g: 0 for g in i["gpus"]}
    for a in i["apps"]:
        g = i["uuid2idx"].get(a["uuid"])
        if g is not None:
            out[g] += a["mib"]
    return out


def other_mib(i) -> dict:
    """per GPU: memory of processes in OTHER containers (not listed) = used - listed."""
    la = guuid_apps(i)
    return {g: max(0, v["used"] - la[g]) for g, v in i["gpus"].items()}


def owner_cpu(i):
    return sum(p["cpu"] for p in i["procs"] if not p["ours"])


def gpu_idle(i, g, lanes_up) -> tuple[bool, str]:
    v = i["gpus"][g]
    if v["util"] >= 10 or v["total"] - v["used"] < 45_000:
        return False, f"gpu{g} util {v['util']} free {v['total'] - v['used']}"
    if other_mib(i)[g] > 1000:
        return False, f"gpu{g} other-container {other_mib(i)[g]} MiB"
    if f"rv_f1_g{g}" not in lanes_up and guuid_apps(i)[g] > 0:
        return False, f"gpu{g} owner in-container app {guuid_apps(i)[g]} MiB"
    return True, "idle"


def global_ok(i) -> tuple[bool, str]:
    if i["raw_err"] or len(i["gpus"]) < 2 or not i["pfss"]:
        return False, f"probe failed: {i['raw_err']}"
    if owner_cpu(i) >= 400:
        return False, f"owner CPU {owner_cpu(i):.0f} %"
    if i.get("load", 99) / max(1, i.get("nproc", 1)) >= 0.5:
        return False, f"load {i.get('load')}"
    return True, "ok"


def complete() -> bool:
    return all((REPO / f"runs/goal_rsi/{r}/ckpt_02000/eval_val/summary.json").exists() for r in RUNS)


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
        pass
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


def claim(g, n, on: bool):
    def up(d):
        k = f"{SHORT}:{g}:{n}"
        if on:
            d["claims"][k] = {"tmux": n, "args": f"goal_f1_lane.sh {g} {n} (F1 shared queue)",
                              "since": time.strftime("%Y-%m-%dT%H:%M:%S")}
        else:
            d["claims"].pop(k, None)
    shells_json(up)


def main():
    t0 = time.time()
    registered = (SHORT in json.loads((REPO / "scripts/shells/shells.json").read_text()).get("allow_only", {}))
    streak = {g: 0 for g in LANES.values()}
    base, base_listed = {}, {}
    cpu_streak, fails = 0, {n: 0 for n in LANES}
    log(event="start", lanes=LANES, poll_s=POLL_S, version="2026-11-22 per-GPU attribution")
    while True:
        if complete():
            log(event="complete")
            finish()
            log(event="finished")
            return
        if not registered and time.time() - t0 > GIVEUP_S:
            log(event="giveup")
            with open(REPO / "experiments/NOTES.md", "a") as f:
                f.write(f"- {time.strftime('%Y-%m-%d %H:%M UTC')}: F1 watcher GAVE UP: 63e0f914 never idle for 2 "
                        "consecutive polls in 72 h (log runs/goal_rsi/f1_watch.jsonl). F1 not run.\n")
            commit("2610.05151 F1: watcher gave up after 72 h (63e0f914 never idle)", ["experiments/NOTES.md"])
            return
        i = probe()
        up = [n for n in LANES if n in i["tmux"]]
        gok, gwhy = global_ok(i)
        oc = owner_cpu(i)
        cpu_streak = cpu_streak + 1 if (not i["raw_err"] and oc >= 600) else 0
        row = {"event": "poll", "lanes_up": up, "global": gwhy, "owner_cpu": oc, "gpus": i["gpus"],
               "other_mib": other_mib(i) if not i["raw_err"] else None}
        if i["raw_err"]:
            log(**row)
            time.sleep(POLL_S)
            continue
        # back-off of running lanes
        killed = set()
        for n in up:
            g = LANES[n]
            free = i["gpus"][g]["total"] - i["gpus"][g]["used"]
            grow = other_mib(i)[g] - base.get(g, 0)
            # listed in-container apps on g = ours (pi0 SFT <= ~35 GB, eval <= ~15 GB) + any owner in-container app; the
            # owner adding a big in-container job shows up as listed memory beyond our max
            lgrow = guuid_apps(i)[g] - base_listed.get(g, 0)
            if grow > 4096 or lgrow > 45_000 or free < 6000 or cpu_streak >= 2:
                out = det("bash", f"{SH}/kill_rv.sh", n)
                killed.add(n)
                log(event="backoff", lane=n, other_growth_mib=grow, listed_growth_mib=lgrow, free_mib=free, owner_cpu=oc,
                    out=out.strip().splitlines()[-2:])
                claim(g, n, False)
                streak[g] = 0
        up = [n for n in up if n not in killed]
        # a lane that ended on its own without finishing the queue counts as a failure (max 3 relaunches)
        for n, g in LANES.items():
            if n not in up and n not in killed and f"{SHORT}:{g}:{n}" in json.loads(
                    (REPO / "scripts/shells/shells.json").read_text())["claims"]:
                fails[n] += 1
                claim(g, n, False)
                log(event="lane_ended", lane=n, fails=fails[n])
        # launch idle GPUs (killed lanes wait >= 2 fresh idle polls)
        for n, g in LANES.items():
            if n in up or n in killed:
                streak[g] = 0
                continue
            ok, why = gpu_idle(i, g, up)
            streak[g] = streak[g] + 1 if (ok and gok) else 0
            row[f"gpu{g}"] = f"{why} streak {streak[g]}"
            if streak[g] >= 2 and fails[n] <= 3:
                if not registered:
                    env = det("bash", f"{SH}/envcheck.sh", timeout=900)
                    good = "OK video decode" in env and "OK sim" in env
                    log(event="envcheck", ok=good, out=[l for l in env.splitlines() if l.startswith("OK")])
                    if not good:
                        streak[g] = 0
                        continue
                    register()
                    registered = True
                out = det("bash", f"{SH}/tmux_start.sh", n, "scripts/goal_f1_lane.sh", str(g), n)
                base[g], base_listed[g] = other_mib(i)[g], guuid_apps(i)[g]
                claim(g, n, True)
                log(event="launch", lane=n, gpu=g, base_other_mib=base[g],
                    out=[l for l in out.splitlines() if "LAUNCHED" in l or "EXISTS" in l or "REFUSE" in l])
                commit(f"shells: F1 lane {n} launched on {SHORT}:{g} (watcher)", ["scripts/shells/shells.json"])
                streak[g] = 0
        log(**row)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
