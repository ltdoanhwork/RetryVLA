"""F1 on the RetryVLA-owned 2-GPU shell (user decision 2026-11-23): local watchdog chain.

python scripts/goal_f1_chain2.py            (local tmux rv_f1_chain2 on the agent box)

- Shell SID (vla_RetryVLA_DoanhLT_2g, full use). Lanes rv_f1_g0 / rv_f1_g1 run scripts/goal_f1_lane.sh on GPU 0 / 1 in
  REMOTE tmux (tmux_start.sh, survives the det ssh session). They share one locked queue of the 6 runs in paired order
  (wm:7 rand:7 wm:8 rand:8 wm:9 rand:9). Resumable: eval summary / DONE markers + resume.pt (compact bf16, 250 steps).
- Every POLL_S: read-only probe of the remote tmux. A lane that is gone while runs remain is relaunched (<= 4 times
  per lane; it first drops its own stale locks). The shell disappearing -> stop and report in NOTES.
- When all 6 evals exist: analyze_goal_f1.py -> RESULTS (auto table, reading by the agent) + NOTES, claims released,
  commit + push, then `det shell kill` SID unless scripts/shells/keep_<short> exists (user queued more work).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SID = "953bb534-56fa-4fdf-9e98-36e94fea2423"
SHORT = SID[:8]
POLL_S = 600
LANES = {"rv_f1_g0": 0, "rv_f1_g1": 1}
RUNS = [f"f1_{a}_s{s}" for s in (7, 8, 9) for a in ("wm", "rand")]
SH = str(REPO / "scripts/shells")
LOG = REPO / "runs/goal_rsi/f1_chain2.jsonl"


def log(**kw):
    kw["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(json.dumps(kw), flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(json.dumps(kw) + "\n")
    except OSError:
        pass


def det(*args, timeout=300) -> str:
    try:
        p = subprocess.run(["env", "-u", "LD_LIBRARY_PATH", "det", "shell", "open", SID, "--", *args],
                           capture_output=True, text=True, timeout=timeout, cwd="/tmp")
        return p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def shell_state() -> str:
    try:
        out = subprocess.run(["env", "-u", "LD_LIBRARY_PATH", "det", "shell", "list"], capture_output=True, text=True,
                             timeout=120, cwd="/tmp").stdout
    except subprocess.TimeoutExpired:
        return "UNKNOWN"
    for l in out.splitlines():
        if l.strip().startswith(SID):
            return l.split("|")[3].strip()
    return "GONE"


def done(r):
    return (REPO / f"runs/goal_rsi/{r}/ckpt_02000/eval_val/summary.json").exists()


def shells_json(update):
    p = REPO / "scripts/shells/shells.json"
    d = json.loads(p.read_text())
    update(d)
    p.write_text(json.dumps(d, indent=1))


def commit(msg, paths):
    subprocess.call(["bash", "-lc", f"cd {REPO} && scripts/git_commit.sh {json.dumps(msg)} {' '.join(paths)} && "
                     "(env -u LD_LIBRARY_PATH git push -q origin main || (env -u LD_LIBRARY_PATH git pull -q --rebase "
                     "origin main && env -u LD_LIBRARY_PATH git push -q origin main))"])


def note(text):
    with open(REPO / "experiments/NOTES.md", "a") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M UTC')}: {text}\n")


def finish():
    subprocess.call(["bash", "-lc", f"cd {REPO} && source env/activate.sh >/dev/null 2>&1 && "
                     "python scripts/analyze_goal_f1.py > runs/goal_rsi/f1_analyze.log 2>&1"])
    md = (REPO / "runs/goal_rsi/f1.md").read_text() if (REPO / "runs/goal_rsi/f1.md").exists() else "(analysis failed)"
    with open(REPO / "experiments/RESULTS.md", "a") as f:
        f.write(f"\n## 2610.05151 follow-up F1: share-matched distillation (table auto-registered "
                f"{time.strftime('%Y-%m-%d %H:%M UTC')}; reading below when written)\n"
                "Pre-registration: RESULTS \"main comparison\" (2026-11-20). Shell 953bb534 (vla_RetryVLA_DoanhLT_2g), "
                "lanes rv_f1_g0 / g1. Bridges only (`--recovery_segments 0`), 36 bridges round-robin (~4 % of anchors), "
                "otherwise = main comparison. Deviations: compact bf16 resume (optimizer moments rounded at resume "
                "points), bf16 final trainable.pt. `scripts/analyze_goal_f1.py` -> runs/goal_rsi/f1.{json,md}.\n\n"
                + md + "\n")
    shells_json(lambda d: [d["claims"].pop(k) for k in [k for k in d["claims"] if k.startswith(SHORT)]])
    keep = (REPO / f"scripts/shells/keep_{SHORT}").exists()
    note(f"F1 finished on {SHORT} (goal_f1_chain2); table appended to RESULTS. Claims released. "
         + ("Shell KEPT (keep file present)." if keep else "Shell killed (det shell kill)."))
    if not keep:
        shells_json(lambda d: (d.setdefault("gone", {}).__setitem__(SHORT, "vla_RetryVLA_DoanhLT_2g killed after F1")
                               if isinstance(d.get("gone"), dict) else d["gone"].append(SHORT),
                               d["allow_only"].pop(SHORT, None), d.get("share_ok", {}).pop(SHORT, None)))
    commit("2610.05151 F1 share-matched distillation: results table (auto) + claims released"
           + ("" if keep else " + 953bb534 killed"),
           ["experiments/RESULTS.md", "experiments/NOTES.md", "scripts/shells/shells.json"])
    if not keep:
        subprocess.call(["env", "-u", "LD_LIBRARY_PATH", "det", "shell", "kill", SID], cwd="/tmp")
        log(event="shell_killed")


def main():
    fails = {n: 0 for n in LANES}       # relaunches after a lane died (first launch not counted)
    started = set()
    log(event="start", sid=SID, lanes=LANES)
    while True:
        if all(done(r) for r in RUNS):
            log(event="all_done")
            finish()
            log(event="finished")
            return
        st = shell_state()
        if st not in ("RUNNING", "UNKNOWN"):
            log(event="shell_lost", state=st)
            note(f"F1 chain2: shell {SHORT} is {st}; F1 incomplete ({[r for r in RUNS if done(r)]} done). "
                 "Resume state is on /pfss (resume.pt); relaunch on a new shell.")
            commit(f"NOTES: F1 shell {SHORT} lost ({st})", ["experiments/NOTES.md"])
            return
        out = det("bash", f"{SH}/probe_tmux.sh", timeout=180)
        tl = [l for l in out.splitlines() if l.startswith("TMUX")]
        if not tl:
            log(event="probe_failed", out=out[-200:])
            time.sleep(POLL_S)
            continue
        up = [n for n in LANES if n in tl[0].split()[1:]]
        log(event="poll", lanes_up=up, done=[r for r in RUNS if done(r)],
            gpus=[l[5:] for l in out.splitlines() if l.startswith("GPUS")])
        remaining = [r for r in RUNS if not done(r)]
        for n, g in LANES.items():
            if n in up or not remaining or fails[n] > 4:
                continue
            if any(n_ in up for n_ in LANES) and len(remaining) <= len(up):
                continue   # the other lane already covers everything left
            out2 = det("bash", f"{SH}/tmux_start.sh", n, "scripts/goal_f1_lane.sh", str(g), n)
            if "LAUNCHED" in out2:
                if n in started:
                    fails[n] += 1
                started.add(n)
            log(event="launch", lane=n, gpu=g, attempt=fails[n],
                out=[l for l in out2.splitlines() if "LAUNCHED" in l or "EXISTS" in l or "REFUSE" in l])
            shells_json(lambda d, n=n, g=g: d["claims"].__setitem__(f"{SHORT}:{g}:{n}", {
                "tmux": n, "args": f"goal_f1_lane.sh {g} {n} (F1 shared queue)",
                "since": time.strftime("%Y-%m-%dT%H:%M:%S")}))
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
