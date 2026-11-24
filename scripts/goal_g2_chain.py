"""Goal G2 + P2-3 orchestrator (local tmux rv_g2_chain on the agent box; pre-registered PLAN 2026-11-24).

python scripts/goal_g2_chain.py

Slots = 4 GPUs on the two RetryVLA-owned shells (953bb534 0/1, 8f473147 0/1). A slot is free when no rv_g2r_* lane of
ours is on it AND none of the bank-generation lanes (rv_g2s*, rv_dp*) is still running on it. Each job = one
`scripts/goal_g2_run.sh` (train + val eval with saved actions + events + E8 / E8on RSR), launched as remote tmux
`rv_g2r_<name>` via tmux_start.sh, resumable (DONE / summary markers), relaunched <= 2 times if it dies.
Dependencies: G2 / G2r jobs need the curated search CLOSED (3 shards); A / B jobs need D_P CLOSED (4 shards), with
n_P = #D_P trajectories (capped at 36). After each group finishes, scripts/analyze_goal_g2.py writes
runs/goal_rsi/g2.{json,md} and the chain appends the table to RESULTS + a NOTES line, then commits + pushes.
Poll every 10 min. Idle-kill rule: a shell with no queued / running work for > 2 h is killed (det shell kill), unless
scripts/shells/keep_<short> exists.
"""

from __future__ import annotations

import glob
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SH = str(REPO / "scripts/shells")
SHELLS = {"953bb534": "953bb534-56fa-4fdf-9e98-36e94fea2423", "8f473147": "8f473147-f766-4ea5-8fe5-32d28567fd52"}
SLOTS = [("953bb534", 0), ("953bb534", 1), ("8f473147", 0), ("8f473147", 1)]
BANK_LANES = {"rv_g2s0": ("953bb534", 0), "rv_g2s1": ("953bb534", 1), "rv_g2s2": ("8f473147", 0),
              "rv_dp0": ("8f473147", 1), "rv_dp1": ("8f473147", 1), "rv_dp2": ("953bb534", 0), "rv_dp3": ("953bb534", 1)}
CUR = "data/recovery/goal_search_onset_cur/s*/p*"
OLD = "data/recovery/goal_search/s*/p*"
DP = "data/recovery/goal_dp_s7/sh*/bank*"
LOG = REPO / "runs/goal_rsi/g2_chain.jsonl"
POLL_S = 600


def log(**kw):
    kw["t"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(json.dumps(kw), flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(json.dumps(kw) + "\n")
    except OSError:
        pass


def det(sid, *args, timeout=240) -> str:
    try:
        p = subprocess.run(["env", "-u", "LD_LIBRARY_PATH", "det", "shell", "open", SHELLS[sid], "--", *args],
                           capture_output=True, text=True, timeout=timeout, cwd="/tmp")
        return p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def tmux_of(sid) -> list[str] | None:
    out = det(sid, "bash", f"{SH}/probe_tmux.sh", timeout=180)
    tl = [l for l in out.splitlines() if l.startswith("TMUX")]
    return tl[0].split()[1:] if tl else None


def search_done():
    return len(glob.glob(str(REPO / "data/recovery/goal_search_onset_cur/s*/p*/CLOSED"))) >= 3


def dp_done():
    return len(glob.glob(str(REPO / "data/recovery/goal_dp_s7/sh*/CLOSED"))) >= 4


def n_dp():
    return sum(sum(1 for _ in open(f)) for f in glob.glob(str(REPO / f"{DP}/retryvla/trajectories.jsonl")))


EVAL_ONLY = {"ev_orig_s7": "runs/goal_rsi/orig_s7/ckpt_02000", "ev_orig_s8": "runs/goal_rsi/orig_s8/ckpt_02000",
             "ev_orig_s9": "runs/goal_rsi/orig_s9/ckpt_02000", "ev_m0": "runs/goal_rsi/m0_s7/ckpt_04000"}


def run_done(name):
    if name in EVAL_ONLY:
        c = REPO / EVAL_ONLY[name]
        return (c / "eval_entries_e8on/summary.json").exists() and (c / "eval_val_ev/events_summary.json").exists()
    return (REPO / f"runs/goal_rsi/{name}/ckpt_02000/eval_entries_e8on/summary.json").exists()


def jobs():
    js = [(n, 0, ["EVAL_ONLY"], lambda: True) for n in EVAL_ONLY]
    for s in (7, 8, 9):
        js.append((f"g2_wm_s{s}", s, [f"{CUR}/wm_diverse@36"], search_done))
    for s in (7, 8, 9):
        js.append((f"g2_rand_s{s}", s, [f"{CUR}/random@36"], search_done))
    for s in (8, 9):   # D5 control (old fixed-cut uncurated bank + continuation) for 3 seeds (s7 = diag_f1wm_full_s7)
        js.append((f"d5_wm_s{s}", s, [f"{OLD}/wm_diverse@36"], lambda: True))
    for s in (7, 8, 9):
        np_ = lambda: min(36, n_dp())
        js.append((f"p23_A_s{s}", s, lambda np_=np_: [f"{CUR}/wm_diverse@{36 + np_()}"], dp_done))
        js.append((f"p23_B_s{s}", s, lambda np_=np_: [f"{CUR}/wm_diverse@36", f"{DP}@{np_()}"], dp_done))
    return js


def commit(msg, paths):
    subprocess.call(["bash", "-lc", f"cd {REPO} && scripts/git_commit.sh {json.dumps(msg)} {' '.join(paths)} && "
                     "(env -u LD_LIBRARY_PATH git push -q origin main || (env -u LD_LIBRARY_PATH git pull -q --rebase "
                     "origin main && env -u LD_LIBRARY_PATH git push -q origin main))"])


def register(tag):
    subprocess.call(["bash", "-lc", f"cd {REPO} && source env/activate.sh >/dev/null 2>&1 && "
                     "python scripts/analyze_goal_g2.py > runs/goal_rsi/g2_analyze.log 2>&1"])
    md = (REPO / "runs/goal_rsi/g2.md").read_text() if (REPO / "runs/goal_rsi/g2.md").exists() else "(analysis failed)"
    with open(REPO / "experiments/RESULTS.md", "a") as f:
        f.write(f"\n## Goal G2 / P2-3 auto table ({tag}; {time.strftime('%Y-%m-%d %H:%M UTC')}; reading by the agent "
                "follows)\nPre-registration: PLAN \"Goal campaign G2 ... P2-3\" (2026-11-24). `scripts/analyze_goal_g2.py` "
                "-> runs/goal_rsi/g2.{json,md}.\n\n" + md + "\n")
    with open(REPO / "experiments/NOTES.md", "a") as f:
        f.write(f"- {time.strftime('%Y-%m-%d %H:%M UTC')}: goal_g2_chain: {tag} finished, auto table appended to RESULTS.\n")
    commit(f"Goal G2 / P2-3: {tag} results (auto table)", ["experiments/RESULTS.md", "experiments/NOTES.md",
                                                            "scripts/shells/shells.json"])


def main():
    fails, started, registered = {}, {}, set()
    idle_since = {s: None for s in SHELLS}
    log(event="start")
    while True:
        tm = {s: tmux_of(s) for s in SHELLS}
        if any(v is None for v in tm.values()):
            log(event="probe_failed", tm=tm)
            time.sleep(POLL_S)
            continue
        busy = set()
        for lane, (s, g) in BANK_LANES.items():
            if lane in tm[s]:
                busy.add((s, g))
        running = {}
        for s in SHELLS:
            for t in tm[s]:
                if t.startswith("rv_g2r_"):
                    running[t[len("rv_g2r_"):]] = s
        for name, sg in started.items():
            if name in running:
                busy.add(sg)
        todo = [j for j in jobs() if not run_done(j[0]) and j[0] not in running]
        free = [sg for sg in SLOTS if sg not in busy]
        for (name, seed, rec, ready), sg in zip([j for j in todo if j[3]() and fails.get(j[0], 0) <= 2], free):
            rec = rec() if callable(rec) else rec
            if any(r.endswith("@0") for r in rec):
                log(event="skip_no_dp", job=name)
                continue
            if rec == ["EVAL_ONLY"]:
                out = det(sg[0], "bash", f"{SH}/tmux_start.sh", f"rv_g2r_{name}", "scripts/goal_eval_only.sh", str(sg[1]),
                          EVAL_ONLY[name])
            else:
                out = det(sg[0], "bash", f"{SH}/tmux_start.sh", f"rv_g2r_{name}", "scripts/goal_g2_run.sh", str(sg[1]),
                          name, str(seed), *rec)
            if name in started:
                fails[name] = fails.get(name, 0) + 1
            started[name] = sg
            busy.add(sg)
            log(event="launch", job=name, slot=sg, rec=rec, out=[l for l in out.splitlines() if "LAUNCHED" in l])
        log(event="poll", running=sorted(running), busy=sorted(busy), todo=[j[0] for j in todo],
            search_done=search_done(), dp_done=dp_done(), n_dp=n_dp() if dp_done() else None)
        for tag, names in (("G2", [f"g2_{a}_s{s}" for a in ("wm", "rand") for s in (7, 8, 9)] + ["d5_wm_s8", "d5_wm_s9"]),
                           ("P2-3", [f"p23_{a}_s{s}" for a in ("A", "B") for s in (7, 8, 9)])):
            if tag not in registered and all(run_done(n) for n in names):
                register(tag)
                registered.add(tag)
        if all(run_done(j[0]) for j in jobs()):
            log(event="all_done")
            return
        for s in SHELLS:   # idle-kill rule (2 h, nothing queued for that shell's slots)
            used = any(sg[0] == s for sg in busy)
            if used or todo:
                idle_since[s] = None
            elif idle_since[s] is None:
                idle_since[s] = time.time()
            elif time.time() - idle_since[s] > 7200 and not (REPO / f"scripts/shells/keep_{s}").exists():
                log(event="kill_idle_shell", shell=s)
                subprocess.call(["env", "-u", "LD_LIBRARY_PATH", "det", "shell", "kill", SHELLS[s]], cwd="/tmp")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
