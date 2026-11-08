#!/usr/bin/env python
"""Cross-WM disagreement sweep chain (user 2026-11-08). Runs on the agent box (CPU only, local tmux `xwm_chain`),
launches 3 light lanes on thanhnv's shells via scripts/shells/launch_remote.sh (claims in shells.json) and babysits:

  lane = (shell, gpu, M5b shard, sub-shard)  ->  rv_xwm_s<shard>[_<sub>]
  python -m retryvla.retry_oom --min_free_gb 20 retryvla.search.compare_arms configs/exp/xwm_sweep.yaml
         m5b.shard=<s> m5b.num_sub_shards=2 m5b.sub_shard=<k> eval.n_workers=<w>

- watchdog: log not written for STALL_MIN min, or the tmux session / shell is gone -> kill that exact session,
  release its claim, relaunch (compare_arms resumes from done.jsonl with paired_rng: pools stay identical).
- back-off (owner thanhnv first): a non-rv_ tmux session or an unknown GPU process on our GPU (GPU used > OWNER_MIB),
  or node load / core > LOAD_MAX -> kill our session on that lane, wait until clear for 2 polls, relaunch.
- all lanes rc 0 -> python scripts/analyze_xwm.py (runs/_analysis/xwm/xwm.md). Console: runs/_logs/xwm_chain_console.log
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHJ = REPO / "scripts/shells/shells.json"
TMUX = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin/tmux"
SHELLS = {"335de36e": "335de36e-7f25-4b34-8979-2e74ded69fce", "b936d1d7": "b936d1d7-8a44-4bb8-8f76-a9edb41eb320"}
# cn06 (335de36e) already carries ~16 load from other containers: 2 lanes x 6 workers pushed it to 1.52 / core
# (2026-11-08 16:09) -> ONE lane there (shard 0 unsplit, 6 workers); cn37 (b936d1d7, load ~2.6): shard 1 split in 2
# lane = (shell, gpu, shard, sub_shard, num_sub_shards, n_workers)
LANES = [("335de36e", 0, 0, 0, 1, 6), ("b936d1d7", 0, 1, 0, 2, 8), ("b936d1d7", 1, 1, 1, 2, 8)]
# back-off on load: load / core EXCLUDING our own sim workers (~1.3 load each incl. render threads + policy)
STALL_MIN, OWNER_MIB, LOAD_MAX, POLL_S, MAX_TRIES, OUR_LOAD_PER_W = 45, 40000, 0.9, 300, 4, 1.3
ENV = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}


def say(msg: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%S"), "[xwm_chain]", msg, flush=True)


def probe(sh: str) -> dict:
    try:
        out = subprocess.run(["det", "shell", "open", SHELLS[sh], "--", "bash", str(REPO / "scripts/shells/probe.sh")],
                             capture_output=True, text=True, timeout=180, env=ENV).stdout.replace("\r", "")
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "err": str(e)}
    info: dict = {"ok": False, "tmux": set(), "gpu": {}}
    for ln in out.splitlines():
        f = ln.split()
        if f and f[0] == "GPU" and len(f) >= 5:
            info["ok"] = True
            info["gpu"][int(f[1])] = (int(f[2]), int(f[4]))
        elif f and f[0] == "CPU" and len(f) >= 3:
            info["nproc"], info["load"] = int(f[1]), float(f[2])
        elif f and f[0] == "TMUX":
            info["tmux"] = set(f[1:])
    return info


def release(sess: str) -> None:
    """pop the claim(s) whose tmux == sess (exact), atomic write"""
    try:
        d = json.load(open(SHJ))
        c = d.get("claims", {})
        ks = [k for k, v in c.items() if v.get("tmux") == sess]
        for k in ks:
            c.pop(k)
        if ks:
            tmp = f"{SHJ}.tmp{os.getpid()}"
            json.dump(d, open(tmp, "w"), indent=1, ensure_ascii=False)
            os.replace(tmp, SHJ)
    except Exception as e:  # noqa: BLE001
        say(f"release {sess} failed: {e}")


def kill(sh: str, sess: str) -> None:
    """kill ONE of our tmux sessions by exact name (never by pattern)"""
    assert sess.startswith("rv_xwm_")
    subprocess.run(["det", "shell", "open", SHELLS[sh], "--", TMUX, "kill-session", "-t", f"={sess}"],
                   capture_output=True, text=True, timeout=180, env=ENV)


def launch(sh: str, gpu: int, sess: str, args: list[str]) -> str | None:
    p = subprocess.run([str(REPO / "scripts/shells/launch_remote.sh"), sh, str(gpu), sess, *args], capture_output=True,
                       text=True, env=ENV, cwd=REPO, timeout=600)
    out = (p.stdout + p.stderr).replace("\r", "")
    m = re.search(r"LAUNCHED \S+ gpu=\d+ log=(\S+)", out)
    if not m:
        say(f"launch {sess} on {sh}:{gpu} failed: {out.strip().splitlines()[-1:] if out.strip() else p.returncode}")
        return None
    say(f"launched {sess} on {sh}:{gpu}: {m.group(1)}")
    return m.group(1)


def rc_of(log: str) -> int | None:
    try:
        m = re.findall(r"RV_DONE rc=(\d+)", open(log, errors="ignore").read())
    except OSError:
        return None
    return int(m[-1]) if m else None


def owner_busy(pr: dict, gpu: int, ours_live: bool, our_w: int = 0) -> str | None:
    """our_w = sim workers of OUR live lanes on this shell (their load is subtracted before the load test)"""
    if not pr.get("ok"):
        return None
    other = sorted(s for s in pr["tmux"] if not s.startswith("rv_"))
    if other:
        return f"owner tmux {other}"
    used = pr["gpu"].get(gpu, (0, 0))[0]
    if used > (OWNER_MIB if ours_live else 20000):
        return f"gpu{gpu} used {used} MiB"
    base = (pr.get("load", 0.0) - OUR_LOAD_PER_W * our_w) / max(1, pr.get("nproc", 32))
    if base > LOAD_MAX:
        return f"load/core excl. ours {base:.2f} (total {pr.get('load', 0):.1f})"
    return None


def main() -> None:
    lanes = {(f"rv_xwm_s{s}_{k}" if ns > 1 else f"rv_xwm_s{s}"): (sh, g, s, k, ns, w) for sh, g, s, k, ns, w in LANES}
    state = {n: {"log": None, "done": False, "tries": 0, "clear": 0} for n in lanes}
    say(f"start: {list(lanes)}")
    while True:
        pr = {sh: probe(sh) for sh in SHELLS}
        for n, (sh, g, s, k, ns, w) in lanes.items():
            st = state[n]
            our_w = sum(w_ for n_, (sh_, *_r, w_) in lanes.items() if sh_ == sh and pr[sh].get("ok") and n_ in pr[sh]["tmux"])
            if st["done"]:
                continue
            live = pr[sh].get("ok") and n in pr[sh]["tmux"]
            if st["log"] is None and live:  # adopt after a chain restart
                logs = sorted((f for f in glob.glob(str(REPO / f"runs/_logs/{n}_*.log")) if re.fullmatch(rf"{n}_[0-9a-f]{{12}}_\d{{8}}-\d{{6}}\.log", os.path.basename(f))), key=os.path.getmtime)
                st["log"] = logs[-1] if logs else None
                say(f"adopted live {n}: {st['log']}")
            if st["log"]:
                rc = rc_of(st["log"])
                if rc is not None:
                    say(f"{n} on {sh}:{g} ended rc={rc}")
                    release(n)
                    st["log"] = None
                    if rc == 0:
                        st["done"] = True
                    else:
                        st["tries"] += 1
                    continue
                idle = (time.time() - os.path.getmtime(st["log"])) / 60 if os.path.exists(st["log"]) else 0.0
                gone = (pr[sh].get("ok") is False and idle > 15) or (pr[sh].get("ok") and not live and idle > 5)
                busy = owner_busy(pr[sh], g, True, our_w)
                if idle > STALL_MIN or gone or busy:
                    why = busy and f"BACK-OFF ({busy})" or ("shell/session gone" if gone else f"stalled {idle:.0f} min")
                    say(f"WATCHDOG {n} on {sh}:{g}: {why} -> kill exact session")
                    if live:
                        kill(sh, n)
                    with open(st["log"], "a") as f:
                        f.write(f"[xwm_chain] killed ({why}) {time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
                    release(n)
                    st["log"], st["clear"] = None, 0
                    if not busy:
                        st["tries"] += 1
                continue
            # not running: launch when the lane is clear for 2 consecutive polls (owner first)
            if st["tries"] >= MAX_TRIES or not pr[sh].get("ok"):
                continue
            if owner_busy(pr[sh], g, False, our_w):
                st["clear"] = 0
                continue
            st["clear"] += 1
            if st["clear"] < 2 and st["tries"] > 0:
                continue
            args = ["retryvla.retry_oom", "--min_free_gb", "20", "retryvla.search.compare_arms",
                    "configs/exp/xwm_sweep.yaml", f"m5b.shard={s}", f"m5b.num_sub_shards={ns}", f"m5b.sub_shard={k}",
                    f"eval.n_workers={w}"]
            st["log"] = launch(sh, g, n, args)
            if st["log"] is None:
                st["tries"] += 1
        if all(v["done"] for v in state.values()):
            say("ALL LANES DONE -> analyze_xwm")
            subprocess.run(["bash", "-c", "source env/activate.sh >/dev/null 2>&1; python scripts/analyze_xwm.py"],
                           cwd=REPO, timeout=1800)
            say("ALL DONE")
            return
        if all(v["done"] or v["tries"] >= MAX_TRIES for v in state.values()):
            say(f"GIVING UP: {[(n, v['tries']) for n, v in state.items() if not v['done']]}")
            return
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
