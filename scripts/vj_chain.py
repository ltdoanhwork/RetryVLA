#!/usr/bin/env python
"""Step-2 chain (foundation WM): runs on the agent box (CPU only, local tmux `vj_chain`), launches GPU work on
dedc84ce via scripts/shells/launch_remote.sh and waits on each job's RV_DONE line. Resumable (skips done steps).

  1. wait collect_v1 V-JEPA latents (144)            [rv_vjenc_c1, already running]
  2. fine-tune V-JEPA 2-AC                            rv_vjwm      (configs/exp/vj_train_wm.yaml)
  3. encode verif_r1 / verif_m5b / rq3 ends           rv_vjenc_*   (as soon as their frames are complete)
  4. C^pi on V-JEPA latents                           rv_vjcrit    (configs/exp/vj_train_critic.yaml)
  5. offline paired re-scoring                        rv_vjresc    (scripts/vj_rescore.py) + analyze_wm_true
  6. online exploitation curve + paired arm, M5b shards 0/1 (49 train instances), 4 sim workers each
                                                      rv_vjsweep_s0 / s1 (configs/exp/vj_rq3_sweep.yaml)
python scripts/vj_chain.py   (console -> runs/_logs/vj_chain_console.log)
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHELL = os.environ.get("VJ_SHELL", "dedc84ce")
LOGS = REPO / "runs/_logs"


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def n_files(d: str) -> int:
    return len(glob.glob(str(REPO / d / "*.pt")))


def launch(name: str, *args: str, tries: int = 60) -> Path:
    """launch_remote (retry while refused, e.g. GPU < 20 GB free) -> path of the job log."""
    for k in range(tries):
        p = subprocess.run(["bash", str(REPO / "scripts/shells/launch_remote.sh"), SHELL, "0", name, *args],
                           capture_output=True, text=True, cwd=REPO)
        m = re.search(r"LAUNCHED \S+ gpu=\d+ log=(\S+)", p.stdout)
        if m:
            log(f"launched {name}: {m.group(1).strip()}")
            return Path(m.group(1).strip().replace("\r", ""))
        log(f"launch {name} refused ({p.stdout.strip()[-120:]}); retry in 5 min")
        time.sleep(300)
    raise RuntimeError(f"could not launch {name}")


def wait(logf: Path, poll: int = 60) -> int:
    while True:
        if logf.exists():
            m = re.search(r"RV_DONE rc=(\d+)", logf.read_text(errors="ignore"))
            if m:
                return int(m.group(1))
        time.sleep(poll)


def release(name: str) -> None:
    p = REPO / "scripts/shells/shells.json"
    d = json.load(open(p))
    for k in [k for k, v in d.get("claims", {}).items() if v.get("tmux") == name]:
        d["claims"].pop(k)
    tmp = f"{p}.tmp{os.getpid()}"
    json.dump(d, open(tmp, "w"), indent=1)
    os.replace(tmp, p)


def run(name: str, *args: str, done=None) -> int:
    if done is not None and done():
        log(f"skip {name} (done)")
        return 0
    rc = wait(launch(name, *args))
    release(name)
    log(f"{name} rc={rc}")
    return rc


def frames_complete(sub: str, n: int) -> bool:
    return n_files(f"data/foundation/frames/{sub}") >= n


def main() -> None:
    sweeps = []
    # 1. collect_v1 latents (encoder job rv_vjenc_c1 started by hand; it exits when 144 files exist)
    while n_files("data/foundation/vj_latents/collect_v1") < 144:
        time.sleep(120)
    log("collect_v1 latents complete")
    # 2. fine-tune
    link = REPO / "data/foundation/latest_vj_wm.txt"
    if run("vjwm", "retryvla.wm.train_vjepa_ac", "configs/exp/vj_train_wm.yaml", done=link.exists) != 0:
        sys.exit("fine-tune failed")
    # 3. encode the eval / critic sets once their frames are complete
    for sub, n in (("verif_r1", 108), ("rq3", 63), ("verif_m5b", 256)):
        while not frames_complete(sub, n):
            time.sleep(120)
        run(f"vjenc_{sub}", "scripts.vjepa_encode", "--src", f"data/foundation/frames/{sub}", "--out",
            f"data/foundation/vj_latents/{sub}", "--ends", done=lambda s=sub, k=n: n_files(f"data/foundation/vj_latents/{s}") >= k)
        if sub == "verif_r1":  # 4. critic as soon as its data exists
            if run("vjcrit", "retryvla.critic.train_critic_vj", "configs/exp/vj_train_critic.yaml",
                   done=(REPO / "data/foundation/latest_vj_critic.txt").exists) != 0:
                sys.exit("critic failed")
            # 6. online sweep can start now (needs only WM + critic); 2 shards in parallel, 4 sim workers each
            for s in (0, 1):
                if not glob.glob(str(REPO / f"data/recovery/vj_rq3_sweep/s{s}/p*/CLOSED")):
                    sweeps.append((f"vjsweep_s{s}", launch(f"vjsweep_s{s}", "retryvla.retry_oom", "--min_free_gb", "10",
                                                            "retryvla.search.compare_arms", "configs/exp/vj_rq3_sweep.yaml",
                                                            f"m5b.shard={s}", "eval.n_workers=4")))
    # 5. offline paired re-scoring (verif_m5b random + wm_diverse; RQ3 picks)
    run("vjresc", "scripts.vj_rescore", "--sets", "data/foundation/vj_latents/verif_m5b:verif_m5b",
        "data/foundation/vj_latents/rq3:true_end_rq3")
    subprocess.run([sys.executable, "scripts/analyze_wm_true.py"], cwd=REPO)
    for name, lf in sweeps:
        log(f"{name} rc={wait(lf)}")
        release(name)
    log("VJ CHAIN DONE")


if __name__ == "__main__":
    main()
