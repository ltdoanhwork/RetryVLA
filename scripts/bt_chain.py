#!/usr/bin/env python3
"""Part 2 chain on ONE shared GPU shell (run inside tmux on that shell via launch_remote.sh):

  wait for both Backtrack-datagen shards (data/recovery/bt_distill/s{0,1}/p*/CLOSED)
  -> in parallel: Backtrack-distill SFT (configs/exp/bt_sft_distill.yaml) | bt_single test eval -> Object bt_retry eval
  -> after SFT: distill RSR (test) + nominal (10 x 50), then delete intermediate checkpoints (002000, 004000).

python -m scripts.bt_chain [--workers 6] [--skip_single] [--skip_obj]
Each step logs to runs/_logs/rv_btc_<step>_<host>_<ts>.log ending in "RV_DONE rc=N" (same convention as remote_tmux).
Steps whose log already has RV_DONE rc=0 are skipped (re-runnable after a crash).
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "runs" / "_logs"


def say(msg: str) -> None:
    try:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [bt_chain] {msg}", flush=True)
    except OSError:
        pass


def done(step: str) -> bool:
    return any("RV_DONE rc=0" in open(f, errors="ignore").read() for f in glob.glob(str(LOGS / f"rv_btc_{step}_*.log")))


def run(step: str, args: list[str]) -> int:
    if done(step):
        say(f"{step}: already done, skip")
        return 0
    log = LOGS / f"rv_btc_{step}_{socket.gethostname()}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    say(f"{step}: start -> {log.name}")
    with open(log, "a") as f:
        f.write(f"[rv] start={time.strftime('%Y-%m-%dT%H:%M:%S')} args: {' '.join(args)}\n")
        f.flush()
        rc = subprocess.run([sys.executable, "-u", "-m", *args], stdout=f, stderr=subprocess.STDOUT, cwd=REPO,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"}).returncode
        f.write(f"[rv] RV_DONE rc={rc} end={time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
    say(f"{step}: rc={rc}")
    return rc


def newest(pat: str) -> str | None:
    m = sorted(glob.glob(str(REPO / pat)))
    return os.path.relpath(m[-1], REPO) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip_single", action="store_true")
    ap.add_argument("--skip_obj", action="store_true")
    a = ap.parse_args()
    W = f"eval.n_workers={a.workers}"
    while not all(glob.glob(str(REPO / f"data/recovery/bt_distill/s{s}/p*/CLOSED")) for s in (0, 1)):
        say("waiting for datagen shards (CLOSED)")
        time.sleep(300)
    n = sum(1 for f in glob.glob(str(REPO / "data/recovery/bt_distill/s*/p*/bt/retryvla/trajectories.jsonl"))
            for _ in open(f))
    say(f"datagen done: {n} trajectories")

    def evals_side():
        if not a.skip_single:
            run("single", ["retryvla.eval.eval_recover", "configs/exp/bt_single.yaml", W])
        if not a.skip_obj:
            run("obj", ["retryvla.eval.eval_recover", "configs/exp/bt_obj_retry.yaml", W])

    side = threading.Thread(target=evals_side)
    side.start()
    rc = run("sft", ["retryvla.retry_oom", "--min_free_gb", "30", "retryvla.train.sft", "configs/exp/bt_sft_distill.yaml"])
    if rc == 0:
        ck = newest("runs/*_BT_sft_bt_distill/checkpoints/006000/pretrained_model")
        for d in glob.glob(str(REPO / "runs/*_BT_sft_bt_distill/checkpoints/00[0-5]*")):
            if not d.endswith("006000"):
                shutil.rmtree(d, ignore_errors=True)
                say(f"deleted intermediate checkpoint {d}")
        t = threading.Thread(target=run, args=("dist_rec", ["retryvla.eval.eval_recover", "configs/exp/m0_baseline_spatial.yaml",
                             "milestone=BT", "recover_tag=sft_bt_distill_recover_test", f"policy.path={ck}",
                             "recover_eval.splits=[test]", W, "eval.save_videos=none", "eval.worker_retries=2"]))
        t.start()
        run("dist_nom", ["retryvla.eval.eval_nominal", "configs/exp/m0_baseline_spatial.yaml", "milestone=BT",
                         "nominal_tag=sft_bt_distill_nominal", f"policy.path={ck}", W, "eval.save_videos=none",
                         "eval.worker_retries=2"])
        t.join()
    side.join()
    say("ALL DONE")


if __name__ == "__main__":
    main()
