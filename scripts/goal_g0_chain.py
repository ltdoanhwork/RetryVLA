"""Unattended chain for the 2610.05151 reverse-direction G0 reproduction gate (local shell 8b889199, GPUs 1-3).

python scripts/goal_g0_chain.py --train_gpus 1,3 --eval_gpu 2 [--seed 7]

Stages (each resumable: a stage is skipped when its output marker exists):
  1. M0: pi0 base -> 5,000 LoRA updates on D0 (bs 48, openpi warmup 1000 / decay 30000), ckpt every 1,000.
     Evaluation of every ckpt on the 100 val scenes runs on --eval_gpu while training continues.
  2. M0 selection = best val successes over 1k-5k (their rule; ties -> earlier step).
  3. Original = M0 + 2,000 updates on D0 only (warmup 200, cosine to 2.5e-6 at 2,000, AdamW restart, seed 7).
  4. Eval Original on val; G0 verdict (pre-registered: M0 in [67, 87] AND Original in [68, 88]) -> runs/<root>/g0.json.
Jobs are subprocesses with explicit CUDA_VISIBLE_DEVICES; a failed stage is retried once, then the chain stops.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENV = f"source {REPO}/env/activate.sh >/dev/null 2>&1; export HF_HUB_OFFLINE=1; cd {REPO}; "


def sh(cmd: str, gpus: str, log: Path, tries: int = 2) -> int:
    for k in range(tries):
        full = f"{ENV} export CUDA_VISIBLE_DEVICES={gpus} MUJOCO_EGL_DEVICE_ID={gpus.split(',')[0]}; {cmd}"
        print(time.strftime("%H:%M:%S"), "RUN", cmd, "gpus", gpus, "try", k, flush=True)
        with open(log, "a") as f:
            rc = subprocess.call(["bash", "-lc", full], stdout=f, stderr=subprocess.STDOUT)
        print(time.strftime("%H:%M:%S"), "rc", rc, flush=True)
        if rc == 0:
            return 0
        time.sleep(60)
    return rc


def train_cmd(out: str, steps: int, ngpu: int, init: str | None, warmup: int, decay: int, seed: int,
              save_every: int, recovery: list[str] | None = None, port: int = 29511) -> str:
    c = (f"torchrun --nproc_per_node {ngpu} --master_port {port} -m retryvla.train.pi0_sft --out {out} "
         f"--steps {steps} --save_every {save_every} --bs 48 --warmup {warmup} --decay_steps {decay} --seed {seed}")
    if init:
        c += f" --init {init}"
    if recovery:
        c += " --recovery " + " ".join(recovery)
    return c


def evaluate(ckpt: Path, gpu: str, logs: Path, workers: int) -> dict | None:
    s = ckpt / "eval_val" / "summary.json"
    if not s.exists():
        sh(f"python -m retryvla.eval.eval_goal --ckpt {ckpt} --split val --workers {workers}", gpu,
           logs / f"eval_{ckpt.parent.name}_{ckpt.name}.log")
    return json.loads(s.read_text()) if s.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/goal_rsi")
    ap.add_argument("--train_gpus", default="1,3")
    ap.add_argument("--eval_gpu", default="2")
    ap.add_argument("--eval_workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    root = REPO / a.root
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    ng = len(a.train_gpus.split(","))
    m0 = root / f"m0_s{a.seed}"

    # 1. M0 training (background) + rolling evaluation of its checkpoints
    tr = None
    if not (m0 / "DONE").exists():
        full = f"{ENV} export CUDA_VISIBLE_DEVICES={a.train_gpus}; " + train_cmd(
            str(m0.relative_to(REPO)), 5000, ng, None, 1000, 30000, a.seed, 1000)
        print(time.strftime("%H:%M:%S"), "START M0 training", flush=True)
        tr = subprocess.Popen(["bash", "-lc", full], stdout=open(logs / "train_m0.log", "a"), stderr=subprocess.STDOUT)
    evals = {}
    while True:
        for k in range(1000, 5001, 1000):
            c = m0 / f"ckpt_{k:05d}"
            if k not in evals and (c / "trainable.pt").exists() and (c / "meta.json").exists():
                r = evaluate(c, a.eval_gpu, logs, a.eval_workers)
                if r:
                    evals[k] = r["successes"]
                    print(time.strftime("%H:%M:%S"), f"M0 ckpt {k}: {r['successes']} / {r['n']}", flush=True)
        if len(evals) == 5:
            break
        if tr is not None and tr.poll() is not None and not (m0 / "DONE").exists():
            if tr.returncode != 0:
                print("M0 training failed rc", tr.returncode, "-> retry once", flush=True)
                full = f"{ENV} export CUDA_VISIBLE_DEVICES={a.train_gpus}; " + train_cmd(
                    str(m0.relative_to(REPO)), 5000, ng, None, 1000, 30000, a.seed, 1000)
                tr = subprocess.Popen(["bash", "-lc", full], stdout=open(logs / "train_m0.log", "a"),
                                      stderr=subprocess.STDOUT)
                tr.retried = True  # type: ignore[attr-defined]
        time.sleep(60)
    best = max(sorted(evals), key=lambda k: (evals[k], -k))
    m0_ck = m0 / f"ckpt_{best:05d}"
    (root / "m0_selection.json").write_text(json.dumps({"val_successes": evals, "selected": best,
                                                        "ckpt": str(m0_ck)}, indent=1))
    print("M0 selected", best, evals, flush=True)

    # 3. Original continuation
    orig = root / f"orig_s{a.seed}"
    if not (orig / "DONE").exists():
        rc = sh(train_cmd(str(orig.relative_to(REPO)), 2000, ng, str(m0_ck), 200, 2000, a.seed, 2000),
                a.train_gpus, logs / "train_orig.log")
        if rc:
            raise SystemExit("Original continuation failed")
    r = evaluate(orig / "ckpt_02000", a.eval_gpu, logs, a.eval_workers)
    m0s, os_ = evals[best], (r or {}).get("successes")
    g0 = {"M0_step": best, "M0": m0s, "Original": os_, "M0_ok": 67 <= m0s <= 87,
          "Original_ok": os_ is not None and 68 <= os_ <= 88, "paper": {"M0": 77, "Original": 78, "Recovery": 88}}
    g0["G0_pass"] = bool(g0["M0_ok"] and g0["Original_ok"])
    (root / "g0.json").write_text(json.dumps(g0, indent=1))
    print("G0", json.dumps(g0), flush=True)


if __name__ == "__main__":
    main()
