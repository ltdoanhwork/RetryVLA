#!/usr/bin/env python3
"""6-arm ablation (plan todo `ablation`): missing seed cells, unattended, on the two RetryVLA shells only.

python3 scripts/abl_chain.py [--dry]        # run on the agent box in tmux (CPU only: launches + polls)

Cells (committed configs, only the seed changes; same recipe / init / steps / eval protocol as seed 0):
  arm 4 free-unfiltered  configs/train/af_arm4_free_unfiltered.yaml  seeds 0-2  (P4_af_arm4_free_unfiltered_s<k>)
  arm 3 free-filtered    configs/train/af_arm3_free_filtered.yaml    seeds 1-2  (seed 0 = af_chain)
  arm 5 hybrid           configs/train/af_arm5_hybrid.yaml           seeds 1-2  (seed 0 = af_chain)
  arm 6 backtrack-distill configs/exp/bt_sft_distill.yaml            seeds 1-2  (BT_sft_bt_distill_s<k>; seed 0 = bt_chain)
  arm 1 demo-only        configs/exp/m9_sft_demo_only.yaml           seeds 1-2  (M9_sft_demo_only_s<k>; seed 0 = M9_sft_demo_only)
Arm 2 (pi1) already has 3 seeds (M5c + M8_seed_pi1_s1/s2) -> reused by scripts/analyze_ablation.py.

Scheduling (re-read every poll; state = files on /pfss, so a restarted chain resumes):
  SFT  (retry_oom --min_free_gb 32, save_every=6000 -> only the final 006000 checkpoint): one at a time on dedc84ce;
       a second lane on c37611dd only while its GPU has >= 45 GB free (other projects' ASR jobs hold ~57 GB now).
  eval (RSR test 560 ep + nominal 500 ep, standard protocol, worker_retries 2): sim-worker budget per shell summed over
       ALL live rv_ claims (sibling agents included): c37611dd 12 (RSR 8 / nominal 4), dedc84ce 8 (RSR 5 / nominal 3).
  done = 006000/pretrained_model/model.safetensors exists (SFT) / summary.json status done (eval).
  Claims are released by exact tmux session name only (never by slot key). Jobs that end rc != 0 are retried twice; launch refusals
  (GPU busy, quota probe, det hiccup) just retry at the next poll.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHJ = REPO / "scripts/shells/shells.json"
LOG = REPO / "runs/_logs" / f"abl_chain_{time.strftime('%Y%m%d-%H%M%S')}.log"
SHELLS = {"dedc84ce": "dedc84ce-dde2-471f-97de-878bdb0a319d", "c37611dd": "c37611dd-dcb4-499b-a025-88c9fd912051"}
SIM_BUDGET = {"c37611dd": 12, "dedc84ce": 8}
WORKERS = {"c37611dd": {"rec": 8, "nom": 4}, "dedc84ce": {"rec": 5, "nom": 3}}
EVAL_PREF = ["c37611dd", "dedc84ce"]
POLL_S = 120
MAX_TRIES = 3

ARMS = {  # key -> (config, milestone, tag stem)
    "a1": ("configs/exp/m9_sft_demo_only.yaml", "M9", "sft_demo_only_s{}"),
    "a4": ("configs/train/af_arm4_free_unfiltered.yaml", "P4", "af_arm4_free_unfiltered_s{}"),
    "a3": ("configs/train/af_arm3_free_filtered.yaml", "P4", "af_arm3_free_filtered_s{}"),
    "a5": ("configs/train/af_arm5_hybrid.yaml", "P4", "af_arm5_hybrid_s{}"),
    "a6": ("configs/exp/bt_sft_distill.yaml", "BT", "sft_bt_distill_s{}"),
}
# order: complete one seed of every arm first (arm 4 seed 0 is the only missing seed-0 cell), then the next seed
ORDER = [("a4", 0), ("a3", 1), ("a5", 1), ("a6", 1), ("a4", 1), ("a3", 2), ("a5", 2), ("a6", 2), ("a4", 2),
         ("a1", 1), ("a1", 2)]  # demo-only had 1 seed only (M9_sft_demo_only), not 3


def say(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [abl_chain] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


class Cell:
    def __init__(self, arm: str, seed: int):
        self.arm, self.seed = arm, seed
        self.cfg, self.ms, stem = ARMS[arm]
        self.tag = stem.format(seed)
        self.short = f"abl_{arm}s{seed}"

    def ckpt(self) -> str | None:
        m = sorted(glob.glob(str(REPO / f"runs/*_{self.ms}_{self.tag}/checkpoints/006000/pretrained_model")))
        m = [p for p in m if Path(p, "model.safetensors").exists()]
        return os.path.relpath(m[-1], REPO) if m else None

    def eval_done(self, kind: str) -> bool:
        suf = "recover_test" if kind == "rec" else "nominal"
        for d in sorted(glob.glob(str(REPO / f"runs/*_{self.ms}_{self.tag}_{suf}"))):
            try:
                if json.load(open(Path(d, "summary.json"))).get("status") == "done":
                    return True
            except Exception:
                pass
        return False

    def sft_args(self) -> list[str]:
        return ["retryvla.retry_oom", "--min_free_gb", "32", "retryvla.train.sft", self.cfg, f"tag={self.tag}",
                f"train.seed={self.seed}", "train.save_every=6000"]

    def eval_args(self, kind: str, w: int) -> list[str]:
        common = [f"policy.path={self.ckpt()}", f"eval.n_workers={w}", "eval.save_videos=none", "eval.worker_retries=2"]
        if kind == "rec":
            return ["retryvla.retry_oom", "--min_free_gb", "6", "retryvla.eval.eval_recover", "configs/exp/m0_baseline_spatial.yaml",
                    f"milestone={self.ms}", f"recover_tag={self.tag}_recover_test", "recover_eval.splits=[test]", *common]
        return ["retryvla.retry_oom", "--min_free_gb", "6", "retryvla.eval.eval_nominal", "configs/exp/m0_baseline_spatial.yaml",
                f"milestone={self.ms}", f"nominal_tag={self.tag}_nominal", *common]


def probe(shell: str) -> dict:
    """read-only: GPU used/total + live tmux sessions of one shell"""
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    try:
        out = subprocess.run(["det", "shell", "open", SHELLS[shell], "--", "bash", str(REPO / "scripts/shells/probe.sh")],
                             capture_output=True, text=True, timeout=180, env=env).stdout.replace("\r", "")
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "err": str(e)}
    info = {"ok": False, "tmux": set()}
    for ln in out.splitlines():
        f = ln.split()
        if f and f[0] == "GPU" and len(f) >= 4:
            info.update(ok=True, used=int(f[2]), total=int(f[3]))
        elif f and f[0] == "TMUX":
            info["tmux"] = set(f[1:])
    return info


def claims() -> dict:
    try:
        return json.load(open(SHJ)).get("claims", {})
    except Exception:
        return {}


def release(sess: str) -> None:
    """pop the claim(s) whose tmux == sess (exact), atomic write; never touches other sessions"""
    try:
        d = json.load(open(SHJ))
        c = d.get("claims", {})
        ks = [k for k, v in c.items() if v.get("tmux") == sess]
        if not ks:
            return
        for k in ks:
            c.pop(k)
        tmp = f"{SHJ}.tmp{os.getpid()}"
        json.dump(d, open(tmp, "w"), indent=1)
        os.replace(tmp, SHJ)
    except Exception as e:  # noqa: BLE001
        say(f"release {sess} failed: {e}")


def sim_used(shell: str, live: set[str]) -> int:
    n = 0
    for k, v in claims().items():
        if k.split(":")[0] == shell and v.get("tmux") in live:
            m = re.search(r"n_workers=(\d+)", v.get("args", ""))
            n += int(m.group(1)) if m else 0
    return n


def sft_live(shell: str, live: set[str]) -> bool:
    """any live SFT claim (ours or a sibling's) on this shell"""
    return any(k.split(":")[0] == shell and v.get("tmux") in live and "train.sft" in v.get("args", "")
               for k, v in claims().items())


def launch(shell: str, sess: str, args: list[str], dry: bool) -> str | None:
    if dry:
        say(f"DRY launch {shell} {sess}: {' '.join(args)}")
        return None
    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    p = subprocess.run([str(REPO / "scripts/shells/launch_remote.sh"), shell, "0", sess, *args], capture_output=True,
                       text=True, env=env, cwd=REPO, timeout=600)
    out = (p.stdout + p.stderr).replace("\r", "")
    m = re.search(r"LAUNCHED \S+ gpu=\d+ log=(\S+)", out)
    if not m:
        say(f"launch {sess} on {shell} failed: {out.strip().splitlines()[-1:] if out.strip() else p.returncode}")
        return None
    say(f"launched {sess} on {shell}: {m.group(1)}")
    return m.group(1)


def rc_of(log: str) -> int | None:
    try:
        m = re.findall(r"RV_DONE rc=(\d+)", open(log, errors="ignore").read())
    except OSError:
        return None
    return int(m[-1]) if m else None


def analyze() -> None:
    cmd = "source env/activate.sh >/dev/null 2>&1; python scripts/analyze_ablation.py > /dev/null 2>&1"
    if (REPO / "scripts/analyze_ablation.py").exists():
        subprocess.run(["bash", "-c", cmd], cwd=REPO, timeout=1800)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    cells = [Cell(*x) for x in ORDER]
    running: dict[str, tuple[str, str, Cell, str]] = {}  # sess -> (shell, log, cell, kind)
    tries: dict[str, int] = {}
    say(f"start: {[c.tag for c in cells]}")
    while True:
        pr = {s: probe(s) for s in SHELLS}
        # 0. adopt our sessions that are alive but unknown (chain restarted)
        for c in cells:
            for kind, s in (("sft", f"rv_{c.short}"), ("rec", f"rv_{c.short}_rec"), ("nom", f"rv_{c.short}_nom")):
                for sh in SHELLS:
                    if s in pr[sh].get("tmux", set()) and s not in running:
                        logs = sorted((f for f in glob.glob(str(REPO / f"runs/_logs/{s}_*.log"))
                                       if re.fullmatch(rf"{s}_[0-9a-f]{{12}}_\d{{8}}-\d{{6}}\.log", os.path.basename(f))),
                                      key=os.path.getmtime)
                        if logs:
                            running[s] = (sh, logs[-1], c, kind)
                            say(f"adopted live {s} on {sh}: {logs[-1]}")
        # 1. reap finished jobs
        for sess, (sh, log, c, kind) in list(running.items()):
            rc = rc_of(log)
            if rc is None:
                continue
            say(f"{sess} on {sh} ended rc={rc}")
            release(sess)
            running.pop(sess)
            if rc != 0:
                tries[sess] = tries.get(sess, 0) + 1  # only real failures count (launch refusals just retry next poll)
            if kind != "sft" and c.eval_done("rec") and c.eval_done("nom"):
                for d in glob.glob(str(REPO / f"runs/*_{c.ms}_{c.tag}/checkpoints/0*")):
                    if not d.endswith("006000"):
                        shutil.rmtree(d, ignore_errors=True)
                        say(f"deleted intermediate checkpoint {d}")
                say(f"{c.tag}: SFT + RSR + nominal DONE")
                analyze()
        # 2. what is left
        todo_sft = [c for c in cells if not c.ckpt() and f"rv_{c.short}" not in running]
        todo_ev = [(c, k) for c in cells if c.ckpt() for k in ("rec", "nom")
                   if not c.eval_done(k) and f"rv_{c.short}_{k}" not in running]
        if not todo_sft and not todo_ev and not running:
            say("ALL DONE")
            analyze()
            return
        todo_sft = [c for c in todo_sft if tries.get(f"rv_{c.short}", 0) < MAX_TRIES]
        todo_ev = [(c, k) for c, k in todo_ev if tries.get(f"rv_{c.short}_{k}", 0) < MAX_TRIES]
        if not todo_sft and not todo_ev and not running:
            say("STOP: remaining jobs failed 3x, see logs")
            return
        if todo_sft or todo_ev:
            # 3. SFT lanes: one SFT per shell (ours or a sibling's)
            for sh in ("dedc84ce", "c37611dd"):
                if not todo_sft or not pr[sh].get("ok"):
                    continue
                if any(v[0] == sh and v[3] == "sft" for v in running.values()) or sft_live(sh, pr[sh]["tmux"]):
                    continue
                need = 45000 if sh == "c37611dd" else 34000
                if pr[sh]["total"] - pr[sh]["used"] < need:
                    continue
                c = todo_sft.pop(0)
                s = f"rv_{c.short}"
                log = launch(sh, s, c.sft_args(), a.dry)
                if log:
                    running[s] = (sh, log, c, "sft")
                    pr[sh]["used"] += 30000
                    pr[sh]["tmux"].add(s)
            # 4. eval slots (sim-worker budget over all live claims, siblings included)
            for c, k in todo_ev:
                for sh in EVAL_PREF:
                    if not pr[sh].get("ok"):
                        continue
                    w = WORKERS[sh][k]
                    if sim_used(sh, pr[sh]["tmux"]) + w > SIM_BUDGET[sh] or pr[sh]["total"] - pr[sh]["used"] < 8000:
                        continue
                    s = f"rv_{c.short}_{k}"
                    log = launch(sh, s, c.eval_args(k, w), a.dry)
                    if log:
                        running[s] = (sh, log, c, k)
                        pr[sh]["used"] += 4000
                        pr[sh]["tmux"].add(s)  # its claim now counts in sim_used
                    break
        if a.dry:
            return
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
