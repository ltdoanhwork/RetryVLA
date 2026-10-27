#!/usr/bin/env python
"""Re-simulate failing LIBERO-Recover test episodes with full telemetry (failure-mode analysis, docs/FAILURE_ANALYSIS.md).

python -m scripts.analyze_failures_resim [--per_model 46] [--workers 8]

Picks a task-stratified sample of FAILED test episodes of M7 base seed 0 and pi1 seed 0 (from their episodes.jsonl),
re-runs each from the exact same failure state with the eval protocol (eval_recover.instance_jobs: same scene,
gripper init, +-2 cm noise seed, policy seed, 520 steps, n_action_steps 10) and records per step:
eef xyz, gripper width, bowl xyz + uprightness, plate xyz, finger-bowl contacts, robot-environment contacts,
bowl-plate contact, action, success. Also saves the agent | wrist video (every 2nd step).
Outputs (resumable): runs/_analysis/failures/resim/<model>/<iid>.npz / .mp4 and results.jsonl.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "runs/_analysis/failures/resim"
RUNS = {  # model -> (eval run of seed 0 whose failures we re-simulate)
    "m7base_s0": "20261017-151822_M7_sft_base_recover_test",
    "pi1_s0": "20261016-054027_M5_m5c_sft_wm_diverse_rr_recover_test",
}
TELE_COLS = ["eef_x", "eef_y", "eef_z", "grip_w", "bowl_x", "bowl_y", "bowl_z", "bowl_up", "plate_x", "plate_y",
             "plate_z", "c_left", "c_right", "c_robot_env", "c_bowl_plate", "a0", "a1", "a2", "a3", "a4", "a5", "a6",
             "success"]

from retryvla.envs.vec import SimPool, _RecordingSim  # noqa: E402


class TeleSim(_RecordingSim):
    """_RecordingSim + per-step object / contact telemetry (kept in the worker, returned with the record)."""

    def _setup(self) -> None:
        import mujoco

        env = self.sim._env.env
        m = env.sim.model._model
        self._m, self._d = m, env.sim.data._data
        ooi = self.sim._ooi
        self._bowl = int(env.obj_body_id[ooi[0]])
        self._plate = int(env.obj_body_id[ooi[1]])
        root = np.asarray(m.body_rootid)
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]
        self._is_bowl = root == root[self._bowl]
        self._is_plate = root == root[self._plate]
        self._is_l = np.array([n in ("gripper0_leftfinger", "gripper0_finger_joint1_tip") for n in names])
        self._is_r = np.array([n in ("gripper0_rightfinger", "gripper0_finger_joint2_tip") for n in names])
        self._is_robot = np.array([n.startswith(("robot0_", "gripper0_", "mount0_")) for n in names])

    def _tel(self, action, success: bool) -> list[float]:
        d, m = self._d, self._m
        st = self._last_obs["state"]
        bp, pp = d.xpos[self._bowl], d.xpos[self._plate]
        q = d.xquat[self._bowl]  # w x y z -> z component of the body z-axis
        up = 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)
        n = int(d.ncon)
        cl = cr = ce = cbp = 0.0
        if n:
            b1 = m.geom_bodyid[np.asarray(d.contact.geom1[:n])]
            b2 = m.geom_bodyid[np.asarray(d.contact.geom2[:n])]
            ib1, ib2 = self._is_bowl[b1], self._is_bowl[b2]
            cl = float(np.any((self._is_l[b1] & ib2) | (self._is_l[b2] & ib1)))
            cr = float(np.any((self._is_r[b1] & ib2) | (self._is_r[b2] & ib1)))
            r1, r2 = self._is_robot[b1], self._is_robot[b2]
            ce = float(np.any((r1 & ~r2 & ~ib2) | (r2 & ~r1 & ~ib1)))
            cbp = float(np.any((ib1 & self._is_plate[b2]) | (ib2 & self._is_plate[b1])))
        a = np.zeros(7) if action is None else np.asarray(action, dtype=np.float64).reshape(-1)[:7]
        return [*st[:3], st[6] - st[7], *bp, up, *pp, cl, cr, ce, cbp, *a, float(success)]

    def begin(self, *a, **k) -> dict:
        out = super().begin(*a, **k)
        self._setup()
        self._tele = [self._tel(None, False)]
        return out

    def step(self, action) -> dict:
        out = super().step(action)
        self._tele.append(self._tel(action, out["success"]))
        return out

    def finish(self, keep_record: bool = False, snapshot: bool = False) -> dict:
        out = super().finish(keep_record, snapshot)
        if out.get("record") is not None:
            out["record"]["tele"] = np.asarray(self._tele, np.float32)
        self._tele = []
        return out


def _tele_worker(conn, resolution: int, env: dict) -> None:
    os.environ.update(env)
    import logging

    logging.basicConfig(level=logging.WARNING)
    sim = TeleSim(resolution)
    while True:
        try:
            cmd, args, kwargs = conn.recv()
        except EOFError:
            break
        if cmd == "__close__":
            sim.sim.close()
            conn.send(("ok", None))
            break
        try:
            conn.send(("ok", getattr(sim, cmd)(*args, **kwargs)))
        except Exception as e:  # report, keep worker alive
            conn.send(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


class TelePool(SimPool):
    def _spawn(self) -> None:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        env = {k: v for k, v in os.environ.items()
               if k.startswith(("MUJOCO", "PYOPENGL", "__EGL", "LD_LIBRARY", "LIBERO", "PYTHONPATH"))}
        self.conns, self.procs = [], []
        for _ in range(self._n_workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_tele_worker, args=(child, self._resolution, env), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self._busy = [False] * self._n_workers


def retry_io(fn, what: str, tries: int = 30):
    """/pfss quota sometimes fills up (EDQUOT): retry instead of dying."""
    for k in range(tries):
        try:
            return fn()
        except OSError as e:
            print(f"IO {what} failed ({e}); retry {k + 1}/{tries}", flush=True)
            time.sleep(30)
    raise RuntimeError(f"IO failed: {what}")


def save_npz(path: Path, **arrs) -> None:
    def f():
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, **arrs)
        os.replace(tmp, path)
    retry_io(f, str(path))


def save_mp4(path: Path, frames: list[np.ndarray], fps: int = 10) -> None:
    import imageio

    def f():
        tmp = path.with_name(path.stem + ".tmp.mp4")
        imageio.mimwrite(tmp, frames, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1,
                         quality=7)
        os.replace(tmp, path)
    retry_io(f, str(path))


def iid_of(r: dict) -> str:
    return f"{r['scene']}__{r['gripper_init']}{r['trial']}"


def pick(rows: list[dict], n: int, rng) -> list[dict]:
    """Task-stratified sample of failures: >= 3 per task (or all), rest proportional; alternate close / open."""
    fails = [r for r in rows if not r["success"]]
    by = collections.defaultdict(list)
    for r in fails:
        by[r["task_folder"]].append(r)
    alloc = {t: min(len(v), 3) for t, v in by.items()}
    rest = n - sum(alloc.values())
    tot = sum(len(v) for v in by.values())
    for t, v in by.items():
        alloc[t] = min(len(v), alloc[t] + int(round(rest * len(v) / tot)))
    out = []
    for t, v in sorted(by.items()):
        v = list(v)
        rng.shuffle(v)
        cl = [r for r in v if r["gripper_init"] == "close"]
        op = [r for r in v if r["gripper_init"] == "open"]
        inter = [x for pair in zip(cl, op) for x in pair] + cl[len(op):] + op[len(cl):]
        out += inter[: alloc[t]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_model", type=int, default=46)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--models", default=",".join(RUNS))
    args = ap.parse_args()

    from retryvla.config import load_config
    from retryvla.envs.splits import load_split, scenes_for
    from retryvla.eval.eval_recover import instance_jobs
    from retryvla.eval.rollout import run_jobs_resilient
    from retryvla.policy import SmolVLAWrapper

    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "results.jsonl"
    done = set()
    if res_path.exists():
        done = {(j["model"], j["iid"]) for j in map(json.loads, open(res_path))}
    with TelePool(args.workers, 256) as pool:
        for model in args.models.split(","):
            run = REPO / "runs" / RUNS[model]
            cfg = load_config(run / "config.yaml")
            rows = [json.loads(l) for l in open(run / "episodes.jsonl")]
            sel = pick(rows, args.per_model, np.random.default_rng(0))
            want = {iid_of(r): r for r in sel}
            (OUT / f"selection_{model}.json").write_text(json.dumps(sorted(want), indent=0))
            scenes = scenes_for(load_split(REPO / cfg.split.file), "test", REPO / cfg.recover.scene_root)
            jobs = instance_jobs(scenes, cfg.recover, int(cfg.eval.seed), cfg.recover.instruction, part="test")
            jobs = [j for j in jobs if iid_of(j.meta) in want and (model, iid_of(j.meta)) not in done]
            for j in jobs:
                j.record, j.record_cfg = "all", {"stride": 2}
            print(f"[{model}] {len(want)} selected, {len(jobs)} to run; policy {cfg.policy.path}", flush=True)
            if not jobs:
                continue
            od = OUT / model
            od.mkdir(exist_ok=True)
            policy = SmolVLAWrapper(str(REPO / cfg.policy.path), "cuda", int(cfg.policy.n_action_steps))

            def on_result(r, model=model, od=od):
                iid = iid_of(r.meta)
                rec = r.record or {}
                tele = rec.get("tele")
                if tele is not None:
                    save_npz(od / f"{iid}.npz", tele=tele, cols=np.array(TELE_COLS), img_t=rec.get("img_t"))
                if "image" in rec:
                    fr = [np.concatenate([a, b], 1) for a, b in zip(rec["image"], rec["image2"])]
                    save_mp4(od / f"{iid}.mp4", fr, fps=10)
                row = {"model": model, "iid": iid, **r.meta, "success": bool(r.success), "steps": int(r.steps),
                       "progress": float(r.progress), "logged_success": bool(want[iid]["success"]),
                       "logged_steps": int(want[iid]["steps"]), "logged_progress": float(want[iid]["progress"])}
                retry_io(lambda: open(res_path, "a").write(json.dumps(row) + "\n"), str(res_path))
                print(f"[{model}] {iid[-60:]} success={r.success} steps={r.steps} "
                      f"progress={r.progress:.3f} (logged {want[iid]['progress']:.3f})", flush=True)
                r.record = None

            run_jobs_resilient(pool, policy, jobs, retries=3, noise_scale=float(cfg.policy.noise_scale),
                               on_result=on_result, desc=model)
            del policy
            import torch

            torch.cuda.empty_cache()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
