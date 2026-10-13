"""M0b: verify LIBERO-Recover scenes restore correctly and deterministically.

For each sampled scene:
  * load with official protocol (close + open gripper), check obs shapes / finite state,
    success checker is False at the failure state;
  * determinism: reload with same seed + same random actions -> identical state;
  * exact branching: restore_snapshot + same actions -> identical obs (state + pixels);
  * saves a contact sheet of the restored failure frames.

python -m retryvla.eval.verify_scenes configs/exp/m0_baseline_spatial.yaml verify.n_scenes=20
"""

from __future__ import annotations

import logging

import numpy as np

from retryvla import REPO_ROOT, tracking
from retryvla.config import cli_config
from retryvla.envs.libero_env import discover_scenes
from retryvla.envs.vec import SimPool
from retryvla.eval.eval_recover import noise_seed
from retryvla.eval.metrics import write_jsonl
from retryvla.utils import setup_logging

log = logging.getLogger("verify_scenes")


def pick_scenes(scenes, n, seed=0):
    rng = np.random.RandomState(seed)
    by_task = {}
    for s in scenes:
        by_task.setdefault(s.task_folder, []).append(s)
    for v in by_task.values():
        rng.shuffle(v)
    out = []
    while len(out) < n and any(by_task.values()):
        for t in sorted(by_task):
            if by_task[t] and len(out) < n:
                out.append(by_task[t].pop())
    return out


def main() -> None:
    setup_logging()
    cfg = cli_config()
    cfg.tag = "verify_scenes"
    vcfg = cfg.get("verify", {})
    n_scenes = int(vcfg.get("n_scenes", 20))
    n_act = int(vcfg.get("n_actions", 60))
    rc = cfg.recover
    scenes = pick_scenes(discover_scenes(REPO_ROOT / rc.scene_root), n_scenes, int(cfg.eval.seed))
    with tracking.start(cfg) as run, SimPool(min(int(cfg.eval.n_workers), len(scenes))) as pool:
        rows, frames = [], []
        todo = list(range(len(scenes)))
        while todo:
            batch, todo = todo[: pool.n], todo[pool.n:]
            # phase 1: load (close) on each worker
            for w, si in enumerate(batch):
                s = scenes[si]
                kw = dict(bddl=s.bddl, npy=s.npy, gripper_close=True, noise_xy=float(rc.obj_noise_xy),
                          noise_seed=noise_seed(0, s.tag, 0), settle_steps=int(rc.gripper_settle_steps))
                pool.send(w, "load_scene", **kw)
            outs = [pool.recv(w) for w in range(len(batch))]
            for w, si in enumerate(batch):
                s, out = scenes[si], outs[w]
                o = out["obs"]
                acts = np.random.RandomState(si).uniform(-1, 1, (n_act, 7)).astype(np.float32)
                ok_shape = o["image"].shape == (256, 256, 3) and o["image2"].shape == (256, 256, 3)
                ok_state = bool(np.isfinite(o["state"]).all())
                succ0 = pool.call(w, "check_success")
                a = pool.call(w, "step_seq", acts, stop_on_success=False)
                s_a = pool.call(w, "get_state")
                # reload determinism
                kw = dict(bddl=s.bddl, npy=s.npy, gripper_close=True, noise_xy=float(rc.obj_noise_xy),
                          noise_seed=noise_seed(0, s.tag, 0), settle_steps=int(rc.gripper_settle_steps))
                pool.call(w, "load_scene", **kw)
                b = pool.call(w, "step_seq", acts, stop_on_success=False)
                s_b = pool.call(w, "get_state")
                # snapshot branching determinism
                pool.call(w, "restore_snapshot", bddl=s.bddl, snap=out["snapshot"])
                c = pool.call(w, "step_seq", acts, stop_on_success=False)
                # open-gripper variant loads too
                open_out = pool.call(w, "load_scene", bddl=s.bddl, npy=s.npy, gripper_close=False,
                                     noise_xy=float(rc.obj_noise_xy), noise_seed=noise_seed(0, s.tag, 5),
                                     settle_steps=int(rc.gripper_settle_steps))
                row = {
                    "scene": s.tag, "task_folder": s.task_folder, "obs_ok": ok_shape and ok_state,
                    "success_at_failure_state": bool(succ0),
                    "reload_state_maxdiff": float(np.abs(s_a - s_b).max()),
                    "reload_obs_maxdiff": float(np.abs(a["obs"]["state"] - b["obs"]["state"]).max()),
                    "snapshot_obs_maxdiff": float(np.abs(a["obs"]["state"] - c["obs"]["state"]).max()),
                    "snapshot_pix_maxdiff": int(np.abs(a["obs"]["image"].astype(int) - c["obs"]["image"]).max()),
                    "gripper_qpos_close": o["state"][6:8].tolist(),
                    "gripper_qpos_open": open_out["obs"]["state"][6:8].tolist(),
                    "deltas": out["deltas"],
                }
                # physics must be bit-exact; allow <=2 intensity levels of GL rasterization jitter
                row["deterministic"] = (row["reload_state_maxdiff"] == 0.0 and row["snapshot_obs_maxdiff"] == 0.0
                                        and row["snapshot_pix_maxdiff"] <= 2)
                rows.append(row)
                frames.append(o["image"])
                log.info("%s | obs_ok=%s succ0=%s det=%s grip(close/open)=%.3f/%.3f", s.tag, row["obs_ok"],
                         row["success_at_failure_state"], row["deterministic"], row["gripper_qpos_close"][0],
                         row["gripper_qpos_open"][0])
        write_jsonl(run.path("verify.jsonl"), rows)
        # contact sheet
        import imageio

        cols = 5
        pad = [np.zeros_like(frames[0])] * ((-len(frames)) % cols)
        grid = np.concatenate([np.concatenate(r, 1) for r in np.array_split(np.stack(frames + pad), max(1, (len(frames) + len(pad)) // cols))], 0)
        imageio.imwrite(run.path("failure_states_grid.png"), grid)
        res = {
            "n_scenes": len(rows),
            "obs_ok_rate": float(np.mean([r["obs_ok"] for r in rows])),
            "deterministic_rate": float(np.mean([r["deterministic"] for r in rows])),
            "success_at_failure_rate": float(np.mean([r["success_at_failure_state"] for r in rows])),
            "gripper_close_mean": float(np.mean([r["gripper_qpos_close"][0] for r in rows])),
            "gripper_open_mean": float(np.mean([r["gripper_qpos_open"][0] for r in rows])),
        }
        run.summary.update(res)
        log.info("verify: %s", res)


if __name__ == "__main__":
    main()
