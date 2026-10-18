"""Evaluate a policy on LIBERO-Recover failure scenes with the official protocol
(per scene: N close-gripper + N open-gripper trials, fresh +-2cm object xy noise, 520 steps).

python -m retryvla.eval.eval_recover configs/exp/m0_baseline_spatial.yaml [recover_eval.splits=[test]] ...

Reports Recovery Success Rate (RSR) overall and broken down by split part, gripper init,
task, and failure level (L1-L4 when a level map is available; otherwise 'unk').
"""

from __future__ import annotations

import logging
import zlib

import numpy as np

from retryvla import REPO_ROOT, tracking
from retryvla.config import cli_config, to_dict
from retryvla.envs.libero_env import Scene
from retryvla.envs.splits import load_split, scenes_for, split_sha
from retryvla.envs.vec import SimPool
from retryvla.eval.metrics import rate_table, save_video, write_jsonl
from retryvla.eval.rollout import Job, run_jobs
from retryvla.policy import SmolVLAWrapper
from retryvla.utils import setup_logging

log = logging.getLogger("eval_recover")


def noise_seed(seed: int, tag: str, trial: int) -> int:
    return (seed * 1_000_003 + zlib.crc32(tag.encode()) * 31 + trial) % (2**31 - 1)


def instance_jobs(scenes: list[Scene], rc, seed: int, instruction: str, part: str = "",
                  trials: list[str] | None = None) -> list[Job]:
    """One job per (scene, trial). trials default = official close x N + open x N."""
    if trials is None:
        trials = ["close"] * int(rc.num_close_trials) + ["open"] * int(rc.num_open_trials)
    jobs = []
    for s in scenes:
        for k, g in enumerate(trials):
            jobs.append(Job(
                "load_scene",
                dict(bddl=s.bddl, npy=s.npy, gripper_close=(g == "close"), noise_xy=float(rc.obj_noise_xy),
                     noise_seed=noise_seed(seed, s.tag, k), settle_steps=int(rc.gripper_settle_steps),
                     num_steps_wait=int(rc.num_steps_wait), build_seed=seed),
                task=s.instruction(instruction), max_steps=int(rc.max_steps),
                policy_seed=noise_seed(seed + 1, s.tag, k),
                meta={"scene": s.tag, "task_folder": s.task_folder, "group": s.group, "level": s.level,
                      "gripper_init": g, "trial": k, "part": part},
            ))
    return jobs


def summarize(rows: list[dict]) -> dict:
    sr = float(np.mean([r["success"] for r in rows])) if rows else float("nan")
    out = {"rsr": sr, "n_episodes": len(rows)}
    for key in ("part", "gripper_init", "task_folder", "level"):
        out[f"by_{key}"] = rate_table(rows, key)
    out["by_scene"] = {k: v["rate"] for k, v in rate_table(rows, "scene").items()}
    for g in ("close", "open"):
        if g in out["by_gripper_init"]:
            out[f"rsr_{g}"] = out["by_gripper_init"][g]["rate"]
    return out


def main() -> None:
    setup_logging()
    cfg = cli_config()
    cfg.tag = cfg.get("recover_tag", f"{cfg.tag}_recover")
    split_path = REPO_ROOT / cfg.split.file
    split = load_split(split_path)
    root = REPO_ROOT / cfg.recover.scene_root
    jobs: list[Job] = []
    for part in cfg.recover_eval.splits:
        scenes = scenes_for(split, part, root)
        tf = list(cfg.recover_eval.get("task_filter") or [])
        if tf:  # only scenes whose task folder contains one of these substrings
            scenes = [s for s in scenes if any(x in s.task_folder for x in tf)]
        if cfg.recover_eval.get("max_scenes"):
            scenes = scenes[: int(cfg.recover_eval.max_scenes)]
        jobs += instance_jobs(scenes, cfg.recover, int(cfg.eval.seed), cfg.recover.instruction, part=part)
    max_videos = int(cfg.eval.max_videos) if cfg.eval.save_videos != "none" else 0
    for j in jobs[:: max(1, len(jobs) // max(1, max_videos))][:max_videos]:
        j.record = "all"
    log.info("evaluating %d episodes over parts %s (instruction=%s)", len(jobs), list(cfg.recover_eval.splits),
             cfg.recover.instruction)

    with tracking.start(cfg) as run:
        run.summary["split_sha"] = split_sha(split_path)
        policy = SmolVLAWrapper(cfg.policy.path, cfg.policy.device, cfg.policy.n_action_steps)
        rows = []

        def on_result(r):
            rows.append({**r.meta, "success": r.success, "steps": r.steps, "progress": r.progress})
            if r.record is not None and (cfg.eval.save_videos == "all" or not r.success):
                m = r.meta
                save_video(r.record["image"], run.path("videos", f"{m['scene']}_{m['gripper_init']}{m['trial']}_"
                                                      f"{'succ' if r.success else 'fail'}.mp4"), cfg.eval.video_fps)
            r.record = None

        with SimPool(int(cfg.eval.n_workers), int(cfg.env.resolution)) as pool:
            run_jobs(pool, policy, jobs, noise_scale=float(cfg.policy.noise_scale), on_result=on_result,
                     desc="recover")
        write_jsonl(run.path("episodes.jsonl"), rows)
        summ = summarize(rows)
        run.log({"recover/rsr": summ["rsr"], **{f"recover/rsr_{g}": summ.get(f"rsr_{g}", float("nan"))
                                                for g in ("close", "open")}})
        for part, v in summ["by_part"].items():
            run.log({f"recover/rsr_{part}": v["rate"]})
        run.summary.update(summ | {"policy": cfg.policy.path, "instruction": cfg.recover.instruction,
                                   "config": to_dict(cfg)})
        log.info("RSR = %.3f over %d episodes | close %.3f | open %.3f", summ["rsr"], len(rows),
                 summ.get("rsr_close", float("nan")), summ.get("rsr_open", float("nan")))
        for part, v in summ["by_part"].items():
            log.info("  part %s: %.3f (%d/%d)", part, v["rate"], v["k"], v["n"])


if __name__ == "__main__":
    main()
