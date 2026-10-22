"""M0a: evaluate a SmolVLA checkpoint on nominal LIBERO tasks.

python -m retryvla.eval.eval_nominal configs/exp/m0_baseline_spatial.yaml [key=value ...]
"""

from __future__ import annotations

import logging

import numpy as np

from retryvla import tracking
from retryvla.config import cli_config, to_dict
from retryvla.envs.vec import SimPool
from retryvla.eval.metrics import rate_table, save_video, write_jsonl
from retryvla.eval.rollout import Job, run_jobs
from retryvla.policy import SmolVLAWrapper
from retryvla.utils import setup_logging

log = logging.getLogger("eval_nominal")


def main() -> None:
    setup_logging()
    cfg = cli_config()
    cfg.tag = cfg.get("nominal_tag", f"{cfg.tag}_nominal")
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[cfg.env.suite]()
    task_ids = list(cfg.nominal.task_ids) if cfg.nominal.get("task_ids") else list(range(len(suite.tasks)))
    jobs = []
    for tid in task_ids:
        task = suite.get_task(tid)
        for ep in range(int(cfg.nominal.episodes_per_task)):
            jobs.append(Job(
                "load_nominal",
                dict(suite=cfg.env.suite, task_id=tid, init_idx=ep, num_steps_wait=cfg.env.nominal_num_steps_wait,
                     seed=cfg.eval.seed + ep),
                task=task.language, max_steps=int(cfg.env.nominal_max_steps),
                policy_seed=cfg.eval.seed * 1000 + tid * 100 + ep,
                meta={"task_id": tid, "task": task.name, "episode": ep},
            ))
    max_videos = int(cfg.eval.max_videos) if cfg.eval.save_videos != "none" else 0
    for j in jobs[:: max(1, len(jobs) // max(1, max_videos))][:max_videos]:
        j.record = "all"

    with tracking.start(cfg) as run:
        policy = SmolVLAWrapper(cfg.policy.path, cfg.policy.device, cfg.policy.n_action_steps)
        if (cfg.get("residual") or {}).get("ckpt"):  # B3.3 residual bridge controller (RL baseline); no-op if unset
            from retryvla.baselines.recovery_rl.bridge import maybe_wrap

            policy = maybe_wrap(cfg, policy, run)
        rows = []

        def on_result(r):
            row = {**r.meta, "success": r.success, "steps": r.steps}
            rows.append(row)
            if r.record is not None and (cfg.eval.save_videos == "all" or not r.success):
                save_video(r.record["image"], run.path("videos", f"t{r.meta['task_id']}_ep{r.meta['episode']}_"
                                                      f"{'succ' if r.success else 'fail'}.mp4"), cfg.eval.video_fps)
            r.record = None

        with SimPool(int(cfg.eval.n_workers), int(cfg.env.resolution)) as pool:
            run_jobs(pool, policy, jobs, noise_scale=float(cfg.policy.noise_scale), on_result=on_result,
                     desc="nominal")
        write_jsonl(run.path("episodes.jsonl"), rows)
        per_task = rate_table(rows, "task_id")
        sr = float(np.mean([r["success"] for r in rows]))
        for tid, v in per_task.items():
            run.log({f"nominal/task_{tid}": v["rate"]})
        run.log({"nominal/success_rate": sr})
        run.summary.update({"success_rate": sr, "n_episodes": len(rows), "per_task": per_task,
                            "config": to_dict(cfg)})
        log.info("nominal %s SR = %.3f over %d episodes", cfg.env.suite, sr, len(rows))
        for tid, v in per_task.items():
            log.info("  task %s: %.2f (%d/%d)", tid, v["rate"], v["k"], v["n"])


if __name__ == "__main__":
    main()
