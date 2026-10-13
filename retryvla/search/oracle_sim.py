"""Oracle scorer: evaluate each candidate recovery segment in the real simulator.

competence objective (default): restore failure snapshot -> execute candidate (H steps) ->
frozen pi0 continues closed-loop -> score by task success (faster = better) + small dense
shaping. This directly measures "does this recovery restore the nominal policy's competence".

goal_progress objective (ablation): execute the candidate only (no pi0 continuation) and score by
object-to-goal progress (+ success bonus).
"""

from __future__ import annotations

import numpy as np

from retryvla.eval.rollout import Job, run_jobs
from retryvla.search.cem import roughness
from retryvla.search.scorers import Instance, Scorer, ScoreOut


class OracleSimScorer(Scorer):
    def __init__(self, pool, policy, objective: str = "competence", horizon: int = 50,
                 continuation_max_steps: int = 300, success_bonus: float = 1.0, time_weight: float = 0.3,
                 progress_weight: float = 0.2, smooth_cost: float = 0.02, noise_scale: float = 1.0,
                 record_success: bool = True):
        assert objective in ("competence", "goal_progress")
        self.pool, self.policy, self.objective = pool, policy, objective
        self.h, self.cont = horizon, continuation_max_steps
        self.bonus, self.tw, self.pw, self.sc = success_bonus, time_weight, progress_weight, smooth_cost
        self.noise_scale, self.record_success = noise_scale, record_success
        self.sim_episodes = 0
        self.sim_steps = 0

    def score(self, inst: Instance, cands: np.ndarray, iteration: int) -> ScoreOut:
        max_steps = self.h + (self.cont if self.objective == "competence" else 0)
        jobs = [
            Job("restore_snapshot", dict(bddl=inst.bddl, snap=inst.snapshot), task=inst.task,
                max_steps=max_steps, prefix=c, record="success" if self.record_success else "none",
                policy_seed=hash((inst.instance_id, iteration, k)) % (2**31 - 1),
                meta={"k": k})
            for k, c in enumerate(cands)
        ]
        policy = self.policy if self.objective == "competence" else None
        res = run_jobs(self.pool, policy, jobs, noise_scale=self.noise_scale, desc=f"cem-it{iteration}")
        self.sim_episodes += len(res)
        self.sim_steps += int(sum(r.steps for r in res))
        succ = np.array([r.success for r in res])
        sstep = np.array([r.success_step if r.success else -1 for r in res])
        prog = np.array([r.progress for r in res])
        t_frac = np.where(succ, sstep / max_steps, 1.0)
        scores = (self.bonus * succ * (1.0 - self.tw * t_frac)
                  + self.pw * np.clip(prog / 0.3 + 1.0, -1.0, 1.0)
                  - self.sc * roughness(cands))
        return ScoreOut(scores, succ, sstep, prog, [r.record for r in res],
                        {"prefix_success": np.array([r.prefix_success for r in res])})
