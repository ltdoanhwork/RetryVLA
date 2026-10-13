---
name: retryvla-search
description: Recovery search in RetryVLA (CEM over open-loop recovery segments, oracle simulator scorer vs world-model+critic scorer, proposal sources, generated recovery datasets). Use when running or modifying run_search, CEM, scorers, primitives, or recovery datasets.
---

# RetryVLA Recovery Search

## Pipeline (`retryvla/search/run_search.py`)

failure instance (scene x gripper_init, fixed protocol seed) -> exact snapshot ->
proposals -> score -> CEM refit -> verified successes -> `RecoveryWriter` (LeRobotDataset) + `branches.parquet`.

- Proposals (`Proposals`): SmolVLA chunk samples at the failure obs (`init.n_vla`, noise scales
  `init.vla_noise_scales`), smooth bounded perturbations of those (`init.n_perturb`, `perturb_std`, 25% gripper flip),
  and primitives (`search/primitives.py`: open, retreat, reposition x4, regrasp, plus randomized variants).
- CEM (`search/cem.py`): Gaussian over [H=50, 7] with temporally smoothed noise, gripper snapped to ±1,
  elites carried across iterations, momentum refit (`alpha`), `min_std` floor.
- `method=random` is the same budget with fresh proposals every iteration and no refit. This is the M1 baseline.

## Scorers (`search/scorers.py`)

Interface: `score(instance, cands [N,H,7], iteration) -> ScoreOut(scores, success, success_step, progress, records)`.
- `oracle_sim` (M1): restore snapshot, execute segment, frozen pi0 continues (`continuation_max_steps`).
  score = bonus x success x (1 - time_weight x t/T) + progress_weight x shaping - smooth_cost x roughness.
  `objective=goal_progress` drops the pi0 continuation (ablation: goal-progress vs competence-targeted).
- `wm_critic` (M3): `WMCriticScorer`, S = V(z_H, q_H, g) - lambda U - beta C. Its successes are predictions and must
  be re-verified in sim before they are written.

## Defaults (pilot, not tuned)

`n_samples=64, n_elites=8, n_iters=4, horizon=50, stop_after_successes=4, max_trajs_per_instance=4`.
The cost per instance is at most n_iters x n_samples sim episodes of up to 350 steps. With 16 workers that is about 1-3 min per
instance at about 20 ms/step. Use `search.max_instances=N` for quick pilots.

## Outputs

`data/recovery/<name>/`: a LeRobotDataset (features like lerobot/libero plus `segment` 0=recovery / 1=continuation, plus
`instance_index`) and `retryvla/{trajectories.jsonl, instances.jsonl, branches.parquet, config.json}`.
Imitate only verified trajectories. Failed branches are for the critic and preference learning only.
