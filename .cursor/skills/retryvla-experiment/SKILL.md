---
name: retryvla-experiment
description: Run, track and document RetryVLA experiments (milestones M0-M6). Use when starting a RetryVLA run, choosing configs, registering results, writing milestone docs, or touching train/pilot/test splits.
---

# RetryVLA Experiment Protocol

## Research plan (read first)

`experiments/PLAN.md` is the agreed framing (2026-10-22): two independently evaluated modules, the policy-aware
recovery planner (WM + competence critic C^pi, verified bridges) and recovery-preserving distillation (constraint:
NSR(pi') >= NSR(pi) - 2 pts over >= 3 seeds). It lists the evidence table, the PoC experiments P1-P5, the scale plan,
and which claims NOT to make. Results go to `experiments/RESULTS.md`, narrative / incidents to `experiments/NOTES.md`.
Spatial test is a dev set from M7 on; held-out splits (Object test) are evaluated once with a frozen recipe.

## Milestones

| M | Question | Entry points |
|---|---|---|
| M0 | Baseline pi0 on nominal Spatial + held-out recovery scenes | `retryvla.eval.eval_nominal`, `retryvla.eval.verify_scenes`, `retryvla.eval.eval_recover` |
| M1 | Oracle (sim) CEM search + distillation helps? | `retryvla.search.run_search`, `retryvla.train.sft`, `eval_recover` |
| M2 | Latent WM + critic accurate on failure states? | `retryvla.wm.*`, `retryvla.critic.*` (skeletons) |
| M3 | WM-CEM finds real recoveries? | `run_search search.scorer=wm_critic` |
| M4-M5 | Distill / advantage weighting | `retryvla.train.sft`, `retryvla.train.awr` |
| M6 | Neural video for selected branches | not started |

Gate: do not start M2 until M1 shows that pi1 (oracle data) beats pi0 and the random-search baseline on test RSR.

## Starting a run

1. Use an experiment config in `configs/exp/` (it composes `defaults:`). Required keys: `milestone`, `tag`.
   Prefer a new `configs/exp/*.yaml` over long CLI overrides for anything you will report.
2. `scripts/launch.sh <session> <module> configs/exp/<x>.yaml [overrides]`.
3. Each run writes `runs/<ts>_<M>_<tag>/{config.yaml, git.txt, metrics.jsonl, tb/, summary.json}`.
   Commit code before launching runs you will report (git.txt records sha + dirty diff).
4. After it finishes: `python scripts/register_run.py runs/<id> --note "..."` and commit `experiments/REGISTRY.md`.
5. Update the milestone doc `experiments/M<k>_*.md` (Hypothesis / Setup / Results / Decision).

## Split rules (non-negotiable)

- Split file: `data/splits/spatial_v1.json` (git-tracked; its sha is stored in every run summary).
  The leakage unit is the failure episode (`task_folder__episode`), because step_0180 and step_0200 of the same episode are near-duplicates.
- `pilot` is a subset of `train`. Search, critic labels, WM data and SFT data come only from train/pilot.
  `assert_allowed()` enforces this. Never bypass it.
- Never use LIBERO-Recover expert demonstrations of test scenes, for any purpose, including tuning.
- Report test RSR only for final comparisons. Tune on pilot.

## Comparisons that must hold equal

- Same number of recovery trajectories across arms (`train.match_trajectory_count_of`).
- Same SFT steps, recovery_ratio, and init checkpoint; same eval seeds and protocol (`configs/env/libero_spatial.yaml`).
- Report the instruction source (`recover.instruction`: `bddl` is official, `canonical` is the original LIBERO wording).
