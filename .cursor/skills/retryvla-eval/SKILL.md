---
name: retryvla-eval
description: Standard evaluation protocol for RetryVLA policies (nominal LIBERO success and LIBERO-Recover RSR). Use when evaluating any SmolVLA checkpoint, comparing pi0/pi1, or interpreting RSR numbers.
---

# RetryVLA Evaluation Protocol

## Nominal (M0a and after every fine-tune)

```bash
scripts/launch.sh nom retryvla.eval.eval_nominal configs/exp/m0_baseline_spatial.yaml policy.path=<ckpt>
```
- libero_spatial, 10 tasks x 50 init states, max 280 steps, 10 settle steps.
- `policy.n_action_steps=10` (replan every 10 of the 50-step chunk). Keep it fixed across compared arms.
- Reference: smolvla_libero is reported around 90% on Spatial. If it is far below that, debug the interface
  (image flip, state layout, task string) before anything else.
- After recovery training, nominal SR must not drop by more than about 2-3 points. Always report it next to RSR.

## Recovery Success Rate (RSR)

```bash
scripts/launch.sh rec retryvla.eval.eval_recover configs/exp/m0_baseline_spatial.yaml \
    policy.path=<ckpt> "recover_eval.splits=[test]"
```
- Official LIBERO-Recovery protocol per scene: 5 closed-gripper + 5 open-gripper trials, fresh ±2 cm object
  xy noise per trial (seeded from scene tag + trial), max 520 steps, success = BDDL goal.
- Seeds are deterministic in `eval.seed`, so all arms see the same perturbations.
- Report: overall RSR with Wilson CI, `rsr_close`, `rsr_open`, per-task, and per-level (L1-L4 once labels exist;
  currently `unk`). `summary.json` has all of them.
- Test split: 56 scenes, 560 episodes. Pilot: 60 scenes. Differences under about 4 points on test are within noise.
- Instruction ablation: `recover.instruction=canonical` vs `bddl` (official).

## Videos

`eval.save_videos=failures|all|none`, `eval.max_videos`. Videos go to `runs/<id>/videos/`.
