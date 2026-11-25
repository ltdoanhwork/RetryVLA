# Review 2026-11-25: LRS-G (full reproduction of the 2610.05151 Recovery arm on LIBERO-Goal)

For the user's review (afternoon UTC+7). Status at writing: 04:00 UTC. The v1 run is finished and diagnosed. The v2
rerun (servo fix) is running on shell `1123d1ea`; its numbers will be appended in section 6.

## 1. What was run (v1, 2026-11-24)
Their pipeline end to end on our G0-validated stack (M0 85, Original 79.3 over 3 seeds; paper: M0 77, Original 78,
Recovery 88). Pre-registration is in `experiments/PLAN.md` "LRS-G" (commit 047d63e), with deviations D-L1..D-L8.
1. Failures: the seed-7 M0 audit (244 / 56 of 300 train scenes; theirs 233 / 67). Entries = 168 fixed cuts at
   t = 75 / 150 / 225, the same entries as our goal_search (D-L1: no LLM auditor).
2. Teacher: Qwen3-VL-8B tool teacher (D-L2) with move_to / move_by / gripper / done / abort. A frozen JSON spec is
   written in the first reply of each stage. Limits: <= 5 stages, <= 300 actions, <= 12 replies per stage, 1 attempt
   per entry.
3. Acceptance = their Eq. 7: M0 runs 60 actions, an LLM observer checks progress every 10 actions, and the first
   advance must be confirmed after 10 more actions. On failure, the teacher endpoint is restored and the stage is
   extended. **Extra:** every hand-off is continued to <= 300 steps and scored against the BDDL goal (sim ground truth).
4. Quality: an LLM grades every command (privileged sim summaries). Windows are runs with q >= 0.75; anchors need
   >= 5 valid targets. Targets past a window's end are masked and the loss is weighted by q.
5. Their recipe: M0 + 2k updates, D0 + windows pooled uniformly, seeds 7 / 8 / 9. The primary arm is volume-matched to
   theirs (1,708 anchors = 4.3 %); `lrs_all` uses every window (10,552 anchors = 21.6 %).

## 2. Teacher-loop statistics (168 entries; `runs/goal_rsi/lrs_goal.json`)
| quantity | value |
|---|---|
| entries accepted by their rule | 132 / 168 (78.6 %); budget 25, turn cap 8, abort 3 |
| hand-off tests | 217 (1.36 stages / entry) |
| LLM-judge accepts | 132 |
| sim ground truth among all hand-offs | 18 successes |
| **judge precision P(sim success \| accept)** | **0.114 [0.070, 0.179]** (Wilson 95 %) |
| judge recall P(accept \| sim success) | 0.83 [0.61, 0.94] |
| entries solved in sim (any hand-off or teacher reaches success) | 0.107 (vs our goal_search wm_diverse 0.589 on the same entries; McNemar 2 / 83, p 2e-22) |
| LLM cost | 15.0 calls, 82 s LLM, 147 s wall per entry per lane; 4 lanes on 2 A100s, ~3.5 h for 168 entries |
| accepted teacher actions | 23,044 (174.6 / accepted path) |

Pre-registered rule "judge over-accepts" (precision < 0.6, Wilson upper < 0.75): **met**. Their short-horizon,
LLM-judged hand-off accepts roughly 9 false paths for every true one on our stack. Most accepts read like "gripper near
handle, drawer slightly open" while the full task later fails.

## 3. v1 SFT result (nominal val, 100 scenes)
| arm | s7 / s8 / s9 | mean | Δ vs Original (scene x seed bootstrap) | RSR_E8 |
|---|---|---|---|---|
| Original | 82 / 77 / 79 | 79.3 | - | 0.147 |
| LRS v1, their volume (4.3 %) | 14 / 21 / 8 | 14.3 | **−65.0 [−75.7, −53.0]** | 0.063 |
| LRS v1, all windows (21.6 %) | 1 / - / - | 1 | −81 | 0.042 |
| paper Recovery | 88 (1 seed) | | +10 vs their Original | n/a |

Every task collapses, including tasks with no teacher windows at all (turn_on_stove 10 -> 0). The damage is
dose-dependent (4 % -> 14; 21 % -> 1).

## 4. Diagnosis (evidence)
Checks that rule out a plumbing bug:
- **Recipe / init:** `args.json` of lrs_s7 = orig_s7 recipe (init `m0_s7/ckpt_04000`, 2k steps, bs 48 = 16 x 3 accum,
  warmup 200, cosine, bf16 / resume path = the validated D2 path, 84) plus `--recovery lrs_goal_ds --masked_recovery`.
- **Loss path:** `scripts/lrs/check_masked_loss.py`. On D0 anchors at M0, the masked loss equals the standard pi0 loss
  (0.13776 vs 0.13776; pairwise within 1e-7 on 6 batches). The mask and weights therefore change nothing for D0.
- **Mix:** anchors per source are [38,240 D0, 1,708 LRS] = 4.3 %, as intended. The L_min filter dropped exactly 4 per
  window.
- **Replay:** the windows replay in sim from the exact entry snapshot. The demo renderer reproduces logged outcomes
  bit-exactly for our bridges (state diff 0.0).
- **Control C0** (masked path, 0 teacher windows, seed 7): running. It must reproduce ~Original (82). The result goes in
  section 6.

Root cause: **the teacher's low-level servo in OUR re-implementation emits out-of-distribution actions.**
- `exec_cmd` (inherited from the 2026-11-17 Spatial pilot) computes a_xyz = clip(err / 1 cm, ±1), i.e. bang-bang
  control, and sets rotation to exactly 0.
- Saturation at ±1: 42 % / 14 % / 41 % of x / y / z targets in the windows. D0: 0 % (|a| q99 0.94). Our WM bridges:
  0.1-0.8 %.
- Rotation std: 0 in the windows. D0: 0.06-0.10. Bridges: 0.10-0.11.
- At the M0 init, the per-target flow-matching loss on the windows is **1.58 vs 0.138 on D0 (11.5x)**. A 4 % anchor
  share therefore carries about 1/3 of the gradient.
- The trained policies copy the style everywhere. Val actions of lrs_s7: rotation std 0.014-0.028 (Original
  0.06-0.08), x saturated 3.6 % (Original 0 %), gripper mean +0.27 (Original −0.20). lrs_all_s7: rotation std 0.003,
  x saturated 14 %, which leaves a frozen orientation and blind pushes.
- Visual check (`docs/research/figs/lrs_goal/*.png`, renders in `runs/_analysis/visual_check/lrs_goal/`). The windows
  also look poor qualitatively: pumping up and down with 9 gripper flips (ep10), closing on air next to the cream
  cheese (ep42), and long detours. Even so, the LLM grader gave them q >= 0.75 (grade inflation). Our WM bridge on the
  same entry is a short, smooth pi0-style motion followed by M0 finishing the task.

Is it their method or our bug? It is our deviation (now logged as **D-L9**). The paper's teacher issues native actions
and motion helpers "with units / limits", and the exact servo is not given. A teacher whose motions resemble
nothing in the demos is not what they ran. The v1 numbers are therefore NOT a measurement of their method. They are
kept as a servo ablation, and they show how fragile a pi0 continuation is to out-of-distribution targets, even at 4 %.

## 5. Fix (v2) and rerun
- `--servo demo`: proportional control a = err / 4 cm with |a_xyz| <= 0.6 (~ the D0 q75-q90 range), plus an orientation
  hold (axis-angle P control, |a_rot| <= 0.3). Unit test: saturation 51 % -> 0 %, |a| q50 1.0 -> 0.59.
- Full re-collection on the 168 entries (same LLM, prompts, judge and grader). Gate before SFT (pre-registered in
  NOTES before any v2 result): window saturation < 2 % and rotation std > 0.01. Then 3 seeds with their recipe and
  the same evals. Chain `scripts/lrs/lrsg2_chain.sh`. ETA: collection ~07:30 UTC, seeds 7 / 8 ~11:00 UTC,
  seed 9 ~14:30 UTC.
- Expectation (written before results): the collapse disappears (≥ Original − 5) if the servo is the cause. A gain of
  +5 or more is unlikely given judge precision 0.11 and the quality of the windows.

## 6. Results as they land
- **C0 control** (masked path, 0 teacher windows, seed 7): val **77 / 100**. That is inside the Original range
  (82 / 77 / 79), so the training and eval path is clean (07:15 UTC).
- **v2 collection** (168 entries, `--servo demo`, done 06:27 UTC; `runs/goal_rsi/lrs_goal_v2_collection.json`):
  - their rule accepts 136; 214 hand-offs; 11 sim-GT successes;
  - **judge precision 0.074 [0.040, 0.130]**, recall 0.91;
  - solved in sim 0.060 vs WM search 0.589 (McNemar 0 / 89);
  - cost 14.2 LLM calls, 74 s LLM, 121 s wall per entry.
  The servo fix does not make the teacher better at the task. It only makes its actions in-distribution.
- **v2 dataset:** 16 paths / 20 windows / 1,725 frames / 1,645 anchors. Gate passed: saturation 0.0 %, rotation std
  min 0.020.
- **v2 visual check** (`docs/research/figs/lrs_goal/v2_window_ep{4,17}_sheet.png`): motions are now smooth (no
  saturated steps, small orientation corrections). Task quality is still weak: ep17 (wine bottle) chatters the gripper
  13 times while hovering; ep4 (cream cheese) closes on the cheese and lifts, then reopens. 2 of the 6 sampled windows
  are 10-20 near-static frames. The LLM grader still passes these at q >= 0.75.
- **v2 SFT** (their recipe, 4.3 % anchors): **lrs2_s7 40, lrs2_s8 29** / 100 (Original 82 / 77); lrs2_s9 running
  (ETA ~11:45 UTC). Better than v1 (14.3), still −40 to −50 vs Original.
  - Per task (s7): open drawer 0, top drawer 1, cream cheese 0, wine rack 0, wine cabinet 1; the other 5 tasks 7-8.
    The collapse concentrates on the drawer / grasp-heavy tasks where the teacher windows come from.
  - Val actions are now in-distribution (0 % saturation; rotation std 0.02-0.06 vs Original 0.06-0.08). The v1 failure
    mode is gone.
  - Why it still hurts: at M0, the per-target FM loss on v2 windows is **1.07 vs 0.131 on D0 (8.2x)**. The teacher's
    multi-command chunks (stop / turn / gripper toggles every 10-40 steps, hovering, regrasping) are not predictable from
    the image, so 4 % of anchors still drive about 1/4 of the gradient. The grader passes them (q >= 0.75), and 93 % of
    the judge-accepted paths are false positives in sim (precision 0.074).
  - Pre-registered follow-up **v2-GT** (PLAN 09:15 UTC): only paths that the judge accepted AND that succeed in sim
    (10 paths / 13 windows / 707 frames), seed 7. It tests whether the judge's false accepts drive the damage.
    ETA ~12:30 UTC.

## 7. Demos
`docs/research/RECOVERY_DEMOS.md`: 6 entries side by side (M0 alone | WM bridge + M0 | LRS teacher + M0), with the
LLM-judge verdict and the sim outcome on the caption. The paper released no demos: v1 only, no code, data, project
page or video (re-checked 2026-11-25). Their 4 figures are synthetic illustrations, copied to
`docs/related_work/figs/2610.05151/`.

## 8. Questions for the user
1. Should v2 also use a stronger, non-local teacher if an API key becomes available? Qwen3-VL-8B solves only 0.107 of
   entries in sim, against 0.589 for our WM search.
2. Should the "judge precision 0.11" finding become a headline comparison point, since their acceptance rule is the
   piece we can measure exactly?
