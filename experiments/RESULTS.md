# RetryVLA results (LIBERO-Recover Spatial)

Eval protocol: test RSR = 56 held-out test scenes x 10 trials (5 close + 5 open gripper init, ±2 cm object noise),
520 steps, no search at test time. Drawer = the 16 drawer test scenes (160 episodes). Nominal = LIBERO-Spatial
10 tasks x 50 inits. One training seed per row unless noted. Updated 2026-10-19.

**Protocol caveat (test-set adaptation):** from M7 on, stove / cookie-box test results were used to choose what to
run next (M7, M8). The 56-scene Spatial test split is therefore a DEVELOPMENT set from M7 onward. Final claims will be
made on a held-out evaluation that is frozen before looking at it (plan: LIBERO-Object scenes with the method frozen),
or reported explicitly as exploratory. M8-balanced (chosen by looking at test) is not run.

## Policies (distillation)

| # | policy | init | recovery data | test RSR | drawer | nominal |
|---|---|---|---|---|---|---|
| 0 | pi0 SmolVLA base (`HuggingFaceVLA/smolvla_libero`) | - | - | 0.118 | 0.03 | 0.656 |
| 1 | pi1-nominal (demo-only SFT) | pi0 | - | 0.138 | 0.04 | 0.796 |
| 2 | M1 oracle-CEM | pi0 | 241 oracle-CEM | 0.286 | 0.10 | 0.690 |
| 3 | M1 random | pi0 | 241 random | 0.279 | 0.11 | 0.698 |
| 4 | M4 A | pi1-nom | 241 random | 0.293 | 0.15 | 0.756 |
| 5 | M4 B | pi1-nom | 241 random + 45 WM | 0.298 | 0.18 | 0.748 |
| 6 | M5 random (matched) | pi1-nom | 241 random | 0.286 | 0.15 | 0.762 |
| 7 | **M5 WM round 1** | pi1-nom | 241 WM | **0.402** | 0.36 | 0.780 |
| 8 | M5b random | pi1-nom | 208 random | 0.252 | 0.06 | 0.752 |
| 9 | M5b WM diverse (top-score subset: drawer dropped) | pi1-nom | 208 WM | 0.298 | 0.11 | 0.778 |
| 10 | M5b WM noise-CEM | pi1-nom | 208 WM | 0.270 | 0.03 | 0.770 |
| 11 | M5b WM QD (top-score subset) | pi1-nom | 208 WM | 0.289 | 0.04 | 0.772 |
| 12 | **M5c WM diverse, round-robin** (= current pi1) | pi1-nom | 208 WM | **0.439** | **0.44** | **0.808** |
| 13 | M5c WM QD, round-robin | pi1-nom | 208 WM | 0.354 | 0.21 | 0.754 |
| 14 | M5c WM diverse, all | pi1-nom | 423 WM | 0.389 | 0.33 | 0.788 |
| 15 | M6 pi2 | #12 | 600 round-robin of r1 (423) + r2 robust (275) | 0.463 | 0.44 | **0.828** |
| 16 | M6 control: #12 + same 6k steps, r1 only | #12 | 208 r1 | 0.480 | 0.45 | 0.804 |
| 17 | **M7 base**: #16 + 6k steps, r1 only | #16 | 208 r1 | **0.504** | 0.49 | 0.810 |
| 18 | M7 weak: #16 + r1 + weak-task recoveries | #16 | 500 = 345 r1 + 155 stove/cookie | 0.495 | 0.49 | 0.810 |
| 19 | M8 aug: #16 + r1 + weak + +-5 cm augmented weak | #16 | 800 = 343 other + 457 stove/cookie | 0.457 | 0.42 | 0.796 |

Paired tests (same 560 test episodes): #7 vs #6 +0.116, scene-bootstrap 95% CI [+0.045, +0.191], McNemar p = 4e-7.
#12 vs #8 +0.187.
M6: #16 vs #12 +0.041 [+0.004, +0.077], p = 0.04 (extra steps help). #15 vs #16 -0.018 [-0.055, +0.021], p = 0.42:
round-2 data gives NO gain beyond extra training.
M7: #18 vs #17 -0.009 [-0.050, +0.034], p = 0.72; on stove + cookie-box test episodes (80) 8/80 vs 7/80.
#17 vs #16 +0.023 (p = 0.21): returns from extra training are flattening.
M8: #19 test weak tasks 16/80 vs 7/80 (#17, p = 0.05) and 8/80 (#18, p = 0.10), next_to_cookie_box 0.05 -> 0.35;
but overall -0.046 vs #17 (p = 0.03): 57% of its recovery data are weak-task trajectories, other tasks lose.

## Search (recovery finding)

| experiment | N failures | sims / failure | arm | solved (>=1 verified) | verified hits / sim |
|---|---|---|---|---|---|
| M1 oracle search (pilot) | 108 | <=256 | oracle CEM / random | ~0.70 / ~0.65 | - |
| M3 hard, WM-unseen | 53 | 16 | WM / random | 0.36 / 0.17 | 0.067 / 0.015 |
| M5 round 1 (pilot, adaptive budget) | 108 | 8-32 | WM diverse | 0.70 | 0.14 |
| M5 round 2 critic v1 / v2 (held-out) | 50 | 16 | WM diverse | 0.60 / 0.62 | 0.161 / 0.186 |
| **M5b paired, fresh trials** | **256** | 16 | random | 0.363 | 0.062 |
| | | | **WM diverse** | **0.547** | **0.171** |
| | | | WM noise-CEM | 0.289 | 0.076 |
| | | | WM QD | 0.453 | 0.113 |

M5b: WM diverse vs random McNemar p = 4e-9. Drawer (83 failures): 0.07 / 0.43 / 0.02 / 0.16.
Analysis: `python scripts/analyze_m5b.py` -> `runs/_analysis/m5b/summary.md`.
| M6 round 2: pi1's own train failures (pi1 SR 0.45 on 400 trials) | 219 | 16 + robust 2 | WM diverse, pi1 continues | 0.584 (robust 0.452) | 0.188 |
| M7 weak tasks (stove, cookie box), best policy continues (SR 0.36 on 160 trials) | 102 | 24 / arm + robust 2 | WM diverse / QD / two-stage | 0.59 / 0.61 / 0.61 (robust 0.47 / 0.50 / 0.50) | 0.145 / 0.138 / 0.138 |
| M8 augmented weak tasks (+-5 cm, 30 trials/scene), best policy SR 0.29 | 340 | 24 + robust 2 | WM QD | 0.36 robust | 0.086 |

## M9: mechanism, baselines, seeds (2026-10-20). Spatial test = dev set.

**3 SFT seeds** (test RSR mean ± sd): pi1 0.418 ± 0.021; M7 base 0.482 ± 0.019; M7 weak 0.468 ± 0.024.
M7 base - pi1 +0.063 [+0.035, +0.090]; M7 weak - M7 base -0.013 [-0.036, +0.011]. Nominal 0.77-0.81.

**Search ablation** (122 paired M5b instances, 16 verifications / arm; transitions per instance incl. extra sims):

| arm | WM | critic | diversity | objective | solved | hits / sim | sim transitions |
|---|---|---|---|---|---|---|---|
| random | - | - | - | - | 0.311 | 0.050 | ~5.0k |
| random_diverse | - | - | yes | - | 0.369 | 0.068 | 5.3k |
| wm_goal | yes | - | yes | goal latent (DINO-WM) | 0.230 | 0.038 | 5.4k |
| wm_goal_comp | yes | yes | yes | goal + competence | 0.451 | 0.121 | 5.1k |
| wm_top | yes | yes | - | competence | 0.475 | 0.157 | 5.0k |
| **wm_diverse** | yes | yes | yes | competence | **0.475** | 0.153 | ~5.0k |
| sim_prefix_critic | REAL | yes | yes | competence | 0.443 | 0.115 | **16.5k** |

wm_diverse vs wm_goal 32/2, p = 7e-8; vs random_diverse p = 0.015; vs sim_prefix_critic n.s. at 3.3x fewer transitions;
vs wm_top identical. Budget 32 (61 instances): wm_diverse 0.541 vs random 0.393.

**Downstream / baselines** (test RSR, 1 seed):
- SFT on competence vs goal-objective recoveries (66 each): **0.427 vs 0.189**, paired 163 / 30, p = 3e-23.
- demo-only SFT, same init + steps: 0.127 (nominal 0.82) vs pi1 0.439 (p = 1e-37).
- online best-of-8 (WM + critic every replanning step): pi0 0.118 (= pi0), pi1 0.466 (+0.027 vs pi1, p = 0.23).
- data selection on the same bank (208): top-score 0.298, random 0.427, round-robin 0.439, task-balanced 0.404.

### Re-done statistics (unit = test scene / failure instance), 2026-10-20 -> runs/_analysis/stats/summary.md
Replaces the earlier episode-level McNemar p-values (which treated the 10 trials of a scene as independent).
- comp vs goal SFT +0.237 [+0.164, +0.311] (1 seed each; seeds 1-2 running); pi1 vs demo-only +0.292 [+0.206, +0.383];
  pi1 vs random SFT +0.167 [+0.103, +0.236]; M5 WM vs random +0.116 [+0.045, +0.191] p = 0.003 (scene level).
- pi1 + best-of-8 vs pi1 +0.027 [-0.016, +0.070]; pi0 + best-of-8 vs pi0 +0.000 [-0.021, +0.021].
- M7 base vs pi1 +0.063 [+0.021, +0.105] (3 vs 3 seeds); M7 weak vs base -0.013 [-0.053, +0.029].
- WM vs sim-prefix (search, 122 failures): +0.033 [-0.049, +0.115]; one-sided 95% lower bound -0.033.
- Selection: top-score keeps only 72 unique failures (vs 105-140) and 4 drawer trajectories (vs 37-44);
  random / round-robin / task-balanced are within noise of each other -> lesson = failure-level coverage.
- **3 seeds, competence vs goal SFT (66 traj each)**: comp 0.417 ± 0.011, goal 0.187 ± 0.002; diff +0.230
  [+0.161, +0.298] (scene x seed bootstrap), sign-flip p < 1e-4; comp > goal on 47 / 56 test scenes, < on 7.

## M10: LIBERO-Object, Setting A (recipe frozen before any Object test number), 2026-10-22
Split object_v1: 77 scenes, train 51 / test 26 (frozen). WM + critic retrained on Object train data (M2 recipe).
Search on train trials 4-9 (247 pi0 failures, 16 verifications / arm), SFT init = Object pi1-nominal, 123 traj each
(= size of the smallest bank, wm_goal), round-robin. Test evaluated ONCE (26 scenes x 10), 1 training seed.

| policy | test RSR | nominal |
|---|---|---|
| pi0 | 0.219 | 0.886 |
| pi1-nominal (Object demos) | 0.200 | 0.944 |
| demo-only (+6k steps) | 0.173 | 0.932 |
| + 123 random-search recoveries | **0.362** | 0.946 |
| + 123 WM competence recoveries | **0.354** | 0.942 |
| + 123 WM goal recoveries | 0.242 | 0.906 |

Scene-level paired (26 scenes): comp - demo-only +0.181 [+0.092, +0.277] p = 0.001; comp - random -0.008
[-0.104, +0.088] p = 0.93; comp - goal +0.112 [-0.004, +0.227] p = 0.08; random - demo-only +0.188 p < 1e-4.
Search: WM competence solved 0.462 vs random 0.405 (30/16, p = 0.054) vs WM goal 0.227 (66/8, p = 2e-12);
hits / sim 0.134 / 0.060 / 0.038. Goal bank covers only 56 unique failures (comp 114, random 100).

## M11: WM as planner without the proposal library (COMPLETE, 2026-10-23: shards 0-3, arm A 124 + arm B 127 instances)
Same pi0 failures as M5b / M9 (shards 0-3 of 8), 16 verified sims / arm, paired continuation seeds.
Arm A (old WM + critic v2): wm_no_prim = WM+critic ranking of VLA samples only (no primitives / perturbations);
wm_plan = 5-round elite CEM in imagination from broad noise. Arm B: wm_plan_v2 = the SAME planner with WM v2 + critic
v3 trained with the broad-action collect_wide data. `scripts/analyze_m11.py --s3_total 34 --v2_total 127` ->
runs/_analysis/m11/summary.md (+ per_arm_v2.csv, paired_tests_v2.csv, only_sets_v2.json).

Arm A, 120 paired instances:

| arm | solved [95% CI] | hits/sim | succ jerk | succ manifold dist | succ cells |
|---|---|---|---|---|---|
| random (M5b) | 0.300 [0.225, 0.383] | 0.048 | 0.135 | 0.519 | 2.25 |
| wm_diverse (M5b, library) | **0.467** [0.383, 0.558] | 0.152 | 0.104 | 0.610 | 2.20 |
| wm_top (M9, library) | **0.467** [0.375, 0.558] | 0.152 | 0.094 | 0.584 | 1.79 |
| sim_prefix_critic (M9) | 0.433 [0.350, 0.525] | 0.111 | 0.082 | 0.510 | 3.12 |
| wm_noise (M5b) | 0.242 [0.167, 0.325] | 0.065 | 0.089 | 0.374 | 2.10 |
| wm_no_prim (M11) | 0.292 [0.208, 0.375] | 0.084 | 0.126 | 0.424 | 3.00 |
| wm_plan (M11) | 0.383 [0.300, 0.467] | 0.090 | 0.228 | 0.948 | 2.30 |

Paired (unit = failure instance, exact McNemar, bootstrap CI): wm_plan - random +0.083 [-0.008, +0.175] (20/10,
p = 0.099; hits +0.66/inst, sign-flip p < 1e-4); wm_plan - wm_diverse -0.083 [-0.158, -0.008] (7/17, p = 0.064;
hits -0.99/inst, p < 1e-4); wm_no_prim - wm_diverse -0.175 [-0.258, -0.092] (6/27, p = 3e-4); wm_no_prim - random
-0.008 (p = 1). Library arms jointly solve 77/120; only 2 failures are solved by wm_plan and no library arm (+3 by
wm_no_prim) -> union 82/120; 33 library-solved failures are missed by wm_plan.

Arm B, 118 paired instances (wm_plan_v2 searched all 127; 9 lack some earlier arm because pi0's precheck differs
between runs). Sim steps = mean verify steps / instance (16 sims); M5b arms did not log steps.

| arm | solved [95% CI] | hits/sim | sim steps | hits / 10k steps | succ jerk | succ manifold dist | succ cells |
|---|---|---|---|---|---|---|---|
| random (M5b) | 0.288 [0.212, 0.365] | 0.041 | - | - | 0.132 | 0.517 | 2.06 |
| wm_diverse (M5b, library) | **0.458** [0.364, 0.551] | 0.145 | - | - | 0.105 | 0.618 | 2.17 |
| wm_plan (old WM + critic v2) | 0.373 [0.288, 0.458] | 0.084 | 5257 | 2.56 | 0.229 | 0.950 | 2.23 |
| wm_plan_v2 (WM v2 + critic v3) | 0.398 [0.305, 0.483] | 0.105 | 5159 | 3.25 | 0.222 | 0.941 | 2.55 |

wm_plan_v2 - wm_plan +0.025 [-0.051, +0.102] (12/9, p = 0.66; hits +0.33/inst, sign-flip p = 0.19);
wm_plan_v2 - wm_diverse -0.059 [-0.144, +0.025] (8/15, p = 0.21; hits -0.64/inst, p = 0.020);
wm_plan_v2 - random +0.110 [+0.042, +0.186] (17/4, p = 0.0072; hits +1.02/inst, p < 1e-4).
Outside the library: library arms jointly solve 75/118, + wm_plan_v2 77/118 (+ every arm 81/118); solved by
wm_plan_v2 and no library arm: 2 (and no other arm at all: 1); library-solved but missed by wm_plan_v2: 30;
83% of wm_plan_v2's success strategy cells are unseen by library arms on the same instance (47 solved instances).

Reading: without the library the WM+critic do NOT match library search. Planning from noise beats ranking VLA
samples (+0.09) and finds new strategy cells, but its successes are the jerkiest and farthest from SmolVLA's
manifold. Retraining WM + critic on broad-action data helps the planner only directionally (+0.025 solved, +25%
hits/sim, +27% hits per 10k sim steps, all n.s.): the solve gap to wm_diverse shrinks from -0.085 to -0.059
(still -0.64 hits / instance, p = 0.02), the union with the library gains just 2 failures, and the jerk /
manifold distance of its successes is unchanged. Broad-action WM data does NOT close the gap; the proposal library
still carries most of the recovery-finding, and the remaining planner deficit is not mainly WM coverage of
off-manifold actions.

## B3.2 RISE-inspired imagination-only baseline (PARTIAL, 2026-10-23: pool arms done, VLA-sample arm running)
Same WM v1 + critic v2 + 256 pi0 failures + 696 imagined 50-step chunks / failure as the PoC, but the VLA learns
directly from IMAGINED advantage A = S(chunk) - V(s_0) (no per-sample sim verification; 0 candidate sim transitions).
Matched recipe: init pi1-nominal, 6k steps, eff. batch 32, lr 5e-5, 50% demo replay, 1 training seed.
`scripts/analyze_b32.py` -> runs/_analysis/b32/summary.md. Unit = test scene (56 x 10), scene (x seed) bootstrap,
paired sign-flip p. pi0 = test part of M0_recover_bddl.

| policy | test RSR | nominal | dRSR vs pi1-nominal | vs pi1 (3 seeds) | vs pi0 | dNSR vs pi1-nominal (eps 2 pt) |
|---|---|---|---|---|---|---|
| pi0 | 0.118 | 0.656 | | | | -0.140 |
| pi1-nominal (init) | 0.138 | 0.796 | | | | |
| pi1 (PoC, 3 seeds) | 0.418 ± 0.021 | 0.785 | +0.280 | | | -0.011 (ok) |
| B3.2 cond, pool | 0.068 | 0.718 | -0.070 [-0.109, -0.037] p<1e-4 | -0.351 [-0.435, -0.270] p<1e-4 | -0.050 [-0.079, -0.023] p=9e-4 | -0.078 [-0.124, -0.036] **FAIL** |
| B3.2 AWR, pool | 0.082 | 0.776 | -0.055 [-0.095, -0.021] p=0.004 | -0.336 [-0.418, -0.252] p<1e-4 | -0.036 [-0.061, -0.012] p=0.008 | -0.020 [-0.060, +0.016] at the margin |
| B3.2 cond, VLA samples | running (ETA ~23:00 UTC) | | | | | |

Scenes worse / better / tied vs pi1-nominal: cond 18 / 0 / 38, AWR 18 / 5 / 33. Nominal CI = 10-task bootstrap.
Diagnosis (critic labels vs the real outcomes of the same chunks that M5b executed in sim, 8740 chunks, 0 new sim):
positive-labelled chunks have imagined V(s_hat_50) 0.81 but real success 0.185 (precision; PoC distils only verified
successes, precision 1.0); argmax-A chunk per state: imagined V 0.94, real 0.205; AUC(A -> real) 0.67. Only 18% of
chunks have A > 0; 85% of cond positives and 90% of AWR mass are latent-CEM chunks (off SmolVLA's manifold: 40%
higher jerk than pi0 samples 0.090 vs 0.065, 1.9 vs 6.0 gripper flips / chunk), pi0's own samples get 3%.
Reading: imagination-only learning is WORSE than not training on recoveries at all (below pi1-nominal and even
pi0); the imagined labels are ~4 / 5 wrong on the positives and push the policy towards off-manifold chunks, which
also costs nominal SR (cond -7.8 pt). Sim verification is what makes the WM data useful.
