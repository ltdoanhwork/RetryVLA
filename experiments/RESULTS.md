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

## B3.2 RISE-inspired imagination-only baseline (COMPLETE 2026-10-24: pool arms + VLA-sample arm)
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
| B3.2 cond, VLA samples | 0.113 (63/560; close 0.114, open 0.111) | 0.770 | -0.025 [-0.052, +0.000] p=0.10 | -0.306 [-0.386, -0.227] p<1e-4 | -0.005 [-0.021, +0.011] p=0.67 | -0.026 [-0.068, +0.018] (point < -eps, CI covers it) |

Scenes worse / better / tied vs pi1-nominal: cond 18 / 0 / 38, AWR 18 / 5 / 33. VLA-sample cond: 13 / 6 / 37 vs pi1-nominal, 7 / 5 / 44 vs pi0;
it beats the pool cond arm (+0.045 [+0.021, +0.073], p = 6e-4) and AWR pool (+0.030, p = 0.035): restricting the
imagined-advantage data to pi0's own samples removes most of the off-manifold damage, but it still only matches pi0
(no recovery learned, -0.31 vs pi1) and does not beat pi1-nominal. Nominal CI = 10-task bootstrap.
Diagnosis (critic labels vs the real outcomes of the same chunks that M5b executed in sim, 8740 chunks, 0 new sim):
positive-labelled chunks have imagined V(s_hat_50) 0.81 but real success 0.185 (precision; PoC distils only verified
successes, precision 1.0); argmax-A chunk per state: imagined V 0.94, real 0.205; AUC(A -> real) 0.67. Only 18% of
chunks have A > 0; 85% of cond positives and 90% of AWR mass are latent-CEM chunks (off SmolVLA's manifold: 40%
higher jerk than pi0 samples 0.090 vs 0.065, 1.9 vs 6.0 gripper flips / chunk), pi0's own samples get 3%.
Reading: imagination-only learning is WORSE than not training on recoveries at all (below pi1-nominal and even
pi0); the imagined labels are ~4 / 5 wrong on the positives and push the policy towards off-manifold chunks, which
also costs nominal SR (cond -7.8 pt). Sim verification is what makes the WM data useful.

## B3.3 residual recovery RL baseline, full budget (2026-10-24; COMPLETE incl. nominal + diagnosis)
Frozen pi0 + residual bridge controller (first 50 steps; DINOv2-small + MLP, PPO, sparse task-success reward) trained
on the same 256 M5b train failure states with the PoC's full sim budget (3.47M env transitions; ckpt
`data/rl/b33_rl_sparse/ckpt_m3469021.pt`). Setup / budget accounting: NOTES "B3.3". `python scripts/analyze_b33.py
--tag B33_b33_rl_sparse_full --rl_dir data/rl/b33_rl_sparse` -> runs/_analysis/b33/summary_B33_b33_rl_sparse_full.md.
Unit = test scene (56 x 10 trials), scene (x seed) bootstrap 95% CI, paired sign-flip p (10k flips; 0 -> < 1e-4).
pi0 = test part of M0_recover_bddl.

| system | test RSR (560 ep) | RL - system [95% CI], p | scenes RL worse / better / tied | nominal Spatial SR (500) |
|---|---|---|---|---|
| pi0 (M0) | 0.118 | +0.177 [+0.102, +0.259], p < 1e-4 | 4 / 20 / 32 | 0.656 |
| pi1-nominal (M1, pi1's init) | 0.138 | +0.157 [+0.075, +0.245], p = 4e-4 | 8 / 21 / 27 | 0.796 |
| pi1 (PoC, 3 seeds: M5c + M8 s1 / s2) | 0.418 ± 0.021 | -0.124 [-0.196, -0.053], p = 7e-4 | 28 / 11 / 17 (vs seed 0) | 0.785 |
| RL residual p25 (867k transitions) | 0.111 | | | 0.400 |
| **RL residual full (3.47M)** | **0.295** (165/560; close 0.289, open 0.300) | | | **0.022** (11/500) |

Learning curve (128 train instances, held-out pi0 seeds, deterministic): 0.039 @0 -> 0.094 @1.06M -> 0.133 @2.1M ->
0.172 @2.45M -> 0.227 @3.48M (still rising at the matched budget). Reading: with the full matched sim budget the
residual RL controller learns real recovery (+0.18 over pi0, +0.16 over pi1-nominal) but stays clearly below
verified-search distillation (-0.12 vs pi1, 3 seeds), needs an extra controller + DINOv2 at test time, and (p25)
hurt nominal badly; the full-budget nominal (below) shows it is a recovery-vs-nominal trade-off, and a steep one.

**Nominal (confirmed).** `runs/20261024-032423_B33_b33_rl_sparse_full_nominal`: 11/500 = 0.022; per task 0/50 on 7 of
10 tasks, 3/50 (task 1), 1/50 (task 3), 7/50 (task 8); 489 of 489 failures run the full 280 steps (no crash / early
stop), the 11 successes are slow (97-280 steps vs pi0's ~60-80). Task-bootstrap deltas: vs pi0 -0.634
[-0.700, -0.560], vs pi1-nominal -0.774 [-0.840, -0.712], vs pi1 (3 seeds) -0.763 [-0.825, -0.693]. p25 controller:
0.400 -> the nominal damage GROWS with RL training while RSR rises (0.111 -> 0.295).

**Why nominal collapses (diagnostic evals, first 10 inits / task = 100 episodes, same seeds as the full eval;
`residual.no_grip` / `residual.cont_mult` ablations + `residual_stats`; runs `*_B33_b33diag_nom_*`, `*_b33diag_rec_full_test12`).**
The bridge is active in nominal too (first 50 policy steps = 5 decisions, by design: a deployed system does not know
it starts in a failure state).

| variant (nominal, 100 ep) | SR | gripper decisions follow / open / close | gripper override that flips pi0 | mean L2 arm delta / step (dec 0 / 1-4) |
|---|---|---|---|---|
| full controller | 0.02 (full-run first 10 inits: 0.04) | 345 / 0 / 155 | 144 / 500 (dec 0: 95%) | 0.74 / 0.29-0.51 |
| arm residual only (no_grip) | 0.03 | 500 / 0 / 0 | 0 | 0.74 / 0.34-0.38 |
| gripper head only (cont_mult 0) | 0.07 | 387 / 0 / 113 | 111 (dec 0: 95%) | 0 |
| neither (wrapper no-op control) | 0.73 (pi0 first 10 inits: 0.68) | - | 0 | 0 |
| (ref) full controller on 12 TEST failure scenes, 120 ep | RSR 0.383 (= the full test run on the same 120 ep 0.375, 1 disagreement; pi0 0.208) | 256 / 0 / 174 | 149 / 430 (dec 0: 39%) | 0.81 / 0.56-0.65 |

Reading: two independent causes, each sufficient on its own: (1) the gripper head forces CLOSE on the very first
decision in 95% of nominal episodes (gripper still open, nothing grasped yet -> pi0 then rarely re-opens and
re-grasps in time); (2) the arm residual adds a large offset (L2 0.74 per step on the first decision, ~0.35 after,
actions are in [-1, 1]) that drives the arm away before pi0 takes over. On failure states the same controller acts
differently (close only 39% at decision 0, larger arm deltas), so it is state-dependent, but it never saw a nominal
initial state in training (all 256 PPO starts are failure states) and its first-decision behaviour on such states is
an off-distribution "recovery manoeuvre". The wrapper itself is harmless (no-op control = pi0). Consequence: as a
deployed system it needs a failure detector / gate (extra component, not budgeted) or training on nominal starts too;
as-is it trades -0.63 nominal for +0.18 RSR, while pi1 keeps nominal (-0.011 vs pi1-nominal) and has +0.12 more RSR.

## B2 / P2 value-critic search, FINAL on all paired instances (2026-10-24): B4 question answered
Same pi0 failures, pool, WM v1, 16 verified sims / instance, continuation seeds; only the critic that ranks the
WM-imagined candidates changes. `python scripts/analyze_b2x.py` -> runs/_analysis/b2x. Unit = failure instance,
exact McNemar on solved (>= 1 of 16 verified), instance-bootstrap 95% CI, sign-flip p on hits. Each pair uses the
instances done by BOTH arms (pair n 122-127; arms have 125-129 instances because shard parts were resumed / salvaged
after the EDQUOT windows; all-arm intersection 119 gives the same conclusions). Critic supervision: NOTES "P2 controls".

| arm (critic) | supervision | n | solved | hits / sim | mean C^pi pred of picks |
|---|---|---|---|---|---|
| random (M5b) | - | 127 | 0.339 | 0.061 | - |
| goal (M9) | goal latent, no critic | 125 | 0.240 | 0.046 | - |
| B3 progress | demos + failure 0 + prefix 1 | 127 | 0.213 | 0.038 | 0.038 |
| **C^pi v2** (PoC, M5b wm_diverse) | pi0 continuation | 127 | 0.480 | 0.156 | 0.724 (own) |
| B4 success | pi0 + pi1 + pi1more + demo (1/4 each) | 127 | 0.559 | 0.182 | 0.262 |
| B4-noPi0 | pi1 + pi1more + demo (no pi0 labels) | 129 | 0.450 | 0.093 | 0.102 |
| Cpi-pool | B4's pi0 labels only (7,160), B4's trainer | 129 | 0.550 | 0.156 | 0.316 |

| paired comparison | pair n | solved A vs B | only A / only B | McNemar p | d hits / instance (sign-flip p) |
|---|---|---|---|---|---|
| B4 - C^pi v2 | 123 | 0.545 vs 0.463, +0.081 [+0.016, +0.146] | 14 / 4 | 0.031 | +0.46 (0.038) |
| B4-noPi0 - C^pi v2 | 125 | 0.440 vs 0.472, -0.032 [-0.112, +0.048] | 12 / 16 | 0.57 | -1.05 (1e-4) |
| Cpi-pool - C^pi v2 | 125 | 0.536 vs 0.472, +0.064 [+0.000, +0.136] | 14 / 6 | 0.12 | -0.12 (0.54) |
| B4 - Cpi-pool | 126 | 0.556 vs 0.540, +0.016 [-0.056, +0.087] | 11 / 9 | 0.82 | +0.46 (0.038) |
| **B4 - B4-noPi0** | 126 | 0.556 vs 0.437, **+0.119 [+0.048, +0.198]** | 20 / 5 | **0.004** | +1.46 (< 1e-4) |
| B3 progress - C^pi v2 | 123 | 0.187 vs 0.463, -0.276 [-0.358, -0.195] | 1 / 35 | < 1e-4 | -1.83 (< 1e-4) |
| B3 progress - random | 123 | 0.187 vs 0.317, -0.130 [-0.195, -0.065] | 2 / 18 | 4e-4 | -0.36 (0.004) |
| B4-noPi0 - random | 125 | 0.440 vs 0.328, +0.112 [+0.040, +0.184] | 18 / 4 | 0.004 | +0.48 (0.012) |
| Cpi-pool - random | 125 | 0.536 vs 0.328, +0.208 [+0.120, +0.296] | 32 / 6 | < 1e-4 | +1.41 (< 1e-4) |

**Conclusion on the B4 question ("does B4 win because it is policy-agnostic, or because of more / re-filtered pi0
continuation labels?"): the latter.** Removing the pi0 labels from B4 (B4-noPi0) costs -0.12 solved (20 / 5,
p = 0.004) and halves the hit rate (0.093 vs 0.182 hits / sim), landing back at (not above) C^pi v2; keeping ONLY
B4's pi0 labels with B4's trainer (Cpi-pool) recovers essentially all of B4 (0.540 vs 0.556, 9 / 11, p = 0.82) and is
+0.064 over C^pi v2 (borderline, p = 0.12; same hit rate 0.156). So the value of the B4 critic comes from the
continuation labels OF THE POLICY THAT WILL CONTINUE (pi0) -- i.e. it is a competence critic with more / differently
weighted pi0 labels and a value-baseline trainer -- not from policy-agnostic success / progress supervision: the
labels from other policies (pi1, pi1more) and demos add nothing on top of pi0's labels, and on their own are only
random+0.11. Task-progress supervision (B3) is worse than random. The +0.08 of B4 over C^pi v2 is a better
trainer / label pool for the same competence objective (worth adopting as "C^pi v3" for future search, after a
held-out check), not evidence against the competence formulation.

## B2 SFT: progress-data vs count-matched competence-data SFT (2026-10-24)
Same recipe as M9 comp (init pi1-nominal, 6k steps, round-robin, 1 seed); the progress bank (B3-progress search,
127 failures) has only 60 verified trajectories, so the competence control is count-matched (n = 60 from the M5b
wm_diverse pool, `B2_sft_comp_n60`). `python scripts/analyze_baselines.py` -> runs/_analysis/baselines/summary.md.
Unit = test scene (56 x 10), scene (x seed) bootstrap, paired sign-flip p.

| comparison | test RSR A vs B | diff [95% CI], p | scenes A worse / better / tied |
|---|---|---|---|
| **progress SFT - competence n=60** | 0.177 vs 0.430 | **-0.254 [-0.334, -0.173], p < 1e-4** | 35 / 9 / 12 |
| progress SFT - competence n=66 (M9, 3 seeds) | 0.177 vs 0.417 | -0.240 [-0.316, -0.165], p < 1e-4 | 44 / 10 / 2 |
| progress SFT - pi1-nominal | 0.177 vs 0.138 | +0.039 [+0.004, +0.077], p = 0.052 | 5 / 18 / 33 |
| progress SFT - goal SFT (M9, 3 seeds) | 0.177 vs 0.187 | -0.010 [-0.062, +0.041], p = 0.72 | 23 / 18 / 15 |
| progress SFT - pi0 | 0.177 vs 0.118 | +0.059 [+0.029, +0.095], p = 7e-4 | 3 / 20 / 33 |
| competence n=60 - pi1-nominal | 0.430 vs 0.138 | +0.293 [+0.214, +0.371], p < 1e-4 | 7 / 38 / 11 |
| competence n=60 - competence n=66 (3 seeds) | 0.430 vs 0.417 | +0.014 [-0.018, +0.046], p = 0.35 | 30 / 22 / 4 |
| competence n=60 - goal SFT (3 seeds) | 0.430 vs 0.187 | +0.243 [+0.177, +0.312], p < 1e-4 | 11 / 40 / 5 |

Reading: at equal trajectory count, recoveries found with the competence critic teach the VLA ~0.25 more RSR than
recoveries found with the progress critic; progress data is barely better than not training on recoveries
(+0.04 vs pi1-nominal, p = 0.05) and equal to goal-objective data. Count matching does not change the M9 result
(n = 60 is within noise of n = 66 and of pi1). Both the search (B3 below random) and the distillation step agree.
Nominal for these two SFTs was not run (not in the DAG; same recipe as M9 comp, nominal 0.77-0.81).

## Baseline summary (Spatial test = dev split; 2026-10-24 14:50 UTC)
`python scripts/analyze_baselines.py` -> runs/_analysis/baselines/summary.md (re-run when pending rows finish; it
skips evals that are still running). RSR: 56 test scenes x 10 trials; dRSR vs pi1 = scene (x seed) bootstrap 95% CI,
paired sign-flip p, pi1 = 3 seeds. Nominal: LIBERO-Spatial 10 tasks x 50 inits; dNSR = 10-task bootstrap vs
pi1-nominal; pre-registered constraint eps = 2 pt (**< -eps** = point estimate violates it). Sim budget = env
transitions spent to produce the training signal (PoC accounting, runs/_analysis/b33/budget.json).

| method | seeds | test RSR | nominal | dRSR vs pi1 [95% CI], p | dNSR vs pi1-nominal [95% CI] | extra test-time cost | sim budget |
|---|---|---|---|---|---|---|---|
| pi0 (SmolVLA base) | 1 | 0.118 | 0.656 | -0.301 [-0.382, -0.218], p < 1e-4 | -0.140 [-0.182, -0.100] **< -eps** | none | 0 |
| pi1-nominal (demo SFT; init of all SFT rows) | 1 | 0.138 | 0.796 | -0.281 [-0.371, -0.196], p < 1e-4 | (reference) | none | 0 |
| demo-only (+6k steps, same init / steps as pi1) | 1 | 0.127 | 0.820 | -0.292 [-0.380, -0.207], p < 1e-4 | +0.024 [-0.028, +0.074] | none | 0 |
| random-data SFT (208 random-search recoveries) | 1 | 0.252 | 0.752 | -0.167 [-0.236, -0.104], p < 1e-4 | -0.044 [-0.082, -0.002] **< -eps** | none | 1.40M |
| goal-data SFT (66 WM latent-goal recoveries) | 3 | 0.187 ± 0.002 | not run | -0.232 [-0.306, -0.161], p < 1e-4 | - | none | 2.09M |
| progress-data SFT (60 WM + B3-progress recoveries) | 1 | 0.177 | not run | -0.242 [-0.323, -0.162], p < 1e-4 | - | none | 2.09M |
| competence SFT, count-matched n=60 | 1 | 0.430 | not run | +0.012 [-0.037, +0.062], p = 0.62 | - | none | 2.90M |
| competence SFT n=66 (M9 comp) | 3 | 0.417 ± 0.011 | not run | -0.002 [-0.050, +0.048], p = 0.95 | - | none | 2.90M |
| **RetryVLA pi1 (208 verified WM recoveries)** | 3 | **0.418 ± 0.021** | **0.785** | (reference) | -0.011 [-0.046, +0.024] | none | 3.47M |
| online best-of-8 on pi1 (WM + critic every replan) | 1 | 0.466 | not run | +0.027 [-0.018, +0.070], p = 0.27 (vs pi1 seed 0) | - | 8 VLA samples + 8 WM rollouts + critic / replan | 3.47M + 0 |
| online best-of-8 on pi0 | 1 | 0.118 | not run | -0.301 [-0.383, -0.221], p < 1e-4 | - | same | 2.21M |
| RISE-inspired cond (imagined adv., pool) | 1 | 0.068 | 0.718 | -0.351 [-0.433, -0.270], p < 1e-4 | -0.078 [-0.124, -0.034] **< -eps** | none | 2.21M (0 verify) |
| RISE-inspired AWR (imagined adv., pool) | 1 | 0.082 | 0.776 | -0.336 [-0.421, -0.255], p < 1e-4 | -0.020 [-0.060, +0.018] | none | 2.21M (0 verify) |
| RISE-inspired cond (imagined adv., VLA samples) | 1 | 0.113 | 0.770 | -0.306 [-0.386, -0.227], p < 1e-4 | -0.026 [-0.068, +0.018] **< -eps** | none | 2.21M (0 verify) |
| residual RL p25 (PPO bridge on frozen pi0) | 1 | 0.111 | 0.400 | -0.308 [-0.388, -0.229], p < 1e-4 | -0.396 [-0.534, -0.266] **< -eps** | DINOv2-small + MLP, first 50 steps | 0.87M |
| residual RL full (PPO bridge on frozen pi0) | 1 | 0.295 | **0.022** | -0.124 [-0.201, -0.056], p = 6e-4 | -0.774 [-0.840, -0.712] **< -eps** | DINOv2-small + MLP, first 50 steps | 3.47M (= PoC) |

Budget detail: random SFT 1.40M = verify 1.26M + precheck 0.13M (256 failures); goal / progress 2.09M = WM data 1.39M +
~0.70M verify / precheck (127 failures; no critic sims); competence 2.90M = + critic 0.82M; PoC 3.47M = WM 1.39M +
critic 0.82M + verify 1.26M + precheck 0.13M (256 failures). Reading: among methods with no extra test-time
component, only competence-searched, sim-verified data reaches pi1's RSR while keeping nominal within eps;
imagination-only (RISE-inspired) is below pi0; the only baseline that learns real recovery at the same budget
(residual RL, +0.18 vs pi0) is still -0.12 below pi1 and destroys nominal (0.022) because its controller fires on
every episode start (B3.3 diagnosis).

## P3 recovery-nominal Pareto (RUNNING, 2026-10-24 14:50 UTC; seed 0, no finished point yet)
Points: recovery ratio r = 0.1 / 0.3 / 0.7 and an L_preserve variant (SFT + test RSR + nominal each; DAG b2p3).
r = 0.1: SFT done, test RSR running (interim 0.33 at 280/560 -- interim, not a result), nominal running.
r = 0.3: SFT done 14:42, evals queued. r = 0.7: SFT at step 3950 / 6000. preserve: SFT at step 4600 / 6000.
