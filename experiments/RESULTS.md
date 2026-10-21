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
