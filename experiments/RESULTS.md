# RetryVLA results (LIBERO-Recover Spatial)

Eval protocol: test RSR = 56 held-out test scenes x 10 trials (5 close + 5 open gripper init, ±2 cm object noise),
520 steps, no search at test time. Drawer = the 16 drawer test scenes (160 episodes). Nominal = LIBERO-Spatial
10 tasks x 50 inits. One training seed per row unless noted. Updated 2026-10-16.

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

Paired tests (same 560 test episodes): #7 vs #6 +0.116, scene-bootstrap 95% CI [+0.045, +0.191], McNemar p = 4e-7.
#12 vs #8 +0.187.
M6: #16 vs #12 +0.041 [+0.004, +0.077], p = 0.04 (extra steps help). #15 vs #16 -0.018 [-0.055, +0.021], p = 0.42:
round-2 data gives NO gain beyond extra training.
M7: #18 vs #17 -0.009 [-0.050, +0.034], p = 0.72; on stove + cookie-box test episodes (80) 8/80 vs 7/80.
#17 vs #16 +0.023 (p = 0.21): returns from extra training are flattening.

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
