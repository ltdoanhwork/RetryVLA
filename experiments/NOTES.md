# RetryVLA: status, findings, and backlog (as of 2026-10-14)

All numbers: LIBERO-Spatial, LIBERO-Recover `selected_bddl_scenes_spatial`, split `data/splits/spatial_v1.json`
(train 112 / pilot 60 / test 56 scenes, grouped by failure episode). Test RSR = 56 scenes x 10 official trials = 560 ep.
Run rows are in `experiments/REGISTRY.md`; details in `M0_baseline.md` and `M1_oracle_search.md`.

## 1. Results so far

### Policies (test RSR, nominal LIBERO-Spatial SR)

| Policy | Recovery data | mix | test RSR [95% CI] | nominal SR |
|---|---|---|---|---|
| pi0 = smolvla_libero | - | - | 0.118 [0.094, 0.147] | 0.656 (lerobot-eval: 0.680) |
| pi1-nominal (SFT demos only, 6k) | - | 0 | 0.138 [0.111, 0.169] | **0.796** |
| pi1-random | 241 random-search traj (pilot, easy+hard) | 0.50 | **0.279** [0.243, 0.317] | 0.698 |
| pi1-CEM (old Gaussian CEM) | 241 CEM traj | 0.50 | **0.286** [0.250, 0.324] | 0.690 |
| r25 | 241 random traj | 0.25 | 0.254 [0.219, 0.291] | 0.726 |
| r25 nontrivial | 170 random traj (>=16 steps) | 0.25 | 0.220 [0.187, 0.256] | 0.766 |

### Search / data generation (pilot instances, simulator-verified)

| Search | Instances | Budget / instance | Solved | Notes |
|---|---|---|---|---|
| random (M1) | 108 | 64 | 0.69 | |
| Gaussian CEM (M1) | 106 | <=256 | 0.72 | refit broken (mode averaging): 11% -> 1% success after refit |
| elite-local CEM (M1-v2), 40 hard | 40 | <=256 | **0.68** vs random 0.55 | success/sample 0.8% -> 3% after refit |
| **WM + critic (M3)**, 53 hard, unseen by WM | 53 | **16 verified** | **0.36** vs random-16 0.17 | 57/848 vs 13/848 verified hits; p=0.04; 35/45 from latent CEM |

### World model / critic (M2)
- C1 data: 144 train instances (hard-instance failure episodes excluded) x 32 candidates = 4,608 traj,
  DINOv2-small 6x6 tokens x 2 cams every 5 steps; 16% positive. `data/wm/collect_v1` (2.7 GB).
- C2 WM: open-loop MSE / copy baseline on held-out failures: 0.63 (5 steps) ... 0.36 (50 steps).
- C3 critic: AUC 0.71 real / 0.67 imagined; top-4 lift 0.33 vs 0.16; **badly calibrated** (ECE 0.15, V~1 for picks).

## 2. Known problems (must address before claims)

1. **Weak base policy.** smolvla_libero = 0.66-0.68 nominal (lerobot-eval agrees), reported SmolVLA ~0.90.
   SFT on demos alone gives 0.80. Reviewers will ask for a stronger base. Use pi1-nominal as the base (init_from).
2. **Nominal regression.** Recovery SFT drops nominal 0.80 -> 0.69-0.73. Trade-off controlled by mix ratio.
3. **WM gain concentrated in one task** (drawer: WM 12/20 vs random 1/20; others 7/33 vs 8/33). N=53, p=0.04.
4. **Data quality.** Searched segments are 4-6x jerkier than demos, with gripper chatter (3-4 flips / 50 steps vs 0.9).
   Trivial (<16-step) recoveries are useful (removing them hurt RSR), so do not filter them.
5. **Imbalanced data.** 29/45 WM recoveries are drawer. Ramekin / drawer dominate the scene set.
6. **Single seed** for every SFT/eval. Differences < ~4 RSR points are noise on test.
7. **Why a WM if the simulator exists?** Need a sim-efficiency curve (verified recoveries vs simulator calls),
   and the "no simulator at the target" argument (real robot / unseen environments) for the paper story.

## 3. Backlog (ideas, not run)

Search yield (successes per verification):
- [ ] Self-improving loop: retrain critic (+WM) on the 848 M3 verifications (labels exactly where the WM is wrong),
      then re-run C4. Cheap; most likely yield gain.
- [ ] Calibrate the critic (temperature scaling on held-out); use uncertainty for exploration, not just penalty.
- [ ] Budget allocation across instances (bandit): stop solved ones, spend on unsolved / under-covered tasks.
Diversity / quality:
- [ ] Diversity-aware top-k (cluster imagined end latents or action modes, best per cluster; DPP / MAP-Elites).
- [ ] Parameterized primitives (lift height, retreat dir, gripper) searched by the WM: smoother, interpretable.
- [ ] Gripper "latch" (only flip when finger state changes) + stronger smoothness cost; smooth segments before saving.
- [ ] Multi-chunk search (beam / MCTS over 10-step chunks) for multi-stage recoveries.
Distillation:
- [ ] C5: SFT with WM-found recoveries (+ random/CEM data); C6: test RSR + nominal.
- [ ] Init from pi1-nominal; lower LR / longer training; recovery segments only; to fix nominal regression.
- [ ] Advantage-weighted SFT (M5) once the critic is calibrated.
- [ ] Iterate pi1 -> mine its remaining train failures -> search -> pi2 (Stage 6).
Scale / rigor:
- [ ] 3 seeds for the final SFT arms; report per-task RSR.
- [ ] More suites (Object, Goal, LIBERO-10) after Spatial is solid.
- [ ] L1-L4 breakdown once LIBERO-Recover releases level labels.

## 4. How to run the pending cells (do NOT run without approval)

```bash
# WM / critic / M3 (C1 data exists)
ALLOWED_SLOTS="<shell8:gpu ...>" scripts/shells/pipeline_m2m3.sh 4 2       # C2 -> C3 -> C4
scripts/shells/launch_remote.sh <shell8> <gpu> m2_wm     retryvla.wm.train_wm         configs/exp/m2_train_wm.yaml
scripts/shells/launch_remote.sh <shell8> <gpu> m2_critic retryvla.critic.train_critic configs/exp/m2_train_critic.yaml
scripts/shells/launch_remote.sh <shell8> <gpu> m3_s0     retryvla.search.wm_search    configs/exp/m3_wm_search_hard.yaml m3.shard=0 m3.num_shards=2
# SFT + eval (pattern)
scripts/shells/launch_remote.sh <shell8> <gpu> sft_x retryvla.train.sft configs/exp/<sft>.yaml
scripts/shells/launch_remote.sh <shell8> <gpu> ev_x  retryvla.eval.eval_recover configs/exp/m0_baseline_spatial.yaml \
    milestone=M1 recover_tag=<tag>_recover_test policy.path=<ckpt> "recover_eval.splits=[test]"
scripts/shells/launch_remote.sh <shell8> <gpu> ev_n  retryvla.eval.eval_nominal configs/exp/m0_baseline_spatial.yaml \
    milestone=M1 nominal_tag=<tag>_nominal policy.path=<ckpt> eval.save_videos=none
```
Free GPUs: `python3 scripts/shells/find_free.py` (avoid-list in `scripts/shells/shells.json`).
Visualizations: `scripts/viz_trajectories.py`, `scripts/viz_wm_data.py`, `scripts/viz_compare_wm.py`, `scripts/analyze_m1.py`.

## 5. Assessment (2026-10-14)

Feasible, with two positive signals: (a) distilling simulator-verified recoveries roughly doubles RSR on held-out
failures beyond a fine-tuning control; (b) a latent WM + competence critic finds recoveries on hard, unseen failures
about 2x more often than random at equal simulator budget, mostly via candidates it proposed in imagination.
Both signals are small-N, single-seed, on one suite, from a weak base policy, and with a nominal-SR cost. The
missing link is (c): does WM-generated data improve the VLA? Recommended order: C5/C6 on a stronger base (pi1-nominal
init) with and without WM data, then the self-improving critic loop, then diversity-aware selection.

## 6. M4 (2026-10-15): does WM-found data help the VLA? (init from pi1-nominal, 6k steps, mix 0.5)

| Arm | Recovery data | test RSR | drawer (test) | nominal SR |
|---|---|---|---|---|
| pi1-random (init pi0) | 241 random | 0.279 (156/560) | 18/160 | 0.698 |
| A (init pi1-nominal) | 241 random | 0.293 (164/560) | 24/160 | 0.756 |
| B (init pi1-nominal) | 241 random + 45 WM (hard instances) | 0.298 (167/560) | 28/160 | 0.748 |

Paired test episodes: B-only success 61, A-only 58 -> **no measurable overall difference** (single seed).
Drawer +4/160, consistent with WM data being drawer-heavy (29/45), but within noise. The 45 WM trajectories are only
~16% of recovery data / ~8% of samples: too small a dose to move test RSR. Stronger init helps a little (0.279 -> 0.293).
Bug found: run_m4 released claims by slot, which killed the following eval on the same GPU ~100 times
(fixed: release by session). Interrupted run dirs were deleted.

## 7. M5 plan (code ready, NOT run): yield + diversity + matched distillation

DAG `configs/dag/m5.json`, runner `scripts/shells/dag_run.py <dag> --slots <shell8:gpu ...>`:
1. `m5_r1_s{0,1}`: `retryvla.search.wm_datagen` on all 108 pilot pi0-failure instances. Diverse top-k (k-means on
   imagined end states), adaptive budget (batches of 8, stop at 4 successes, max 32 sims), every verification logged in
   C1 latent format (`data/wm/verif_r1`). Data -> `data/recovery/m5_wm_r1_s*`.
2. `m5_critic_v2`: critic retrained on C1 + round-1 verifications (round-2 held-out instances excluded from verif only).
3. `m5_r2_v1` / `m5_r2_v2`: same fixed 16-sim budget on the 50 held-out instances, critic v1 vs v2 (self-improvement).
4. `m5_sft_wm` / `m5_sft_rand`: SFT from pi1-nominal, WM-only vs random-only recoveries, SAME trajectory count
   (`@ntraj` = min(241, WM count)). Then test RSR + nominal for both.
Estimated: ~6 h on 2 GPUs (R1 ~2.5 h, critic 15 min, R2 ~40 min, SFT 45 min x2, evals ~45 min x4).

## M5b: scaled paired comparison of search arms (launched 2026-10-15 11:25 UTC)

Question: does WM guidance give both efficiency AND better-distributed recovery data? N = 288 fresh instances
(train scenes x object-noise trials 2..9, never seen by WM/critic) -> ~250 pi0 failures, 8 shards.
Per instance, 4 arms each get 16 sims (paired: same instance, same continuation seeds per slot):
`random` (pool uniform), `wm_diverse` (M5 default), `wm_noise` (CEM in SmolVLA flow-matching noise space, pCN
moves -> on-manifold by construction), `wm_qd` (MAP-Elites over 60 strategy cells). Logged: hits/solved,
critic calibration, jerk / roughness / distance to SmolVLA samples / pairwise spread / strategy cells of the
verified successes. Then matched-count SFT per arm (init pi1-nominal) -> test RSR + nominal SR.
Code: `retryvla/search/{arms,compare_arms}.py`, DAG `configs/dag/m5b.json` (also finishes the M5 evals that
were cut when 04a5937c / aa1cf25d were reclaimed), analysis `scripts/analyze_m5b.py`.
Smoke (5 inst, budget 4): solve random 0.4 / wm_diverse 0.8 / wm_noise 0.6 / wm_qd 0.4 (no signal, N tiny).

### M5 / M5b results (2026-10-16)
Test RSR (560 ep, init pi1-nominal, paired by episode; scene-bootstrap CI):
- M5: WM r1 241 traj **0.402** vs random 241 traj 0.286: +0.116 [+0.045, +0.191], McNemar p=4e-7. Nominal 0.78 vs 0.762.
  Gain is drawer (0.36 vs 0.15) and on_the_ramekin (0.62 vs 0.35).
- M5b search (N=256 paired fresh failures, 16 sims/arm): solve random 0.363 / wm_diverse **0.547** / wm_noise 0.289 /
  wm_qd 0.453. Hit rate 0.062 / **0.171** / 0.076 / 0.113. wm_diverse > random p=4e-9 (McNemar). Drawer (83 inst):
  0.07 / 0.43 / 0.02 / 0.16. wm_noise is the smoothest, lowest jerk and closest to SmolVLA, but it rarely solves the hard ones (it stays on
  pi0's failing manifold). wm_qd has the most strategy cells (4.0 per solved instance).
- M5b SFT (208 matched, score top-n): 0.252 / 0.298 / 0.270 / 0.289. BUG: the score top-n cut dropped drawer
  trajectories (wm_diverse 4 left, M5 r1 had 42), so drawer test RSR went to 0.03-0.11. Fixed: `train.subset_mode=roundrobin`.
  Rerun as DAG m5c: wm_diverse_rr, wm_qd_rr (208), wm_diverse_all (423).
- m5c (round-robin subset, init pi1-nominal), test RSR / nominal: wm_diverse_rr 208 **0.439 / 0.808**; wm_qd_rr 208
  0.354 / 0.754; wm_diverse_all 423 0.389 / 0.788 (more data != better; 1 seed each).
- E2E viz (`scripts/viz_e2e.py`, 5 seeds, no search at run time), drawer: train ep031 open6: pi0 0/5, pi1-nom 0/5,
  pi1-random 0/5, pi1-WM 2/5. TEST ep040 close1: 0/5, 0/5, 0/5, **3/5**. pi1-WM lifts out of the drawer, then regrasps.

### M6 round 2 (pi1 -> pi2), 2026-10-17
- Mined 219 pi1 failures (400 train trials, pi1 SR 0.45). WM diverse + pi1 continuation: solved 0.58, robust (>=2/3 seeds)
  0.45, hits/sim 0.188, 275 robust trajectories. Cookie-box / stove failures: few and rarely robust.
- Test RSR: pi1 0.439 | control (pi1 + 6k steps on r1 data) **0.480** | pi2 (r1+r2, 600) 0.463. Nominal 0.808 / 0.804 / 0.828.
  pi2 vs control -0.018 (p=0.42): the loop adds nothing beyond more training; control vs pi1 +0.041 (p=0.04).
- Likely reasons: r2 data concentrates on tasks pi1 already handles (drawer/ramekin); the test failures that remain
  (stove 0.05-0.10, cookie box 0.00) have almost no robust recoveries in train; critic is still pi0-competence-trained.

### M7 weak tasks, 2026-10-18
- Search on ALL train trials of stove / cookie-box scenes (160, best policy SR 0.36 -> 102 failures). Robust solve: stove ~0.55,
  on_cookie_box 0.72, next_to_cookie_box ~0.05. Arms nearly tied (diverse 0.47, QD 0.50, two-stage 0.50 robust).
- SFT with 155 weak-task trajectories: test weak tasks 8/80 vs control 7/80, overall 0.495 vs 0.504 (n.s.).
  Search WORKS on train scenes but does not transfer to the 8 test scenes of these tasks: only 10 stove / 6 cookie-box
  train scenes, so recoveries are scene-specific. Next: check train-split RSR of M7 weak (memorize vs not learned),
  and more scene diversity (augment object/plate positions) for these tasks.
- Every SFT variant hurts next_to_cookie_box vs pi0 (0.25 -> 0.05-0.15).

### M8 step 1: memorize vs not learned (2026-10-18)
Train-split stove/cookie-box (160 trials): pi1 0.331, M7 base 0.431, M7 weak 0.487. On the 57 trials whose robust
recovery is in the SFT data: 0.25 / 0.33 / 0.47; on searched-but-unsolved trials 0.00-0.04. Test weak: 0.04 / 0.09 / 0.10.
=> partial in-scene learning that does not transfer. Also pi0 itself: pilot weak 0.26 vs test weak 0.09 (test scenes
of these tasks are intrinsically harder). Step 2: +-5 cm, 30 trials/scene augmentation (DAG m8bc), then 3 SFT seeds.
### M8 step 2 (+-5 cm augmentation), 2026-10-19
480 trials -> 340 failures, robust solve 0.36, 316 trajectories. SFT (800: 457 weak / 343 other):
test weak 16/80 (M7 base 7/80, p=0.05; M7 weak 8/80), next_to_cookie_box 0.35, stove 0.20; train weak 0.525.
Overall test 0.457 vs 0.504 (p=0.03): other tasks drop (cabinet 0.38->0.30, drawer 0.49->0.42, ramekin 0.68->0.60).
=> scene diversity DOES transfer to test for the weak tasks; the cost is data imbalance. Next: rebalance (cap weak share
~25-30%, or mix ratio per task) and/or start from the M8-aug data but upweight other tasks.
