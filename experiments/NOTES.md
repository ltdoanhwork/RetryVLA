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

## M9 plan (2026-10-19, after external review): mechanism > scale
Q1 why competence, Q2 why a WM, Q3 why distill. Queue (DAG m8m9, 3 lanes on 49b70230):
- Phase 1: 3 SFT seeds (pi1, M7 base, M7 weak) [running]; retrospective analysis of M5b logs (scripts/analyze_retro.py):
  budget curve k=1..16, critic calibration, dG vs dV on real end states.
- Phase 2 Exp1+2 (m9_mech, shards 0-3 of the M5b instances, paired with M5b random / wm_diverse): wm_goal (DINO-WM
  goal-latent objective, goals = final frames of successful demos, data/wm/goal_feats.pt), wm_goal_comp, wm_top
  (no diversity), random_diverse (no WM / critic), sim_prefix_critic (real dynamics + critic, prefix transitions charged).
  Downstream: SFT on wm_goal vs competence recoveries, matched count (m9_sft_goal / m9_sft_comp).
- Exp3: budget 32 for random / wm_diverse (2 shards) + sim transitions logged per arm.
- Phase 3 Exp4: demo-only SFT with matched steps; online best-of-8 (WM + critic at every replanning step) on pi0 and pi1.
- Exp6: data selection on the same bank: score / random / round-robin / task-balanced.
- Not yet: RL baseline, LIBERO-Object final held-out, M8-balanced (dropped: would be tuned on test).

### M9 Phase-1 retrospective (M5b logs, N=256), runs/_analysis/retro/
- Budget: WM diverse with k=4 verifications solves 0.39 of failures > random with k=16 (0.36) -> ~4x fewer sims.
- Critic on imagined end states: AUC 0.69-0.80 but over-confident on WM picks (mean pred 0.735 vs 0.171 verified, ECE 0.16).
- Real end states: dV predicts verified success (AUC 0.694), goal-latent progress dG does not (AUC 0.447, below chance).
  50% of verified recoveries moved AWAY from the goal latent while raising competence (ramekin 77%, drawer 12%).
  Illustrative for Q1; the controlled test is the wm_goal arm in m9_mech.
- 2026-10-21: Object search shards all crashed at ~01:00 UTC on a /pfss write error (torch.save of the verif log).
  Salvaged parts (152 instances done, datasets readable) marked CLOSED, verif-log save now retries; resumed 95 remaining.
  dag_run left stale local claims after the crash (deadlock) -> now drops dead local-lane claims every poll.

### M10 Object outcome (held-out, frozen recipe)
Replicates: recovery distillation >> demo-only (+0.18); competence >> goal in SEARCH (66/8) and directionally in SFT
(+0.11, p=0.08, 26 scenes). Does NOT replicate: WM competence data > random data for the VLA (0.354 vs 0.362, n.s.).
On Object random search is already strong (pi0 continuation good: solved 0.405 vs Spatial 0.31) and WM search gain is
small (0.46 vs 0.41). With equal count (123) the extra WM recoveries (340 vs 215 bank) are unused. Claim to drop or
reframe: "WM data quality > random"; keep: WM = more recoveries per sim (2.2x hits/sim), competence > goal.

## B3.2 RISE-inspired baseline (2026-10-22): WM + advantage learning, NO search verification

Question: can the same WM + critic improve the VLA directly from IMAGINED outcomes (RISE, RSS 2026 style:
advantage from a learned value on imagined rollouts, no per-sample real verification), instead of the PoC's
verify-in-sim-then-distill? Inspired by RISE, not a reproduction.

Adaptation (and the KEY limitation): our WM predicts DINOv2 latents (6x6 patches x 2 cams + state), not RGB, so
SmolVLA cannot act closed-loop inside the WM. The baseline is therefore CHUNK-LEVEL and one-step: at each real
failure state s_0, candidate 50-step chunks are imagined open-loop, A = S(chunk) - V(s_0) with
S = V(s_hat_50) - lambda U_n std(V) - beta roughness (the PoC search objective, critic v2 on the imagined end state;
V(s_0) = critic on the real failure latent), and the VLA is trained on (obs_0, chunk) pairs. Training never sees an
intermediate observation (no rendering of imagined states) and no continuation frames.
- Data (`retryvla/baselines/rise_lite.py`, `configs/exp/b32_rise_datagen.yaml`): the 256 M5b pi0 failures
  (compare_arms.select_jobs with the M5b keys; pi0-failure labels from the M5b pre-check; states re-created with the
  official restore, 0 policy steps). `proposal=pool` replays the PoC Proposals RNG in M5b shard order and runs the
  same arm_wm_diverse latent CEM, i.e. EXACTLY the PoC candidate set (smoke: 94-100% of the 16 PoC wm_diverse picks
  reproduced bit-for-bit). `proposal=vla` = 696 SmolVLA samples (scales 1.0 / 1.5), closer to RISE (policy's own samples).
- Main variant = advantage CONDITIONING (`retryvla/baselines/sft_rise.py mode=cond`): per failure state, top 30% of A
  with A > 0 -> "<task> Advantage: positive.", bottom 50% -> "... negative."; failure states sampled uniformly,
  50/50 pos/neg; nominal demos tagged positive; at test time every instruction gets the positive suffix
  (`retryvla/baselines/eval_cond.py`, reads `<ckpt>/rise_condition.json`). Chosen because it is what RISE / RECAP do,
  and because it also learns from the NEGATIVE imagined outcomes (the information imagination adds over best-of).
- Second variant = AWR (`mode=awr`): w = exp(clip(A_hat, +-2.5) / 0.5), A_hat standardized within the failure state,
  implemented as importance sampling of the chunk (same expected gradient, lower variance); plain eval.
- Imagined chunks are clipped to [-1, 1] for training (latent-CEM samples reach +-2; the env clips anyway).

Matched to the PoC (pi1 = M5c wm_diverse_rr): init pi1-nominal; 6k optimizer steps; effective batch 32 (16 x 2
grad accumulation, shared GPU); lr 5e-5 cosine, warmup 300; 50% nominal LIBERO-Spatial demo replay; same 256 failure
states; same frozen pi0, WM v1, critic v2, lambda / beta; same WM budget per failure state = 696 rollouts (the M5b
Proposals(n_total=256) pool is 232 = 96 SmolVLA + 96 perturbations + 40 primitives, + 2 x 232 latent CEM; NOT 768);
same eval (56 test scenes x 10, nominal 10 x 50). Not matched / different by design: ZERO simulator transitions for
candidates (the PoC spends 16 verifications / failure, ~5k transitions incl. continuation, ~1.3M for the bank) and no
pi0 pre-check (reused); recovery samples are single (obs_0, chunk) pairs, not trajectories with continuation; every
failure state is used (the PoC only distils the ~55% it solved). Not done: the multi-step on-policy variant (execute
the best imagined chunk in the sim, re-plan from the next real state), skipped because the node is CPU-saturated.
Smoke (3 failures -> 50-step SFT -> eval_cond on 1 test scene, 10 ep): runs end to end. Per-source loss of pi1-nominal:
demos 0.36, SmolVLA samples 0.66-1.74, primitives 0.76, perturbations 14.7, latent-CEM 6.9; positives are ~93%
latent-CEM chunks, i.e. imagination-only positives are off the policy manifold (RISE-faithful `vla` arm tests this).
DAG `configs/dag/b32_rise.json` (lane local:0:e, tmux rv_dag_b32): datagen pool -> cond SFT -> test RSR + nominal ->
AWR SFT + evals -> datagen vla -> cond(vla) SFT + evals. Analysis: `python scripts/analyze_b32.py` (scene-level paired
bootstrap vs pi1 seed 0 and the 3 pi1 seeds, vs demo-only; nominal; precision of imagined labels on the chunks the
PoC verified -- analysis only, never used for training).
- 2026-10-22 incident: `b32_sft_cond_pool` died at step 5150/6000 (12:02 UTC) in the /pfss EDQUOT window
  (11:39-12:03): root cause = TensorBoard's async event writer hit `[Errno 122] Disk quota exceeded`, and its
  re-raised OSError from `run.log` killed training (checkpoints 2000 / 4000 were intact but had no optimizer state).
  Fix in `sft_rise.py`: metric logging is non-fatal (TB disabled on error), checkpoints are written to
  `<step>.tmp/` + renamed (retry 60..480 s back-off, ~36 min), newest checkpoint carries `training_state.pt`
  (optim / sched / step / EMA / RNG), and `train.resume=auto` continues the newest unfinished run dir with an
  identical recipe; data stream is a pure function of (seed, item), so a resumed run sees the same batches
  (smoke: kill after step 3 of 6, relaunch -> resumed at 3, same lr / recovery-fraction trace, status done).
  The failed run could not be resumed (no optimizer state) -> retrained from scratch; failed log -> `failed_*`;
  its partial 2000/4000 checkpoints deleted to free quota. DAG relaunched 2026-10-23 02:25 UTC (lane local:0:e);
  that first relaunch died at once: after the container restart the shared tmux server carries another tool's
  ~90-char TMPDIR, so DataLoader workers failed with `OSError: AF_UNIX path too long` (sockets under $TMPDIR);
  `sft_rise` / `eval_cond` now fall back to `/tmp/rv_<uid>` when TMPDIR is longer than 60 chars (smoke OK under
  that TMPDIR; plain eval_recover is unaffected, SimPool uses socketpairs). Relaunched 02:40 UTC.
- 2026-10-23 results (pool arms; RESULTS.md B3.2, runs/_analysis/b32/summary.md): cond RSR 0.068 / NSR 0.718,
  AWR RSR 0.082 / NSR 0.776, both BELOW pi1-nominal (0.138, scene-paired -0.070 p<1e-4 / -0.055 p=0.004) and below
  pi0 (0.118 test part; -0.050 / -0.036), vs pi1 (3 seeds) -0.35 / -0.34. NSR: cond -7.8 pt (fails eps = 2 pt),
  AWR -2.0 pt (exactly at the margin; 1 seed). Diagnosis, zero new sim cost (the 8740 imagined chunks that M5b
  executed in sim carry their real chunk + pi0-continuation outcome, matched bit-for-bit): critic is badly
  over-confident on imagined end states -- positives V 0.81 vs real 0.185, V > 0.9 -> real 0.23, AUC 0.67; argmax-A
  chunk / state real 0.205. 18% of chunks have A > 0; positives / AWR mass are 85% / 90% latent-CEM chunks (jerk 0.090
  vs pi0 samples 0.065; gripper flips 1.9 vs 6.0), pi0's own samples only 3%. Caveat: verified chunks are the
  PoC-picked (high-score) subset, so precision is measured where the critic is most confident. No new sim
  verification was run (the reused labels cover 91% of the argmax-A chunks). Interpretation: imagination-only learning
  distils ~80% false positives that are off the policy manifold; RISE-faithful `vla` arm (positives drawn from SmolVLA
  samples) tests whether staying on-manifold helps -- its datagen shows frac(A > 0) ~0.02 per state, so most states
  will have almost no positives. Remaining DAG (lane local:0:e): datagen_vla (34 s / state, done ~17:00 UTC) ->
  sft_cond_vla (~2.7 h) -> rec (~2.3 h) -> nom (~1 h); ETA ~23:00 UTC.

## B3.3 residual recovery RL baseline (ReCoVLA-inspired), 2026-10-22

Name in the paper: **residual recovery RL baseline (ReCoVLA-inspired)**. Simple level: task-success (+ optional shaped)
reward, no VLM reward compiler, not a ReCoVLA reproduction. Systems-level comparison: it needs an extra controller at
test time (DINOv2-small 22M frozen + 2.4M trainable MLP); pi1 has no extra module.

**Architecture** (`retryvla/baselines/recovery_rl/`). Frozen pi0 (smolvla_libero, n_action_steps 10, noise 1.0) +
residual BRIDGE controller, active only for the first H = 50 env steps of an episode (= RetryVLA's bridge length),
then pi0 alone until success or 520 steps. Chosen over an always-on residual because it matches RetryVLA's structure
(bridge, then hand back to the unchanged policy) and cannot hurt pi0 after the window; always-on is the obvious
alternative (not run). One decision per pi0 chunk (5 per episode, closed loop at chunk boundaries like pi0 itself).
Input: DINOv2-small patch tokens of both cameras pooled 2x2 (3072), 8-D state, hashed bag of uni/bi-grams of the
instruction (256-D, fixed; test instructions are paraphrases), pi0's planned 10x7 chunk, decision index. Output:
6-D Gaussian arm residual, executed as 0.5 * clip(a, -1, 1) added to every step of pi0's chunk, and a 3-way gripper
head {follow pi0, force open, force close}. Init: residual mean ~0, gripper biased to follow (p 0.79), so the
untrained controller == pi0 at eval (deterministic: mean / argmax). Separate actor / critic MLPs.
Eval hook: `residual.ckpt=<ckpt>` in eval_recover / eval_nominal (wrapper `ResidualBridgePolicy`; episode step decoded
from the run_jobs seed convention, rollout.py untouched). In nominal eval the bridge is also active for the first 50
steps (a deployed system does not know whether it starts in a failure state).

**RL.** PPO, gamma 1, GAE 0.95, clip 0.2, 4 epochs, minibatch 64, lr 3e-4, ent 0.003, 32 episodes / iteration
(8 instances x 4 pi0 seeds, worker-scene affinity: an env rebuild costs 5-30 s on this node). Reward on the last
bridge decision: +1 if the episode succeeds (in the bridge or during pi0's continuation); `b33_rl_shaped` adds
2.0 x progress() delta per decision - 0.05 x mean(delta^2) (configured, not queued: not cheap on this node).
Episodes start from the SAME 256 M5b train failure states pi1's data came from (select_jobs of compare_arms, all
8 shards, filtered to the M5b instance ids; exact load_scene snapshot restored with `restore_snapshot`, as in the
compare_arms verification jobs). Training pi0 seeds in [0, 1e9); learning-curve eval: 128 of the train instances,
held-out pi0 seeds (1.5e9 + i), deterministic residual, at 0 and every 350k transitions (not charged).

**Budget accounting** (`scripts/b33_budget.py` -> `runs/_analysis/b33/budget.json`; env transitions after the
failure state, gripper settle excluded, exact from logged success_step except the precheck):

| component | rollouts | transitions | note |
|---|---|---|---|
| WM v1 + critic v1 data (`collect_v1`, 144 train inst x 32, cap 350) | 4,608 | 1,390,909 | amortized, charged fully |
| critic v2 extra data (`verif_r1`, 108 pilot inst, 8-32 sims) | 2,552 | 815,119 | critic v2 is the M5b critic |
| verification, wm_diverse arm (`verif_m5b`, 256 failures x 16) | 4,096 | 1,262,993 | 4.93k / failure (log: ~5.0k) |
| **PoC total = RL budget** | | **3,469,021** | 13.6k / failure |
| precheck (find pi0 failures, 288 x <=520) | 288 | ~135k (estimated) | shared by both, not matched |
| RL residual PPO | | 3,469,021 (25% = 867,255 first) | all bridge + continuation steps |

M5b per_instance.jsonl has no `sim_steps_verify` (only newer arms): verification was recomputed from the verif logs.
Charging all WM data to pi1 is conservative in favour of RL (collect_v1 also served M3, M5, M5b's 4 arms, M9, M11).
The curve also marks the verify-only budget (1.26M).

**Matched:** frozen pi0 + its sampling, the 256 failure states and restore protocol, bridge length 50, transition
counting. **Not matched:** RL continuation cap 520 (official RSR cap) vs 350 in PoC verification (charged anyway);
GPU compute (WM / critic / SFT training vs PPO updates + DINO at every bridge decision) not equalized; pi1 also
trains on nominal demos (pi1-nominal init), RL never changes pi0's weights; RL needs the controller + DINOv2 at test;
the RL learning curve is on train instances (held-out seeds), the test RSR measures transfer to unseen failures.

**Smoke (2026-10-22):** 3 instances, ~1.4k transitions PPO (cap 120): snapshots, eval, 6 PPO iterations,
milestone / final checkpoints, resume state OK; `eval_recover residual.ckpt=...` on 1 test scene: RSR 8/10, bridge
applied exactly on decisions 0-4 (unit check). Throughput on the loaded node (load 100-140 on 32 cores): 5-10 env
steps/s with 3 workers; snapshot building for 256 instances ~15 min (cached in `rl.out_dir/instances.pt`).

**Queued:** DAG `configs/dag/b33_rl.json` on lane local:0:f (tmux rv_dag_b33): PPO to 25% of the budget ->
test RSR (`B33_b33_rl_sparse_p25_recover_test`, save_videos none) + nominal with residual -> resume PPO to 100% ->
test RSR + nominal. Analysis: `python scripts/analyze_b33.py --tag B33_b33_rl_sparse_p25 --rl_dir data/rl/b33_rl_sparse`
(scene-level paired bootstrap vs pi1 M5c wm_diverse rr (+ seeds) and vs pi0 M0_recover_bddl test part; curve plot).

**p25 results (2026-10-22, ckpt `data/rl/b33_rl_sparse/ckpt_m867255.pt`, 867k transitions = 25% of PoC budget;
`runs/_analysis/b33/summary_B33_b33_rl_sparse_p25.md`, `curve.png`).** PPO wall 8.5 h (10:17-18:50 UTC, 3 MuJoCo
pool segfaults survived via restart + ckpt_last), 35-55 env steps/s.

| system | test RSR (560 ep, 56 scenes) | diff vs RL p25, 95% CI scene bootstrap, p | nominal Spatial SR (500) |
|---|---|---|---|
| pi0 (M0) | 0.118 | RL - pi0 = -0.007 [-0.030, +0.016], p 0.66 | 0.656 |
| pi1 (M5c wm_diverse rr, 1 seed found) | 0.439 | RL - pi1 = -0.308 [-0.386, -0.228], p < 1e-3 | 0.808 |
| **RL residual p25** | **0.111** (close 0.089, open 0.132) | | **0.400** |

Learning curve (128 train instances, held-out pi0 seeds, deterministic residual): SR 0.039 @0 -> 0.086 @355k ->
0.086 @712k -> 0.055 @883k (n = 128, so +-0.05 is noise); stochastic train SR flat at 0.06-0.10. The policy is
NOT learning recovery at this budget; what it does learn is to override the gripper more (force open/close rate
0.21 -> 0.44 over iterations, |delta| 0.19 -> 0.21). That gripper/arm override in the first 50 steps is what hurts
nominal: -0.256 overall, tasks 4 (0.66 -> 0.04) and 5 (0.44 -> 0.00) collapse, 1 and 8 unchanged. So with the sparse
reward at 25% budget the residual is ~pi0 on recovery and clearly worse than pi0 on nominal. Not a final claim:
the 100% run is still running.

**Restart 2026-10-23 02:12 UTC:** container restart killed all tmux (~22:00 UTC) during `b33_rl_sparse_full`
(it 62, 961k transitions). Partial logs renamed `failed_rv_b33_*`, dead claim local:0:f dropped, DAG relaunched
(tmux rv_dag_b33); PPO resumes from ckpt_last (it 62, 961,081 transitions; the interrupted iteration is re-collected
and not charged). ETA: PPO to 3.47M ~18-20 h (42 steps/s + 7 curve evals) -> ~22:00 UTC 10-23; test RSR + nominal
~2.5 h after.

## P2 policy-agnostic value critics (2026-10-22, DAG b2p3, lane local:0:g)
Question: is the search gain from POLICY-GROUNDING (C^pi) or from "any learned value"? Two critics with the SAME
architecture (RecoveryValue, width 256, depth 3), the SAME WM (v1, latest_wm.txt), the same token space (DINOv2-small
6x6 x 2 cams + 8-d state, WM normalizer), the same train recipe (8k steps, bs 128, lr 2e-4, imagined_frac 0.5), swapped
into the search objective S = V - 0.5 U - 1.0 R. Code: `retryvla/critic/{value_data,train_value_baseline}.py`,
demo latents `scripts/build_demo_latents.py` -> `data/wm/demo_latents_spatial.pt` (432 Spatial demos, every 5th frame +
last, 11,112 frames). Arms `wm_progress` / `wm_success` = `arm_wm_value` = wm_diverse with the critic swapped
(same pool, 2 latent-CEM rounds, T = 0.05, diverse top-16, same seeds). Every M5b / M9 search instance id (259) is
excluded from every critic's training data.

Exact supervision:
| critic | samples (train+val) | label | continuation-policy labels? |
|---|---|---|---|
| C^pi v2 (reference) | collect_v1 + verif_r1 segments (pi0 continuation) | pi0 finished after the segment | yes, pi0 only |
| **B3 progress P_phi** | demo frames 11,112 (80% of each batch); failure states 221 (15%); prefix-success end states 585 (5%) | demo: t/(T-1), soft BCE; failure state (frame 0 of the LIBERO-Recover train instances): 0; segment that finished the task ITSELF (sim success during the 50 steps, no continuation): 1 | **none** |
| **B4 success V_mix** | pi0 7,160 (collect_v1 + verif_r1), pi1 880 (verif_m6_r2, trials 0-1), pi1more 8,904 (verif_m7_weak trials 0-1 + verif_m8_aug), demo 11,112 | task finished after the segment under the policy that continued (pi0 / M5c pi1 / M6 pi1-more); demo states: 1 (the expert finishes from each) | pooled; batches 25% per source, so 75% of labels are NOT pi0 outcomes |

Why failure states in B3: demos alone never show off-manifold states; the regressor then extrapolates arbitrarily on
exactly the states search scores. Labelling benchmark failure states 0 is policy-free (they are failure states by
construction of LIBERO-Recover). Prefix successes give a 1-anchor off the demo manifold, also without a continuation.
Why B4 differs from C^pi: same pi0 data is only 25% of each batch; the rest are other continuation policies (stronger,
pi1-family) and the expert. Real / imagined inputs for all critics (demo: WM rollout of the 50 demo actions that led to
frame t). Smoke (300 steps) rank agreement with C^pi on imagined val candidates: Spearman 0.55 (B3), 0.58 (B4): the
critics do differ from C^pi. +-2 cm logs keep trials 0-1 only (C^pi exposure), M8 (+-5 cm) is a different state
distribution so kept whole.

Validation (held out: WM-val failure episodes, 15% of demo episodes; "pi0-diag" = C^pi val set: collect_v1 candidates of
the WM-val instances, labels = pi0 continuation; "noprefix" = candidates whose segment did not finish the task itself,
i.e. the part where competence matters):
| critic | demo progress Spearman (real / imag) | demo-vs-failure AUC | pi0-diag AUC real / imag | pi0-diag AUC noprefix real / imag | top-4 success (mean 0.155) |
|---|---|---|---|---|---|
| C^pi v2 (reference) | - | - | 0.809 / 0.787 | 0.748 / 0.663 | 0.328 |
| B3 progress (final, 8000 steps) | 0.984 / 0.981 | 0.915 | 0.721 / 0.716 | 0.638 / 0.683 | 0.203 |
| B4 success (final, 8000 steps) | - | demo mean pred 0.9998 | 0.623 / 0.604 (pi1 val 0.558 / 0.568) | 0.693 / 0.644 | 0.320 |
Final numbers: `runs/20261022-085906_B2_train_progress/summary.json`, `runs/20261022-094644_B2_train_success/summary.json`
(B4 smoke row was 0.746 / 0.688, 0.733 / 0.673, 0.297 at 300 steps). Caveat: in train_value_baseline the diag dict
overwrites B4's OWN-val `pi0_auc_real` (same key), so the B4 pi0 own-val real AUC is not stored (own-val imag 0.639,
ECE 0.138); the table uses diag only, so it is unaffected.
B3 shows the expected pattern: excellent on its own target (task progress), weaker at ranking which bridge pi0 can
finish (top-4 success 0.20 vs C^pi 0.33; real noprefix AUC 0.64 vs 0.75; Spearman vs C^pi on imagined val 0.59).
B4 matches C^pi on top-4 (0.32 vs 0.33) despite a lower diag AUC (0.62 vs 0.81) and ranks differently from C^pi
(Spearman on imagined val 0.34); demo states are saturated at 1.

Smoke (`runs/20261022-080756_B2_smoke_value_search_s0of8`, 3 instances, budget 4, smoke progress critic used for both
new arms): wm_progress 1/3 solved, wm_success 1/3, wm_diverse 1/3 (4/4 hits on the solved one vs 1/4). B3 predicts
~1.0 for its picks while C^pi predicts 0.005 / 0.0002 / 0.25 for the same picks (logged as pred_cpi_* per arm).

Queued: b2_search_s0-3 (`configs/exp/b2_value_search.yaml`, out data/recovery/b2_value, verif data/wm/verif_b2_value,
eval.n_workers 8), then SFT on wm_progress / wm_success recoveries (`b2_sft_{progress,success}`, init pi1-nominal,
count = @ntrajmin:66 over the arm's bank and M5b wm_diverse s0-3 = the M9 comp / goal count, round-robin) + test RSR.
If a bank has < 66 trajectories, the count is lower than the 3-seed comp runs: then add a comp SFT at that count.

**Incident + restart (2026-10-22/23).** Critics done (rc 0, 09:34 / 10:19). b2_search_s0: attempt 1 hit CUDA OOM on
the shared GPU (empty part p0, retry_oom waited), attempt 2 (part p1) ran 11:09-13:39 and died at 13:39:04 on EDQUOT:
lerobot's async image writer only logged `[Errno 122] Disk quota exceeded` for episode 40 of wm_success (21st
instance), then `save_episode -> compute_episode_stats` raised `PIL.UnidentifiedImageError` on a truncated PNG (the
verif `.pt` save had retried). dag_run stopped (FAILED); the container's tmux server died later, so nothing ran until
2026-10-23. Salvage of `data/recovery/b2_value/s0/p1`: 20 done instances; wm_progress 12 eps / 1235 frames,
wm_success 40 eps / 4502 frames, all parquet images decode, every episode belongs to a done instance (no
`delete_episodes` needed: episode 40 never reached the parquet); removed the half-written `images/*/episode-000040`
dirs, the undone instance's row in both instances.jsonl and a stale verif `.pt.tmp`; marked CLOSED (+ SALVAGED).
`b2_value_search.yaml` now sets `m5b.write_retries: 8` (RecoveryWriter PNG verify + rewrite, added for M11 in
018b1c9). Failed log -> `failed_rv_b2_search_s0_..._20261022-102026.log`; no local:0:g claim was left. Relaunched
2026-10-23 02:29 UTC (tmux rv_dag_b2p3, lane local:0:g; container host is now ee580a52a73c): s0 resumes with 16 left.
Preliminary s0 (20 instances, NOT a result; paired with M5b on 19): solved wm_progress 0.25 (hit 0.07),
wm_success 0.70 (hit 0.20) vs M5b wm_diverse 0.47 (hit 0.13), random 0.32. C^pi rates the picks low:
pred_cpi mean 0.02 (progress picks, own pred 1.00) / 0.23 (success picks, own pred 0.71).

## P3 recovery-nominal Pareto (2026-10-22, DAG b2p3)
pi1 recipe (m5c_sft_wm_diverse_rr): init pi1-nominal, 6k steps, bs 32, lr 5e-5, bank data/recovery/m5b/s*/p*/wm_diverse,
round-robin 208 trajectories. Only train.recovery_ratio changes: 0.1 / 0.3 / 0.7 (`configs/exp/p3_sft_ratio.yaml`);
0.5 = pi1 (runs/20261016-042447_M5_m5c_sft_wm_diverse_rr: RSR 0.439, nominal 0.808; M8 seeds 1-2 exist).
L_preserve (`configs/exp/p3_sft_preserve.yaml`, `retryvla/train/sft.py::build_preserve / preserve_forward`, default OFF):
ratio 0.5 + on NOMINAL samples beta * MSE(v_student, v_teacher) at the SAME noise + time (velocity of the flow-matching
head, real action dims, valid steps), teacher = frozen init (pi1-nominal). beta = 1.0, chosen a priori: same units as
the FM loss, starts at 0, and in the smoke grows to ~0.05 after 50 steps vs FM loss ~0.6 (it regularizes, does not
dominate); single pre-registered value, not tuned on test.
Smoke: 50 steps bs 8 with L_preserve -> `runs/20261022-080313_P3_smoke_preserve` (rc 0; teacher costs ~6 GB, ~1.3x step).
Each point: test RSR + nominal SR (eval.n_workers 8, save_videos none). Reference nominal: pi1-nominal 0.796, pi0 0.656;
pre-registered eps = 2 pts: a point passes if NSR >= 0.776 (vs pi1-nominal) over >= 3 seeds.
Seeds 1-2: `python3 scripts/make_dag_b2p3.py --seeds 0 1 2` APPENDS tasks (seed-0 names unchanged), then restart the
runner (done tasks are skipped by their RV_DONE logs).

Ops: one lane local:0:g, all tasks wrapped in `retryvla.retry_oom` (waits for free GPU memory, retries on CUDA OOM: the
first B4 smoke died of OOM when the shared GPU was full). Order: critics -> 4 search shards -> P2 SFT progress + RSR ->
P3 preserve, 0.1, 0.7, 0.3 (SFT, RSR, nominal) -> P2 SFT success + RSR. ETA (load ~90 on 32 cores): critics ~1.5 h,
search ~2-2.5 h / shard (~31 instances x 2 arms x 16 sims) = ~9-10 h, SFT ~1.3 h, RSR ~1 h, nominal ~0.4 h each ->
P3 seed 0 ~11 h; whole DAG ~25-28 h.

Analysis:
- P2 search: paired McNemar on the 122 instances (shards 0-3), wm_progress / wm_success vs M5b wm_diverse + random
  (`runs/*_M5_m5b_compare_arms_s[0-3]of8/per_instance.jsonl`) and M9 wm_goal (`runs/*_M9_mech_search_s*`), hits / sim,
  pred_cpi_* (does C^pi rate the policy-agnostic picks low?).
- P2 SFT: scene-level paired bootstrap vs comp / goal SFT (3 seeds), `scripts/stats_cluster.py` style.
- P3: RSR vs nominal scatter (one point per ratio / preserve, seeds as error bars) with the eps band
  [0.776, inf) around pi1-nominal 0.796 and pi0 0.656 marked; RSR differences scene-bootstrapped vs pi1 (0.5).

## M11 incident 2026-10-22: workspace quota (EDQUOT) ~11:39-12:03 UTC
- Root cause: the shared `/pfss` workspace fileset hit its quota (`[Errno 122] Disk quota exceeded` in the lerobot image
  writer; torch.save -> `iostream error` / `unexpected pos 2048`). Not a code bug, but `wm.collect` wrote straight to
  the final `.pt`: collect_wide_s2 and _s3 died and each left a 2 KB truncated `.pt` that a resume would have SKIPPED.
  search_a_s3 died in `save_episode` (PNG write failed); search_a_s2 survived via `_save_retry` (4th try).
  dag_run only reported collect_wide_s2 (it exits on the first failure; s3 / search_a_s3 failed after it had exited).
- Fix (`retryvla/wm/collect.py`): tmp+rename writes with retry (8 tries, 60 s linear backoff, then raise), resume
  deletes stale `.tmp` and re-collects any existing `.pt` whose zip directory is unreadable.
- Cleanup: deleted the 2 truncated collect_wide `.pt` (114 valid kept; all 99 verif_m11_a files load). Salvaged
  `data/recovery/m11_a/s3/p0` (16 done instances, all 5938 frames decode): dropped wm_no_prim episodes 11-13 of the
  unfinished 17th instance (lerobot `delete_episodes`, side jsonl filtered), marked CLOSED -> resumed with 20 left.
- Relaunched DAG m11 12:24 (failed logs -> `failed_*`). Quota headroom is unknown (no quota tool in the container):
  if EDQUOT recurs, prune intermediate SFT checkpoints (runs/*/checkpoints/00[0-9]000, ~2.3 GB/run x ~25 runs).

## M11 incident 2026-10-22/23: second EDQUOT hit (12:45) + tmux server loss overnight
- 12:45:44 `m11_search_a_s3` (resumed part p1) died again on EDQUOT: lerobot's async image writer only LOGS
  `[Errno 122] Disk quota exceeded`, leaves a truncated PNG, then `save_episode -> compute_episode_stats` raises
  `PIL.UnidentifiedImageError` (verif `.pt` had survived via `_save_retry`). dag_run exited (FAILED); collect_wide
  s2 / s3 kept running and finished rc=0 (13:24 / 13:56). Overnight the container's tmux server was restarted
  (all rv_* sessions gone), so nothing ran after that. 2026-10-23 02:10: `/pfss` writable again (dd 50 MB OK).
- Fix (opt-in, other callers unchanged): `RecoveryWriter(write_retries=N)` -> before `save_episode` every PNG of
  the episode is `PIL.verify`-checked and any unreadable one is rewritten synchronously (tmp+rename) from the
  in-memory frame, up to N rounds with linear backoff (60 s), then raises. compare_arms passes
  `m5b.write_retries` (default 0 = old behaviour); set to 8 in `m11_search_a/b.yaml` only. Tested with a
  simulated 1-in-7 truncated-PNG writer: all frames repaired and decode bit-exact.
- Salvage: `data/recovery/m11_a/s3/p1` (1 done instance; wm_no_prim 4 eps / 501 frames, wm_plan 2 eps / 368
  frames, all decode): removed the half-written wm_no_prim episode-000004 image dirs + the undone instance's
  row in instances.jsonl, marked CLOSED. All 144 collect_wide and 108 verif_m11_a `.pt` pass zip tests.
- State at relaunch (02:16 UTC, lanes local:0:a-d, tmux rv_dag_m11): done = collect_wide s0-s3, search_a s0-s2;
  search_a_s3 resumes (17 done); train_wm_v2 started (collect_v1 + collect_wide); critic_v3 + search_b s0-s3 queued.
  Failed log -> `failed_rv_m11_search_a_s3_..._20261022-122441.log`; dead claims for local:0:a-d removed.

## M11 incident 2026-10-23 02:52: MuJoCo segfault in a sim worker (m11_search_a_s3, part p2)
- Not quota / OOM / write_retries: during the `verify` batch of the 6th instance of p2 one SimPool worker died with
  `Fatal Python error: Segmentation fault` inside robosuite `_reset_internal` (LiberoSim.restore_snapshot -> _build,
  env re-construction); the main process got a bare `EOFError` from the pipe -> rc=1 -> dag_run FAILED. 5 instances
  of p2 were complete; nothing of the 6th was written (writers add after all sim batches). RAM 1.6 TB free.
- Fix (additive): `SimPool` raises `WorkerDied` (subclass of EOFError, so old callers unchanged) with pid +
  exitcode, and has `respawn()`. compare_arms `m5b.worker_retries` (default 0 = old behaviour; 2 in
  m11_search_a/b.yaml): on WorkerDied respawn the pool and re-run the whole batch (same jobs + seeds -> same paired
  result); if it still dies, the instance is SKIPPED, logged to `<part>/skipped.jsonl` with the reason, and never
  retried on resume. Tested with a SIGSEGV-killed worker (respawn OK, retry/re-raise logic OK).
- Salvage `data/recovery/m11_a/s3/p2`: 5 done; wm_no_prim 9 eps / 528 frames, wm_plan 12 eps / 1453 frames, all
  decode (`scripts/validate_recovery_part.py --decode`, new), no orphan rows / images -> CLOSED + SALVAGED.
  Failed log -> `failed_rv_m11_search_a_s3_ee580a52a73c_20261023-021635.log`; dead claim local:0:b
  (rv_m11_train_wm_v2, finished rc=0 06:14) dropped. m11_train_wm_v2 was NOT affected: finished 20k steps, val
  open-loop WM/copy 0.35-0.49, linked in data/wm/latest_wm_v2.txt. Note: retryvla/wm/train_wm.py has NO resume
  (only a final save) -> a killed WM run restarts from step 0.
- Relaunched 06:31 UTC (tmux rv_dag_m11, lanes local:0:a-d): search_a_s3 resumes (22 done), critic_v3 started.
  If an instance is skipped it will appear in s3/p*/skipped.jsonl (none so far).

## P2 controls for B4 (B4-noPi0, Cpi-pool), partial results 2026-10-23 12:05 UTC (DAG b2x lane local:0:h, b2p3 lane g)
Critic validation (same held-out sets as the P2 table; C^pi val set = pi0-diag; final 8000 steps):
| critic | supervision | pi0-diag AUC real / imag | noprefix real / imag | top-4 success (mean 0.155) | Spearman vs C^pi (imag) |
|---|---|---|---|---|---|
| C^pi v2 | pi0 continuation (train_critic) | 0.809 / 0.787 | 0.748 / 0.663 | 0.328 | 1 |
| B3 progress | demos + failure 0 + prefix 1 | 0.721 / 0.716 | 0.637 / 0.683 | 0.203 | 0.589 |
| B4 success | pi0 + pi1 + pi1more + demo (1/4 each) | 0.623 / 0.604 | 0.693 / 0.644 | 0.320 | 0.345 |
| B4-noPi0 | pi1 + pi1more + demo (1/3 each) | 0.633 / 0.564 | 0.612 / 0.577 | 0.234 | 0.308 |
| Cpi-pool | B4's pi0 labels only (7,160) | 0.766 / 0.655 (own val 0.785 real) | 0.725 / 0.640 | 0.312 | 0.519 |
Runs: `runs/20261023-075605_B2_train_success_nopi0`, `runs/20261023-082325_B2_train_cpi_pool` (B4-noPi0 own pi1 val AUC
0.48 / 0.53 on 128 samples: no signal on its own held-out continuation labels).

Search (`python scripts/analyze_b2x.py` -> runs/_analysis/b2x; solved = >= 1 of 16 verified sims; exact McNemar).
B2 shards 0-3 COMPLETE (127 instances); B2x PARTIAL (s0 31 + s1 12 = 43 instances; s1-s3 running).
| arm | n | solved | hits/sim | mean C^pi pred of picks |
|---|---|---|---|---|
| random (M5b) | 127 | 0.339 | 0.061 | - |
| C^pi v2 wm_diverse (M5b) | 127 | 0.480 | 0.156 | - |
| goal (M9) | 125 | 0.240 | 0.046 | - |
| B3 wm_progress | 127 | 0.213 | 0.038 | 0.038 |
| B4 wm_success | 127 | 0.559 | 0.182 | 0.262 |
| B4-noPi0 (partial) | 43 | 0.465 | 0.077 | 0.081 |
| Cpi-pool (partial) | 43 | 0.535 | 0.132 | 0.228 |
Paired (arm vs reference, pair n, only-arm / only-ref, exact McNemar p):
- B4 vs C^pi v2: n 123, 0.545 vs 0.463, 14/4, p 0.031 (hits sign-flip p 0.038). Final for B2.
- B3 vs C^pi v2: n 123, 0.187 vs 0.463, 1/35, p < 1e-4; B3 vs random 2/18 p 0.0004 (worse than random).
- B4 vs random 30/2 p < 1e-4; B4 vs goal 40/1 p < 1e-4.
- PARTIAL n 42-43: B4-noPi0 vs C^pi 7/4 p 0.55; Cpi-pool vs C^pi 7/1 p 0.070; B4 vs Cpi-pool 4/3 p 1.0;
  B4 vs B4-noPi0 6/2 p 0.29 (hits +1.44 / instance, sign-flip p 0.0004).
Interim reading (not a claim until n = 127): Cpi-pool (pi0 labels only, B4's pool + trainer) already tracks B4
(0.535 vs 0.558, 4/3) while B4-noPi0 loses most of the hit-rate advantage (0.077 vs 0.182 hits/sim) and C^pi rates its
picks low (0.08 vs 0.26). So far the B4 gain looks like "more / differently-filtered pi0 continuation labels with the
value-baseline trainer", NOT policy-agnostic value; B3 (no continuation labels) is below random.

Ops 2026-10-23: b2_sft_progress_s0 (launched 11:07) hung: every DataLoader worker died with `OSError: AF_UNIX path too
long` (tmux inherited the ~90-char TMPDIR, same as the B3.2 incident). `retryvla/train/sft.py::main` now calls
`sft_rise.short_tmpdir()` (also covers the P3 SFTs). Killed the session, log -> `*.log.dead_tmpdir`, run dir ->
`runs/20261023-110817_B2_sft_progress_DEAD_tmpdir`; dag_run relaunched it 11:59 (training, 6k steps). Note the progress
bank yields only 60 trajectories (< 66): a matched comp SFT at n = 60 is still to be added.
Queue / ETA (shared GPU): b2x s1 ~13:35, s2 ~16:05, s3 ~18:35 UTC; b2p3: SFT progress + RSR ~14:30, P3 preserve / r0.1 /
r0.7 / r0.3 (SFT + RSR + nominal, ~2.7 h each) ~01:30 Oct 24, SFT success + RSR ~03:30 Oct 24.
Update 12:12 UTC: count-matched comp control added to DAG b2p3 (`scripts/make_dag_b2p3.py::comp_matched`) right after
the progress SFT + RSR, before the P3 points: `b2_sft_comp_progressn_s0` = M9 comp recipe (configs/exp/m9_sft_comp.yaml,
init pi1-nominal, round-robin, pool data/recovery/m5b/s[0-3]/p*/wm_diverse) with max_recovery_trajectories = the same
@ntrajmin expression as the progress SFT (-> 60), tag `sft_comp_n60`, + test RSR (`b2_rec_comp_progressn_s0`).
Success bank is final (B2 shards 0-3 done): 224 trajectories -> success SFT n = 66 = the existing 3-seed M9 comp runs, so
no extra comp control (comp_matched adds one automatically if the count were < 66). Runner restarted (24 tasks); the
running progress SFT kept its lane (0.84 it/s). ETA: progress SFT ~14:00, RSR ~15:00, comp n60 SFT ~17:00, RSR ~18:00,
P3 x4 (~3.4 h each at the current speed) ~07:30 Oct 24, success SFT + RSR ~10:30 Oct 24; b2x s1-s3 unchanged (~18:35).
Incident 12:09-12:15 UTC: /pfss EDQUOT again (dd of 1 MB wrote 0 bytes; git could not write objects). The relaunched
progress SFT died at step ~650 (rc 1): TensorBoard's async writer re-raised `[Errno 122]` in run.log. `sft.py` now uses
sft_rise's `safe_log` (TB dropped on failure, training continues) and `save_checkpoint` (checkpoints/<step>.tmp ->
rename, 8 retries, ~36 min backoff), so retryvla.train.sft (P2 + P3 SFTs) survives quota windows like B3.2. Failed log
-> `failed_rv_b2_sft_progress_s0_..._115936.log`, run dir -> `runs/20261023-120039_B2_sft_progress_DEAD_edquot`;
runner relaunched it 12:15:43 from step 0. ETAs shift +15 min (progress SFT ~14:15 ... success SFT + RSR ~10:45 Oct 24).
b2x_search_s1 hit verif `.pt` save retries 1-5 in the same window (watch for rc != 0; parts resume).

## M11 incident 2026-10-23 06:55: CUDA OOM in m11_search_b_s1 precheck (shared GPU full) -> DAG stopped
- Not a sim-worker crash (WorkerDied path not involved), not quota, not arm_wm_plan / ranker2: 3 min after launch,
  in the FIRST `precheck` batch (pi0 rollouts), the main process's batched SmolVLA call (`run_jobs ->
  predict_chunk -> vision_model`) raised `torch.OutOfMemoryError` (78 MiB requested, 135 MiB free of 79.3 GiB):
  search_b s0/s1/s2 + search_a_s3 + critic_v3 + the other lanes' jobs had all just started on the one A100.
  Nothing was written (s1/p0 has 0 instances -> marked `ABANDONED`). dag_run stops the whole DAG on any failure,
  so m11_search_b_s3 was never launched; s0 (29) and s2 (32) finished normally (08:27 / 08:34, CLOSED).
- Fix (additive, opt-in): compare_arms `make_safe_run(..., oom_retries, oom_wait_s)` also catches
  `torch.OutOfMemoryError` in any sim batch: `empty_cache`, wait `oom_wait_s * k`, respawn the pool (workers may be
  mid-batch) and re-run the whole batch (deterministic seeds -> same paired result); default `m5b.oom_retries: 0`
  = old behaviour, 6 in m11_search_b.yaml. Process-level backstop: the m11 search_b DAG tasks now run through
  `retryvla.retry_oom --min_free_gb 16 --wait_s 300` (as b2p3 / b2x). Tested with a fake OOM-raising run_jobs
  (retry, re-raise after N, N=0 old behaviour). An OOM in the WM planner itself (outside sim batches) is only
  covered by retry_oom (restart; the unclosed part is then redone).
- Failed log -> `failed_rv_m11_search_b_s1_ee580a52a73c_20261023-065159.log`; dead claims local:0:a/b/d dropped.
  Relaunched 12:05 UTC (tmux rv_dag_m11, lanes local:0:a-d): search_b_s1 on a, search_b_s3 on b.
- Third /pfss EDQUOT window ~12:11-12:16 UTC (not caused by RetryVLA: runs+data ~200 GB, no >500 MB file written in
  the workspace in the last 2 h): both fresh jobs were still loading and died because the log filter `grep` in the
  remote_tmux job hit `write error: Disk quota exceeded`, exited, and the python pipe got SIGPIPE (no RV_DONE ->
  dag_run relaunched both at 12:15). Their parts s1/p1, s3/p0 had 0 instances -> `ABANDONED`; logs -> `failed_*`.
  Fix: `remote_tmux.sh` restarts the grep filter on the same pipe when it exits 2 (write error), so a job survives
  an EDQUOT window (log lines during the window are lost); tested against /dev/full (producer rc 0) + a real launch.
- 14:29 UTC: m11 DAG ALL DONE. search_b s1 (p2, 32) and s3 (p1, 34) finished rc=0, no OOM retries / worker
  respawns / skips needed; with s0 (29) + s2 (32) wm_plan_v2 covers 127 instances. `validate_recovery_part.py`
  OK on all 4 closed parts (s0 30 / s1 39 / s2 39 / s3 49 trajectories; `--decode` run on s0 + s2, all frames decode).
  Results: RESULTS.md "M11" (arm B).
Incident 13:59-14:12 UTC: EDQUOT again; both of my runners exited (FAILED).
- b2_sft_progress_s0 died at the step-4000 checkpoint: `safetensors SafetensorError ... Disk quota exceeded`, which is
  neither OSError nor RuntimeError, so sft_rise.save_checkpoint did not retry it. `sft.py` now wraps save_checkpoint in
  an outer 8-try retry on any exception (60 s linear backoff). Run dir -> `runs/20261023-121732_B2_sft_progress_DEAD_edquot`
  (checkpoints deleted, 1.2 GB freed); log -> failed_*. Relaunched 14:35 from step 0 (no optimizer-state resume in sft.py).
- b2x_search_s1 died at instance 29/31 in lerobot's parquet writer (`[Errno 122] Error writing bytes`). The 28 done
  instances' search results are intact in `runs/20261023-110724_B2_valuex_search_s1of8/per_instance.jsonl` (56 rows),
  but the part's arm datasets are not loadable (wm_cpi_pool file-005.parquet unfinalized; both arms contain episodes of the
  undone instance). These datasets have no downstream use (b2x = search comparison only), so instead of a 2.5 h re-run
  the part `data/recovery/b2x_value/s1/p0` was closed by hand: arm dirs -> `wm_*_BROKEN` (never train on them),
  `DATASET_BROKEN` note, CLOSED -> resume skips the 28 done instances. Relaunched 14:35.
Revised ETA (UTC): b2x s1 ~14:55, s2 ~17:30, s3 ~20:00; progress SFT ~17:00, RSR ~18:00, comp n60 SFT ~20:30, RSR
~21:30; P3 x4 ~11:00 Oct 24; success SFT + RSR ~14:00 Oct 24.

## Incident 2026-10-23 ~15:39-2026-10-24 ~03:20 UTC: fourth /pfss EDQUOT window + tmux server loss -> relaunch 03:24
- 15:39 lerobot image writer `[Errno 122]` in b2x_search_s2 (write_retries repaired the PNGs), 15:46 b32_datagen_vla
  died in `save_retry` (6 tries) on `inst/<id>.pt` (99th instance), 16:45 b2x_search_s2 died in pyarrow
  (`Error writing bytes ... Disk quota exceeded`, 15th instance). b2_sft_progress_s0 had finished (rc 0 16:25, 006000
  checkpoint loads: 788 tensors) but `remote_tmux.sh` refused b2_rec_progress_s0 every poll (`QUOTA_EXCEEDED`), same
  for b33_rl_sparse_full_nom from 00:22 (test RSR had finished 00:22, rc 0). The b2p3 (16:49) and b33 (00:52) runners
  then died on a bare `OSError` from `dag_run.say` writing their own log (EDQUOT). The container's tmux server (pid
  399, up since 10-18) was found defunct at ~03:20 (other users' sessions also gone).
- 03:18 dd 64 MB: only 36 MiB written (EDQUOT); 03:20-03:24 64 MB x 3, 512 MB, 1 GB, 2 GB writes OK (headroom came back
  without deleting anything; it is shared workspace-wide, cause outside RetryVLA, RetryVLA = runs 61 GB + data 145 GB).
- Fix (additive): `dag_run.say` ignores OSError on print / log write; shells.json claim writes go through tmp +
  `os.replace` with 8 retries (an EDQUOT mid-`json.dump` can no longer truncate shells.json for all runners).
- Salvage: b32 `data/recovery/b32_rise/vla/inst`: 98 `.pt` all pass zip test, index.jsonl = the same 98 ids; removed the
  192-byte `.pt.tmp` of the 99th. b2x `s2/p0`: 14 instances in done.jsonl = 28 rows in
  `runs/20261023-145912_B2_valuex_search_s2of8/per_instance.jsonl`; `validate_recovery_part.py`: wm_cpi_pool OK
  (28 eps / 4605 frames), wm_success_nopi0 NOT loadable (file-002.parquet unfinalized + episode 13 of the undone
  15th instance) -> renamed `wm_success_nopi0_BROKEN` + `DATASET_BROKEN` note + CLOSED (b2x datasets have no
  downstream use, as for s1/p0). Failed logs -> `failed_rv_b32_datagen_vla_..._20261023-143121.log`,
  `failed_rv_b2x_search_s2_..._20261023-145904.log`. No dead claims were left (claims already empty).
- Relaunch 03:24-03:25 UTC (same session names; the new tmux server was started by another user's agent, its global env
  has another conda env on PATH; jobs `source env/activate.sh`, checked: retryvla python, torch cu128, CUDA OK):
  rv_dag_b33 (lane f) -> b33_rl_sparse_full_nom; rv_dag_b32 (e) -> b32_datagen_vla resumes (98 done, 158 left);
  rv_dag_b2x (h + c, c added) -> search_s2 resumes (14 done, new part p1) and search_s3 in parallel;
  rv_dag_b2p3 (g + a + b, a / b added) -> b2_rec_progress_s0, b2_sft_comp_progressn_s0, p3_preserve_seed0_sft in
  parallel; the next P3 SFTs start on free lanes through retry_oom (--min_free_gb 32 / 38 gates them while the GPU is
  full). Done tasks were all skipped (status from RV_DONE logs). GPU 03:31: 65 GB used (ours ~36 GB, rest = other
  containers on the same A100), load ~120 on 32 cores -> SFT 0.38 it/s (~4.4 h / 6k steps).

## Incident 2026-10-24 ~08:36-09:30+ UTC: fifth /pfss EDQUOT window -> 4 SFTs lost, both runners FAILED; relaunch 09:47
- Root cause = quota again (not code, not OOM / segfault / TMPDIR): every failure is a checkpoint write
  `safetensors SafetensorError: Error while serializing: I/O error: Disk quota exceeded (os error 122)`.
  * b32_sft_cond_vla (sft_rise) died AT ONCE at step 4000 (08:36): `sft_rise.save_checkpoint` retried only
    OSError / RuntimeError, and SafetensorError is neither (the 10-23 outer-retry fix was only in train/sft.py).
  * p3_preserve_seed0 (step 6000 = training finished!), p3_r01_seed0 (step 4000), p3_r07_seed0 (step 2000):
    sft.py's outer retry (8 tries, 60..420 s, ~36 min) ran out at 09:08 / 09:10 / 09:30 -> the window lasted
    > 55 min. Runners: b32 FAILED 08:39, b2p3 FAILED 09:09 (rv_p3_r07 kept running until it died 09:30).
  * evals were unaffected (b2_rec_comp_progressn_s0 finished rc 0 08:42). /pfss writable again at 09:44
    (dd 64 MB / 512 MB / 2 GB OK; nothing deleted to get there; not caused by RetryVLA).
- Fixes (additive):
  * `sft_rise.save_checkpoint` retries ANY exception and waits out long windows: >= `tries` attempts, then every
    <= 10 min until `max_wait_s` (default env `RV_CKPT_MAX_WAIT_H` = 12 h) has elapsed (the 10-23 window was ~12 h);
    model stays in memory meanwhile. Covers sft.py (P2 / P3 SFTs) too. Tested with a fake writer (3 failures ->
    saved; endless failures -> raises after the budget, no partial dir left).
  * `retry_oom`: free-memory gate shared by all retry_oom processes (flock on /tmp/rv_gpu_gate_<uid>; a waiter
    passes only if no other passed in the last `--gate_hold_s`, default 240 s when --min_free_gb > 0): before, several
    waiting SFTs passed together the moment memory freed (the new job allocates only after 1-2 min) and overshot.
  * `dag_run --stagger_s N` (default 0 = old): sleep between launches in one poll.
- Salvage / cleanup: b32 `runs/20261024-061541_B32_rise_sft_cond_vla` KEPT (sft_rise auto-resume: complete
  checkpoints/002000 with training_state.pt; removed 004000.tmp) -> resumed at optimizer step 2000 (lr 4.03e-05,
  item offset 64000), verified in the log. sft.py has no optimizer-state resume (its training_state is only
  {"step"}), so the 3 P3 runs restart from step 0 with the exact recipe (resuming from a weights-only checkpoint
  with a fresh AdamW would change the recipe): run dirs -> `*_DEAD_edquot`, their checkpoints deleted (3.5 GB).
  Failed logs -> `failed_rv_{b32_sft_cond_vla,p3_preserve,p3_r01,p3_r07}_seed0..._2026102[4]-*.log`. Dead claims
  local:0:a (rv_p3_r07) / local:0:b (rv_p3_preserve) dropped (tmux sessions gone).
- Relaunch: rv_dag_b32 09:47 (lane e; restarted 09:55 with lanes e + f so rec + nom run in parallel),
  rv_dag_b2p3 09:50 (lanes g, a, b, c, d; c / d added; `--stagger_s 240`). Running: b32_sft_cond_vla (resumed),
  p3_preserve, p3_r01; p3_r07 / p3_r03 / b2_sft_success wait in retry_oom (--min_free_gb 32) for GPU memory.
  GPU 10:27: 73 GB used incl. ~13 GB of other containers; per-job: SFT 12-14 GB (bs32), eval 2-7 GB.
- B3.3 nominal diagnostics (not DAG tasks; tmux rv_b33diag_*, no lane claims, 6 sim workers each, 10 inits per
  task = first 10 of the 50 used by the full nominal eval): `residual.no_grip` / `residual.cont_mult` (opt-in
  ablations in `ResidualBridgePolicy`, defaults = trained controller) + `residual_stats` in the eval summaries.
  Results: RESULTS "B3.3".
- Ops check 2026-10-24 14:45 UTC (no new failures since the 09:47 / 09:50 relaunch; dag logs have no FAILED after
  it): b32 sft_cond_vla finished (resumed from 2000), nom_cond_vla done 13:25 (0.770), rec_cond_vla 422/560.
  b2p3: p3_r01 SFT done 13:29 -> rec 280/560 + nom running; p3_r03 SFT done 14:42 -> evals next; p3_r07 SFT 3950/6000
  (0.43 it/s); p3_preserve SFT 4600/6000 (0.27 it/s); b2_sft_success_s0 still gated in retry_oom (--min_free_gb 32,
  10-15 GB free since 10:06; GPU 56-73 GB used incl. other containers) -> starts when a P3 SFT finishes.
  ETAs (UTC): RISE VLA RSR ~15:30; P3 r0.1 ~16:00; r0.3 ~17:00; r0.7 / preserve ~17:30-18:00; B2 success SFT
  ~20:00 + RSR ~21:30.
- `scripts/analyze_baselines.py` (new): compact baseline table + B2 SFT paired tests; rows whose newest eval is still
  running (episodes.jsonl missing / < 560 or < 500 rows) are reported as pending / running, never as partial numbers.
- 15:39 UTC: DAG b32_rise ALL DONE (cond_vla RSR 0.113, nominal 0.770) -> lane e free; rv_dag_b2p3 restarted 15:42
  with lanes g, a, b, c, d + e (done tasks skipped, running sessions kept) -> p3_r03_seed0_rec started on e at once
  (p3_r03_nom follows on the next free lane).
- 16:24 UTC: new DAG `configs/dag/nom_fill.json` (tmux rv_dag_nomfill, lanes local:0:f + local:0:h, stagger 240 s):
  missing nominal evals for B2 progress / comp n=60 SFT and M9 goal / comp SFT x 3 seeds (8 evals; M9 demo-only
  nominal already exists). Each goes through retry_oom --min_free_gb 14 (shared gate) so total GPU stays < ~74 GB.
  b2_sft_success_s0 finally started 16:08 (0.41 it/s -> 6k steps ~20:10, RSR ~21:30). analyze_baselines picks up
  the new nominal tags, the B2 success SFT row and a P3 Pareto section automatically.

## Incident 2026-10-24 16:44 UTC: MuJoCo segfault in a nominal-eval sim worker -> b2p3 runner FAILED; relaunch 2026-10-25 12:00
- `p3_r03_seed0_nom` died at 316/500 episodes (SR so far 0.813): one SimPool worker hit `Fatal Python error: Segmentation
  fault` in robosuite `_reset_internal` (LiberoSim._reset <- load_nominal), the main process got `WorkerDied ... exitcode
  -11` -> rc=1 -> dag_run stopped (FAILED). Same failure mode as the M11 search segfault (2026-10-23 02:52), not quota /
  OOM. Because the runner stopped, `p3_r07_seed0_nom` (dep done 12:09) and `b2_rec_success_s0` (SFT finished rc 0
  20:01, `runs/20261024-160616_B2_sft_success/checkpoints/006000`) were never launched; everything else in b2p3 had
  finished (P3 rec r0.3 0.375 / r0.7 0.436 / preserve 0.429; preserve nominal 0.806).
- Fix (additive): `retryvla.eval.rollout.run_jobs_resilient` (on WorkerDied: respawn the pool, re-run ONLY the jobs
  without a delivered result; each job's outcome depends only on its own load / policy seeds) used by eval_nominal and
  eval_recover via `eval.worker_retries` (default 0 = old behaviour). `make_dag_b2p3.py` passes `eval.worker_retries=2`
  to every rec / nom eval. Tested: SIGSEGV to a worker mid-run (40 nominal loads, 2 workers) -> respawn, the 23
  unfinished jobs re-run, all 40 results delivered once with the original job ids.
- Also added `b2_nom_success_s0` (nominal of the B2 success SFT; the other B2 / M9 SFT nominals ran in nom_fill).
- Cleanup: failed log -> `failed_rv_p3_r03_seed0_nom_ee580a52a73c_20261024-154825.log`, partial run dir ->
  `runs/20261024-154925_P3_p3_r03_seed0_nominal_DEAD_segv`; dead claims local:0:a/b/d/e/g dropped (tmux sessions gone).
- Relaunch 2026-10-25 12:00 UTC: rv_dag_b2p3 on lanes local:0:a-h (`--stagger_s 60`); the first try at 11:58 hit a
  sixth short /pfss EDQUOT window (`remote_tmux.sh` quota probe refused; headroom back by 12:00, nothing deleted).
  Running: p3_r07_seed0_nom, p3_r03_seed0_nom, b2_rec_success_s0, b2_nom_success_s0 (8 sim workers each).
- `analyze_baselines.py`: P3 pending points are labelled 'pending' (not 'running'), the B2 success row reads
  `B2_sft_success_nominal`, and `p3_plot` writes `p3_pareto.png` (runs/_analysis/baselines + experiments/figs).

### Disk cleanup: superseded SFT checkpoints deleted, 2026-10-25
- Removed `checkpoints/` contents (006000 + last; M5 sft_wm_r1 also 002000/004000) of 17 finished SFT runs. All of them had
  recover_test and nominal evals done, and no config, script, doc or slide uses them as a model to reuse. Each run keeps
  summary/config/metrics/tb, all its eval dirs are kept, and a `DELETED_CHECKPOINTS.txt` note is added. runs/ went from 85 GB to 63 GB.
- Deleted: M1 sft_random, sft_oracle_cem, sft_random_r25, sft_random_r25_nontrivial; M4 sft_B_base_nom_random_wm;
  M5 sft_wm_r1, sft_random_matched; M5b sft_wm_diverse, sft_wm_noise, sft_wm_qd; M5c sft_wm_qd_rr, sft_wm_diverse_all;
  M6 sft_pi2; M8 sft_aug; M9 sft_demo_only; M10 sft_demo_only_obj, sft_random_obj.
- Kept on purpose: M4 sft_A (used by `scripts/viz_distill.py`), M5b sft_random (used by `scripts/viz_e2e.py`), and
  M9 sft_sel_random / sft_sel_task_balanced (they have no nominal eval). All key models, P3, B2, B32 and M10 comp/goal/nominal are also kept.

### Disk cleanup round 2, 2026-10-26 / 2026-10-28 (RetryVLA 174 GB -> 151 GB apparent)
- 2026-10-26 16:44 UTC (before the user's stop on deleting checkpoints): `checkpoints/` of 13 finished, evaluated
  ablation SFT runs were removed (each has `DELETED_CHECKPOINTS.txt`): M9 sft_sel_random, sft_sel_task_balanced;
  M10 sft_goal_obj; B32 rise_sft_cond_pool, rise_sft_awr_pool, rise_sft_cond_vla; B2 sft_progress, sft_comp_n60,
  sft_success; P3 r01 / r03 / r07 / preserve seed0. Since 2026-10-28 NO checkpoint is deleted locally (another agent
  uploads unneeded checkpoints to the remote store first); data/rl PPO checkpoints are untouched.
- 2026-10-28: verification logs that are NOT critic training sources were slimmed in place: per-candidate DINO latents
  `z`, `q`, `valid` dropped (atomic tmp + rename; all other fields kept, plus a `slimmed` note), 25.3 GB -> 0.06 GB:
  verif_m9_mech, verif_obj, verif_m9_b32, verif_m11_a, verif_m11_b, verif_b2_value, verif_b2x_value, verif_r2_v1/v2
  (script: `runs/_logs/slim_verif_20261028.py`). KEPT full (critic / WM data or loaded by viz scripts): collect_v1,
  collect_wide, collect_obj, verif_r1, verif_m5b, verif_m6_r2, verif_m7_weak, verif_m8_aug.
- Workspace caches: `conda clean -a` (pkgs 17 GB; most of it had already been cleaned) and pip caches
  (~/.cache/pip 1.9 GB, doanh/.cache/pip 1.8 GB) emptied; `__pycache__` dirs removed.
- Nothing to delete: eval videos (only runs/M0_recover_bddl/videos 0.4 MB, used by collect_slide_videos.py; the
  runs/_analysis videos are slide / failure-analysis sources), smoke banks (none left), m11_b ABANDONED parts (empty).

## RQ3 (plan Part 3): verifier optimism + uncertainty penalty, 2026-11-01
- **Finding: the WM "ensemble" never was one.** In every LatentWM checkpoint (WM v1 `20261014-103001_M2_train_wm_v1`,
  WM v2 `M11_train_wm_v2`, Object WM) the 4 `z_heads` / `q_heads` are BIT-IDENTICAL (max |W_e - W_0| = 0): heads are
  zero-initialised (identity dynamics) and get identical gradients from the shared trunk and identical targets, so the
  symmetry is never broken. Hence U_WM == 0 exactly and the `- lambda_unc * U_n * V.std()` term of `WMCriticRanker.score`
  (every WM search arm since M3, lambda 0.5) has been a no-op; all earlier WM results are effectively lambda = 0.
  Not fixed in `dynamics.py` (would change future WMs silently); RQ3 adds a separate ensemble instead.
- **U_ens used for RQ3** (`retryvla/wm/boot_heads.py`, `scorers.ensemble_values`): 4 bootstrap linear heads (Poisson(1)
  trajectory weights, closed-form ridge 1e-2) fitted on the FROZEN WM v1 trunk features of the WM's own training
  transitions (collect_v1 train split; val failures unseen). The WM / its mean prediction / C^pi v2 are unchanged.
  Member e rolls out on its own trajectory (trunk + head e); U_ens = std_e of C^pi(z_H^e) with MC dropout p 0.1 on the
  critic (C^pi v2 trained with dropout 0.1), i.e. WM + critic disagreement in C^pi units. Val 1-step z MSE: WM 0.1564,
  boot heads 0.1565-0.1566. `data/wm/boot_heads_20261014-103001_M2_train_wm_v1.pt` (fit took 10 s on dedc84ce).
  Caveat seen in the smoke: when C^pi saturates at 1.0 every member agrees -> U_ens = 0 on most CEM-optimised picks.
- Offline (0 new sim, `python -m scripts.verifier_rescore` on dedc84ce + `python scripts/analyze_verifier.py` ->
  runs/_analysis/verifier): 43,632 verified candidates of the M5b-instance logs re-scored with WM v1 + C^pi v2 (matches
  the logged pred_value to ~5e-4 median). Results in RESULTS "RQ3".
- **Ops incident 18:09 UTC (mine):** `remote_ctl.sh stop dedc84ce:0` to release my 10-s boot-heads job killed the
  Part-4 agent's `rv_af_dg` (wm_datagen af_futures_m5b, 6 min into its precheck): on a shared GPU the bare slot key
  `dedc84ce:0` belonged to THAT session (mine was `dedc84ce:0:rv_rq3_boot`). Relaunched identically at 18:20
  (`rv_af_dg`, same args, nothing written yet by the killed run). Rule: on shared GPUs release claims by tmux session
  name only (pop the claim whose `tmux == name`), never `remote_ctl.sh stop <shell:gpu>`.
- Sweep launched 19:06 UTC on dedc84ce (2 x 4 sim workers, `rv_rq3_sweep_s0/s1`, configs/exp/rq3_verifier_sweep.yaml,
  M5b shards 0-1 of 8, pi0 failures restored from the M5b precheck without re-rolling).

## Part 4 (RQ2) action-free futures, ops log 2026-11-01
- Shell: only dedc84ce (0097c39c gone). First `rv_af_dg` (18:05) was killed 18:09 by a sibling's slot-key release
  and relaunched 18:20 by them; I restarted it 19:03 with `futures.verify_every=4` (CPU-bound shell: ~4 sibling sim jobs,
  load 14-36 / 32; full verification of filtered + random was ~2 min / state -> 8.5 h). Rate ~45 s / state -> ~3.3 h.
- One short /pfss EDQUOT window ~18:25 (smoke checkpoint write retried 2 min and succeeded). Smoke checkpoints deleted.
- Chain `af_chain` (agent machine tmux, CPU only) launches on dedc84ce: SFT arm 3 -> SFT arm 5 + arm 3 evals (RSR 8
  workers, nominal 4) -> arm 5 evals -> probe (pi1-nominal, pi1, arm 3, arm 5). SFTs use `save_every=6000` (no
  intermediate checkpoints). Worktree `../RetryVLA_af_base_wt` (c6d3f8c, for the default-path check) can be removed.

## Review + disk cleanup round 3, 2026-11-02 (~07:40-08:00 UTC)
- Job status at 07:40 UTC: nothing of ours running on dedc84ce (GPU 0 MiB, no live rv_ process); 0097c39c is gone
  (a new `vla_RetryVLA_DoanhLT_1` shell `c37611dd` exists but is NOT in `allow_only`, not used).
  Backtrack chain DONE (bt_single, bt_retry, Object bt_retry, bt_distill SFT + RSR + nominal; rc 0, 04:25 UTC).
  RQ3 sweep shards 0/1 DIED at 00:53 / 01:08 UTC on `[Errno 122] Disk quota exceeded` (RecoveryWriter parquet write)
  after 24/30 + 25/33 instances (49 used by analyze_verifier); parts `data/recovery/rq3_sweep/s*/p0` are NOT
  `CLOSED`, so a plain relaunch would redo them (to resume: check the p0 datasets, write `p0/CLOSED`, relaunch).
  Not relaunched (sibling-owned; 49 / 63 instances already give the curve + lambda sweep).
- **af_chain bug (Part 4):** `launch()` kept the `\r` that `det shell open` (pty) appends, so `wait_log` polled a
  non-existent "<log>\r" forever: arm 3 SFT finished 23:59 UTC but the chain sat idle ~8 h (GPU idle). Fix: `tr -d '\r'`;
  plus skip an SFT whose 006000 checkpoint exists, eval workers 5 (RSR) + 3 (nominal) and arm-5 evals after arm-3
  evals (<= 8 sim workers on the shared shell). Old local tmux `af_chain` killed; relaunched 07:54 UTC
  (`bash scripts/af_chain.sh dedc84ce 0`, local tmux `af_chain`): arm 5 SFT + arm 3 RSR / nominal running.
- Released the 5 finished claims (rq3_sweep_s0/s1, bt_chain, bt_single_rec, bt_chain2) by exact key.
- Deleted (dry-run first; ~128 GB freed, almost all workspace-level cache):
  - `~/.cache/huggingface/datasets/parquet`: 397 of 448 Arrow cache dirs last written before 2026-10-29
    (131 GB apparent; dir 135 GB -> 7.1 GB). Pure `datasets` cache of local parquet loads, regenerated on next load.
    List with dates/sizes: `runs/_logs/cleanup_20261102_hfcache_list.txt`. Dirs written 10-29..11-02 kept.
  - worktree `../RetryVLA_af_base_wt` (c6d3f8c, af_smoke default-path check; `git worktree remove` + prune), 81 MB.
  - smoke runs: runs/*_P4_af_futures_smoke, *_P4_af_smoke_{default_base,default_new,af_arm5_hybrid,af_arm3_free_filtered},
    *_SMOKE_verif_sweep_s0of1, and `data/futures/af_smoke` (~6 MB; results already in the Part 4 / RQ3 notes above).
  - 4 stale `*.pt.tmp` in data/wm/verif_b2x_value (2.5 MB), all `__pycache__` outside env/.
- Checked, nothing to free: `conda clean -a --dry-run` (pkgs 20 GB are all in use by envs), `uv cache prune`
  (no unused entries; archive-v0 86 GB backs other projects' venvs), runs/ checkpoints (every SFT run already has only
  006000 + last; no intermediate step dirs). torch hub (4.2 GB model weights) and HF hub (170 GB, other projects) kept.
- Kept on purpose in data/ (129 GB): every data/recovery and data/wm dir is either a config-referenced SFT / critic /
  WM source or a registered search output; data/futures/af_m5b (RQ2), data/recovery/bt_distill, data/recovery/rq3_sweep
  + data/wm/verif_rq3_sweep (dead-sweep outputs, needed to resume), data/wm/latest_*, boot heads, init checkpoints
  (M1 nominal, M5c pi1 s0, M6, M10 nominal obj) and the arm-3 / Backtrack-distill checkpoints. Unreferenced-but-old
  candidates left for the user to decide: data/recovery/m1b_hard_random_s{0,1} + m1b_hard_cem_local_s1 (~2.2 GB).

## 6-arm ablation chain + shell change, 2026-11-02 (~08:58-09:15 UTC)
- User 15:57 (+07): RetryVLA runs ONLY on `dedc84ce` (cn69) and `c37611dd` (`vla_RetryVLA_DoanhLT_1`, label 'VLA').
  Read-only probe of c37611dd: node cn43, container 497d28ac2bd7 = the Cursor agent box itself; 1x A100 80 GB with
  ~57 GB / 99 % util held by other projects' ASR jobs (Qwen3-ASR evals, nemo `train_u`; their tmux abl1-q4b, x1-*, x3-*,
  w5-final, report8790 untouched), load ~11.5 / 32. envcheck: OK video decode, OK sim 50 steps 1.27 s (NVIDIA EGL),
  headless cv2 OK. -> shells.json allow_only + share_ok (evals only, <= 12 sim workers; SFT only if >= 45 GB free);
  0097c39c + 8a2808f9 moved to `shells.json:gone` (bdfaf71). TMPDIR is unset in both images (= /tmp).
- Inventory at 09:00 UTC: bt_chain ALL DONE 04:25 (Backtrack-distill s0 RSR 0.182 / nominal 0.806); RQ3 sweeps dead
  since 00:53 / 01:08 (EDQUOT, not relaunched by their owner); af_chain (local tmux) has arm 3 s0 RSR (250/560) +
  nominal (done) and arm 5 s0 SFT (done 09:01) -> arm 5 evals + probe next, all on dedc84ce (8 sim workers).
- **Arm 1 demo-only had only seed 0** (M9_sft_demo_only; the M8 seeds are pi1 and M7 base) -> seeds 1-2 added.
- `scripts/abl_chain.py` in local tmux `abl_chain` (console runs/_logs/abl_chain_console.log): 11 SFTs one at a time on
  dedc84ce (retry_oom 32 GB, save_every 6000 -> only the final checkpoint), order a4s0, a3s1, a5s1, a6s1, a4s1, a3s2, a5s2,
  a6s2, a4s2, a1s1, a1s2; evals (RSR 8 + nominal 4 workers on c37611dd, 5 + 3 on dedc84ce) scheduled by a sim-worker budget
  over ALL live rv_ claims (c37611dd 12, dedc84ce 8, siblings included). Sessions `rv_abl_<arm>s<seed>[_rec|_nom]`,
  claims released by exact session name. SFT speed on the shared dedc84ce GPU ~1.2 it/s (~85 min / SFT).

## WM-vs-critic diagnosis + V-JEPA 2-AC foundation WM, ops log 2026-11-02/03 (results: RESULTS "RQ3 follow-up")
- Replays: `scripts/replay_bridges.py` re-simulates logged 50-step bridges from the exact failure snapshots (compare_arms
  restore-only path / collect build_instances, policy=None). Physics is bit-exact (state max|dq| = 0, prefix_success agree
  1.000). Mesa (CPU shell) vs NVIDIA render + fp32 vs bf16 DINO: latent rel. MSE ~2e-2 (control rows:
  runs/_analysis/wm_vs_true/control_rows.parquet). `LP_NUM_THREADS=1` per sim worker is ~1.5x faster on Mesa.
- Shells: a 1-GPU shell was refused (workspace slot limit 29/29); two CPU-only shells (slots 0) created from c37611dd's
  template (scripts/shells/retryvla_shell_cpu.yaml): bd8bdcf6 `vla_RetryVLA_DoanhLT_cpu`, 89995d8d `..._cpu2`.
  89995d8d killed 2026-11-03 06:02 UTC after its replays; bd8bdcf6 killed after the CPU diag of the RQ3 picks.
- CUDA context creation on c37611dd hung in the NVIDIA driver (D state) on 2026-11-02 16:00-16:30 UTC (box at load 64-91 /
  32, GPU 70 GB used) -> all V-JEPA GPU work ran on dedc84ce (~14 GB peak for the fine-tune, 0.72 it/s).
- /pfss reads with `torch.load(mmap=True)` ran at ~0.4 MB/s (page faults); plain loads at ~1 GB/s.
- V-JEPA 2: code third_party/vjepa2 (git clone, commit 204698b, untracked), weights split into
  data/foundation/vjepa2/vjepa2_ac_vitg_{encoder (bf16 2.0 GB), predictor (fp32 1.2 GB)}.pt; the 11.7 GB release file was
  deleted (re-download: dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt). Latents data/foundation/vj_latents (13 GB:
  collect_v1 8.6 GB only needed to re-fine-tune), JPEG frames data/foundation/frames (2.6 GB).
- Chain `scripts/vj_chain.py` (local tmux) DONE 22:54 UTC 2026-11-02: fine-tune 18:09-19:20, critic 19:32-19:48, encodes,
  offline rescoring, online sweeps s0/s1 (rv_vjsweep_*, rc 0, 22:38 / 22:54). dedc84ce was TERMINATED ~23:00 UTC:
  nothing of this work was lost; stale dedc84ce claims removed, dedc84ce moved to shells.json:gone.

## thanhnv shells probe + pending-work audit, 2026-11-08 (~09:35-09:50 UTC)
- User: two Determined shells `vla_roboworm_thanhnv_2-1` (335de36e, cn06) and `vla_roboworm_thanhnv_2-2` (b936d1d7,
  cn37) belong to thanhnv; a FEW SMALL RetryVLA jobs may use idle capacity. Read-only probe 09:37 UTC:

  | shell | node / container | GPU 0 used / peak util | GPU 1 used / peak util | load / nproc | owner procs / tmux |
  |---|---|---|---|---|---|
  | 335de36e (2-1) | cn06 / 2ff49dc615a0 | 0 MiB / 0 % | 0 MiB / 0 % | 15.9 / 32 (other containers on the node) | none (only stale `git log` pagers from sshd) / none |
  | b936d1d7 (2-2) | cn37 / 21a1ed348945 | 0 MiB / 0 % | 0 MiB / 0 % | 2.6 / 32 | none / none |

  envcheck: both OK video decode + OK sim 50 steps 1.05 s / 0.94 s (NVIDIA EGL). -> shells.json allow_only + share_ok
  ("light jobs only, owner thanhnv": <= 8 sim workers / job, retry_oom, never touch non-rv_ sessions).
- c37611dd and bd8bdcf6 are no longer in `det shell list` (terminated) -> shells.json:gone; the stale
  `bd8bdcf6:cpu:rv_diagcpu` claim (job finished rc 0 2026-11-03 06:50) removed; dedc84ce / 89995d8d already in gone.
  bd8bdcf6 needed no kill. The agent box is now 8e980338 (`vla_RetryVLA_DoanhLT_1`, cn55), not in allow_only.
- Pending-work audit -> **nothing left to launch**:
  1. Ablation arm 4 seed 2 RSR: finished on c37611dd (rv_abl_a4s2_rec rc 0, 09:37 UTC 2026-11-03), abl_chain ALL DONE,
     runs/_analysis/ablation/ablation.md already the full 6 x 3 table (re-run by the chain). RESULTS section finalised.
  2. V-JEPA 2-AC work was NOT lost with dedc84ce: vj_chain DONE 22:54 UTC 2026-11-02 before the shell died (vj sweeps
     s0 30/30, s1 33/33 instances, p0 CLOSED; RESULTS "RQ3 follow-up"). Decision there: not clearly better than WM v1
     (McNemar p 0.45 / 0.77) -> plan gate says no distill, so no SFT / test RSR is due. The Step-1 high-budget
     true-end diag (rows_rq3.parquet, 1920 picks, CPU on bd8bdcf6, done 06:50 UTC 11-03) had never been folded into
     summary.md: re-ran `scripts/analyze_wm_true.py` (CPU, 20 s) -> section true_end_rq3; numbers in RESULTS.
  3. RQ3 sweep: the relaunch (rv_rq3_sweep_s0/s1, 2026-11-02 10:00) finished rc 0 at 11:03 / 11:16 UTC; parts p0 + p1
     CLOSED, 30 + 33 = 63 instances, same 63 as the V-JEPA sweep. Done.
- No jobs launched on the thanhnv shells (none pending); they stay registered for the next RetryVLA eval work.

## Cross-WM disagreement sweep (WM v1 x V-JEPA 2-AC), ops log 2026-11-08 (results: RESULTS "Cross-WM disagreement")
- User chose the "next lever" of the V-JEPA test: penalise disagreement between two DIFFERENT world models. New arms in
  `retryvla/search/arms.py` (`xwm_values`, `xwm_objective`, `arm_xwm`): every candidate (pool + 2 elite-local CEM rounds,
  exactly the wm_unc_l0 pipeline, diverse top-8 on WM v1 end features) is imagined by WM v1 + C^pi v2 AND by V-JEPA 2-AC
  + C^pi_VJ; J = mean(C_v1, C_vj) - lambda |C_v1 - C_vj| - mu R with lambda 0 (average) / 0.5 (= min: both must agree) /
  1. In-run controls `wm_unc_l0` (WM v1 alone) and `vj_unc_l0` (V-JEPA alone, = the vj_sweep arm). Both verifiers score
  every arm's picks (`order_v1` / `order_v2`). Config `configs/exp/xwm_sweep.yaml` (inherits rq3_verifier_sweep: same
  63 train instances = M5b shards 0-1, precheck restored, 8 verified / arm, same continuation seed per slot).
- `compare_arms`: opt-in `m5b.paired_rng` + `num_sub_shards` / `sub_shard`: precheck ALL instances of the shard and draw
  every instance's proposal pool in order, skipping other-sub-shard / done instances only AFTER drawing -> pools identical
  to an un-split, un-resumed run (the Proposals RNG is sequential; the old rq3 sweep's resumed p1 instances had NOT been
  paired this way). Done set = done.jsonl of every part. Default path unchanged (flags off).
- Chain `scripts/xwm_chain.py` (agent box, local tmux `xwm_chain`, console runs/_logs/xwm_chain_console.log): launch via
  launch_remote (claims), watchdog (no log write 45 min / session or shell gone -> kill exact session, relaunch, resume),
  owner back-off (non-rv_ tmux, unknown GPU memory > 40 GB on our GPU, or node load / core EXCLUDING our workers > 0.9 ->
  kill our lane, relaunch after 2 clear polls), then `scripts/analyze_xwm.py` -> runs/_analysis/xwm/xwm.md.
- 16:03 UTC first launch with 4 lanes (2 per shell, 6 + 6 / 8 + 8 workers). cn06 (335de36e) already carries ~16 load
  from other containers: two lanes pushed it to 1.52 / core, the back-off (then counting our own load) killed both cn06
  lanes at 16:09 before their first instance finished (partial dirs deleted). 16:16 chain restarted with ONE lane on
  cn06 and our own load subtracted in the back-off test:
  | tmux | shell:GPU | work | sim workers |
  |---|---|---|---|
  | rv_xwm_s0 | 335de36e:0 | shard 0 (30 instances) | 6 |
  | rv_xwm_s1_0 | b936d1d7:0 | shard 1 sub 0 of 2 (17) | 8 |
  | rv_xwm_s1_1 | b936d1d7:1 | shard 1 sub 1 of 2 (16) | 8 |
  First instance 223 s (5 arms x 8 verifications, 8 workers). GPU ~10-15 GB / lane (WM v1 + V-JEPA ViT-g + SmolVLA).

## RGR (recursive geometric recovery) PoC, ops log 2026-11-08 (results: RESULTS "RGR")
- Shell: the thanhnv shells are carrying the xwm sweep (other agent; untouched), so a 1-GPU shell was requested under
  the user's own account: `det shell start --detach -w IAS_SAMMerge --config-file <copy of retryvla_shell_1gpu.yaml,
  description vla_RetryVLA_DoanhLT_rgr>` (without `-w` it fails: "resource pool 42_Compute ... not available to workspace
  id 1") -> d2ecd68c (cn28, container 87970a1a39b6, 1x A100 80 GB, node load ~11-13 from other containers). envcheck OK
  (video decode, sim 50 steps 0.97 s NVIDIA EGL). Registered in shells.json allow_only + share_ok; `scripts/rgr_chain.py`
  kills it (`det shell kill`) after phase 2 (set RGR_KEEP_SHELL=1 to keep).
- Geometry label replays (rv_rgr_lab0/1, 2 x 8 workers) 17:06-17:37 UTC. First smoke caught a feature bug:
  "infeasible" (floating object) used the height above the TABLE, but objects resting on the cabinet / stove sit
  ~16 cm high -> fixed to the lift relative to the segment start (b9a7c0d). Smoke runs + their outputs deleted.
- Budget note: in the first smoke the recursive arm executed only 4 first bridges and spent the rest on low-value
  2-bridge chains; changed to adaptive screening (execute first bridges while the chains whose OBSERVED s_1 reaches B_0
  do not fill the sim budget, <= 10 executions = 500 env steps), verification ordered by observed / re-planned C^pi,
  replans capped by the 768-rollout imagination budget. Only smoke instances were looked at (train shard 0), no tuning
  on results beyond this budget-accounting fix.
- Chain `scripts/rgr_chain.py` (agent box local tmux `rgr_chain`, console runs/_logs/rgr_chain_console.log): head
  (done 17:42) -> sweep shards 0-3, 2 at a time (rv_rgr_sw<k>, retry_oom, stall watchdog 45 min) -> analyze_rgr after
  each shard -> phase 2: pre-registered decision -> optional matched SFT pair + test RSR / nominal -> kill d2ecd68c.
- FINAL (checked 2026-11-09 02:50 UTC): shards sw0-sw3 rc 0 (19:48 / 20:04 / 21:49 / 22:22 UTC), 127 paired instances;
  phase 2 decision `arm: null` (no candidate has n01 > n10 with p < 0.10; best rec_geo_nofc 11/12, p 1.0) -> no SFT
  launched; `det shell kill d2ecd68c` rc 0 at 22:22 (state TERMINATED). shells.json: d2ecd68c removed from allow_only /
  share_ok (no claims left) -> gone. RESULTS "RGR" marked FINAL (negative).

## Cross-WM disagreement sweep, ops log continued 2026-11-08/09 (start: section "Cross-WM disagreement sweep" above)
- 2026-11-08 17:04 UTC: owner-side tmux `quang-vgp` appeared on b936d1d7 -> back-off killed rv_xwm_s1_0 / s1_1 (25 / 33
  shard-1 instances done). 19:26 rv_xwm_s0 rc 0 (30 / 30). The chain then waited (tmux test kept failing) and no report
  was produced overnight (coordinator check 2026-11-09 02:50).
- 2026-11-09 02:51 resume2 (`XWM_LANES=resume2`): shard 1 on 335de36e, 2 x 4 workers; 03:02 back-off again: a triquang
  SpatialLIBERO / robocasa job arrived in that container (GPU0 45 GB 99 %, load 71 / 32). Read-only check of
  b936d1d7: GPUs 0 MiB, load 2.5, `quang-vgp` = an interactive `claude` CLI idle ~10 h (1 % CPU) -> 03:08 resume3 there
  (2 x 6 workers, `XWM_IGNORE_TMUX=quang-vgp`, GPU / load back-off kept). 03:30 triquang video-gen jobs took both
  b936d1d7 GPUs (100 %) -> back-off killed s1_1 (3 instances left); s1_0 had 1 instance left (~10 GB, 6 workers) and was
  let finish (rc 0 03:42). 03:43 chain STOPPED by me, claims released, no rv_ session left on either shell.
  **60 / 63 paired instances** (missing: shard 1 sub 1 episode_002 open7, episode_005 open7, episode_013 close4).
  The per-instance pools stayed paired through all resumes (paired_rng).
- Controls vs the earlier sweeps (same pools / seeds): identical per-slot outcomes on 78 % (WM v1) / 72 % (V-JEPA) of
  instances, solved 0.367 vs 0.350 / 0.400 vs 0.417: GPU non-determinism in the bf16 WM rollouts changes some CEM picks
  -> only the IN-RUN controls are used for the paired tests.

## ACR (Active Causal Recovery) PoC, ops log 2026-11-09 (design: PLAN "ACR"; results: RESULTS "ACR")
- Shells: the thanhnv shells are unavailable (335de36e: owner job at 99 % / load 78; b936d1d7 not in the probe), the
  agent box (8e980338, cn55) GPU is someone's 100 %. So, as for RGR, a 1-GPU shell was requested under the user's own
  account: `det shell start --detach -w IAS_SAMMerge --config-file <copy of retryvla_shell_1gpu.yaml, description
  vla_RetryVLA_DoanhLT_acr>` -> e26be342 (cn48, container d0ebdd764ec7, 1x A100 80 GB, node load ~20 / 32 from other
  containers). envcheck OK (video decode, sim 50 steps 0.94 s NVIDIA EGL). shells.json allow_only + share_ok.
- Code checks (single process, agent box, 1 sim, no policy): the OSC delta is world-aligned, ~1 cm / step at |a| = 1
  (probe / servo scale fixed to that); the bowl weighs 5.6 g, gripper = position actuator, force range +-20 N;
  MuJoCo combines contact friction by MAX -> "slippery" needs a raised geom priority. Lowering the OBJECT friction made
  the bowl slide onto the plate by itself (prefix success at step 1) -> slippery redefined as FINGER friction (resting
  state unchanged; tests/test_causal.py::test_static_equivalence). All 5 tests pass (render-identical first obs,
  bit-exact nominal after a hypothesis, every hypothesis changes some probe response, static equivalence, lengths).
- Calibration (rv_acr_calib, 12 workers, 3 calibration states x 22 hypotheses x 3 scripted prefixes x 3 variants, no
  policy, 2.7 min): ranges frozen (PLAN "Calibration outcome"); anchored dropped.
- Smoke (rv_acr_smoke, 1 CALIBRATION state, tier B pipeline, 12 workers): 663 s / state (7 particles x 6 families x 2
  planning trials + 4 cases x 36 outcome-table trials + 99 probes); analysis end-to-end OK (index bug fixed).
  Smoke outputs deleted (calibration state, never reported).
- 16:14 UTC chain `scripts/acr_chain.py` (agent box local tmux `acr_chain`, console runs/_logs/acr_chain_console.log):
  rv_acr_B0 / B1 (tier B, 25 states, 2 lanes x 7 workers) -> rv_acr_A0 / A1 (tier A, 60 states) -> analyze_acr (3
  feature sets) -> postprobe B / A -> analyze. Expected ~4 h tier B + ~1.5 h tier A. Shell kept for review follow-ups.
- 2026-11-09 16:46 UTC (user request): acr_chain stopped (tmux kill-session =acr_chain), then `det shell kill
  e26be342` -> TERMINATED. Tier B was at 2 / 25 states (data/acr/v1/B, complete per-state files; the in-flight 3rd
  states of both lanes are lost). shells.json: e26be342 -> gone, no claims left. Resume: on any allowed shell,
  re-run `scripts/acr_chain.py` with SH / SH_FULL edited (acr_search skips finished state files).

## ACR-N (Act Before You Imagine, natural failures), ops log 2026-11-16 (design: PLAN "ACR-N"; results: RESULTS "ACR-N")
- Shells: the local agent box GPUs are someone else's (95-99 %); thanhnv shells not used. Two 1-GPU shells requested
  under the user's own account (`det shell start --detach -w IAS_SAMMerge --config-file <copy of
  retryvla_shell_1gpu.yaml>`): cb2b6ab1 (vla_RetryVLA_DoanhLT_acr2) and f02ea5c5 (..._acr2b); envcheck OK on both
  (sim 50 steps 0.99 / 1.01 s, NVIDIA EGL). shells.json allow_only + share_ok; `scripts/acrn_chain.py` kills both at
  the end (ACRN_KEEP_SHELL=1 to keep).
- Reused the 2026-11-09 ACR code (probes, families, SimPool opt-in commands); new: oracle cause tests, `release`
  probe, visual probe response (DINOv2 latent change agent / wrist, C^pi change), `place_goal` / `pull_free` options,
  WM options (`wm_single` = fresh 256-pool + 2 CEM rounds + diverse top-8 = the `single` scorer; `wm_prim`).
- Smoke (1 L3 U state, cause free, out_root n1_smoke, deleted): 342 s / state at 12 workers, 0 / 80 table trials
  succeeded. The movability test flung a free 5.6 g bowl ~5 m (3 m g for 20 steps) -> shortened to 5 control steps
  before the real run (threshold 1 cm unchanged; it only has to separate "moves" from "blocked").
- 04:26 UTC chain `acrn_chain` (agent box tmux, console runs/_logs/acrn_chain_console.log): rv_acrn_c0..c3 (2 lanes
  per shell x 12 workers, 97 states) -> analyze --write-decisions -> postprobe (evi / random sequences re-run from the
  true post-probe state) -> analyze -> kill both shells.

## T2Mem PoC (test-time memory for fail -> recovery), 2026-11-16 (user request; ops + pre-registrations)
- Idea: T²Mem (arXiv 2609.36720, fast-weight test-time memory in a VLA) applied to RetryVLA. Approach A = memory in the
  policy, trained on fail -> recovery episodes; approach B = instance-level test-time memory of VERIFIED outcomes in the
  recovery search (the verifier's FP ~0.8 is the documented bottleneck: RQ3 / xWM / RGR).
- Shell: user authorized thanhnv / quang / doanhlt shells. b8c028b9 (vla_roboworm_thanhnv_1-1, cn35, 1x A100, idle at
  04:00 UTC probe) added to shells.json allow_only. Jobs: rv_t2m_cache (1 min), rv_t2m_smoke, rv_t2m_poc1 (53 min).
- PoC 0 (`scripts/poc_mem_repeat.py`, 0 sim, rule written in the docstring before running: drop A if < 20 % of failed
  episodes repeat an earlier failed grasp). The close-command attempt definition is chatter-dominated (10-80 close
  onsets / episode, 15 % touch the bowl) and gave 4-79 % depending on thresholds -> NOT used. Finger-bowl contact
  segments (primary): 8-15 % of 73 failed episodes (pi1 s0 + M7 base s0) contain a repeat at tol 2-3 cm; within-episode
  repeat rate 23-41 % vs cross-episode null 0-14 %. -> approach A dropped (few episodes have >= 2 real attempts).
- PoC 1 (`scripts/t2mem_cache.py` + `scripts/t2mem_poc1.py`, 0 sim, verif_m5b train, 4 folds by failure episode):
  pre-registered gate FAILED on condition 3 (t2m_label < its static control). The static control (no write; a linear
  outcome recalibrator on C^pi's imagined-state embedding, trained on other folds) reaches 2.327 hits r2-4 vs
  oracle_true 2.387 vs static 1.859.
- Follow-up F1 (pre-registered here BEFORE running; `t2mem_poc1 --followup`): does memory add anything ON TOP of that
  recalibrator? Stage A = recalibrator (used from round 1), Stage B = A frozen, only the write path learned (T2Mem-style
  alternation). Rule: memory adds value iff mem_on_recal - recal_r1 > 0 hits r2-4 with sign-flip p < 0.10. If NO, the
  test-time-memory idea is reported negative at n = 256 (offline replay) and the recalibrator is reported separately as
  a lead (not a memory result; needs its own held-out check before any search use).
- Review (experiments/reviews/T2MEM_review.md, fresh-context agent): A close-as-tested; B redesign (F1 could only test
  the label channel: latent read-out o = 0 had no gradient; label write = residual vs RAW critic, overwriting the
  recalibrator); recalibrator = separate lead, picks-only label access shrinks it (+0.18). F1's NO is therefore NOT
  evidence against test-time memory in general.
- Follow-up F2 (pre-registered here BEFORE running; `scripts/t2mem_f2.py`, train only, cluster = failure episode,
  all p-values = episode-clustered sign flips):
  * Base = logistic recalibrator on [l_wm, h_wm] (fold-trained; cross-fitted logits for meta-training).
  * Memory ResMem on top of the FROZEN base: W starts at 0 (not learned), delta-rule write of the residual on the
    CURRENT score (y - sigmoid(score)), live latent read-out (o random init); arms mem_fixed (label + latent, primary),
    mem_ss (latent only), mem_pos (writes successes only); plus GP-UCB residual surrogate and knn_pos (hyper-params by
    train-fold grid). Regimes K4 x R4, K1 x R16, K4 x R8.
    Rule: B continues iff mem_fixed - base >= +0.10 hits after round 1, clustered p < 0.10 and solved not lower, in
    K4R4 or K1R16. Secondary arms passing the same rule -> "redesign around that variant" (multiplicity noted).
    All fail in all regimes -> B closed (negative at n = 256, offline fixed pool, 16-32 verifications).
  * Recalibrator, realistic labels: prequential over the instance stream (3 episode orders), trained only on its own
    16 picks per earlier instance. Lead continues iff second-half gain >= +0.15 hits r2-4 (K4R4), clustered p < 0.10.
  * Budget reallocation (Hyperband-style, static ranking): r1-miss instances capped at 8 verifications, saved budget
    to r1-hit instances, same total. Positive iff distinct hits / verification +20 % with <= 2 pt solved loss.

## ACR-N follow-ups (review docs/reviews/ACRN_REVIEW.md), ops log 2026-11-17 (results: RESULTS "ACR-N")
- Pre-registration of F1 / F1b / F2 committed (45b1b1f) before any run. Clarification (no result seen): "fresh
  single on new seeds" uses `rgr.seed_offset 1000` only; `eval.seed` is NOT changed because it also seeds the
  LIBERO-Recover protocol (object noise) and would change the instances.
- 09:46-09:53 UTC: /pfss workspace quota (EDQUOT) blocked writes and the git commit (6th EDQUOT window); F1 was
  developed in /tmp meanwhile; writes recovered by 09:53 without deleting anything.
- Postprobe bug (2026-11-16 chain): `KeyError: nan` (pandas NaN probe in no-probe decision rows) -> all postprobe
  attempts rc 1; fixed in acr_search.postprobe_n (string check); not re-run (G1 failed).
- F1 preliminary (09:51, before latents existed; obs_x only): learned 0.301 / learned_sub 0.301 vs fixed 0.274;
  probe sets +0.04-0.05 AUC. Final F1 (10:20, with the pre-registered DINOv2 latent PCA-16): see RESULTS; the
  preliminary is reported there only as the "probe info is in the frame" observation.
- F2 chain `acrn_f2_chain` (agent box = 8b889199 local tmux; console runs/_logs/acrn_f2_chain_console.log), 09:53 UTC:
  rv_acrnf2_0 / 1 (GPU 3 / 1, 5 workers; head-pose table; ~240 s / state) + rv_acrnsf_0..3 (GPU 2, 2, 3, 1; 3 workers;
  fresh single, ~70-150 s / instance). GPUs 1-3 are shared with other users' cab_prod2_label jobs (8-20 GB, up to
  99 % util); GPU0 untouched. Node load 26-35 / 32.
- 2026-11-17 12:45 UTC: acrn_f2_chain finished, all 6 lanes rc 0 (F2 table 97 / 97, fresh single 127 / 127; no
  watchdog restarts); analyze_acrn_f2 ran automatically. Claims released by the chain; nothing of ACR-N is running
  on 8b889199 any more (GPUs 1-3 free for the 2610.05151 reproduction agent). Analysis + RESULTS F2 / final verdict
  written 2026-11-18 (F2 FAILED its rule; no distillation proposal).

## 2610.05151 reverse direction (RetryVLA on their pi0 / LIBERO-Goal protocol), ops log 2026-11-18 (design: PLAN "2610.05151 comparison"; doc: docs/related_work/COMPARE_2610.05151.md)
- 03:55 UTC: coordinator says GPUs 1-3 free (ACR-N F2 done). Invisible processes from other containers keep 6-14 GB /
  20-60 % util on GPUs 1-3 at times. We share them, memory is ample.
- Infra (logged as D-deviations in the doc section 3.2b): split goal_rsi_v1, D0 built from the original hdf5 records
  (38,240 windows). A replay-built D0 was dropped after the first 40 demos: only 82 % of demo replays succeed in our
  sim. Also built: pi0 LoRA (451.5 M trainable), DDP trainer, eval_goal, Pi0Wrapper, G0 chain.
- Smoke tests: 12 SFT steps at bs 24 on 1 GPU take 2.7 s/it; eval on 4 val scenes runs end to end (63 s, 4 workers).
  The checkpoint round-trip works (LoRA merge on load).
- 04:20 UTC: first M0 launch on GPUs 1, 3 (bs 24 / GPU) measured 3.3 s/it, ETA 4.5 h. It was killed at step ~60
  (orphan torchrun children killed by PID) and relaunched at 04:33 UTC on GPUs 1-3, 16 / GPU, global bs 48 unchanged,
  with eval co-located on GPU 2 (8 sim workers). Claims `8b889199:{1,2,3}:rv_goal_g0`. The stale claim
  `8b889199:2:rv_lrs_smoke` (session gone) was dropped.
- 04:50 UTC: M0 runs at 2.43 s/it on 3 GPUs (global bs 48), so 5k steps take ~3.4 h (ETA ~08:00 UTC). Each 1k ckpt
  gets a val eval on GPU 2 while training continues. Then Original (2k steps, ~1.4 h) and its val eval. **G0 verdict
  ETA ~10:00 UTC** (`runs/goal_rsi/g0.json`). Next, only if G0 passes: M0 audit on the 300 train scenes, then C^pi
  data and the Goal WM.
- 08:56 UTC: G0 chain finished rc 0, **G0 PASS** (M0 4k 85, Original 82; RESULTS "G0"). Eval wall-clock per 100
  episodes was 6-15 min. GPUs 1-3 then sat idle 08:56-17:15 UTC (agent turn ended, nothing queued; lesson: queue the
  next stage inside the chain). 17:20 UTC: stale claims `8b889199:{1,2,3}:rv_goal_g0` released.
- 17:18-17:34 UTC: M0 audit on the 300 train scenes (GPUs 1 / 2, tmux rv_goal_audit_s7 / _s8). Both seeds give
  244 / 300 successes (0.813) and 56 failures, i.e. 168 entries each. Failures concentrate on cream_cheese (18 / 18)
  and wine_rack (14 / 8). The paper has 67 / 300 failures. Claims released.
- Smoke tests of the stage-2 path:
  - Goal collect (1 entry, 8 candidates): 72 s, 5 / 8 succeed. Caveat: entries have no M0 pre-check, so early
    entries (T/4) may be states M0 could still solve. This is accepted per the pre-registration and noted for the
    reading of solved rates.
  - Goal search (1 entry, Spatial WM as a plumbing stand-in): random 0 / 2, wm_diverse 2 / 2.
  - Distillation with a bank (3 steps, max_traj 2): runs end to end.
- 17:42 UTC: **chain `rv_goal_rv` launched** (`scripts/goal_rv_chain.py`, watchdog wrapper restarts it up to 3x).
  Claims `8b889199:{1,2,3}:rv_goal_rv`.
  - Stages: collect (3 shards, GPUs 1-3, 8 workers each) -> WM + C^pi (GPU 1) -> search (3 shards) -> 6 distillation
    runs (wm / rand x seeds 7-9, one per GPU, accum 3) + val evals -> analyze_goal_rsi.
  - In parallel, `rv_goal_orig` on GPU 3 trains Original seeds 8 and 9 (4.8 s/it shared, ~2.8 h each).
  - ETA: collect ~21:15, WM / C^pi ~22:45, search ~01:45, distillation + evals ~08:00 UTC 2026-11-19, then
    `runs/goal_rsi/main.{json,md}`.
- 2026-11-19 03:19-09:31 UTC: distillation waves ran (rec_wm s7-9, then rec_rand s7-9, one per GPU, ~3.3 h / ~2.9 h
  per wave incl. val eval). All rc 0, **CHAIN DONE 09:31 UTC**. Original seeds 8 / 9 (rv_goal_orig) also finished:
  77 / 79. Results: RESULTS "main comparison" (verdict "below": Δ −11.3).
- 2026-11-20 05:58 UTC (coordinator: "shell killed?"): `det shell list` shows 8b889199 (vla_cvpr_DoanhLT_4g)
  **TERMINATED** ("allocation exited successfully"). It ended AFTER the chain finished, so **nothing was lost**: every
  run has DONE + ckpt_02000/eval_val/summary.json (orig_s7-9, rec_wm_s7-9, rec_rand_s7-9) and main.json exists.
  Stale claims `8b889199:*` released; 8b889199 moved to `shells.json:gone`.
- Roboworm shells (user: continue on a roboworm shell with 2 GPUs): the only RUNNING roboworm-named shell is
  `63e0f914` (vla_roboworm_1). Read-only probe 06:00 UTC: 2x A100, GPU0 99 % util / 7.9 GB, GPU1 29-71 % / 5.4 GB,
  owner tmux egoprm / nativemem / robotood + 3 `search.py` jobs at 100 % CPU, so **not idle**. No other
  roboworm / RoboWorM shell is running. Not registered, no job launched there. The follow-up F1 (pre-registered in
  RESULTS) is ready to launch on any 2 free GPUs: 6 runs x ~3 h single-GPU, i.e. ~9 h with 2 GPUs.
- 2026-11-20 07:20 UTC: user decision: run F1 on 63e0f914 (vla_roboworm_1) ONLY when it is idle. Local watcher
  **tmux `rv_f1_watch`** on the agent box (42f3ad7fd1c0, Determined task $DET_TASK_ID 1400feb4) runs
  `scripts/goal_f1_watch.py`. Poll log: runs/goal_rsi/f1_watch.jsonl.
  - Poll: every 12 min, read-only `probe_owner.sh` via det shell open.
  - Launch rule, 2 consecutive polls of:
    - both GPUs < 10 % max util over 5 samples and >= 45 GB free;
    - no owner GPU process > 1 GB;
    - no owner process >= 50 % CPU (3 s top sample);
    - load per core < 0.5.
  - At the first launch: envcheck, then shells.json allow_only / share_ok + claims. Lanes `rv_f1_g0` (wm s7-9) and
    `rv_f1_g1` (random s7-9) run `scripts/goal_f1_lane.sh`: bridges only, 36 bridges (~1,645 anchors ≈ 4.1 %;
    the pre-registration's "≈ 33" assumed 50-frame bridges, the measured mean is 45.7). Trainer `--resume_every 250`.
  - Back-off (kill only rv_f1_* via `kill_rv.sh`, requeue): an owner GPU process > 1 GB, or owner CPU >= 150 % on
    2 consecutive polls.
  - Give-up: 72 h without a launch.
  - At the end: `analyze_goal_f1.py`, auto-appended RESULTS / NOTES, claims released, commit + push.
  - First probe 07:15 UTC: GPU0 100 % (owner 3 GPU procs, 12.7 GB), owner CPU ~300 % -> busy, waiting.
  - Deviation (logged): the action chunks of bridge anchors near the bridge end still contain M0-continuation
    actions (no validity mask as in theirs).

## Disk cleanup round 4, 2026-11-21 (~04:15-04:35 UTC; ~15.6 GB freed on /pfss)
- Trigger: the F1 watcher's probes failed with `[Errno 122] Disk quota exceeded` at 03:58 and 04:10 UTC (det CLI
  temp write). A write test at 04:20 succeeded. F1 lanes were in back-off at that time (owner GPU > 1 GB, 03:46).
  `rv_f1_watch` and every other tmux session / process were left untouched.
- Measured (du): runs 66 G (goal_rsi 23.6 G, _logs 24 M, _analysis 273 M), data 176 G (recovery 117 G, wm 39 G,
  foundation 19 G, goal_rsi 1.4 G, acr 9 M, geom 5 M). Workspace ~/.cache 484 G: HF hub 250 G (mostly other projects),
  HF datasets 14 G, xet 7.5 G, openpi 12 G, torch 4.3 G, uv 6.4 G, pip 29 M. miniforge3/pkgs 22 G, ~/.conda-pkgs-gpu 8.1 G.
- Deleted (dry-run table first):
  - `runs/goal_rsi/m0_s7/ckpt_{01000,02000,03000,05000}/trainable.pt` (4 x 1.8 GB = 6.7 GB). These are non-selected
    M0 checkpoints of a finished run. ckpt_04000 (the M0 init used by F1) is kept in full. Each ckpt keeps eval_val/,
    meta.json and its processors. A note is in `m0_s7/DELETED_CHECKPOINTS.txt`.
  - `~/.cache/huggingface/datasets/parquet`: the 78 of 101 Arrow cache dirs last written before 2026-11-14 (8.9 GB,
    14 G -> 4.7 G). They are regenerated on the next load. List: `runs/_logs/cleanup_20261121_hfcache_list.txt`.
  - Scratch: `/tmp/diagram-design` git clone (57 MB, container overlay; the flow-diagram skill is self-contained),
    the 2 RGR smoke runs `runs/20261108-17*_RGR_rgr_smoke_s0of8` (166 KB), and 19 `__pycache__` dirs outside env/.
- Checked, nothing to free: orig_s7-9 / rec_wm_s7-9 / rec_rand_s7-9 already hold only the final ckpt_02000.
  Their `trainable.pt` is fp32 weights only (451.5 M params, no optimizer state). `conda clean -a --dry-run` found
  nothing. `uv cache` is only 6.4 G and shared, so it was not pruned. No SFT run has intermediate step dirs
  (006000 + last only). runs/_logs is 24 M (no rotation needed). No /tmp/acrn_* or /tmp/lrs* left.
- Left for the user to decide (not deleted): the 9 goal_rsi final checkpoints (15 G; results already in main.json);
  closed-idea raw data (RGR rgr_sweep 9.4 G, xwm_sweep 3.7 G + verif_xwm_sweep, ACR-N acrn_single_fresh 1.7 G,
  rq3_sweep 3.9 G + verif_rq3_sweep, vj_rq3_sweep 1.7 G, foundation/vj_latents 12.8 G + frames 2.6 G);
  old M1 / M1b / M3 / M5 recovery banks (~13 G); data/recovery/obj 19.6 G and m5b 14 G (code-referenced);
  ~/.cache/openpi pi0_base (12 G, JAX copy next to HF lerobot/pi0_base 14 G; owner unclear);
  HF paligemma-3b-pt/mix-224 (22 G); HF xet chunk cache 7.5 G. The .env HF token is read-only, so offloading
  to HF is not possible yet.
- 2026-11-22 09:50 UTC: **F1 watcher fixed and restarted** (local tmux `rv_f1_watch`).
  - Symptom: 6 launch -> back-off cycles (11-20 13:52 / 14:30, 11-21 21:40 / 22:17 / 22:54 / 23:32 UTC). Each one
    backed off after ~13 min (the first RUN poll) with "owner_gpu_mib ~60 GB, owner_cpu 0". The runs reached only
    step ~100 (f1_wm_s7 / f1_rand_s7 = args.json + metrics.jsonl).
  - Root cause = OUR OWN memory. Inside the Determined container nvidia-smi reports HOST pids, which are not in the
    container's /proc. `probe_owner.sh` could not resolve them to our tmux trees, so our two pi0 SFT jobs
    (2 x 30.3 GB = 60.6 GB, exactly the logged value) were counted as owner memory, and the watcher killed them.
  - The lane-log SignalException "signal 1" (SIGHUP) + DataLoader "Terminated" is that back-off kill (kill_rv.sh
    TERM, then tmux kill-session HUP). It is NOT the det session closing: the lanes were alive 12 min after launch,
    and tmux on the remote side survives the det session.
  - Fix (goal_f1_watch.py docstring):
    - Per-GPU attribution: other-container memory = used − listed apps. In-container listed apps count as ours while
      our lane is on that GPU.
    - Back-off when other(g) grows > 4 GB or listed(g) grows > 45 GB since launch, or free < 6 GB, or owner CPU
      >= 600 % on 2 polls.
    - Per-GPU launch after 2 idle polls of that GPU.
    - The 6 runs form a shared, locked queue in paired order (wm:7 rand:7 wm:8 ...), so GPU1 alone runs them in
      sequence.
    - Owner-CPU gate relaxed to < 400 % (the owner's single 100 % CPU search.py has run for days).
  - Stale claims released.
  - Current: GPU0 owner-busy (in-container app 9.8 GB, 30-60 % util); GPU1 idle (0 MiB / 0 %), first idle poll
    09:50, so the expected launch on 63e0f914:1 is ~10:03 UTC.
  - The 09:29 "Disk quota exceeded" was a transient /pfss quota error inside det's ssh setup (probe failed, no write
    by us). Later polls are fine.
- 2026-11-22 11:03 / 11:16 UTC: **both F1 lanes died on /pfss quota** (EDQUOT) while writing the 250-step resume
  checkpoint.
  - Root cause: the workspace-wide /pfss quota is full. Other projects consume it; write tests show 0-3 GB headroom,
    fluctuating. Our resume format then needed two full copies of 5.26 GB (trainable fp32 1.8 GB + AdamW moments
    3.6 GB, plus a .tmp before the rename). `torch.save` raised `inline_container.cc:668 unexpected pos` and killed
    the run. The lane logs also show `--- Logging error ---`: even log writes fail under EDQUOT.
  - State saved: f1_wm_s7 resume step 500 (training reached ~700), f1_rand_s7 resume step 250 (reached ~450). Both
    resume.pt are valid zip files; the truncated resume.pt.tmp (0.99 + 2.3 GB) were deleted.
  - Space: converting both resume.pt to the compact format went via the agent box's local /tmp (the /pfss write
    failed at 0 GB headroom). Measured headroom afterwards: >= 7 GB.
  - Freed in total: 3.3 GB (truncated .tmp) + 5.3 GB (compact resume: 2 x 5.26 -> 2 x 2.63 GB) = 8.6 GB.
  - Fix:
    - `save_resume` stores trainable params + AdamW moments in bf16 (2.63 GB), ONE copy (old deleted before the
      write), and skips on failure instead of crashing. Moments are cast back to fp32 on load (logged deviation).
    - The final ckpt is retried on EDQUOT (12 x 5 min) and written in bf16 for F1 (`--save_dtype bf16`, 0.9 GB).
      Logged deviation: eval loads bf16-rounded trainable weights; inference already runs in bf16 autocast.
    - metrics.jsonl and eval episode / summary writes are EDQUOT-tolerant (retry / warn).
    - tmux_start.sh pipes the job output through a restartable grep, so a log-file EDQUOT can't raise inside the job.
    - No intermediate ckpts (save_every 2000 = final only).
  - Peak F1 footprint: 2 lanes x (2.63 resume + 0.9 final) ≈ 7 GB, ~5.4 GB of final ckpts at the end.
  - NOT deleted (cleanup round-4 decision list, needs the user's OK):
    - the 9 finished goal_rsi final ckpts (orig / rec_wm / rec_rand x 3, 1.7 GB each = 15 GB; could also be
      halved to bf16);
    - data/wm/goal_verif 3.1 GB (search verification log with latents);
    - data/wm/goal_collect 3.1 GB (WM / C^pi training data);
    - the closed-idea data listed in round 4.
    The M0 init `m0_s7/ckpt_04000` is required by F1.
  - Owner at 11:28 UTC: in-container GPU jobs on both GPUs (4.8 / 6.1 GB, 29-59 % util) -> the watcher waits per its
    rules (lanes relaunch after 2 idle polls; fails 1 / 3 each).

## Disk cleanup round 5, 2026-11-23 (~02:50-03:10 UTC; 136.5 GB freed, RetryVLA ~241 G -> 104 G)
- User-approved groups (1)-(4). Dry-run table first; `lsof` showed no open file under any target; no symlink from a
  kept path points into a target; no target is git-tracked. `rv_f1_watch` (`scripts/goal_f1_watch.py`, only touches
  `runs/goal_rsi/`) and all other tmux sessions left running. Not touched: `runs/goal_rsi`, `data/wm/goal_*`,
  `data/recovery/goal_search`, HF hub, paligemma, xet, conda pkgs, other projects.
- (1) SFT checkpoints, 32.9 GB: `checkpoints/*` emptied (config.yaml, git.txt, metrics.jsonl, summary.json, tb, eval
  dirs kept) in 29 runs: M4_sft_A_base_nom_random; M5_m5b_sft_random; M7_sft_{base,weak}; M8_seed_{base,weak}_s{1,2};
  M9_sft_{goal,comp} (+ seed1 / seed2); M10_sft_comp_obj; P4_af_arm{3_free_filtered,4_free_unfiltered,5_hybrid}_s{0,1,2};
  BT_sft_bt_distill (+ _s1 / _s2); M9_sft_demo_only_s{1,2} (all 2026-10-14 .. 2026-11-02, results in RESULTS).
  Kept (init / pending SMOOTH-V): M1_sft_nominal_only (pi1-nominal init), M5_m5c_sft_wm_diverse_rr (pi1 seed 0, SMOOTH-V
  `control` + smoothness proxy), M8_seed_pi1_s{1,2} (pi1 seeds 1 / 2 = SMOOTH-V `control` seeds), M6_sft_pi1_more,
  M10_sft_nominal_obj. WM / critic runs referenced by `data/wm/latest_*.txt` hold no checkpoints/ dirs and were untouched.
- (2) Closed ideas, 38.8 GB: `data/recovery/{rgr_sweep, xwm_sweep, rq3_sweep, vj_rq3_sweep, acrn_single_fresh}`,
  `data/wm/{verif_rq3_sweep, verif_xwm_sweep, verif_vj_rq3_sweep}`, `data/foundation/{vj_latents, frames, vjepa2}`
  (V-JEPA 2 weights re-downloadable). `runs/_analysis/*` outputs kept (none read these dirs; `wm_vs_true` / t2mem use
  `verif_m5b`, kept). `data/wm/true_end_rq3` kept (wm_vs_true source). `data/foundation/latest_vj_*.txt` kept.
- (3) Old banks / verif logs, 64.9 GB: `data/recovery/{m7_weak, m9_mech, m6_r2, m8_aug, m11_a, m9_b32, b2x_value,
  b2_value, m1_oracle_cem_spatial_{s0,s0b,s1}, m1_random_spatial_{s0,s0b,s1}, m1b_hard_{cem_local,random}_{s0,s1},
  m3_wm_search_hard_{s0,s1}, m5_r2_{v1,v2}, m5_wm_r1_s{0..3}}`, `data/wm/{verif_m8_aug, verif_m7_weak}`.
  Kept: `data/recovery/m5b` (pi1 bank `m5b/s*/p*/wm_diverse`, SMOOTH-V + smoothness scripts), `data/recovery/obj`,
  `data/wm/verif_m5b` (9.4 G; read by analyze_verifier / wm_vs_true / t2mem / vj scripts, i.e. the SMOOTH-V candidate pool
  for `smooth_reverify` fill-ins), `data/wm/verif_r1`, collect_*, m11_b, b32_rise, bt_distill.
  Note: configs for M8-M11 / B2 searches and SFTs that read the deleted banks can no longer be re-run without regenerating.
- (4) `~/.cache/openpi` (12 G, pi0_base JAX + paligemma tokenizer): NOT deleted. RetryVLA does not use it (HF
  `lerobot/pi0_base`), but other projects in the workspace use openpi (`samdata/pi0_*` robomimic / aloha runs with
  base `gs://openpi-assets/checkpoints/pi0_base/params`, `cross_emb_2/RA-VLA`, `minhduc/*` pi05) -> ask the user.
- 2026-11-23 03:25 UTC (user decision): F1 moved to a NEW RetryVLA-owned 2-GPU shell.
  - Shell: `953bb534` `vla_RetryVLA_DoanhLT_2g` (scripts/shells/retryvla_shell_2gpu.yaml = the 1-GPU template with
    slots 2), started 03:26 UTC, RUNNING at 03:27 UTC, node cn62.
  - envcheck GPU0 / GPU1: OK video decode, OK sim 50 steps 0.98 / 0.96 s. Registered in shells.json allow_only /
    share_ok (full use, kill when done).
  - rv_f1_watch (63e0f914) stopped. Its last polls showed the owner on both GPUs (60 / 20 GB, 99 %) and no lanes of
    ours. Stale F1 locks were removed, and 63e0f914 is marked light-share / unused (no claims).
  - Local watchdog `rv_f1_chain2` (scripts/goal_f1_chain2.py, 10-min polls via probe_tmux.sh) launched remote tmux
    lanes `rv_f1_g0` (GPU0) and `rv_f1_g1` (GPU1) at 03:36 UTC.
    - Same shared locked queue and quota-safe checkpointing (compact bf16 single-copy resume, bf16 final ckpt,
      EDQUOT-tolerant writes).
    - Resumed f1_wm_s7 at step 1750 and f1_rand_s7 at step 250 (the last complete 250-step resume points; training
      had reached ~1850 / ~450). 4.3 s/it.
  - The first chain2 start (03:28) timed out on a `bash -c` probe (det argv re-join trap) and was replaced by a script
    probe before any launch.
  - ETA: all 6 runs + val evals ~11:30 UTC 11-23, then automatic analysis -> RESULTS / NOTES, commit + push,
    `det shell kill 953bb534` (unless scripts/shells/keep_953bb534 exists).
- 2026-11-23 (KIN-1 / Kintsugi novelty agent): research + pre-registration only; no GPU used.
  - `docs/research/NOVELTY_VS_KINTSUGI.md`: Kintsugi verified from the arXiv HTML v1; no official code (GitHub
    `Artifex2002/project_Kintsugi` is unrelated).
  - PLAN "KIN-1" written, about 115 GPU-h primary. Queue on 953bb534 AFTER F1 and the Back on Track evals, and
    coordinate with those agents before claiming GPUs.
  - RESULTS: appended a main-table note.
  - PAPER_FRAMING_2026-11.md was not present at commit time, so nothing was appended there.
- 2026-11-23 08:25-08:35 UTC (Goal campaign):
  - Second RetryVLA-owned shell `8f473147` (vla_RetryVLA_DoanhLT_2g_b, cn36, 2x A100) RUNNING; envcheck OK on
    GPU0 / 1.
  - Lanes: `rv_onset` (GPU0, onset scans of audit seeds 7, 8) and `rv_rsr` (GPU1, stage-1b E8 RSR of 13 ckpts),
    started via tmux_start.sh, claims registered.
  - `scripts/shells/keep_953bb534` created, so goal_f1_chain2 keeps 953bb534 after F1.
  - Diagnostic D1: orig_s7 cast to bf16 = 83 / 100 (fp32 82), so bf16 final ckpts are not the cause of the F1 drop.

## Back on Track + ForkCast-style test-time arms, ops log 2026-11-23 (design: PLAN "Back on Track + ForkCast-style"; doc: docs/related_work/BACK_ON_TRACK_REPRO.md)
- 04:55 UTC: `scripts/shells/keep_953bb534` created (user request), so goal_f1_chain2's finish() keeps 953bb534 alive
  after F1. 953bb534 belongs to the Goal-improvement campaign (coordinator 08:05 UTC): BoT never uses its GPUs.
- 08:05 UTC coordinator: Goal protocol first, ForkCast-style arm added, LIBERO-Recover secondary, one 1-GPU shell OK.
- 08:08 UTC: requested `92927808` `vla_RetryVLA_DoanhLT_bot1g` (user account, -w IAS_SAMMerge,
  scripts/shells/retryvla_shell_bot1g.yaml = 1-GPU template, new description). RUNNING, envcheck OK (video decode, sim
  50 steps 0.94 s). Registered in shells.json allow_only / share_ok. KILL when the BoT queue is done.
- CPU unit test (fake policy, Goal WM + C^pi): all 4 wrappers run; batch-1 CPU bench 0.41 s / 1,280 plans.
- Smoke on 92927808 (8 collection episodes, 6 val episodes / arm, `runs/bot_goal_smoke`): collect -> train -> BoT (C^pi),
  ForkCast (C^pi / progress), BoT (liveness, forced trigger t_c -100) all end-to-end. A100 batch-1 latency: monitor
  16-40 ms, BoT planning cycle (256 x 5 = 1,280 ten-step plans) 100-137 ms, ForkCast cycle (5 candidates x 50 steps)
  34 ms, pi0 3B chunk 0.33-0.38 s. First smoke bug: train_progress with 0 successes (fallback added).
- Deviation logged before the full run: PCA play data = LIBERO-Goal TRAIN-split teleop demos (their play data is the
  LIBERO-Goal demos) instead of our WM's goal_collect actions (k = 8 of 60 dims on demos alone).
- 08:36 UTC: full chain `rv_bot_goal` (scripts/bot_goal_chain.sh) launched on 92927808 GPU0; local watchdog
  `rv_bot_watch` (scripts/bot_watch.py, 10-min probes, <= 3 relaunches, re-asserts claim `92927808:0:rv_bot_goal`,
  runs scripts/analyze_bot_goal.py at CHAIN_DONE). Queue: collect 900 M0 rollouts (train inits x seeds 8-10) ->
  liveness + progress + PCA -> 8 val evals (m0_bot_cpi, m0_bot_liv, m0_fc_cpi, m0_fc_prog, orig7_bot_{cpi,liv},
  recwm7_bot_{cpi,liv}). ETA ~11:30-12:30 UTC. Secondary (LIBERO-Recover Spatial / Object) is queued after it on the
  same shell.
- 2026-11-23 11:07 UTC: BoT / ForkCast Goal chain DONE on 92927808 (bot_watch); table runs/bot_goal/main.md (reading + RESULTS by the agent).
- 2026-11-23 11:59 UTC: F1 finished on 953bb534 (goal_f1_chain2); table appended to RESULTS. Claims released. Shell KEPT (keep file present).
- 2026-11-24 ~08:50 UTC (coordinator): BoT / ForkCast-style Goal chain (`rv_bot_goal` on 92927808, CHAIN_DONE
  2026-11-23 11:00 UTC; local watchdog `rv_bot_watch` exited "done" 11:07) analysed: RESULTS "Main table: test-time
  arms on the 2610.05151 LIBERO-Goal protocol". M0 85 -> BoT(liveness) 79 (6 / 12, p 0.24, inconclusive),
  BoT(C^pi) 51, ForkCast-style C^pi 62 / progress 32; Original s7 + BoT(liveness) 82 = 82; cost 1.2-1.7x of plain
  pi0 per episode, 90-100 ms per planning cycle. Deviations as pre-registered (BACK_ON_TRACK_REPRO D1-D15); no
  further Goal arms queued (F1 below Original -> best-RetryVLA + BoT skipped per coordinator). Spatial / Object BoT
  arms NOT run (secondary). Shell 92927808 killed (`det shell kill`), claim released, moved to shells.json gone.
  Data kept: data/bot/goal_m0 (900 rollouts, full tokens, 1.6 GB; delete if disk is needed).
- 2026-11-24 08:45 UTC: **F1 diagnosis** (coordinator priority 1). Control runs are on 4 GPUs, all M0 + 2k, seed 7,
  val eval after.
  - D2 `diag_orig_newpath_s7` (953bb534:0): Original recipe through the NEW code path (single GPU accum 3,
    compact bf16 resume every 250 with a FORCED restart at step 500 via `--stop_at`, bf16 final ckpt).
  - D3 `diag_f1wm_oldpath_s7` (953bb534:1): F1 wm recipe on the OLD path (no resume, fp32 final).
  - D4 `diag_d0dup_s7` (8f473147:0): pooling control, D0 + 36 episodes of D0 itself as the "recovery" root.
  - D5 `diag_f1wm_full_s7` (8f473147:1): 36 wm trajectories with bridge + continuation anchors.
  - Launched 08:46-08:48; this rule block was written 08:58, before any of these results (D1 = bf16 cast of orig_s7
    = 83, done 11-23).
  - Rules (Original s7 = 82, seed range 77-82; F1 wm s7 = 58):
    - code-path regression iff D2 < 72 (then fix + rerun F1 x 3);
    - resume / bf16 path explains F1 iff D3 >= 72;
    - Pool / second-root bug iff D4 < 72;
    - otherwise "data effect, no pipeline bug".
    - D5 vs D3 = continuation vs bridge-only at equal trajectories (36), 1 seed (directional only).
  - Determinism check: D3's step-0 / step-50 loss (0.1294 / 0.1922) equals f1_wm_s7's, so training is bit-repeatable
    up to the first resume.
  - Static checks done (runs/goal_rsi bank vs D0):
    - same feature keys, 128 px images, 8-d state, 7-d action, same task strings;
    - image channel means 0.487 / 0.454 / 0.419 vs 0.495 / 0.461 / 0.425;
    - action-to-state alignment: corr(a_t, s_{t+1} − s_t) is 0.70-0.91 per axis for bridges vs 0.89-0.91 for D0, and
      lag 0 is the best lag in both, so no off-by-one;
    - fps differs (bank 10, D0 20), but the chunk delta_timestamps are per dataset, so both index consecutive frames;
    - gripper in [-1.1, 1.1] (unclipped M0 continuation outputs; the env clips).
  - Bridge content: 137 / 308 wm bridges contain a gripper flip (proposal perturbations flip gripper intent 25 %);
    11 % are primitives.
  - Bridge frames are concentrated on cream_cheese (4.0k) and wine_rack (3.4k). These are exactly the tasks F1
    breaks most (cream_cheese 7.0 -> 2.0, wine_rack 4.0 -> 1.3).
- 2026-11-24 11:21-11:24 UTC: F1 diagnosis runs finished: D2 84, D3 63, D4 80, D5 80. Verdict "data effect, no
  pipeline bug" (RESULTS "F1 diagnosis"). Registered 12:45 UTC. Lesson: register finished chains immediately.

## CRIT-V critic variants (2026-11-23/24)
- Shell a3da0634 (vla_RetryVLA_DoanhLT_critic, cn28, user account) requested 2026-11-23 08:20 UTC, registered; used for the
  rich replay (11-23 08:30-09:37 UTC, 5 banks, 33,296 bridges, eef replay <= 2.7 mm, prefix agreement 100 %) and the
  critic-variant training (11-24 08:55-10:09 UTC, 2 suites x 7 kinds x 3 seeds). Killed 2026-11-24 ~12:45 UTC after all
  pre-registered jobs finished; claims released, moved to shells.json:gone. No GPU of 953bb534 / 92927808 used.
- Deviations: (1) bridge-only replay (no pi continuation), decided before results (PLAN). (2) First training launch
  (08:47) restarted at 08:55 after 3k steps: CPU-side batch gather (110 ms/step); fix = bank on GPU, same seeds /
  batches / recipe, the killed partial run was deleted. (3) eval run on CPU locally, bootstrap 400 reps (Goal) / 200
  (Spatial) instead of 2,000 (time).
- Result: no variant passes the pre-registered rule (RESULTS "CRIT-V") -> **no hand-off to the LIBERO-Goal campaign**.
  For information only: a plain C^pi retrained on all goal_collect (`runs/20261124_CRITV_goal/ckpt/base_s{0,1,2}.pt`,
  `retryvla.critic.variants.CriticV(kind="base")`) has goal_verif imagined AUC 0.714 vs 0.639 for the shipped critic,
  but that comparison is confounded by independence (shipped C^pi selected half of those candidates).
- New opt-in instrumentation for future verification runs: `m5b.rich_record: true` (+ `m5b.rich_out`) in compare_arms
  -> per candidate actions, proprio, pooled DINOv2 latents every 10 steps, symbolic / BDDL / contact timeline every 5
  steps over bridge + continuation, point clouds every 10 bridge / 50 continuation steps (~0.1 MB / candidate).
- Pending: adversarial review (idea-loop step 4) before any "close".
- 2026-11-24 12:45-13:10 UTC: G2 / P2-3 launched (PLAN pre-registration, written before any result).
  - Smoke tests passed:
    - curated search: 1 onset entry, random 0 / 2, wm 1 / 2; the verified bridge has 0 gripper flips, 50 + 263 steps;
    - D_P: 1 pre-onset entry, p(a-) = 0, 6 / 12 screen hits, 2 positives, 7.5k sim steps, 222 s;
    - detector E-v1: 4 val episodes of Original s7, 1 stall event, incidence 0.25.
    One fix along the way: a- shorter than 50 steps when the audit rollout ends.
  - Lanes:
    - curated onset search `rv_g2s0` / `rv_g2s1` (953bb534:0 / 1) and `rv_g2s2` (8f473147:0), 17-18 entries each,
      ~70-210 s per entry;
    - D_P generation `rv_dp0..3`: 8f473147:1 x2 and 953bb534:0 / 1, 13 pre-onset entries each, ~4 min per entry.
  - Orchestrator: local `rv_g2_chain` (scripts/goal_g2_chain.py). Its queue fills the 4 GPUs as the bank lanes
    finish:
    - eval-only for Original s7-9 / M0 (events + E8on);
    - G2 wm / rand x 3, D5 s8 / s9;
    - P2-3 A / B x 3.
    It auto-registers the G2 and P2-3 tables (analyze_goal_g2.py) and kills an idle shell after 2 h.
  - New code:
    - compare_arms `m5b.curated` (opt-in family / gripper-flip filter);
    - retryvla/search/goal_prevent.py, retryvla/eval/goal_events.py (detector E-v1, frozen) + vec.py `events_scan`;
    - eval_goal `--save_actions`;
    - pi0_sft per-group caps `--recovery glob@N`.

## LRS-G (full 2610.05151 Recovery reproduction on Goal), ops log
- 2026-11-24 13:20 UTC: 1-GPU shell bb15ca35 (demos) requested, killed unused 13:50 when the scope grew. 13:45 UTC:
  2-GPU user-account shell `1123d1ea` (vla_RetryVLA_DoanhLT_lrs2g, node 1fc4d0846a68) requested, registered in
  shells.json (allow_only + claims `1123d1ea:{0,1}:rv_lrsg`). envcheck OK (sim 50 steps 1.0 s).
- Code: `retryvla/baselines/lrs_goal.py` (+ `lrs_goal_prompts.py`), `scripts/lrs/{collect_lanes.sh, build_lrs_dataset.py,
  lrs_run.sh}`, trainer opt-in `pi0_sft --masked_recovery --l_min 5`, demos `scripts/demos/*`.
- Smoke (1 entry): 8 LLM calls / 32 s; the 8-reply cap ended the stage before any `done` -> D-L4b + cap 12.
- 14:00 UTC: 6 lanes (3 / GPU) OOM at load (~26 GB / lane while loading pi0 + Qwen3-VL-8B). 14:11 UTC: relaunched with
  4 lanes (2 / GPU, `rv_lrsg_a0_l{0..3}`, 42 entries each); model load takes ~10 min with 4 concurrent loads from /pfss.
  Throughput ~1.5-2.5 min / entry / lane (8-15 LLM calls), ETA collection ~15:45 UTC.
- First rows: the LLM observer accepts the first hand-off in most entries while the sim ground truth (M0 <= 300 steps)
  mostly fails (to be quantified at the end; pre-registered judge-precision rule).
- Dataset + masked Pool checked on 25 accepted entries: 53 windows / 1,335 frames, L_min drops 4 / window, recovery
  anchors carry validity masks (5-50 valid targets) and q weights; D0 anchors unchanged (mask 1, w 1).
- 2026-11-24 15:00-15:15 UTC (coordinator additions):
  - CRIT-V `base` critic swap wired in (WMCriticRanker loads CriticV); link data/wm/goal_critic_base.txt (gitignored)
    -> runs/20261124_CRITV_goal/ckpt/base_s0.pt.
  - G2 / D_P were generated with the shipped C^pi (deviation; D_P uses no critic).
  - Bank sizes: G2 wm 132 trajs / 43 entries, random 115 / 44; D_P 82 positives at 44 / 51 s^-, p(a-) mean 0.11.
  - Lane `rv_physfc` (8f473147:1, co-located, CPU-bound sim) runs the G2-phys scan (132 + 115 replays), then the FC
    labelling of seeds 7, 8.
  - INCIDENT 15:08: restarting goal_g2_chain lost its job -> slot map and launched 4 P2-3 jobs onto the 4 GPUs that
    were already training G2. Caught within 2 min: the P2-3 lanes were killed (rv_* only, by session) and their run
    dirs deleted before any training step. The G2 runs were untouched (1 app / GPU, ~30 GB each, verified).
  - Fix: the slot map is persisted (runs/goal_rsi/g2_slots.json), and running lanes not in the map make their shell
    busy.
  - Queue reordered to the pre-registered order (P2-3 before G2 random); G2-phys arm added (gated on phys_summary).
- 2026-11-24 17:51 UTC: goal_g2_chain: G2 wm finished, auto table appended to RESULTS.
- 2026-11-25 03:09 UTC: goal_g2_chain: G2 wm finished, auto table appended to RESULTS.
- 2026-11-25 03:09 UTC: goal_g2_chain: P2-3 finished, auto table appended to RESULTS.

## V2R (RetryVLA v2 training track on LIBERO-Recover Spatial), ops log
- 2026-11-24 15:05 UTC: 2-GPU user-account shell `ea7f9ee3` (vla_RetryVLA_DoanhLT_v2rec, host 894e2ccf9223) requested,
  envcheck OK (sim 50 steps 1.0 s); 15:57 UTC second 2-GPU shell `f72a5ddd` (_v2rec_b, host 442c67d05c0d). Both in
  shells.json allow_only. Never used 953bb534 / 8f473147 / 1123d1ea.
- 15:59 UTC pre-registration committed (PLAN "V2R", a547a63) before any V2R result.
- 16:01-~19:00 UTC R1 search (critic swap, 8 shards as 4 tmux lanes rv_v2r_s0..3 on ea7f9ee3): all 8 shards DONE.
  Search-level vs M5b wm_diverse (same 256 instances / slot seeds, historical control): solved 0.570 vs 0.547
  (McNemar 25 / 19, p 0.45: no detectable ranking gain at n = 256), hits / verification 0.178 vs 0.171, mean
  predicted value on picks 0.837 vs 0.735 (base critic more optimistic in absolute terms on its own picks; real hit
  rate 0.18). 436 verified trajectories (6.0 GB bank, slim verification log 25 MB).
- Smokes (16:03-16:23 UTC): eval save_actions + E-v1s replay (6 nominal episodes, replay_ok 6 / 6, events tipped /
  displaced found), rec_phys + visual_bank on a 6-trajectory copy of an M5b shard, sft groups + aux (failed only
  because the smoke bank had no CLOSED marker), rec_prevent: KeyError 'akita_black_bowl_1' -> the scenes_full
  bddl dumps declare one object per line, which the LIBERO parser collapses; fix = rewrite the (:objects) block into
  the benchmark-scene form (rec_prevent.fix_bddl, cached under data/recovery/v2r_dp/bddl_fix; init regions untouched).
- **Incident (agent): the chain was NOT launched after the smokes on 11-24; both shells sat idle from ~19:00 UTC
  (search done) to 03:10 UTC 11-25 (~8 h x 4 GPUs lost).** The previous agent turn ended before `rv_v2r_watch`
  was started. Lesson (already in retryvla-run-lessons 1): start the watchdog in the same turn as the first lane.
- 2026-11-25 03:10 UTC: local watchdog `rv_v2r_watch` (scripts/v2r/watch.py) started; it started `rv_v2r_chain`
  (scripts/v2r/chain.py, shared claim queue on /pfss) on both shells. 03:10 dp_sh0 / dp_sh1 (ea7f9ee3), r1_phys +
  vis_r1 (f72a5ddd); 03:13 R1 SFT s0 / s1 (ea7f9ee3), s2 (f72a5ddd); r0 nominal event re-evals as sim slots free up.
- Visual check R1 bank (runs/_analysis/visual/recover_v2r_r1/, 8 stratified renders; looked at 5 sheets 03:20 UTC):
  successes are genuine (bowl visibly on the plate at the end); the worst bridges are gripper chatter (11-14 flips in
  50 steps, perturbation family) with little EE motion, a 1.79 action jump at one handoff, and the pi0 continuations
  themselves chatter (15-53 flips). Same character as the M5b bank behind pi1 -> distilled as R1 (pi1 recipe); the
  R2 filter targets exactly these (flips > 1, jerk, seam).
- 2026-11-25 03:20 UTC: goal_g2_chain: G2 wm finished, auto table appended to RESULTS.
- 2026-11-25 03:20 UTC: goal_g2_chain: P2-3 finished, auto table appended to RESULTS.
- 2026-11-25 03:10-03:45 UTC:
  - goal_g2_chain had died at 22:54 UTC 11-24: the agent box changed container (local tmux gone). GPUs were idle ~4 h.
  - The chain now runs ON 953bb534 (tmux `rv_g2_chain`, scripts/goal_g2_chain_remote.sh, 5 restarts; det works from
    inside the shell). It relaunched g2phys_wm_s9, g2_wm18_s7 / s8 and g2_rand_s7 at 03:09 and auto-registered the
    G2 wm + P2-3 tables.
  - Follow-ups queued: g2physns_wm x3, g2_wm5 x2.
  - Visual check (runs/_analysis/visual_check/goal_g2):
    - G2 bridges are smooth, but M0 continuations chatter (ep26: 21 gripper flips over 280 steps; ep44: 9; handoff
      jumps up to 1.59).
    - The phys filter removes seam / jerk / object-jump cases that also look bad.
    - Its survivors are short near-goal bridges without continuation.
  - Review doc: docs/research/REVIEW_2026-11-25_GOAL.md.
- **Deviation D-V1 (2026-11-25 03:35 UTC, before any R2 training / eval result): R2 filter.** The pre-registered
  rec_phys rules keep only 5 / 436 R1 trajectories (4 / 146 instances): Spatial D0 thresholds d3 > 0.195 (2 x demo
  median 0.098) and seam > 0.246 (3 x median ||Δa|| 0.082) reject 362 / 331 trajectories, because searched bridges are
  ~4.6x jerkier than demos (bank d3 median 0.446; cf. SMOOTH-V 4.7x). Other rejects: replay mismatch 49 (11 %, open-loop
  replay from the rebuilt snapshot does not reproduce success; to investigate), gripper flips > 1 50, object jump
  24, penetration 7, off table 2. The pre-registered fallback ("all survivors + R1 re-matched to the same count") at
  n = 5 tests nothing, so R2 is redefined BEFORE any result: hard physical rejects stay (replay mismatch, penetration,
  off table, impulse, gripper flips > 1) -> 307 trajectories / 118 instances; jerk and seam become the SMOOTH-V
  smooth-first ranking instead of rejects (round-robin over instances, lowest bridge d3 first) -> 208 selected,
  118 instances, bridge d3 median 0.374 vs 0.446 for all R1 trajectories (−16 %), seam median 0.79. Keep list
  `data/recovery/v2r_r1/phys_wm_diverse_hard.jsonl` (strict flags kept as `reject_strict`). The strict 5-trajectory
  survivors are reported only as a bank statistic. Visual check of the strict survivors: runs/_analysis/visual/
  recover_v2r_r2/ (5); of the hard-filter selection: recover_v2r_r2h/. Chain restarted on both shells to pick this up
  (running children kept; restarted chain now counts live claims of its host as occupied slots).

### LRS-G v1 results + diagnosis (2026-11-25 03:10-03:40 UTC; coordinator: "val looks broken")
- v1 chain finished 2026-11-24 23:08 UTC. Val (100 scenes): lrs_s7 14, lrs_s8 21, lrs_s9 8 (mean 14.3), lrs_all_s7 1
  vs Original 82 / 77 / 79 (79.3), M0 85. Per task: open drawer 0, push plate 0, cream cheese 0, wine rack 0, stove 0.
- Teacher-loop statistics (168 seed-7 entries, 1 attempt; `runs/goal_rsi/lrs_goal.json`): accepted 132 (78.6 %),
  budget 25, turn cap 8, abort 3; 217 hand-off tests, judge accepts 132, sim-GT successes 18 -> **judge precision
  0.114 [0.070, 0.179]**, recall 0.83 [0.61, 0.94]; entries solved in sim 0.107 vs goal_search wm_diverse 0.589
  (McNemar 2 / 83, p 2e-22). 15.0 LLM calls, 82 s LLM, 147 s wall / entry / lane; 1.36 stages; 23,044 accepted teacher
  actions; datasets: `lrs_goal_ds` 18 paths / 50 windows / 1,708 anchors (4.3 %), `lrs_goal_ds_all` 118 paths / 292
  windows / 10,552 anchors (21.6 %).
- Checks that PASS (no pipeline bug in the trainer / eval path):
  - args.json lrs_s7 = orig_s7 recipe + `--recovery lrs_goal_ds --masked_recovery --l_min 5` (init m0_s7/ckpt_04000,
    2k, bs 48 accum 3, warmup 200, cosine; the bf16 / resume path is the validated D2 path, 84).
  - masked loss == standard pi0 loss on D0 anchors (M0, 6 batches: 0.13776 vs 0.13776, pairwise equal to 1e-7):
    `scripts/lrs/check_masked_loss.py`, `runs/_analysis/lrs_goal_masked_loss_check.json`.
  - windows replay in sim from the exact entry snapshot (dataset builder; visual check renders replay them).
- ROOT CAUSE (data, not plumbing): **the teacher's low-level servo (`exec_cmd`, inherited from the 2026-11-17 Spatial
  LRS pilot) emits bang-bang actions**: a_xyz = clip(err / 1 cm, +-1) -> 42 % / 14 % / 41 % of x / y / z targets are
  saturated at +-1 (D0: 0 %, |a| q99 0.94; WM bridges 0.1-0.8 %); rotation targets are exactly 0 (D0 std 0.06-0.10).
  At the M0 init the per-target FM loss on these windows is 1.58 vs 0.138 on D0 (11.5x), so a 4 % anchor share carries
  ~33 % of the gradient. The trained policies copy it: val actions of lrs_s7 have mean gripper +0.27 (Original -0.20),
  |a_x| saturated 3.6 % (Original 0 %), rotation std 0.014-0.028 (Original 0.06-0.08); lrs_all_s7 (21 % share):
  saturated x 14 %, rotation std 0.003 -> near-frozen orientation, SR 1. The damage is broad (tasks never in the
  windows also collapse, e.g. turn_on_stove 10 -> 0): out-of-distribution, max-magnitude targets with a zero-rotation
  prior shift the whole action head.
- This is our implementation's deviation, not the paper's method: their teacher uses "native actions" + motion
  helpers with units / limits (unknown servo). Logged as **D-L9 (v1 servo)**. Fix v2 (`--servo demo`): proportional
  a = err / 4 cm, |a_xyz| <= 0.6 (~D0 q75-q90), orientation hold (axis-angle P control, |a_rot| <= 0.3); unit test on
  the cream-cheese entry: saturation 51 % -> 0 %, |a| q50 1.0 -> 0.59, rotation drift 2.6 -> 1.2 deg.
- Controls launched (shell 1123d1ea): C0 = masked path with 0 teacher windows, seed 7 (GPU1, `lrsdiag_c0_masked_nowin_s7`;
  must reproduce ~Original 82); v2 re-collection lanes 0-1 on GPU0 (`rv_lrsg2_l*`, data/recovery/lrs_goal_v2), lanes
  2-3 after C0. C1 (v1 windows, unmasked loss) dropped: the masked loss is proven identical, so it adds nothing.
- Pre-registered for v2 (written BEFORE any v2 result): same rules as PLAN "LRS-G"; v2 is the primary LRS arm, v1 is
  reported as the servo ablation. Gate before SFT: v2 windows have |a_xyz| saturation < 2 % and rotation std > 0.01
  (visual check of 6 windows), else no SFT.
- 2026-11-25 03:33-03:39 UTC: the chain briefly launched the pre-registered "R1 re-matched to the R2 count" check
  (`v2r_r1m_s0`) because its count still read the STRICT survivors (5); killed after ~2 min (no checkpoint, run dir
  deleted), count now reads the D-V1 hard keep list (307 >= 208 -> no re-match needed). R2 s0 SFT launched 03:39
  (group: 307 eligible, 208 selected over 118 instances, 23.2k recovery frames).
- D_P progress 03:45 UTC: 7 / 56 source episodes, t_last in 40-160, 6 verified positives at 5 s^- (p(a-) mostly 0),
  5-12 min per episode -> D_P closes ~06:30 UTC.
- 2026-11-25 06:22 UTC: goal_g2_chain: G2-phys finished, auto table appended to RESULTS.
- 2026-11-25 07:15 UTC: C0 (masked path, 0 windows, s7) val 77 (Original 82 / 77 / 79): trainer / eval path clean.
  v2 collection done 06:27 (`runs/goal_rsi/lrs_goal_v2_collection.json`): accepted 136 / 168, judge precision 0.074
  [0.040, 0.130], solved in sim 0.060; dataset 16 paths / 20 windows / 1,645 anchors; gate PASS (sat 0.0 %, rot std
  0.020). Visual check v2 (6 windows): smooth, but gripper chatter (13 flips, ep17) and 2 / 6 near-static windows;
  grader inflation persists. SFT lrs2_s7 / s9 on GPU0, lrs2_s8 on GPU1 (chain rv_lrsg2_chain), ETA ~11:40 UTC.
- 2026-11-25 08:43 UTC: goal_g2_chain: G2 wm finished, auto table appended to RESULTS.
- 2026-11-25 08:43 UTC: goal_g2_chain: P2-3 finished, auto table appended to RESULTS.
- 2026-11-25 08:44 UTC: goal_g2_chain: G2-phys finished, auto table appended to RESULTS.
- 2026-11-25 08:40-09:05 UTC: visual check (full-fps MP4, every step at the bank's 10 fps, plus consecutive-frame
  strips; runs/_analysis/visual_check/goal_g2/{g2physns_keep, g2phys_rand_keep}).
  - **G2-physns** (no seam rule, 6 samples: 2 worst-jerk, worst seam, 3 mid):
    - Bridges are smooth and physically clean.
    - Continuations still carry the M0 behaviour the seam rule had removed. ep8 (cream cheese): 13 gripper flips
      (7.8 / 100 steps) in the continuation; consecutive frames t 72-94 show open / close pulses while the EE
      descends towards the cheese, with no grasp until ~t 120. Handoff jump 0.81.
    - ep31 / 36 / 42 (wine rack): smooth bridge -> handoff 0.39-0.45 -> short clean continuation.
    - Verdict: the filter is OK for the physics checks it applies. Continuation chatter remains, and G3-c20
      (cont_max 20) targets exactly this.
  - **Strict random survivors** (feed G3-phys-strict-r, 5 samples): mostly short, clean, near-goal bridges.
    - 5 / 28 are trivial (2-10 steps; ep16: 4 steps, the plate is already at the stove front, success at step 4).
    - ep2 (middle drawer, 281 steps): a slow, clean continuation with a 1-step gripper spike at t 195.
    - Verdict OK to distill. Caveat for G3-phys-strict-r: ~11 % of its trajectories are trivial completions with
      almost no action content (not filtered, pre-registration unchanged).
- **Incident 2 (2026-11-25 ~04:15 -> 08:35 UTC): chain idle with zombie claims.** Tasks launched by a chain process
  that was later restarted became `<defunct>` when they finished (nobody reaps them in the container), /proc/<pid>
  stayed, and both the stale-claim check and the slot accounting counted them as alive -> after the R1 SFTs (done
  ~04:15) and D_P (done ~06:30) no eval was launched; only f72a5ddd GPU1 kept training (R2 s0-s2 SFT, done 06:25).
  ~4 h x 3 GPUs lost. Fix: `alive()` reads /proc/<pid>/stat and treats Z / X as dead (both checks); chains restarted
  08:39 UTC. Also: up to 3 sim jobs per shell (9 workers each, least-loaded GPU) instead of 1 per GPU.
- D_P CLOSED (both shards): 56 train source episodes -> 52 with t_last (>= 2 / 8 pi0 successes) -> 48 s^- with a
  pi0 failure -> 42 with >= 1 verified pair -> 78 positive trajectories (n_P = 70 for R3), 1.63 M sim steps.
  dp_phys (aux targets for R4) failed 3x: rec_phys read `<bank>/sminus.json` but rec_prevent writes one per shard
  (`sh*/sminus.json`); fixed, re-queued.
- Visual check D_P (runs/_analysis/visual/recover_v2r_dp/, full-fps renderer, looked at 4 / 8 sheets 08:50 UTC): the
  "FAIL" banner is a renderer artefact (D_P meta has no success_step; only re-run-confirmed successes are written).
  Late s^- (t140-t160): bowl already held above the plate, a+ = 5-24 steps of lowering / release ending in success
  inside the bridge, while pi0's own a- fails there (p(a-) mostly 0): prevention concentrated at the place phase.
  Early s^- (t060-t080): 50-step reach / regrasp + pi0 continuation. Smoother than R1 (bridge d3 median 0.24 vs
  0.45, no > 1 flips). OK to distill. R3 SFT s0-s2 launched 08:43 (138 R2-selected + 70 D_P = 208, 21.6k frames).
- R0 event re-evals (E-v1s, nominal 500 episodes, replay ok 480-482): s0 SR 0.804, incidence 0.068, first-attempt
  0.784; s1 SR 0.770, incidence 0.088, first-attempt 0.746 (s2 queued). Registered R0 nominal stays 0.785.
- 2026-11-25 09:50 UTC: v2 val: lrs2_s7 40, lrs2_s8 29 (Original 82 / 77). Per task (s7): drawer 0, top drawer 1,
  cream cheese 0, wine rack 0, wine cabinet 1; other tasks 7-8. Val actions now in-distribution (saturation 0 %,
  rotation std 0.024-0.055 vs Original 0.06-0.08), so the v1 failure mode is gone, but nominal still drops 40-50 pt.
  M0 per-target FM loss on v2 windows 1.07 vs 0.131 on D0 (8.2x; v1 11.5x): still far off-manifold (the policy cannot
  predict the teacher's chunks) -> 4 % of anchors still carry ~1 / 4 of the gradient. Masked == pi0 loss on D0 (1e-8).
  lrs2_s9 started 09:25 (GPU0, ETA ~11:45 val), lrs2gt_s7 (13 GT-verified windows, 707 frames) queued on GPU1 after
  lrs2_s8's entry evals (ETA ~12:30 UTC).
- 2026-11-25 10:15 UTC: goal_g2_chain: G2-phys follow-ups finished, auto table appended to RESULTS.
- 2026-11-25 10:19 UTC: v2r chain (agent): stage R1 complete, auto table appended to RESULTS (reading pending).
- 2026-11-25 11:20 UTC: v2r chain (agent): stage R2 complete, auto table appended to RESULTS (reading pending).
- 2026-11-25 12:01 UTC: v2r chain (agent): stage R3 complete, auto table appended to RESULTS (reading pending).
