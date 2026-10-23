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
