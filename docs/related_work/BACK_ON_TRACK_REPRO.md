# Back on Track (arXiv 2610.19579) and its liveness value (arXiv 2605.11479): extraction + RetryVLA re-implementation

Written 2026-11-23, before any BoT result. Both papers were read in full from the arXiv HTML (2610.19579v1,
2605.11479) plus the project page https://policy-improvement-wm.pages.dev/ (no code released as of 2026-11-23; the
project page is "Anonymous Authors"). Pre-registration: `experiments/PLAN.md` "Back on Track + ForkCast-style
test-time arms (2026-11-23)". Code: `retryvla/baselines/bot/`, `retryvla/policy/bot.py`.

## 1. What the papers do (exact extraction)

| aspect | Back on Track (2610.19579) | source |
|---|---|---|
| setting | black-box nominal policy pi_b, improved at deployment by an external monitor + fallback planner; no fine-tuning | Sec. III |
| state | s = [psi(o), q]; psi = SigLIP2 image embedding, q = proprio | Sec. IV-A |
| value | policy-conditioned discounted liveness value of 2605.11479: V(s) = (1 - gamma) + gamma min{l(s), V(s')}, l = -1 at a goal state, +1 otherwise; l built from episode outcomes (final state of a successful episode = goal). V in [-1, 1], low = close to completion under pi_b | Eq. 1 |
| value training data | 200 rollouts of pi_b per task under the test perturbation distribution, capped at 300 steps, success / timeout labels from the simulator | Sec. V-A |
| value architecture / training (2605.11479 Appx. C, LIBERO-Spatial) | MLP 5 layers x 512, GELU, LayerNorm, input 1544 (SigLIP2 + proprio), lr 1e-5, batch 512, 5,000 epochs, gamma 0.993, prioritized replay (capacity 10k, alpha 0.6, beta 0.4 -> 1, 2 gradient steps / batch); two-stage algorithm: anchor values on successful episodes (1 - 2 gamma^k), then bootstrapped target l_bar = V_bar and the Bellman operator again over all data (reduces truncation bias) | 2605.11479 Sec. V-C, Alg. 1, Tab. I |
| monitor | ensemble of 5 value nets (mean), temporal smoothing W = 3 (Eq. 2); trigger iff V_tilde > eta_t, eta_t = 1 / (1 + exp((t - 60) / 10)), t = frame index, terminal threshold 0; same schedule for all tasks | Sec. IV-A |
| monitor ablations | no schedule -8.5 pt, 1-member -8.3, 2-member -4.5, no smoothing -4.3 (4-task subset, n = 100) | Tab. IV |
| WM | 29.6 M-param 4-layer ViT over [psi(o), q] (DINO-WM-style feature prediction), s_{t+1} = phi(s_t, a_t), trained on the 200 rollouts / task + 53 min of LIBERO-Goal teleop demos (play data, pooled over 10 tasks), autoregressive curriculum kappa 1 -> 16 | Sec. IV-B1, V-A |
| planner | CEM minimising J = sum_{i=1..H} V_bar(s_hat_i) (Eq. 3); proposal centred on the nominal chunk (pi_b queried ONCE per intervention); perturbation delta = s V_k Lambda_k^1/2 eps in the PCA basis of training action chunks (k explaining 98 % variance; k = 13 for H = 10, D = 6 on pi0.5 LIBERO), s = 0.6; gripper unperturbed (= nominal) | Sec. IV-B2/3 |
| planner size | 1,280 ten-step plans per cycle in ~108 ms (project page); "a batch of 256 rollouts takes 44.2 ms"; i.e. 256 samples x 5 iterations (our reading; #elites not stated) | V-F, project page |
| hand-back | execute the first h <= H actions (h = 5; h = 2 costs -1.3 pt), return control to the monitor, which re-invokes the planner if pi_b has still not recovered | IV-B3, Tab. IV |
| compute | 108 ± 3 ms planning + 180 ms policy query (hardware, RTX 5090); 27 ms / control step amortised, 13.9 Hz; 0 overhead when the monitor is inactive | V-F |
| benchmark | LIBERO-Goal, pi0.5 LIBERO checkpoint (98 % nominal) on a PERTURBED start distribution (joints 4/5 ± 0.1 chi rad, wrist roll ± 0.2 pi chi, objects ± Upsilon m; per-task chi, Upsilon in Tab. I), 300 paired episodes / task, pi0.5 sampling seed fixed across methods | V-A |
| numbers | pi0.5 72.6 -> 81.5 (+8.9; rescued 369 / broken 102 of 3,000); monitor fires on 94 % of would-fail and 16 % of would-succeed trials; 4-task subset 65.2 -> 80.9 vs TD3+BC 39.6, IQL 42.2, LatentSafe 59.2, Filtered-BC (white-box, full 3.3 B FT) 78.2; filter on top of Filtered-BC 78.2 -> 79.8; MC-D value instead of liveness -13.1; iCEM noise -4.7; no play data (t0) still +14.7; hardware DP lunchbox 40 -> 60 | Tab. I-V |
| stated limitations | false triggers break successes; WM errors compound and are exploited by the planner (wine bottle t2 -1.0); needs labelled rollouts + play data | Sec. VI |

## 2. Our re-implementation and every deviation

| # | item | theirs | ours | why / expected effect |
|---|---|---|---|---|
| D1 | benchmark | perturbed-start LIBERO-Goal, pi0.5 | (a) **2610.05151 Goal protocol** (nominal demo inits, 100 val scenes, pi0 3B M0, seed 7, 300 steps, replan 5); (b) LIBERO-Recover Spatial / Object (failure snapshots, 520 steps), SmolVLA | the comparison the user asked for; much less headroom on (a) (M0 85 / 100) |
| D2 | state encoder | SigLIP2 global embedding (+ proprio) | our WM latent: DINOv2-small 6x6 patch tokens per camera (normalized) + proprio; the value sees them average-pooled to 2x2 per camera (8 x 384) + proprio + task one-hot (3,096 dims) | the value must score our WM's imagined states |
| D3 | value input / per-task | one value per task (likely) | one value over all tasks with a task one-hot | 10 tasks, ~90 rollouts each |
| D4 | value training | lr 1e-5, 5,000 epochs, PER, 2 grad steps / batch | lr 3e-4 AdamW, 8k regression steps (stage 1) + 20k fitted-Bellman steps with target net (sync 500) and PER (alpha 0.6, beta 0.4 -> 1), batch 512, 5 members | same algorithm, fewer passes (compute); convergence checked by the 2605.11479 success / failure metrics on held-out scenes |
| D5 | frames | every control step | latents every 5 env steps (= our WM chunk); one frame transition = 5 Bellman steps with l held constant: V(s_f) = 1 - gamma^5 (1 - min{l_bar(s_f), V(s_{f+1})}) | storage / encoding cost |
| D6 | rollout data | 200 / task under the test distribution | Goal: 900 M0 rollouts (300 train inits x policy seeds 8, 9, 10; ~90 / task); Spatial: 4 trials x 112 train scenes (448) per policy | budget; train split only, val / test never used |
| D7 | WM | 29.6 M ViT on rollouts + play data, curriculum to 16 steps | **our existing latent WM** (6-layer transformer over DINO tokens, 5-step action chunks, 4-head ensemble; Goal WM trained on 168 failure entries x 32 candidates of M0; Spatial WM v1). Not retrained on the BoT rollouts (the user asked for our own WM). | same WM as the RetryVLA generator -> the test-time vs distillation comparison holds the WM fixed. Likely weaker on nominal mid-task states than theirs (trained on failure-state bridges) |
| D8 | PCA basis | rollout chunks + play data | rollout chunks (executed actions) + our WM's own training actions (Goal: `goal_collect`, Spatial: `collect_v1`) as play data; k = 98 % variance, s = 0.6, gripper = nominal | same role as their play data |
| D9 | H / h | H = 10, h = 5 | same (H = 10 = 2 WM steps, h = 5); J sums V over the 2 imagined WM states (not over 10 per-step states) | WM granularity |
| D10 | CEM elites | not stated | 32 of 256, refit mean / std in PCA-coefficient space, std floor 0.05, best-ever plan kept, the unperturbed nominal chunk is sample 0 of iteration 0 | standard choices |
| D11 | monitor cadence | every frame | at every policy query (chunk boundary: every 5 env steps on Goal, 10 on Spatial; every 5 after a planned segment); W = 3 over those | our rollout engine observes only at chunk boundaries |
| D12 | threshold schedule | t_c 60, sigma 10, terminal 0, frame index | same, t = env steps since episode start | Spatial / Object start at a failure state: the 60-step grace period delays interventions there (kept, faithful) |
| D13 | C^pi variant | - | V = 1 - 2 C^pi(s) with our critic (same threshold, same objective); Goal C^pi has no training data for 3 tasks without audit failures (`open_middle_drawer`, `bowl_on_stove`, `bowl_on_cabinet`) -> V := -1 there (never triggers) | labelled variant 'BoT (C^pi)' |
| D14 | other policies + BoT | filter on Filtered-BC reuses pi_b's value | Original / RetryVLA policies + BoT reuse M0's value (Goal); on Spatial pi1 gets its own liveness from its own rollouts (policy-conditioned, faithful) | |
| D15 | latency hardware | RTX 4090 / 5090 | A100 80 GB, batched across 8-10 sim workers; we report per-episode GPU wall time (VLA / monitor / planner) and batch-1 latency | |

## 3. ForkCast-style selector (2610.18650), shares the same code
Commit (the policy's chunk, open loop) + 4 library primitives (retreat20, regrasp, reposition_0, reposition_180;
"Replan" analogue; no Reobserve since LIBERO cameras are fixed), imagined to a common 50-step horizon with the same WM,
scored by (a) a ForkCast progress head (mean-pooled tokens per camera + proprio + task, LayerNorm, 2-layer MLP; label
t / T on successful M0 rollouts, because their privileged object-to-goal distance is not defined for most LIBERO-Goal
predicates) or (b) C^pi; argmax with Commit winning ties; first decision at step 60 (theirs 64); a chosen primitive
runs to its last active step, then control returns to the policy. No detector, no threshold (as ForkCast).
V-JEPA 2-AC: not used on Goal (our adapter is trained on Spatial only).

## 4. Fairness note
- Same base policies, same val scenes, same pi0 seeds; BoT queries pi0 with the same seed stream, so episodes in
  which the monitor never fires reproduce the base trajectory (up to GPU numeric nondeterminism) and differences come
  from interventions (their paired design). We report rescued / broken counts.
- Same WM and the same C^pi as the RetryVLA generator, so "test-time planning vs distillation" holds the model fixed.
  This is not BoT's own WM (trained on rollouts + play data). If BoT underperforms, a weaker WM on nominal mid-task
  states is the first suspect (deviation D7), and we say so.
- No hyper-parameter was tuned on val / test; all BoT constants are the paper's (D10 fills unstated ones).
- Their gain is on a perturbed-start distribution; on the 2610.05151 protocol (M0 at 85 / 100) the ceiling is +15.
- Inference cost is measured on the same GPU in the same run (plain VLA time vs monitor + planner time).
