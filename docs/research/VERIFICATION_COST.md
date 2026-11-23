# Verification cost: real robot vs simulator vs world models

Written 2026-11-23 (literature check + CPU-only log reading; no GPU jobs, no code changes). Trigger: user challenge to
our sentence "sim verification is expensive, so the selector (WM + C^pi ranking) decides where to spend the sim
budget". User position: sim verification is cheap; REAL-world verification is what is expensive. Verdict below.

## TL;DR

- **User is right on the ordering.** Per verification: real robot (minutes of robot time + human reset/labeling,
  hours of human labor per few hundred trials) >> video world model (tens of s to minutes of H100/A100 per rollout)
  >~ our LIBERO sim with a VLA in the loop (~6-9 s wall per verification on 1 A100 + 8 CPU workers) >> latent WM +
  C^pi imagination (~0.2-1.5 ms per candidate, batched).
- **But "cheap" is relative.** In our sim one verification still costs ~10^4x a latent-WM candidate score
  (6 s vs 0.3 ms). That gap is the only honest justification for the selector in sim: it is an *efficiency* device
  (more verified successes per sim-hour), not a necessity. In sim we could afford to brute-force verify every
  candidate for a fair upper bound, and we should show that.
- **Recommended wording (section 5):** do NOT write "sim verification is expensive". Write that verification cost
  is the axis along which the selector matters; in sim it buys efficiency (measured x verified hits per sim-hour);
  it becomes essential when verification is real-robot or video-WM based, which is the deployment target.
- **Suggested experiment (section 6):** brute-force verify ALL (or a large fixed subset of) candidates in sim as an
  upper bound, then plot hits / solved vs number of verifications for selector vs random order; report sim-hours
  saved at matched solved rate.

## 1. References (each number checked on 2 sources: paper text + venue page / project page / README / second paper)

### 1a. Cost of REAL-robot evaluation / data verification

| claim | source | venue | quote / number | 2nd source | link |
|---|---|---|---|---|---|
| Real eval of a VLA costs > 100 h of human labor | AutoEval (Zhou, Atreya, ... Levine) | CoRL 2025 (PMLR v305) | "a comprehensive evaluation of ... OpenVLA ... required more than 2,500 rollouts across four robot setups and three institutions, and a total of more than 100 hours of human labor for resetting scenes, rolling out policies, and recording success rates" (=> ~2.4 min human labor / rollout) | Berkeley tech report EECS-2025-42 (same text) | [2503.24278](https://arxiv.org/abs/2503.24278) |
| Automated real eval throughput: ~850 episodes / 24 h per cell, 3 human interventions / 24 h; same trials by hand ~16 h of human time; >99 % human-time saving | AutoEval | CoRL 2025 | Fig. 9: "24 hour AutoEval evaluation run with ~850 total evaluation episodes"; "compared to ~16 hours if a human evaluator wanted to run the same number of trials by hand" (=> ~100 s robot wall-clock per episode, ~68 s human time per trial if manual) | project page auto-eval.github.io | [2503.24278](https://arxiv.org/abs/2503.24278) |
| Real eval is "expensive and inefficient", not scalable or reproducible; sim lets sweeps run "at negligible added cost" | SIMPLER (Li, Hsu, Gu, Pertsch, ...) | CoRL 2024 | "real-world evaluation is expensive and inefficient"; sim "can reduce the need for costly real-world evaluations ... at negligible added cost"; sim runs "1.5k simulation steps per second on a consumer NVIDIA 4090 GPU at 640x512" | Veo report and SureSim cite SIMPLER for the same point | [2405.05941](https://arxiv.org/abs/2405.05941) |
| Real eval needs distributed labs: 612 pairwise A/B episodes, 4,284 rollouts over 7 institutions | RoboArena (Atreya et al.) | CoRL 2025 (PMLR v305) | "612 pairwise comparisons ... 4284 evaluation episodes" | proceedings.mlr.press/v305/atreya25a | [2506.18123](https://arxiv.org/abs/2506.18123) |
| Real eval is expensive: limited parallelism, manual logging, careful resets; sim + prediction-powered inference saves 20-25 % of hardware trials | SureSim (Badithela et al.) | ICRA 2026 | "Real-world evaluation is expensive because it offers limited parallelism while often requiring manual logging of outcomes and careful resetting of environments"; "saves over 20-25% of hardware evaluation effort" | ICRA DOI 10.1109/icra57385.2026.11697329; project page | [2510.04354](https://arxiv.org/abs/2510.04354) |
| Every real trial needs a manual reset; task switches need re-arranging scenes; active selection halves estimation error at equal cost | Anwar et al., active experiment selection | CoRL 2025 (PMLR v305) | "each trial requires a manual environment reset, and each task change involves re-arranging objects or even changing robots"; OpenVLA L1 error 0.017 vs 0.036 (random) at cost 500 | OpenReview page | [2502.09829](https://arxiv.org/abs/2502.09829) |
| Statistically meaningful real eval needs "hundreds of thousands of expensive real-world evaluation trials"; Veo validated with 1,600+ real trials | Gemini Robotics Team, Veo world simulator | arXiv tech report 2025 | quote verbatim from Sec. 1; "1600+ real-world evaluations of eight Gemini Robotics policy checkpoints and five tasks" | project page veo-robotics.github.io | [2512.10675](https://arxiv.org/abs/2512.10675) |
| Large VLA papers use only small real budgets: OpenVLA 170 Bridge rollouts (17 tasks x 10) + 60 Google-robot rollouts per method; pi0 10 episodes per task and method | OpenVLA (Kim et al.); pi0 (Black et al.) | CoRL 2024; RSS 2025 | "we evaluated each method in 170 rollouts (17 tasks with 10 trials each) ... and 60 rollouts (12 tasks with 5 trials each)"; pi0: "normalized score averaged over 10 episodes per task and method" | AutoEval quotes the OpenVLA effort (above) | [2406.09246](https://arxiv.org/abs/2406.09246), [2410.24164](https://arxiv.org/abs/2410.24164) |
| Real data collection: 76k trajectories = 350 h of interaction, 50 collectors, 12 months; "substantial investments in hardware and human labour", "logistical and safety challenges" | DROID (Khazatsky et al.) | RSS 2024 | abstract | Edinburgh research portal + UT RPL page (same numbers, DOI 10.15607/RSS.2024.XX.120) | [2403.12945](https://arxiv.org/abs/2403.12945) |
| Real-robot RL "requiring millions of interactions that are impractical, unsafe, and time-consuming"; "VLA tasks require repeated rollouts in the physical world, which incur high hardware costs, safety concerns, and limited scalability" | WMPO (Zhu et al.) | ICLR 2026 | quotes from Sec. 1 | ICLR proceedings PDF | [2511.09515](https://arxiv.org/abs/2511.09515) |
| Evaluating / improving generalist policies "requires a large number of real-world rollouts"; both processes "slow, costly, and difficult to scale" | Ctrl-World (Guo, Shi, Chen, Finn) | ICLR 2026 | abstract / Sec. 1 | ICLR proceedings PDF | [2510.10125](https://arxiv.org/abs/2510.10125) |
| "real-world evaluation can sometimes take days to complete", WM rollouts "under an hour on a single GPU" | WorldGym (Quevedo et al.) | ICLR 2026 | Sec. 4 | ICLR proceedings PDF | [2506.00613](https://arxiv.org/abs/2506.00613) |

### 1b. Cost of SIMULATION rollouts with a VLA in the loop, and of world-model rollouts

| claim | source | venue | quote / number | 2nd source | link |
|---|---|---|---|---|---|
| VLA RL in LIBERO sim still costs GPU-days: 8x A800-80GB, LIBERO RL "approximately 1 or 2 days" (one-traj SFT), ~1 day (full-traj); max 512 env steps / episode, 8 samples / prompt | SimpleVLA-RL (Li et al.) | ICLR 2026 | Appendix, training time | ICLR proceedings PDF (same text) | [2509.09674](https://arxiv.org/abs/2509.09674) |
| For CPU simulators like LIBERO "CPU cores often become the primary bottleneck"; "the number of physical CPU cores per node becomes the upper bound for efficient rollout"; rollout time dominates; on LIBERO the colocated mode is 1.34x-2.27x faster than SimpleVLA-RL | RLinf-VLA (Zang et al.) | arXiv 2025 | Sec. 3 / 5.2 | pi_RL builds on RLinf with env + rollout + actor co-located on the same GPU, run serially | [2510.06710](https://arxiv.org/abs/2510.06710) |
| pi0 / pi0.5 RL in parallel sim (LIBERO, ManiSkill 320 parallel envs) on 8x H100 80GB | pi_RL (Chen et al.) | arXiv 2025 | Sec. 5.1 / App. B.2 | RLinf docs page (placement configs) | [2510.25889](https://arxiv.org/abs/2510.25889) |
| PPO on OpenVLA in ManiSkill: "about 42 hours on a single NVIDIA A100 GPU to converge" | RL4VLA (Liu et al.) | NeurIPS 2025 | Sec. 4.2 | project page rlvla.github.io | [2505.19789](https://arxiv.org/abs/2505.19789) |
| OpenVLA-7B RL on LIBERO: "after only 48 GPU hours of rl training ..."; 1 GPU dedicated to vLLM inference for rollouts | VLA-RL (Lu et al.) | arXiv 2025 | Sec. 4.1 / 3.5 | papers-with-code summary (same numbers) | [2505.18719](https://arxiv.org/abs/2505.18719) |
| Sim physics alone is cheap: SIMPLER 1.5k sim steps / s on a 4090 at 640x512; ManiSkill3 up to 30k+ FPS GPU-parallel | SIMPLER; ManiSkill3 (Tao et al.) | CoRL 2024; RSS 2025 | numbers from papers | ManiSkill3 abstract | [2405.05941](https://arxiv.org/abs/2405.05941), [2410.00425](https://arxiv.org/abs/2410.00425) |
| Video WM, policy in the loop: Ctrl-World "One interaction step takes around ~10s on A100 or ~5s on H100" (1 step = 1-s, 15-action chunk; 3 views 192x320) | Ctrl-World | ICLR 2026 | official README tip | PrimeSim benchmark table: Ctrl-World 0.87 rollout frames / s on one A100-80GB (5 frames / chunk -> ~5.7 s / chunk), same order | [README](https://github.com/Robert-gyj/Ctrl-World), [PrimeSim](https://primebottech.github.io/PrimeSim-website/) |
| Video WM: WorldGym 40-frame rollout on one A100 = 93 s (h=1) or 33 s (h=4) | WorldGym | ICLR 2026 | App. F.2, Table 9 | ICLR proceedings PDF (same table) | [2506.00613](https://arxiv.org/abs/2506.00613) |
| Video foundation WM: Cosmos-Predict2 Video2World, one 5-s 480p/16fps clip = 79.9 s (2B) / 286.5 s (14B) on H100 PCIe; 720p/16fps 2B 228.8 s on H100 SXM | NVIDIA Cosmos-Predict2 | official docs / HF model card | performance tables | HF card Cosmos-Predict2-14B-Video2World (same tables) | [perf docs](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/documentations/performance.md) |
| Video-WM RL motivated by interaction cost: WMPO replaces "costly real-world rollouts with model-generated trajectories"; WM training + policy optimisation on 32 H100 | WMPO | ICLR 2026 | Sec. 1 / impl. details | ICLR proceedings PDF | [2511.09515](https://arxiv.org/abs/2511.09515) |
| Veo / WorldEval do not publish per-rollout latency; success is still judged by humans (Veo) or Gemini-2.0 (WorldEval) | Veo report; WorldEval (Li et al.) | arXiv 2025 | Veo: "Each episode consists of an 8-second rollout, which is scored ... by human evaluators"; WorldEval: WAN 2.1 14B I2V + LoRA | emergentmind summary of Veo ("human assessors are currently used") | [2512.10675](https://arxiv.org/abs/2512.10675), [2505.19017](https://arxiv.org/abs/2505.19017) |

### 1c. Ranking / pre-filtering candidates before an expensive verifier

| claim | source | venue | quote / number | 2nd source | link |
|---|---|---|---|---|---|
| Sample K actions from a generalist policy, re-rank with a learned Q; overhead 1.28x (K=10) to 1.59x (K=50) of Octo-small inference | V-GPS (Nakamoto, Mees, Kumar, Levine) | CoRL 2024 (PMLR v270) | Sec. 5.2, Table 9 | project page | [2410.13816](https://arxiv.org/abs/2410.13816) |
| Test-time best-of-N with a learned action verifier; "sample and verify 16 candidate actions in approximately 650 ms" on one H100; verifier 46 actions / s at batch 16 | RoboMonkey (Kwok et al.) | CoRL 2025 | Sec. 5.4, App. G Table 3 | project page robomonkey-vla.github.io | [2506.17811](https://arxiv.org/abs/2506.17811) |
| Latent WM imagines outcomes of K candidate plans, a VLM verifier filters them (100 plans -> 6 after NMS) | FOREWARN (Wu, Tian, Swamy, Bajcsy) | RSS 2025 | Sec. IV | RSS proceedings p076 | [2502.01828](https://arxiv.org/abs/2502.01828) |
| Allocate an expensive real-eval budget actively (cost-aware EIG) instead of uniformly | Anwar et al. | CoRL 2025 | ~50 % lower error at equal cost | OpenReview | [2502.09829](https://arxiv.org/abs/2502.09829) |
| Combine many cheap (sim) predictions with few expensive (real) trials, keep valid confidence intervals | SureSim | ICRA 2026 | 20-25 % hardware trials saved | project page | [2510.04354](https://arxiv.org/abs/2510.04354) |
| Rank policies in a cheaper proxy (video WM) instead of real trials | Ctrl-World; Veo; WorldGym; WorldEval | ICLR 2026 / arXiv | Veo: Pearson 0.86, MMRV 0.06 on generalization axes, validated against 1,600+ real trials | project pages | see 1a / 1b |

Gap we can claim: these works prefilter *actions* at test time (V-GPS, RoboMonkey, FOREWARN) or allocate *policy-eval*
trials (Anwar, SureSim). None ranks *recovery-segment candidates from a failure state* by a WM + policy-conditioned
critic to decide which ones get verified (sim or real) for data generation. That is our niche; it is still a
"cheap proxy before expensive verifier" argument and should be framed with these as the prior art.

## 2. Our own sim verification cost (read from logs, CPU only)

Setup: every job holds ONE A100-SXM4-80GB; the VLA runs batched on the GPU in the main process, MuJoCo + EGL
render runs in `eval.n_workers` CPU worker processes (`retryvla/eval/rollout.py::run_jobs`: one batched
`predict_chunk` for all workers that need a chunk, then an open-loop `step_seq`). So 1 job = 1 GPU-hour per wall-hour.
A "verification" = restore the failure snapshot, execute the candidate recovery segment (50-step bridge), then let
the policy continue to success / time-out (<= 520 steps on LIBERO-Recover, <= 300 on Goal). Numbers are from
`runs/<run>/summary.json` (wall_time_s), `per_instance.jsonl` (order_success, sim_steps_*, t_arm_s, n_scored) and
`metrics.jsonl` (per-instance timestamps). Wall time includes model load, WM imagination and logging.

| run (policy, res, replan, workers) | instances x arms x verifs | wall | s / verif (amortised) | verifs / GPU-h | sim steps / verif | median s / instance |
|---|---|---|---|---|---|---|
| M5b compare_arms s0-7 (SmolVLA, 256 px, n_act 10, 14 workers) | 256 x 4 x 16 = 16,384 | 21.3 GPU-h total (8 jobs) | 4.1-5.2 | 690-870 (mean 771) | n/a (not logged) | 276 (64 verifs) |
| RQ3 xwm_sweep s0 (SmolVLA, 256 px, n_act 10, 6 workers, V-JEPA scoring in-loop) | 30 x 5 x 8 = 1,200 | 3.09 h | 9.3 | 388 | 316 | 373 (40 verifs) |
| RGR rgr_sweep s0-3 (SmolVLA, 256 px, n_act 10, 8 workers) | 127 x 6 x 5.3-8 = 4,980 | 8.6 GPU-h total | 5.9-6.4 | 560-615 | 339-371 (verif + exec) | 243 (~39 verifs) |
| GOAL goal_search s0-2 (pi0 3B, 128 px, replan 5, 8 workers) | 168 x 2 x 16 = 5,376 | 9.45 GPU-h (3 GPUs, 3 h 23 m wall) | 5.9-6.8 | 529-609 (mean 569) | 274-327 | 198 (32 verifs) |

Single-rollout latency (not amortised) is much longer: `[verify]` log lines show the first of 40 verifications
finishing after ~30 s and a 40-verification batch taking ~150-200 s with 6 workers; a failing continuation runs the
full 520 steps. Aggregate throughput is only ~45-60 sim steps / s per job (e.g. GOAL s0: 491,669 steps / 10,591 s), far
below raw MuJoCo / SIMPLER physics (1.5k steps / s), i.e. the cost is VLA inference (~60 pi0 calls per Goal
verification) + CPU rendering, matching the RLinf finding that LIBERO rollouts are CPU / inference bound.

Latent-WM imagination cost per candidate (same runs, `t_arm_s / n_scored`, batched on the same GPU):

| scorer | candidates / instance | time / instance | ms / candidate | candidates / GPU-h |
|---|---|---|---|---|
| WM v1 (DINOv2 latent) + C^pi, `wm_diverse` / `single` | 696 | 0.15-0.21 s | 0.2-0.3 | ~1.2-1.8 x 10^7 |
| WM v1 + C^pi with bootstrap heads / MC dropout (xwm `wm_unc_l0`) | 696 | 1.03 s | 1.5 | ~2.4 x 10^6 |
| V-JEPA 2-AC + C^pi_VJ (xwm `vj_unc_l0`) | 696 | 21.4 s | 31 | ~1.2 x 10^5 |
| RGR recursive chains (`rec`, `rec_geo`) | 700-750 | 7-9 s | 9-13 | ~3-4 x 10^5 |

**Cost ratio, one sim verification vs one latent-WM candidate score: ~6 s / 0.25 ms ~ 2 x 10^4** (vs V-JEPA: ~200x).
So in sim the selector is cheap relative to what it rations, but sim verification itself is also cheap in absolute
terms: ~570 verifications per A100-hour with pi0 3B.

### 2b. Per-verification cost, all regimes side by side (order-of-magnitude, best public numbers)

| verifier | per verification | throughput per unit | extra cost | sources |
|---|---|---|---|---|
| latent WM + C^pi (ours) | 0.2-1.5 ms GPU (batched) | ~10^6-10^7 / GPU-h | none | section 2 |
| LIBERO sim + VLA (ours) | ~6 s amortised (30-90 s latency) | ~400-800 / GPU-h + 8-14 CPU cores | none; exact snapshot restore | section 2 |
| video WM + VLA, 15-s horizon (= our ~300 steps at 20 Hz) | Ctrl-World ~15 chunks x 5-10 s = 75-150 s; WorldGym 33-93 s per 40 frames; Cosmos-2B 80 s per 5-s clip (H100) | ~25-50 / GPU-h | success label by human (Veo, Ctrl-World) or VLM (WorldEval, WorldGym) | 1b |
| real robot, automated (AutoEval) | ~100 s robot time | ~35 / robot-h (~850 / 24 h) | reset policy + success classifier per task (1-3 h human setup); 3 interventions / day | 1a |
| real robot, manual | minutes of robot time + ~1-2.4 min human labor | tens / human-h | human resets, labeling, safety, hardware wear | 1a |

Two caveats the paper must state:
1. Sim with a VLA is NOT free: one verification costs ~10^4x a latent-WM score. But it is only ~12-25x cheaper
   than a video-WM rollout of the same horizon (~6 GPU-s vs Ctrl-World 75-150 GPU-s for 15 s), not 1000x, and
   ~2-3 orders of magnitude cheaper than a real trial once human time is counted.
2. For RECOVERY verification the real-world cost is worse than for policy evaluation: AutoEval-style reset policies
   return the scene to the *initial* distribution; verifying a recovery candidate needs the *same mid-episode
   failure state* re-created for every candidate (see `docs/research/CUT_POINT_STUDY.md`), which only sim (exact
   snapshot) or a WM (from the logged observation) can do. This is the strongest argument for the selector.

## 3. What the selector buys today, at a fixed budget of 16 verifications (paired, same pool and seeds)

| benchmark (policy) | arm | solved | hits / verification | p@1 | mean verifs to first hit (solved only) |
|---|---|---|---|---|---|
| LIBERO-Recover M5b, 256 inst (SmolVLA) | random | 0.363 | 0.062 | - | - |
| | wm_diverse (WM + C^pi) | 0.547 | 0.171 (2.8x) | - | - |
| LIBERO-Goal, 168 entries (pi0 3B) | random | 0.571 | 0.163 | 0.185 | 4.47 |
| | wm_diverse | 0.589 | 0.243 (1.49x) | 0.286 | 3.57 |

Reading: on Goal the selector does NOT raise reach at budget 16 (solved +0.018), it raises yield per verification
(1.49x hits / sim, p@1 0.19 -> 0.29). On LIBERO-Recover it raises both, but with 16 random verifications on a
256-candidate pool, random under-samples; whether wm_diverse beats random at a LARGE budget is unknown. In
hits / sim-hour (569 verifs / GPU-h on Goal): ~93 -> ~138 verified successful recoveries per A100-hour.

## 4. Verdict on the claim

- "Sim verification is expensive" is **not defensible** as written. ~570 pi0-3B verifications / A100-hour is cheap
  relative to every alternative except latent imagination, and it parallelises with CPU cores.
- "Verification is the expensive step relative to proposing and scoring candidates" **is** defensible: 2 x 10^4 x
  latent scoring in sim, and real-robot / video-WM verification is 1-3 orders of magnitude worse again.
- Therefore the selector's role differs by regime: sim = efficiency (and a testbed where the oracle upper bound is
  affordable); real / video WM = necessity (budget of tens of verifications per failure state at most).

## 5. Recommended wording for the paper

> Verification, i.e. executing a candidate recovery and checking task success, is the dominant cost of recovery
> data generation, but how dominant depends on the verifier. In simulation it is cheap in absolute terms (about
> 570 verifications per A100-hour with a 3B VLA in the loop on LIBERO-Goal), yet still ~10^4 times more expensive
> than scoring a candidate with the latent world model and policy-conditioned critic. On a real robot each
> verification additionally requires re-creating the same mid-episode failure state and a human or learned reset
> and success check; real-world policy evaluation already costs on the order of 100 h of human labor for a few
> thousand trials (AutoEval, CoRL 2025; SureSim, ICRA 2026), and action-conditioned video world models cost tens of
> GPU-seconds per second of rollout (Ctrl-World, WorldGym, ICLR 2026). We therefore treat the selector as a
> budget-allocation device: in simulation it buys efficiency (1.5-2.8x verified successes per simulator hour at a
> 16-verification budget), and we quantify that against an exhaustive-verification upper bound; it becomes
> essential when verification is a real robot or a video world model, which is the setting that motivates it.

Do NOT say: "sim verification is expensive"; "the selector is necessary in sim"; any claim about reach (solved)
improvement on Goal at budget 16 (not significant).

## 6. Suggested experiment: exhaustive sim verification as an upper bound + efficiency curve

Goal: turn "selector buys efficiency" into a measured curve, and get the oracle reach that no ranking can exceed.

- Protocol (train / pilot only, never test). Take the existing pools (pool_size 256 per entry, same proposals RNG
  and continuation seeds as goal_search / M5b). Verify EVERY candidate once in sim (`budget = pool_size`, arm
  `all`), store per-candidate success. Then every budgeted arm is an offline re-ordering of the same bank: for
  k in {1, 2, 4, 8, 16, 32, 64, 128, 256} compute solved@k and hits@k for (a) selector order (WM + C^pi, J),
  (b) random order (mean over 100 permutations), (c) oracle order (all successes first) = upper bound,
  (d) diversity-only order. Paired by construction; McNemar on solved@k and bootstrap CIs over entries.
- Report: verifications (and sim-hours) needed by random to match the selector's solved@16 and hits@16;
  fraction of the exhaustive union reach achieved at k = 16; AUC of solved@k.
- Noise ceiling: re-verify the successful candidates and a random 10 % of failures with 2 more continuation seeds
  to estimate per-candidate success probability (separates selector error from VLA continuation stochasticity).
- Cost from section 2 (~6 s / verification / A100 with 8 workers): Goal 168 x 256 = 43,008 verifs ~ 72 GPU-h
  (~1 day on 3 GPUs); pilot version 56 entries (T/4 cut) x 256 = 14,336 verifs ~ 24 GPU-h; LIBERO-Recover 60 train
  instances x 256 ~ 26 GPU-h. Cheap enough to run; this is itself evidence that sim verification is cheap.
- Pre-registered reading: if selector solved@16 >= 0.9 x union reach and random needs >= 2x the verifications to
  match, the "efficiency" claim stands; if random@16 already matches selector@16 on solved (as on Goal), claim only
  yield (hits / sim) and lead with the real / video-WM motivation.
- Optional cost-transfer table: re-express the curve with per-verification cost from section 2b (real ~100 s robot
  + human reset; Ctrl-World ~100 GPU-s) to show the budget at which the selector stops being optional.

## 7. Limitations of this note

- Video-WM latencies come from READMEs / appendices on different GPUs and resolutions; Veo and WorldEval publish
  no per-rollout latency. Treat 2b as order-of-magnitude.
- Our s / verification are amortised wall times of whole jobs (include model load, WM scoring, logging) on shared
  shells with variable CPU load (xwm run was throttled by the load back-off, hence 9.3 s).
- M5b runs did not log sim steps per verification; steps are from xwm / RGR / Goal.
