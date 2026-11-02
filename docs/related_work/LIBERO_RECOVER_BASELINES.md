# LIBERO-Recover: who reports results, runnable baselines, protocol check (2026-11-02)

Scope: methods with simulation results on LIBERO-Recover ([arXiv 2609.05178](https://arxiv.org/abs/2609.05178),
[code](https://github.com/liulin815/LIBERO-Recovery), [project](https://liulin815.github.io/LIBERO-Recovery/)),
papers that cite it (Sep-Nov 2026), closely related sim recovery methods we could run as baselines, and how our
protocol / numbers line up. No GPU jobs were run for this note. Sources: the arXiv PDFs (text extracted locally),
arXiv API, GitHub API (repo + issues), Hugging Face / ModelScope metadata. Semantic Scholar's citation API was
rate-limited (HTTP 429) every time, so the citing-paper list comes from arXiv keyword search + full-text grep of
~20 candidate PDFs + web search. It may miss papers that cite LIBERO-Recover without using recovery/LIBERO/VLA
words in the abstract.

## TL;DR
- **Only the LIBERO-Recover paper reports LIBERO-Recover numbers.** None of the citing papers we found (Kintsugi-VLA,
  FIT-MAC) and none of the related recovery papers (Backtrack, CoRe, RESETTLE, SafeLoop, B2FF, RoboRecover, ...)
  evaluate on it. There is no published SmolVLA number and no published method that *improves* RSR besides the
  paper's own "Combined" (human teleop recovery data) and "temporal" variants. So our RetryVLA rows are the first
  method results on this benchmark that we know of.
- **The paper's baselines are public.** The StarVLA checkpoints the paper evaluates (`StarVLA/Qwen3-VL-PI-LIBERO-4in1`,
  `Qwen3-VL-OFT-LIBERO-4in1`, `Qwen2.5-VL-GR00T/FAST-LIBERO-4in1`, `WM4A-*`) are on Hugging Face. The human recovery
  data (`ataier/LIBERO_Recovery_Expert`, 3,184 episodes, LeRobot v2.1) is on ModelScope. It contains about 1,185
  Spatial-task and about 1,150 Object-task episodes, so the paper's "Combined" baseline can be reproduced on SmolVLA.
- **Protocol mismatch to keep in mind:** the paper evaluates 2,178 scenarios (Spatial 196, Object 321). The released
  assets have 880 (Spatial 168, Object 77). We evaluate a 56-scene Spatial / 26-scene Object *test subset*. Our
  test scenes have no L4 and only 1 (Spatial) or 4 (Object) L1 scenes. Suite-level RSR is comparable in kind but
  not identical; per-level numbers are not comparable.
- **Recommended baselines** (details in section 5): (1) SFT on the paper's human teleop recovery data (the paper's
  "Combined" recipe, on SmolVLA), (2) zero-shot anchors with the paper's StarVLA checkpoints on our split,
  (3) a Kintsugi-style "recoverability-frontier" start-state baseline built from our own pipeline.

## 1. The LIBERO-Recover paper's own baselines (arXiv 2609.05178 v3, 28 Sep 2026)

**Protocol (paper, Sec. 4.1 + App. D.4).** Policies start from a recorded failure state (`.bddl` + flat MuJoCo state).
For each scenario there are 5 closed-gripper and 5 open-gripper trials, each with a fresh xy perturbation of up to
±2 cm on free-joint objects. The gripper is driven for 20 settle steps, and a run gets at most **520 control steps**.
A trial counts as a success if the env's `done` fires. RSR is averaged over episodes, so every rollout has equal
weight. Levels are L1 action retry, L2 action adaptation, L3 object-state recovery and L4 environmental recovery.
They come from a Qwen3.5-27B video annotation (the README / our notes say Qwen3-VL-27B), with a manual spot check.
The paper also reports RD (pre- vs post-failure degradation) and RC (1 - std of per-task RSR).
**Scenarios:** 2,178 in total. Spatial has 196 (L1 64 / L2 58 / L3 48 / L4 26), Object 321 (130 / 108 / 48 / 35),
Goal 380, LIBERO-10 590 and LIBERO-90 691. The failures come from six policies, with no manual perturbation.
Scenarios used to collect teleop data are excluded from evaluation (Stage 4).

**Policies.** All six are re-implemented in the **StarVLA** framework and trained on the union of the 4 LIBERO suites
(6,193 demos). They take 224 px third-person + wrist images and proprioception, output 8-step action chunks, and are
re-queried every 8 steps at 20 Hz, in bf16 on one H100. Models: OpenVLA-OFT (Qwen3-VL-4B + DiT-B, 50k steps),
π0 (Qwen3-VL-4B + DINOv2 + DiT flow matching, 100k), π0-FAST (Qwen2.5-VL-3B, FAST tokens, 30k), GR00T-N1.5
(Qwen2.5-VL-3B + DiT, 30k), Wan2-Policy (Wan2.2-TI2V-5B WAM, 60k) and Cosmos-Predict2-Policy (2B WAM, 50k).
These are **StarVLA re-implementations, not the original openpi / OpenVLA-OFT / GR00T code**.

**Table 1: RSR (%) per level** (v3 HTML; the v1 PDF has slightly different numbers, e.g. PI Spatial L1 40.0):

| Policy | Spatial L1 / L2 / L3 / L4 | Object L1 / L2 / L3 / L4 | Goal L1 / L2 / L3 / L4 | LIBERO-100 L1 / L2 / L3 / L4 |
|---|---|---|---|---|
| π0 | 40.8 / 11.2 / 13.7 / 6.0 | 56.4 / 41.9 / 24.6 / 8.4 | 50.6 / 21.9 / 5.9 / 2.7 | 39.4 / 20.6 / 10.1 / 0.0 |
| π0-FAST | 29.3 / 13.1 / 6.9 / 3.2 | 49.2 / 41.7 / 35.2 / 12.2 | 56.9 / 25.8 / 5.6 / 0.0 | 24.5 / 10.8 / 16.2 / 0.0 |
| GR00T-N1.5 | 7.4 / 13.2 / 7.3 / 3.4 | 47.9 / 26.8 / 30.6 / 1.8 | 51.9 / 41.8 / 5.8 / 0.0 | 14.3 / 20.6 / 17.6 / 0.0 |
| OpenVLA-OFT | 40.8 / 12.1 / 12.8 / 8.6 | 56.4 / 41.9 / 25.1 / 7.9 | 56.9 / 30.8 / 8.9 / 3.9 | 39.4 / 20.6 / 10.1 / 0.0 |
| Wan2-Policy | 16.7 / 3.6 / 11.8 / 12.4 | 9.3 / 33.1 / 20.1 / 0.0 | 34.4 / 20.8 / 4.2 / 2.9 | 10.7 / 7.6 / 13.9 / 0.0 |
| Cosmos-Predict2 | 7.4 / 4.4 / 14.2 / 0.0 | 10.8 / 24.5 / 8.5 / 0.0 | 30.7 / 19.4 / 5.7 / 0.0 | 11.9 / 10.9 / 11.2 / 0.0 |

Note: the π0 and OFT rows are identical on Spatial L1, Object L1-L2 and all of LIBERO-100, which looks like a copy
error in the paper. Treat them as low-precision numbers.

**Fig. 5: suite-level RSR (%)** (LIBERO SR in brackets):

| Policy | Spatial | Object | Goal | LIBERO-100 |
|---|---|---|---|---|
| π0-FAST | 10.0 (96.4) | 34.0 (96.8) | 23.3 (88.6) | 13.3 (60.2) |
| GR00T-N1.5 | 5.8 (92.0) | 26.0 (92.0) | 26.3 (86.0) | 13.3 (76.0) |
| OpenVLA-OFT | 15.7 (97.6) | 32.7 (98.4) | 24.7 (97.9) | 19.3 (94.5) |
| π0 | 14.7 (96.8) | 32.3 (98.8) | 21.3 (95.8) | 18.0 (85.2) |
| Wan2-Policy | 11.1 (98.9) | 15.7 (98.3) | 16.3 (94.4) | 8.3 (90.4) |
| Cosmos-Predict2 | 6.8 (97.4) | 11.0 (98.0) | 14.7 (95.1) | 8.7 (90.4) |

**Methods that improve RSR (Tables 2-3), suite-level RSR:**

| Method | Spatial | Object | Goal | LIBERO-100 | Training |
|---|---|---|---|---|---|
| GR00T-N1.5 | 0.057 | 0.260 | 0.263 | 0.133 | LIBERO only |
| + Combined (teleop recovery data) | 0.116 | 0.262 | 0.264 | 0.206 | LIBERO + 3,184 teleop recoveries (40% of frames), 50k steps |
| + temporal (8 past frames) | 0.066 | 0.260 | 0.264 | 0.166 | same + pre-failure history |
| OpenVLA-OFT | 0.156 | 0.326 | 0.246 | 0.193 | LIBERO only |
| + Combined | 0.178 | 0.333 | 0.270 | 0.236 | as above |
| + temporal | 0.193 | 0.333 | 0.300 | 0.247 | as above |

Combined recipe (App. E.3): 9,377 episodes / 1.57M frames, with all components sampled at equal weight. Effective
batch 16 on 4x H20 96 GB with DeepSpeed ZeRO-2, 50k steps (~0.5 epoch), ~5 h. Only the vision encoder is frozen.
The paper's main lesson is that human recovery data gives only **+2 to +6 pt** on Spatial. Their "Key finding 5" is
that smaller action chunks help recovery (chunk sizes 4 to 32 were tested).

**Code / data / compute for us.** Evaluator: `examples/LIBERO/eval_files/eval_libero_custom_scene.py` (StarVLA
server + LIBERO client). Our `retryvla/envs/libero_env.py::load_scene` re-implements the same protocol, and the
upstream repo is vendored in `third_party/LIBERO-Recovery`. Checkpoints: `StarVLA/*-LIBERO-4in1` on HF. Running one
zero-shot policy on our test split costs 820 episodes (56 x 10 + 26 x 10) x 520 steps max. With a 3-4B VLA on one
A100 that is ~2-4 GPU-h per model, plus a second conda env (`starvla`: transformers 4.57, flash-attn). Their
"Combined" SFT at full scale is ~20 GPU-h (H20).

## 2. Papers that cite LIBERO-Recover (found by 2026-11-02)

| Paper | Uses LIBERO-Recover? | Setting | Policy | Result | Code |
|---|---|---|---|---|---|
| **Kintsugi-VLA** ([2609.31048](https://arxiv.org/abs/2609.31048), Snegirev et al., 25 Sep 2026) | **No**, related work only | Isaac Lab Franka cube lift-and-transport (one task) | SmolVLA; PPO privileged teacher generates recoveries | Aggregate recovery success 34.6% / 38.4% (difficulty / frame-budget matched), +5.8 / +6.7 pt over uniform sampling of start states; clean SR 76.8 -> 74.7 | none found |
| **FIT-MAC** ([2610.28109](https://arxiv.org/abs/2610.28109), Li et al., Oct 2026) | **No**, related work only | LIBERO-PRO perturbations | OpenVLA + others, multi-agent closed-loop wrapper | SOTA on LIBERO-PRO | none found |
| **HELM** ([2604.18791](https://arxiv.org/abs/2604.18791)) | **Name clash**: its "LIBERO-Recovery" is a different protocol, predates 2609.05178 | LIBERO-Long with ±5 cm displacement or gripper flip injected at a subgoal boundary | OpenVLA + memory / verifier | Not comparable | none found |

**Kintsugi-VLA in more detail (closest in spirit to RetryVLA).** It restores saved sim states along failed rollouts,
estimates *interventional recoverability* = P(the privileged expert completes the task from state s) with adaptive
Monte-Carlo continuations and Wilson intervals, finds the "terminal low-recoverability frontier", and starts expert
recovery demos just before it. SmolVLA is then fine-tuned on these demos. Differences from us: (i) recoverability is
measured for a *privileged PPO expert* rather than the deployed VLA; we use C^π of the frozen VLA itself; (ii) the
recovery actions come from the expert, with no search and no WM; (iii) one Isaac Lab task, not LIBERO. The idea
"use sim branching to choose *where* to start recovery data" is portable to our pipeline (section 5, baseline 3).

**Not citing it, but adjacent (Sep-Nov 2026, sim, no LIBERO-Recover):**
- **RoboRecover** ([2609.28952](https://arxiv.org/abs/2609.28952), [repo](https://github.com/RUCKBReasoning/RoboRecover)):
  a *competing* recovery benchmark. It has 1,000 LIBERO + 1,000 RoboTwin scenarios with a fixed 800 / 200 split,
  built by replaying action prefixes. SmolVLA is evaluated on RoboTwin, π0.5 gets 64.4% RSR on LIBERO, and
  "Reversion" (restart 30 steps earlier) gives 74.0 and Monitor+Reversion 77.0. It is a second benchmark we could
  report on later.
- **Backtrack** ([2610.32941](https://arxiv.org/abs/2610.32941)): failure head + rewind on frozen SmolVLA / π0.5 /
  π0-FAST, evaluated on clean LIBERO + MetaWorld. No official code (it builds on lerobot). **We already re-implemented
  it** (`retryvla/policy/backtrack.py`): on Spatial test, single 0.134 / retry 0.129 / distill 0.182 vs pi1 0.418.
- **RESETTLE** ([2610.12185](https://arxiv.org/abs/2610.12185), [code](https://github.com/JIA-Lab-research/RESETTLE)):
  action-disagreement trigger + V-JEPA (ViT-g, 1B) retrieval of a same-task demo frame + one corrective action.
  Evaluated on LIBERO-Plus / Pro, Meta-World (with SmolVLA) and RoboCasa. The code covers LIBERO. Training used
  8x H800.
- **SafeLoop** ([2609.26313](https://arxiv.org/abs/2609.26313), [code](https://github.com/Loule0-0/SafeLoop)): risk
  predictor + joint-space rollback on frozen VLA, 24 LIBERO tasks.
- **B2FF** ([2606.09258](https://arxiv.org/abs/2606.09258)): milestone bank of imagined futures for foresight VLAs;
  failure-injected LIBERO 56.3 -> 74.0. Needs a goal-image-conditioned VLA, so it does not fit SmolVLA.
- **VLA-Corrector** ([2609.06508](https://arxiv.org/abs/2609.06508)): learned verifier + stage-conditioned recovery
  prompts, LIBERO / LIBERO-Plus. Same name as ZJU's NeurIPS 2026 repo `ZJU-OmniAI/vla-corrector`, which may be a
  different paper.

## 3. Related sim recovery methods: can we run them on LIBERO-Recover (SmolVLA, Spatial / Object, 520 steps)?

Effort uses our unit costs (NOTES): SFT 6k steps ≈ 1.3 h on 1 A100; RSR eval on Spatial test ≈ 1 h, Object ≈ 0.5 h;
nominal ≈ 0.4 h. One SFT arm with 3 seeds on one suite ≈ 8 GPU-h.

| Method | What it needs | Code | Fit to our setup | Effort | Value as a baseline |
|---|---|---|---|---|---|
| **LIBERO-Recover "Combined"** (teleop recovery SFT) | human recovery episodes (released), SFT | data on ModelScope; recipe in App. E.3 | direct: LeRobot v2.1, 256², 20 fps, 7-D delta EEF actions, same tasks | 1-1.5 days eng + ~8 GPU-h per suite (3 seeds) | **high**: it is the benchmark's own improvement method, and "self-generated vs human recovery data" is the obvious reviewer question |
| **StarVLA zero-shot anchors** (paper's π0 / OFT) | public HF checkpoints + StarVLA server | yes | server/client adapter into our rollout; 8-step chunk | ~1 day eng + 2-4 GPU-h per model | **high** for protocol calibration and context |
| **Kintsugi-VLA** (recoverability frontier start states) | sim restore + MC continuations + a data generator | no | portable: our C^π labels / continuations already measure recoverability for π0; the generator would be our random or oracle search (no privileged expert in LIBERO) | 1-2 days eng + ~1.3M transitions + 8 GPU-h | medium-high: isolates "choose start states by measured recoverability" vs "C^π-guided bridge end state" |
| **Backtrack** | failure head on VLA features + rewind | no (we re-implemented it) | done | 0 | done: single 0.134, retry 0.129, distill 0.182 (Spatial); retry 0.165 (Object) |
| **SafeLoop** (risk predictor + rollback) | hazard labels from rollouts, joint-space rollback | yes | close to Backtrack; hazard labels (collisions / drops) would need defining in LIBERO | 2-3 days | low: rewind baselines are already ≈ pi0 on LIBERO-Recover |
| **RESETTLE** (disagreement trigger + V-JEPA retrieval + 1 corrective action) | 2 flow samples per step (SmolVLA OK), ViT-g module trained on LIBERO demos | yes (LIBERO) | test-time wrapper on frozen pi0 / pi1; must check whether a LIBERO module checkpoint is released | 2-4 days + module training (paper: 8x H800; we would need a much smaller run) | medium: strongest recent test-time module, but costly |
| **RACER** (ICRA 2025) | perturb-and-recover data + LLaVA supervisor + RVT actor | yes (RLBench / RVT) | keyframe RVT is incompatible with SmolVLA chunks; only the data idea ports, and it is close to our random-search arm | ≥1 week (faithful) / done in spirit (random arm) | low |
| **FailSafe** (IROS 2026) | motion-planner failure generation + LLaVA-OV-7B recovery-delta VLM | yes (ManiSkill only) | LIBERO has no motion-planner stack; 7B VLM fine-tune; VLM queried online | 1-2 weeks | low-medium |
| **CCIL** (ICLR 2024) | Lipschitz dynamics model + corrective labels near demos | yes (state-based; image version "in development") | labels are imagined *states*, and SmolVLA needs images; our DINOv2 WM cannot render pixels | ≥1 week, not faithful | low |
| **RePO-VLA** | recovery data (planner / human) + PAS-VF value + value-conditioned VLA | project page says code; no repo found | could use the teleop expert data as recovery data, but needs a value token in SmolVLA and a value model | 4-6 days | medium (reported on RoboTwin) |
| **CoRe** | OOD detector + image editing / voxel re-render + motion planner to restore the scene | link not found | needs a sim motion planner and scene-rearrangement; training-free | ≥1 week | low-medium |
| **ReCoVLA** | Qwen3-VL reward compiler + PPO residual on frozen VLA | no (only openpi-comet dependency) | our B3.3 residual-PPO baseline already covers the spirit (0.295 RSR, nominal 0.022) | 3-5 days for the VLM part | low (residual RL covered) |
| **FLARE (CVPR 2026)** | Retry: pose-noise + bridging augmentation of demos; Reset: MLLM + human reset skills | no | the Retry half ports as a DART-style augmentation (perturb demo state, scripted bridge back, keep verified) | 2-3 days + 8 GPU-h | medium-low |
| **B2FF** | goal-image-conditioned foresight VLA | no | SmolVLA has no visual-goal input | n/a | n/a |

## 4. Our protocol and numbers vs the paper

**Same as the paper:** 5 close + 5 open trials per scene, ±2 cm xy object noise, 20 gripper-settle steps, 520-step
cap, `done`-based success, instruction = the BDDL `(:language)` line, RSR averaged over episodes
(`configs/env/libero_spatial.yaml`, `retryvla/envs/libero_env.py::load_scene`).

**Differences:**

| Item | Paper | Ours | Effect |
|---|---|---|---|
| Scenario set | 2,178 (Spatial 196, Object 321), not released as such | released assets: Spatial 168 (84 failure episodes, 9 / 10 tasks), Object 77 | Released ≠ evaluated, especially Object (77 vs 321). Issue [#2](https://github.com/liulin815/LIBERO-Recovery/issues/2) asks for the frozen manifest (2,117 vs 2,178); the maintainer pointed to the ModelScope release, and no manifest was given |
| Split | no split; all scenarios are test, and the 413 teleop scenarios are excluded | our own grouped split: Spatial train 112 / test 56 (28 failure episodes), Object train 51 / test 26 | We report a subset. Spatial test is a **dev set** from M7 on; Object is held out |
| Levels | L1-L4 for all scenarios | official labels only for `step_0180` (Spatial) / `step_0240` (Object) scenes. Spatial test 28 labeled = 1 L1 / 11 L2 / 16 L3, Object 19 = 4 L1 / 15 L2; no L4 | Our per-level RSR is not comparable to Table 1; L4 cannot be measured |
| Policy / chunk | StarVLA 3-5B VLAs / WAMs, 8-step chunks, 224 px, 4-suite training | SmolVLA (`HuggingFaceVLA/smolvla_libero`), chunk 50, `n_action_steps = 10`, 256 px | The paper finds RSR drops with longer executed chunks; our 10 is close to their 8 |
| First observation | official evaluator passes the obs returned by `set_init_state`, i.e. **before** gripper settling | we pass the **post-settle** obs | Affects only the first chunk; closed-gripper trials see the right gripper state in ours. Issue #2 also flags this |
| Object xy noise | `perturb_object_xy` adds noise at `flat_state[jnt_qposadr]`, but robosuite's flat state is `[time, qpos, qvel]` | **we copied the same indexing** (`_free_joint_xy_adrs`) | Off by one in both: "dx" goes to the previous qpos entry (a quaternion component of the previous object, or a robot joint for the first object), "dy" goes to object x, and object y is never perturbed. Issue #2 also flags it. It is consistent with the official numbers, but the "±2 cm xy noise" claim is not literally true. Fix or document before the paper |
| Seeds | `--seed 7`, 30 sampled scenes by default (README) | our per-scene `noise_seed(…)` | different perturbation draws; fine for paired comparisons inside our tables only |
| Nominal | LIBERO SR of 4in1 models (92-99%) | LIBERO-Spatial 10 × 50 (pi0 0.656, pi1 0.785) | SmolVLA is a weaker nominal policy |

**Numbers side by side** (suite-level RSR; the paper's values are over its full suite, ours over our test subset):

| Model | Spatial | Object |
|---|---|---|
| paper: π0 / OFT / π0-FAST / GR00T / Wan / Cosmos (zero-shot) | 0.147 / 0.157 / 0.100 / 0.058 / 0.111 / 0.068 | 0.323 / 0.327 / 0.340 / 0.260 / 0.157 / 0.110 |
| paper: OFT + teleop (Combined) / + temporal | 0.178 / 0.193 | 0.333 / 0.333 |
| ours: pi0 = SmolVLA base | 0.118 | 0.219 |
| ours: pi1-nominal / demo-only | 0.138 / 0.127 | 0.200 / 0.173 |
| ours: Backtrack (retry) on pi0 | 0.129 | 0.165 |
| ours: **RetryVLA pi1** (3 seeds) | **0.418 ± 0.021** | 0.354 (1 seed, Object recipe) |
| ours: M7 base (3 seeds) | 0.482 ± 0.019 | - |

Reading: SmolVLA pi0 (0.118 Spatial) falls inside the paper's zero-shot range (0.06-0.16), so our split is not
obviously easier. On Spatial, pi1 is +0.30 over pi0, while human-data SFT in the paper gives +0.02 to +0.06. That gap
is large, but it compares a different policy and a different scene set. The clean comparison is baseline 1 below.
On Object, our zero-shot SmolVLA (0.219) is below the paper's 4B models (0.26-0.34), and pi1 (0.354) is about at
their level. **Naming caveat:** our "pi0" / "pi1" are RetryVLA policy generations (SmolVLA base / after distillation).
They are not Physical Intelligence π0, and not the paper's StarVLA "π0" (Qwen3-VL-PI). Say this explicitly in the
paper.

## 5. Recommended runnable baselines (in priority order)

**B-1. Human teleop recovery SFT (the paper's "Combined" recipe on SmolVLA).** *Most valuable.*
- Data: `ataier/LIBERO_Recovery_Expert`, filtered to the Spatial tasks (the 7 paraphrased "pick the akita black bowl
  … place it on the plate" tasks, 1,185 episodes) and the Object tasks (about 1,150 "pick the … place it in the
  basket" episodes). The images are AV1 video, so decoding needs pyav / ffmpeg with dav1d.
- Arms, all with the pi1 recipe (init pi1-nominal, 6k steps, bs 32, 50% nominal replay, 3 seeds):
  (a) all expert episodes of the suite; (b) count-matched to pi1 (208 Spatial / 123 Object).
  Optional (c): expert + our verified bridges.
- Leakage check first. The mapping from the 3,184 episodes to the 413 source scenarios is unreleased (issue #2,
  Q3). Match each expert episode's frame 0 (`first_frames/` + `observation.state`) to our scene sim states and drop
  episodes from our test failure episodes. Also check the action conventions (gripper sign, rotation units) and
  20 Hz vs the SmolVLA LIBERO dataset.
- Effort: ~1-1.5 days of engineering (download, convert to our `MixDataset`, leakage check) + ~8 GPU-h per suite
  (Spatial and Object) ≈ 16 GPU-h total.
- Why: it answers "is search + verification better than (much more) human recovery data at the same SFT budget?"
  on the benchmark's own data, and it reproduces the paper's only RSR-improving method on our policy.

**B-2. Zero-shot anchors with the paper's checkpoints on our split.**
- `StarVLA/Qwen3-VL-PI-LIBERO-4in1` (the paper's "π0") and `StarVLA/Qwen3-VL-OFT-LIBERO-4in1` ("OFT"), run on the
  Spatial 56 × 10 and Object 26 × 10 test scenes through our rollout code. Our rollout would call their policy
  server through `model2libero_interface.py`, with the 8-step chunk, so env / noise / seeds stay identical to our
  other rows. Optionally run one model once with the official evaluator to measure the effect of the two
  evaluator quirks above.
- Effort: ~1 day of engineering (`starvla` conda env, websocket adapter) + ~2-4 GPU-h per model per both suites
  ≈ 6-8 GPU-h.
- Why: it calibrates our protocol against the paper (expected Spatial ~0.15 and Object ~0.32 if the subset is
  representative), and it shows that SmolVLA + RetryVLA beats 3-4B VLAs that have 4-suite training. This is
  context, not a method baseline.

**B-3. Kintsugi-style recoverability-frontier start states (no WM).**
- From pi0's failed continuations on train scenes, we already log states + outcomes. Measure π0 recoverability at
  checkpoints along each failed trajectory by MC continuations, using the existing verify machinery. Pick the
  frontier state, generate recoveries from it with the random-search proposal library, and keep only verified
  ones. SFT with the pi1 recipe at a matched trajectory count and sim budget, 3 seeds.
- Effort: ~1-2 days of engineering (`run_search` already supports restore + continuation) + ~1.3M transitions of
  search (CPU) + ~8 GPU-h SFT / eval on Spatial.
- Why: it is the only recent method that also uses sim branching + measured recoverability to make recovery data,
  and it tests whether C^π-guided *bridge end-states* matter beyond picking *start states* by measured
  recoverability.

Not recommended now: RESETTLE (best recent test-time module, code available, but ViT-g module training is heavy
and Backtrack-type wrappers already sit at pi0 level on this benchmark), SafeLoop (≈ Backtrack), RACER / FailSafe /
CCIL / CoRe / B2FF (no practical port to SmolVLA + LIBERO), ReCoVLA (covered by our residual-RL baseline).
Total for B-1 + B-2 + B-3: about 4 days of engineering and ~30 GPU-h, which fits one week on the
c37611dd / dedc84ce GPUs.

## Links
- LIBERO-Recover: [arXiv 2609.05178](https://arxiv.org/abs/2609.05178) · [code](https://github.com/liulin815/LIBERO-Recovery) · [project](https://liulin815.github.io/LIBERO-Recovery/) · data: [assets](https://www.modelscope.cn/datasets/ataier/LIBERO_Recovery_Assets), [expert](https://www.modelscope.cn/datasets/ataier/LIBERO_Recovery_Expert), [LIBERO_10_LL](https://www.modelscope.cn/datasets/ataier/LIBERO_10_LL) · checkpoints: [StarVLA on HF](https://huggingface.co/StarVLA)
- Citing: [Kintsugi-VLA 2609.31048](https://arxiv.org/abs/2609.31048) · [FIT-MAC 2610.28109](https://arxiv.org/abs/2610.28109)
- Adjacent: [RoboRecover 2609.28952](https://arxiv.org/abs/2609.28952) · [Backtrack 2610.32941](https://arxiv.org/abs/2610.32941) · [RESETTLE 2610.12185](https://arxiv.org/abs/2610.12185) · [SafeLoop 2609.26313](https://arxiv.org/abs/2609.26313) · [B2FF 2606.09258](https://arxiv.org/abs/2606.09258) · [VLA-Corrector 2609.06508](https://arxiv.org/abs/2609.06508) · [HELM 2604.18791](https://arxiv.org/abs/2604.18791)
- Classic: [FailSafe 2510.01642](https://arxiv.org/abs/2510.01642) ([code](https://github.com/Jimntu/FailSafe_code)) · [RACER 2409.14674](https://arxiv.org/abs/2409.14674) ([code](https://github.com/sled-group/RACER)) · [CCIL 2310.12972](https://arxiv.org/abs/2310.12972) ([code](https://github.com/personalrobotics/CCIL)) · [RePO-VLA 2605.09410](https://arxiv.org/abs/2605.09410) · [CoRe 2608.14822](https://arxiv.org/abs/2608.14822) · [ReCoVLA 2606.09630](https://arxiv.org/abs/2606.09630) · [FLARE 2608.26645](https://arxiv.org/abs/2608.26645)
