# World-model-guided recovery: who already does what RetryVLA does (overlap audit, 2026-11-16)

> Trigger: advisor feedback "check related work carefully; several recent papers already use a world model in a similar way".
> Question: which papers already do RetryVLA's core (after a VLA fails, use a latent / video world model to imagine or
> search corrective actions, score them with a value / progress / success / policy-competence critic, verify them, and use
> them at test time or for training)?
> Scope: 2024 to Nov 2026; this file adds to, and does not repeat, `RELATED_WORK.md` (sections A-K) and
> `LIBERO_RECOVER_BASELINES.md`. No GPU jobs were run.

**Tóm tắt cho thầy (VI).** Đúng là đã có nhiều bài làm gần như cùng hướng. Gần nhất: *Back on Track* (Stanford,
10/2026) dùng latent WM + value có điều kiện theo policy (liveness, học từ rollout của chính policy) để CEM tìm hành
động sửa rồi trả quyền cho policy, nhưng làm lúc test, không verify, không distill. *SAILOR* (NeurIPS 2025): WM +
reward + critic ensemble, search để sửa lỗi của base policy rồi distill. *2610.05151*: verify bằng việc cho chính policy
đóng băng chạy tiếp rồi distill (không có WM). *Rollback*, *TACO*: search / tưởng tượng sửa sai trong WM rồi post-train.
Phần còn bảo vệ được: (1) ghép WM + C^pi để chọn, verify bằng continuation của pi trong sim, distill vào VLA, giữ
nominal, trên LIBERO-Recover; (2) đo định lượng WM lạc quan dưới áp lực search (tách phần WM / critic); (3) so sánh
cùng ngân sách: tín hiệu chấm điểm và distill vs planning lúc test (phải tự chạy baseline kiểu Back on Track).

## How the sweep was done and verified
- Queries (WebSearch, several phrasings): "world model failure recovery VLA", "corrective action world model",
  "recovery data generation world model", "policy-aware / policy-conditioned value recovery", "imagined rollouts recovery",
  "self-correction VLA world model", "counterfactual recovery", "value-guided recovery planning VLA", "test-time scaling
  VLA world model verifier", "world model exploitation / optimism under search", "WM-RL keyframe failure states",
  "recoverability world model", OpenReview ICLR 2027 / CoRL 2026 / NeurIPS 2026 and the related-work sections of the
  closest papers (Back on Track's bibliography gave DreamAvoid and VLA-in-the-Loop).
- Citations: Semantic Scholar citation API for Back on Track, Dream2Fix, WM-DAgger, Hi-WM, Kintsugi-VLA, CoRe,
  CureWM and the liveness-OPE paper (it found 2610.05151, 2610.06921, RoboCoach, XPACE, EgoRecovery and others).
  Back on Track had no indexed citations yet (Oct 2026 paper). The arXiv export API returned HTTP 429 all session,
  so metadata come from the arXiv abs pages (`citation_title / author / date` meta + comments field), fetched 2026-11-16.
- Cross-check rule (Gemini is unavailable here): every arXiv ID below was opened on its arXiv abs page (title,
  first / last author, v1 date) **and** on a second source: project page, GitHub, proceedings page, Semantic Scholar,
  alphaXiv or a conference PDF host. The "2nd src" column says which. "arXiv only" = abs page + arXiv HTML full text
  only (no independent second source found); treat those claims as single-source. Numbers are from the abstracts or the
  HTML full text and were not reproduced.
- Not verifiable: *Generative Failure-to-Success Bridging* (project page + author CV only, "work in progress", no
  arXiv); *World-in-Loop / VLA-in-the-Loop* (ECCV 2026 PDF on the conference host + a blog note, no arXiv ID found);
  MeRL's "CoRL 2026" (project page only, arXiv comment empty). OpenReview ICLR 2027 listing pages did not render
  through the fetch tool, so ICLR 2027 coverage relies on arXiv comments ("Submitted to ICLR 2027": Needlework).

## TL;DR (verdict)
1. **The core loop is no longer ours alone.** "After the frozen policy goes off track, search corrective actions in a
   learned latent WM, scored by a value learned from the policy's own outcome-labelled rollouts, aiming for states from
   which the policy itself can finish, then hand control back" is exactly **Back on Track** (2610.19579, Stanford,
   14 Oct 2026, LIBERO-Goal + Franka), whose value is the policy-conditioned liveness value of 2605.11479 (RSS 2026).
   "Search in a latent WM against a learned reward + critic ensemble with an uncertainty penalty to recover from the base
   policy's mistakes, then distill the search into the policy" is **SAILOR** (2506.05294, NeurIPS 2025).
   "Search / branch in a video WM, score outcomes, distill into the policy" is **Rollback** (2610.36068) and **TACO**
   (2607.02840). Test-time WM steering of frozen VLAs is a crowded line (DreamAvoid, ForkCast, DreamSteer, VeriVLA,
   World-in-Loop, FOREWARN, LPS).
2. **What still looks defensible** (no paper found that does all of it): (a) the *combination* of a WM + policy-
   competence critic as a cheap filter over many candidate bridges **and** acceptance only when the *frozen policy's
   own continuation* succeeds from a bit-exact restored sim state **and** distillation into the same VLA with nominal
   success measured, on LIBERO-Recover (still no other method numbers on that benchmark). Each piece exists separately:
   the "frozen student must continue from the teacher endpoint" acceptance test is in **2610.05151** (LLM tool-teacher,
   no WM, no critic, LIBERO-Goal), sim restore + MC recoverability is in Kintsugi-VLA (expert, not the deployed VLA),
   WM + policy-conditioned value is in Back on Track (test time, no verification, no distill). (b) A quantitative study
   of *optimism under search*: the gap between imagined and true C^pi on search-selected vs random candidates, split
   into WM part and critic part, with FP / AUC / ECE and sim ground truth on the same bridges. Closest: CureWM (passive
   rollouts), 2610.37196 (selectable headroom, no optimism decomposition), SAILOR / Back on Track (state the risk, do
   not measure it). (c) Matched-budget comparisons of the scoring signal (C^pi vs goal-latent vs progress value) and of
   distill vs test-time planning. (c) is defensible only if we run the Back-on-Track-style test-time arm ourselves.
3. **What is taken**: "first to use a WM to plan recovery for a VLA", "first policy-conditioned value / competence
   critic for recovery", "search + distill instead of RL", "branch from a failure state into many corrections",
   "structured perturbation proposals beat native VLA samples" (2610.37196 and DreamSteer report the same).
4. **Recommended positioning**: present RetryVLA as *train-time amortisation and verification* of Back-on-Track-style
   policy-conditioned recovery planning (plus a measurement of how much imagined recovery survives verification under
   search pressure), not as a new planning objective. Cite Back on Track, liveness OPE, SAILOR, Kintsugi-VLA,
   2610.05151, Rollback, TACO, Hi-WM, WM-DAgger, Dream2Fix, DreamAvoid, ForkCast, CureWM, 2610.37196 and PLD as the
   closest prior work. Details in the last two sections.

## 1. All papers found, sorted by closeness to RetryVLA
Legend. **WM type**: latent / video (pixel) / WAM (world-action model, policy that also predicts futures) / sim
(physics simulator used as the model) / none. **WM use**: TTP = test-time planning or selection; DG = data
generation; RL = RL in imagination; EV = evaluation. **Train use**: none = frozen policy. "new" = not in
`RELATED_WORK.md` / `LIBERO_RECOVER_BASELINES.md` before this note. Code: GH = GitHub repo seen; PP = project page
only; "-" = none found.

### Tier A: same core loop (WM or sim branching + policy-conditioned or learned value + corrective actions)

| # | Paper | arXiv / link | v1 date | Venue | WM type | WM use | Scorer / critic | Verification | Train use | Benchmark | Code | 2nd src | in repo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | Back on Track: Deployment-time Policy Improvement via WM Planning | [2610.19579](https://arxiv.org/abs/2610.19579) | 2026-10-14 | preprint | latent (SigLIP2 + 4-layer ViT, 29.6M) | TTP: monitor triggers CEM, then hand back | **policy-conditioned liveness value** V_pi_b (ensemble mean, learned from on-policy rollouts with episode outcomes) | none (executes directly) | none (black-box policy) | LIBERO-Goal perturbed init, pi0.5 72.6 -> 81.5; Franka DP 40 -> 60 | PP | SIA lab page, alphaXiv | (K1) |
| A2 | Offline Policy Evaluation via Discounted Liveness | [2605.11479](https://arxiv.org/abs/2605.11479) | 2026-05-12 | RSS 2026 (comment + roboticsproceedings) | none | EV | policy-conditioned liveness Bellman value from finite rollouts | - | none | 2 sim tasks (VLA + DP), cloth folding | PP | RSS proceedings | new |
| A3 | SAILOR: Robust Imitation via Learning to Search | [2506.05294](https://arxiv.org/abs/2506.05294) | 2025-06-05 | NeurIPS 2025 (proceedings) | latent (Dreamer) | TTP (MPPI over residuals around base plan) + periodic distill (ExIt) | discriminator reward (expert vs learner) + lambda-return critic ensemble, mean minus std penalty | online env data to refit WM / RM, no per-sample check | ExIt distill; post-hoc distilled DP = search on Lift / Can, worse on Square | 12 tasks RoboMimic / RoboSuite / ManiSkill, Diffusion Policy; 8x test budget "robust to reward hacking" | GH | NeurIPS proc., GH | new |
| A4 | Recursive Self-Improvement of Visuomotor Policies through Local Recovery Supervision | [2610.05151](https://arxiv.org/abs/2610.05151) | 2026-10-04 | preprint | none (sim restore) | DG | LLM tool-teacher + auditor; **frozen student must continue from teacher endpoint**, else extend | sim continuation by the frozen policy | quality-weighted SFT, recursive rounds | LIBERO-Goal pi0 78 -> 88 / 100; robomimic Can | - | Semantic Scholar | new |
| A5 | Kintsugi-VLA | [2609.31048](https://arxiv.org/abs/2609.31048) | 2026-09-25 | preprint (ICRA 2027 sub.) | none (sim branching) | DG | MC recoverability of a privileged expert, Wilson CIs; frontier start states | sim (expert completes) | SFT SmolVLA | Isaac Lab cube lift, +5.8 / +6.7 pt | - | S2 | (LRB) |
| A6 | Rollback: Experience Backtracking in WMs | [2610.36068](https://arxiv.org/abs/2610.36068) | 2026-10-26 | preprint | video (GE-Sim 2.0) | DG: MCTS rewinding to decision points | fine-tuned success judge on leaves; sibling-relative values, no value net | none per sample; real transitions LoRA-refine WM | distill action-level labels (ExIt style) | 3 real tasks, +10.7 / 11.1 / 11.9 pt | PP | project page | new |
| A7 | TACO: Tactile WM as a Self-Corrector | [2607.02840](https://arxiv.org/abs/2607.02840) | 2026-07-03 | preprint | video + tactile | DG: imagine local corrections at failure-adjacent states | IDM + **progress** value; keep top progress gain | kinematic / tactile plausibility; real replay of a subset | CFG-RL (binary advantage), iterative | real tasks 0.375 -> 0.825 | PP | emergentmind, S2 | new |
| A8 | Hi-WM: Human-in-the-World-Model | [2604.21741](https://arxiv.org/abs/2604.21741) | 2026-04-23 | preprint | video, action-cond. | DG (rollback / branching in WM) | human | human | SFT | 3 real tasks, +37.9 pt | PP | S2 | (F5) |
| A9 | WM-DAgger | [2604.11351](https://arxiv.org/abs/2604.11351) | 2026-04-13 | preprint | video (Cosmos-Predict-2.5, eye-in-hand) | DG: deviate-and-return around expert | none; DINOv2 terminal-frame consistency filter | filter vs real frame | DAgger aggregation (BC) | real few-shot, 93.3 % bag pushing | GH | GH | (report) |
| A10 | Dream2Fix (Counterfactual Failure Synthesis) | [2603.13528](https://arxiv.org/abs/2603.13528) | 2026-03-13 | preprint | video (Ctrl-World) | DG: perturb actions to make failures + paired corrections | VLM + IDM + kinematic verifier | learned verifiers | SFT a VLM corrector | real Franka, 46 % closed-loop | PP | project page | (A2) |

### Tier B: WM used at test time to pick or plan corrective actions for a frozen policy

| # | Paper | arXiv / link | v1 date | Venue | WM type | WM use | Scorer / critic | Verification | Train use | Benchmark | Code | 2nd src | in repo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | DreamAvoid: Critical-Phase Test-Time Dreaming | [2605.11750](https://arxiv.org/abs/2605.11750) | 2026-05-12 | preprint | video (DreamDojo) | TTP at triggered critical phases | value (progress delta, ranking loss) trained on success / failure / boundary data | none | none (frozen pi0.5); WM + value adapted | real 48.8 -> 72.5 %; LIBERO, SimplerEnv, RoboTwin | GH | GH, alphaXiv | new |
| B2 | ForkCast: Predictive Recovery via Counterfactual Futures (Wen, Zeng ... Zuo, MBZUAI) | [2610.18650](https://arxiv.org/abs/2610.18650) | 2026-10-13 | preprint | latent (V-JEPA 2-AC, 3-channel cond.: EE incr. / cum. disp. / camera offset) | TTP every chunk: argmax over Commit (policy's own chunk) / Reobserve (camera move + re-query) / Replan (planner re-grasp); N = 2-3; no detector | progress head on predicted 64-step endpoint latent; label = normalised object-to-goal distance (privileged geometry, not P(success)) | none (privileged oracle only as an upper bound: 31.7 vs 30.0) | none (frozen VLA; WM + head trained on sim "mining" branches) | RoboTwin 2.0 occl. / slip, 30 eps per cond.: pooled 13.3 -> 30.0 % (CI 19.9-42.5, overlaps fixed Replan 21.7); transfers to RDT / XR-0; real 4 settings | "will be released" | author homepage, alphaXiv; **note [`notes/2610.18650.md`](notes/2610.18650.md)** (overlap low-moderate: 2 SAME / 3 SIMILAR / 13 DIFFERENT; Phase 2 untouched) | (L7) |
| B3 | DreamSteer | [2607.02865](https://arxiv.org/abs/2607.02865) | 2026-07-03 | preprint | latent | TTP: VLA samples **+ motion primitives** | VLAC progress (InternVL2-2B) | none | none | 4 real benchmarks 23.75 -> 66.25 % | PP | project page | new |
| B4 | VeriVLA | [2610.21985](https://arxiv.org/abs/2610.21985) | 2026-10-15 | preprint | latent (JEPA) | TTP: two-axis trigger, rank chunks | predicted progress change | none | none | LIBERO + real (+14.7 pt) | PP | project page | new |
| B5 | World-in-Loop / VLA-in-the-Loop | [ECCV 2026 PDF](https://media.eventhosts.cc/Conferences/ECCV2026/pdfs/5018.pdf) | 2026 | ECCV 2026 (host PDF) | video (I2V) + VLM judge | TTP at grasp events: rollback to healthy state, imagine success, IDM decodes | VLM feasibility judge | none | none | SimplerEnv WidowX / Google Robot + real | - | blog note; **no arXiv ID found** | new |
| B6 | How Much Can Best-of-N Help a WAM? | [2610.37196](https://arxiv.org/abs/2610.37196) | 2026-10-27 | preprint | WAM (Cosmos Policy etc.) | TTP + measurement | linear readout of imagined future; fail-gated | **bit-exact branching, repeated continuations, held-out oracle** | none | RoboCasa, LIBERO-Plus; 76.9 -> 81.6 % | - | arXiv only | new |
| B7 | When Does Test-Time Search Help WAMs? (audit) | [2610.41970](https://arxiv.org/abs/2610.41970) | 2026-10-30 | preprint | WAM (Cosmos Policy) | TTP audit | value selector **worse than random** (-1.66 pp); 82 % of states identical outcomes | hindsight oracle, pre-registered | none | RoboCasa | scripts released (per abstract) | arXiv only | new |
| B8 | Latent Policy Steering (LPS) | [2507.13340](https://arxiv.org/abs/2507.13340), [2609.22521](https://arxiv.org/abs/2609.22521) | 2025-07-17 / 2026-09-18 | preprint | latent (Dreamer, optical-flow pretrain) | TTP | robust value that penalises distribution shift, trained on WM-simulated mistakes | none | none | Robomimic + real (DP, pi0.5) | PP | CMU RI page | new |
| B9 | FOREWARN (Foresight to Forethought) | [2502.01828](https://arxiv.org/abs/2502.01828) | 2025-02-03 | RSS 2025 (roboticsproceedings) | latent (Dreamer-v3) | TTP | VLM aligned to WM latents | none | none | real manipulation | GH | GH, RSS proc. | new |
| B10 | CoRe: Counterfactual Realignment | [2608.14822](https://arxiv.org/abs/2608.14822) | 2026-08-14 | preprint | synthesized obs (RGB-D + FK), not learned | TTP: imagine policy continuation from a viable state | rejoin feasibility | virtual probes | none | LIBERO-Long perturb., real | - | S2 | (F2) |
| B11 | B2FF: Back to the Familiar Future | [2606.09258](https://arxiv.org/abs/2606.09258) | 2026-06-08 | CoRL 2026 (comment) | foresight VLA milestones | TTP: recoverability-aware milestone selection | recoverability selector | none | none | failure-injected LIBERO 56.3 -> 74.0 | - | alphaXiv | (LRB) |
| B12 | FAVOR: Future Anchored Verification and Online Recovery | [2610.06280](https://arxiv.org/abs/2610.06280) | 2026-10-05 | preprint | WAM predicted frames as anchors | TTP: anchor verifier + VLM corrective instruction | anchor deviation | none | none | LIBERO 97.85 -> 98.10; LIBERO-Plus 72.60 -> 72.98 | - | S2 | new |
| B13 | DREAM-Chunk | [2606.18589](https://arxiv.org/abs/2606.18589) | 2026-06-17 | preprint | latent | TTP: switch to chunk whose dream matches reality | latent match | none | none | Kinetix + 4 real / sim tasks | - | arXiv only | new |
| B14 | GeoBoN (zero-shot geometric verification) | [2607.17454](https://arxiv.org/abs/2607.17454) | 2026-07-20 | CVPR 2026 EAI workshop ext. | WAM | TTP BoN | cross-view depth reprojection consistency (VGGT) | none | none | RoboCasa, LIBERO-Long, RoboTwin | - | arXiv only | new |
| B15 | WISE (illusion-aware selector for WAMs) | [2611.03585](https://arxiv.org/abs/2611.03585) | 2026-11-02 | CoRL 2026 workshop (comment) | WAM | TTP | IDM action-future consistency + progress | none | none | not checked (abstract only) | - | arXiv only | new |
| B16 | V-VLAPS | [2601.00969](https://arxiv.org/abs/2601.00969) | 2026-01-02 | preprint | sim (MCTS in simulator) | TTP | value head on Octo latents (MC returns) | sim | none | 5 LIBERO suites | - | arXiv only | new |
| B17 | Reflect-VLM; Value-guided multi-path reflection | [2502.16707](https://arxiv.org/abs/2502.16707), [2602.19372](https://arxiv.org/abs/2602.19372) | 2025-02 / 2026-02 | ICRA 2026 (2nd, comment) | video diffusion dynamics | TTP: imagine, critique, revise | VLM / goal-distance critic | none | VLM policy SFT | multi-stage assembly | PP | project page | new |
| B18 | SC-VLA; CycleVLA; DDP | [2602.21633](https://arxiv.org/abs/2602.21633), [2601.02295](https://arxiv.org/abs/2601.02295), [2603.21017](https://arxiv.org/abs/2603.21017) | 2026-01 / 02 / 03 | preprints | sparse imagination / none / WM-regularised DP | TTP self-correction | dense reward / VLM / imagination | none | SFT / none | LIBERO, LIBERO-Plus, MetaWorld | PP | arXiv only | new |

### Tier C: WM in the training loop of a VLA (RL or filtered data in imagination; recovery is a side effect)

| # | Paper | arXiv / link | v1 date | Venue | WM type | WM use | Scorer / critic | Verification | Train use | Benchmark | Code | 2nd src | in repo |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | WoVR (KIR: rollouts start near failure-critical states) | [2602.13977](https://arxiv.org/abs/2602.13977) | 2026-02-15 | preprint | video (Wan) | RL | success classifier / Qwen3-VL progress | WM-policy co-evolution | GRPO | LIBERO 39.95 -> 69.2; real +30 | GH (RLinf) | project page | (report) |
| C2 | WoVR+ | [2610.17922](https://arxiv.org/abs/2610.17922) | 2026-10-13 | preprint | video, multi-view | RL, stage-local resets | stage rewards | recalibration | GRPO | long-horizon sim + real | - | arXiv only | new |
| C3 | WMPO | [2511.09515](https://arxiv.org/abs/2511.09515) | 2025-11-12 | ICLR 2026 | video | RL | success classifier | policy behaviour alignment | GRPO; emergent self-correction | MimicGen, ALOHA | GH | ICLR proc. | (C2) |
| C4 | RISE | [2602.11075](https://arxiv.org/abs/2602.11075) | 2026-02-11 | RSS 2026 (comment) | video + value | RL | progress value | none | advantage-conditioned | 3 real tasks | GH | project page | (C1) |
| C5 | MeRL: Trust-Calibrated Post-Training with Evolving Imagination | [2610.16517](https://arxiv.org/abs/2610.16517) | 2026-10-12 | CoRL 2026 (project page only) | video + reward proxy | RL | progress proxy + chunk-level trust | online grounded data | RL with trust-weighted replay | LIBERO-PRO 79.7 %; real | PP | project page | new |
| C6 | HaWMPO | [2609.09941](https://arxiv.org/abs/2609.09941) | 2026-09 | preprint | video | RL | hallucination-aware reward | none | GRPO | LIBERO, G1 real | - | arXiv only | (report) |
| C7 | DRILL | [2610.30814](https://arxiv.org/abs/2610.30814) | 2026-10-22 | NeurIPS 2026 (comment) | video | RL; WM trained for policy utility | return after inner GRPO | target simulator return | GRPO | 5 sims, 2 VLAs, +13.1 over WMPO | - | alphaXiv | new |
| C8 | World-VLA-Loop | [2602.06508](https://arxiv.org/abs/2602.06508) | 2026-02-06 | preprint | video + reward head | RL | binary reward from diffusion latents | co-evolution | RL | LIBERO-Object, real | - | arXiv only | (report) |
| C9 | WISE: WM-guided Imagination Scheduling | [2609.03681](https://arxiv.org/abs/2609.03681) | 2026-09-03 | preprint | video, multi-view | DG at interaction-relevant states | progress + completion | none | action refinement | pi0 / pi0.5 sim + real | - | arXiv only | new |
| C10 | Ctrl-World / VLAW / WEAVER | [2510.10125](https://arxiv.org/abs/2510.10125), [2602.12063](https://arxiv.org/abs/2602.12063), [2606.13672](https://arxiv.org/abs/2606.13672) | 2025-10 / 2026-02 / 2026-06 | ICLR 2026 (Ctrl-World, notes) | video / latent | DG (success-filtered) | human / VLM P(yes) > 0.8 / reward + critic head | real anchors | SFT / weighted FM | DROID real | GH | project pages | (G2, report) |
| C11 | RoboCoach | [2609.39685](https://arxiv.org/abs/2609.39685) | 2026-09-30 | preprint | video (CoachWorld) | EV of imagined failures -> which demos to request | progress judge | real deploy corr. rho 0.84 | expert adapters | 2 sims + 2 real | PP | S2 | new |
| C12 | XPACE | [2609.17372](https://arxiv.org/abs/2609.17372) | 2026-09-15 | preprint | joint world-action model | DG: deviation-recovery around expert demos | filtered | none | SFT | IRON humanoid | - | S2 | new |
| C13 | World2Act | [2603.10422](https://arxiv.org/abs/2603.10422) | 2026-03-11 | preprint | video WM | residual policy aligned to WM dynamics | contrastive, reward-free | none | residual | LIBERO, RoboCasa | PP | arXiv only | new |
| C14 | Failure-Aware RL (FARL) | [2601.07821](https://arxiv.org/abs/2601.07821) | 2026-01-12 | preprint | latent + constraint head | safety filter -> recovery policy | constraint-violation risk | real | offline-to-online RL | real manipulation | PP | emergentmind | new |
| C15 | Simulation Distillation (SimDist) | [2603.15759](https://arxiv.org/abs/2603.15759) | 2026-03-16 | RSS 2026 (comment) | latent, sim-pretrained | WM pretrain on perturbed failure / recovery rollouts | expert value | real adaptation | planning | real | - | arXiv only | new |

### Tier D: sim-verified recovery data and distillation without a learned WM (threatens "verified SFT" novelty)

| # | Paper | arXiv / link | v1 date | Venue | How recoveries are made | Verification | Train use | Benchmark | Code | 2nd src | in repo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | PLD: Probe, Learn, Distill | [2511.00091](https://arxiv.org/abs/2511.00091) | 2025-10-30 | ICLR 2026 (proceedings) | residual RL specialists take over at base-policy states | sim success | SFT of the VLA | LIBERO 99 %, SimplerEnv | PP | ICLR proc. | new (spirit = our B3.3) |
| D2 | SimPT | [2610.29161](https://arxiv.org/abs/2610.29161) | 2026-10-21 | preprint | CuRobo expert from retained snapshots, falls back to earlier stages | sim success | intervention-based post-training | sim VLA | - | arXiv only | new |
| D3 | Needlework | [2610.02339](https://arxiv.org/abs/2610.02339) | 2026-10-01 | ICLR 2027 submission (comment) | goal-conditioned IDM bridges between logged observations (incl. failed episodes) | learned action-conditioned reachability verifier | DP training on bridges + logged data | robomimic, real (+21 pt) | GH | GH, project page | new |
| D4 | Generative Failure-to-Success Bridging | [project page](https://junhahyung.github.io/FailureToSuccess/) | 2026 | WIP, no arXiv | VLM stage parse + success-suffix retrieval + IDM bridge + WM fills frames | endpoint consistency | planned | RoboCasa (WM part only) | - | **unverified** (author page only) | (notes) |
| D5 | Does a Learned Corrector Beat a Simple Retreat? | [2610.06921](https://arxiv.org/abs/2610.06921) | 2026-10-02 | preprint | learned runtime corrector vs scripted retreat on frozen pi0.5 | **paired placebo re-run**; 43 % of base failures succeed on a plain re-run | none | RoboTwin, 3,888 paired episodes | - | S2 | new |
| D6 | RoboAware | [2610.11480](https://arxiv.org/abs/2610.11480) | 2026-10-08 | preprint | state-locked counterfactual branching over skill families | sim execution | MCTS + Q-learning coordinator | RoboSuite, LIBERO-Pro, RoboTwin | - | S2 | new |
| D7 | (already listed) Backtrack, RESETTLE, SafeLoop, Recova, FLARE, CARE ([2609.24118](https://arxiv.org/abs/2609.24118)), VLA-Corrector, RoboRecover | see `LIBERO_RECOVER_BASELINES.md` | | | | | | | | | CARE new |

### Tier E: WM reliability and exploitation (bears on RQ3 / over-optimism under search)

| # | Paper | arXiv / link | v1 date | Venue | Finding relevant to RetryVLA | 2nd src | in repo |
|---|---|---|---|---|---|---|---|
| E1 | CureWM | [2610.09134](https://arxiv.org/abs/2610.09134) | 2026-10-06 | preprint | released robot WMs predict success on 80 % of verified LIBERO failures; fix by execution-verified counterfactual fine-tuning; passive rollouts, not search | GH, S2 | (G1) |
| E2 | Learning Counterfactual World Models (CLWM) | [2609.05834](https://arxiv.org/abs/2609.05834) | 2026-09-05 | NeurIPS 2026 workshop (comment) | contrastive counterfactual objective cuts "exploitative planning failures" 18.4 -> 9.7 % | arXiv only | new |
| E3 | RENEW | [2607.14180](https://arxiv.org/abs/2607.14180) | 2026-07-15 | preprint | repairs model exploitation with human preferences over imagined rollouts (non-robot envs) | arXiv only | new |
| E4 | OpTTA | [2610.35636](https://arxiv.org/abs/2610.35636) | 2026-10-26 | CoRL 2026 workshop (comment) | deliberately *optimistic* test-time WM adaptation helps JEPA planning; opposite sign of our concern | arXiv only | new |
| E5 | Do Robotic WMs Really Follow Actions? (WorldEcho / WorldSync) | [2608.24885](https://arxiv.org/abs/2608.24885) | 2026-08-25 | preprint | expert-trained WMs fail on off-expert actions (exactly the bridge regime) | S2 | new |
| E6 | QWM: Q-Learning With World Models | [2608.17163](https://arxiv.org/abs/2608.17163) | 2026-08-17 | preprint | search over imagined trajectories on top of Q-learning; policy / value trained only on real transitions to avoid compounding WM bias | alphaXiv | new |
| E7 | IPD: Imaginary Planning Distillation | [2603.04289](https://arxiv.org/abs/2603.04289) | 2026-03-04 | preprint | WM + uncertainty-filtered MPC rollouts distilled into a policy (D4RL; offline RL analogue of WM-search-then-distill) | arXiv only | new |

## 2. The closest papers: exact overlap and what is left

**A1. Back on Track (2610.19579).** *Overlap*: almost the whole RetryVLA search stage. A small latent WM (frozen
image encoder + proprio state) is trained on the frozen policy's own rollouts plus play data; a policy-conditioned value
learned from a few hundred episode-outcome-labelled rollouts (liveness formulation of A2, ensemble) is both the trigger
and the CEM objective; the planner "does not seek to complete the task on its own; it searches for states that are
favorable under the subsequent behavior of pi_b", runs a short horizon and hands control back. That is our C^pi + short
bridge + hand-back, and our "frozen VLA" framing. They also name the planner exploiting WM errors as a limitation.
*Left*: (i) they act at deployment with no verification; we verify every bridge by the policy's own continuation in a
restored sim state and distill, so the deployed policy has no WM / critic / monitor (nominal kept within ~2 pt);
(ii) failure states from LIBERO-Recover (L1-L3, displaced / knocked objects) rather than perturbed LIBERO-Goal inits;
(iii) our measurements of optimism under search (RQ3) and of library vs native proposals. Their result is our strongest
"why not just plan at test time?" reviewer question: we must report a test-time arm with our own C^pi + WM.

**A3. SAILOR (2506.05294, NeurIPS 2025).** *Overlap*: learned latent WM + learned reward + critic ensemble with an
uncertainty penalty (our U_ens idea), sampling-based search around the base policy's plan "to recover from mistakes
the base policy makes", and expert-iteration distillation of the search back into the policy; they also scale the
test-time budget 8x and report no reward hacking. *Left*: their reward is a discriminator "expert vs learner state"
(closer to "return to demo support" than to P(pi succeeds)), search is MPC at test time with distillation as an add-on
(the distilled policy loses on the hardest task), there is no per-sample verification, policies are task-specific
Diffusion Policies, not a VLA, and the benchmarks are nominal robomimic / robosuite / ManiSkill. Their "robust to hacking
at 8x" contradicts our RQ3 finding of strong optimism on search-selected bridges; the likely reasons (small residuals
around the base plan, discriminator reward, online WM refits) are worth one sentence in the paper.

**A4. Recursive Self-Improvement through Local Recovery Supervision (2610.05151).** *Overlap*: restore a sim state
before the failure, generate a correction, accept it only if **the frozen student policy can continue from the endpoint**
(else restore and extend the teacher segment), distill, repeat in rounds; evaluated with pi0 on LIBERO-Goal and with
compute-matched "original-data continuation" controls. This is our verification rule (bridge + continuation of pi) and
our distillation recipe. *Left*: no WM, no learned critic and no search: recoveries come from an LLM tool-using teacher;
small preliminary study (88 vs 78 / 100). It removes "verify by the policy's own continuation" as a novelty claim; our
contribution has to be selection (WM + C^pi) at scale and the analysis.

**A5. Kintsugi-VLA (2609.31048).** *Overlap*: exact sim restore + branching, Monte-Carlo recoverability with Wilson
intervals, failed rollouts turned into recovery SFT data for SmolVLA, clean-task success reported. *Left*: recoverability
is for a privileged expert, used to pick *where* recovery starts; recoveries come from that expert, no WM, no search,
one Isaac Lab task. RetryVLA scores *bridge end states* by the deployed policy's own competence. Their idea is the
cleanest non-WM baseline (already B-3 in `LIBERO_RECOVER_BASELINES.md`).

**A6. Rollback (2610.36068).** *Overlap*: training-time search inside a WM from real success / failure trajectories,
rewinding to decision points and branching (our "one failure state, many bridges"), outcome-scored leaves turned into
action-level labels and distilled into the policy, with real data refitting the WM. *Left*: success judge on full
rollouts instead of a policy-competence critic on short bridges; no per-sample ground-truth verification (WM verdicts
become labels, the risk we measured as RQ3); not recovery-specific; real tasks, no LIBERO-Recover.

**A7. TACO (2607.02840).** *Overlap*: failure-adjacent states found with a value model, local corrections imagined in a
WM, candidates filtered for feasibility and ranked by predicted value gain, aggregated with demos for iterative
post-training of the VLA. This is our "WM-generated recovery -> filter by critic -> train" loop. *Left*: the critic is
task progress (not policy competence), the WM generates the correction (IDM labels actions) instead of scoring proposed
action bridges, verification is plausibility checks plus real replay of a subset, training is CFG-RL; tactile, real only.

**A8 + A9 + A10. Hi-WM, WM-DAgger, Dream2Fix (already in RELATED_WORK, re-checked).** Together they cover "WM as a
corrective substrate": branching corrections from a failure state inside the WM (Hi-WM, human-labelled), deviate-and-
return recovery synthesis with a consistency filter that is critical (WM-DAgger, 96.7 -> 46.7 % without the filter per
our report), and counterfactual failure + correction pairs with learned verifiers (Dream2Fix). None scores corrections by
the policy's own competence or verifies them in ground truth; all are real-robot, none uses LIBERO-Recover.

**B6. How Much Can Best-of-N Help a WAM? (2610.37196).** *Overlap* is methodological and on a result, not on the
method: bit-exact MuJoCo snapshots, repeated continuations and held-out oracles to measure what a selector can really
realise; they find native policy samples have almost no selectable headroom while structured displacements of the
policy chunk create 4.5-16 pt, i.e. our M11 result "proposal library >> VLA samples / WM planning from noise", and that
imagined futures carry ranking signal that the value head leaves unused. *Left*: test-time selection for nominal tasks,
no recovery data, no distillation. Cite it as independent support for our proposal library and for paired, denoised
evaluation (and 2610.06921 for paired placebo re-runs: 43 % of base failures succeed on a plain re-run).

**B2. ForkCast (2610.18650), read in full 2026-11-23** (details and component table in
[`notes/2610.18650.md`](notes/2610.18650.md)). *Overlap*: a V-JEPA 2-AC latent WM (the same family as our drop-in WM)
is trained on sim branches that execute every candidate from the same state. It compares the policy's own chunk
(Commit) against scripted repairs (camera move, planner re-grasp) and hands back to the frozen VLA, with no detector.
*Left*: its scorer is a geometric progress proxy at a 64-step horizon, not an outcome / competence critic, which is
our losing "progress" arm. Search pressure is N = 2-3 (near its oracle, 30.0 vs 31.7), there is no GT verification,
no distillation ("does not add new manipulation skills"), and failures are injected, not natural. Our V-JEPA 2-AC FP
curve (~0.70 at N = 8, ~0.84 at N = 512) explains why low-N selection works without verification and large-N data
search does not. Action: a ForkCast-style test-time row (Commit + primitives; progress head vs C^pi; N = 3 vs 512).

**Test-time cluster (B1-B5, B8-B9).** DreamAvoid, ForkCast, DreamSteer, VeriVLA, World-in-Loop, LPS and FOREWARN all
keep the VLA frozen and rank candidate chunks by WM-imagined outcomes with a progress / value / VLM scorer; ForkCast
explicitly compares the policy's own continuation against repairs, and DreamSteer mixes VLA samples with motion
primitives (our library idea). None verifies or distils; together they make "WM steering of a frozen VLA after
deviation" a standard baseline category that reviewers will expect us to compare with or argue against.

## 3. Which RetryVLA claims are taken and which remain defensible

Our own numbers below are from `experiments/RESULTS.md` (Spatial test = dev split; see the caveat there).

| RetryVLA claim | Status | Who takes it / evidence | What we can still say |
|---|---|---|---|
| Use a WM to search corrective actions after a VLA failure | **taken** | Back on Track, SAILOR, DreamAvoid, ForkCast, World-in-Loop, DreamSteer, VeriVLA, Rollback | nothing as a novelty claim; it is the setting |
| Policy-competence critic C^pi(s) = P(pi finishes from s), learned from the policy's own continuations, as the search objective ("bring the robot back where pi can continue") | **taken in substance** | Back on Track (policy-conditioned liveness value, "states favorable under the subsequent behavior of pi_b"), liveness OPE (RSS 2026), Kintsugi (MC recoverability, expert); ForkCast compares the policy continuation vs repairs but scores with a geometric *progress* head (2026-11-23 re-read), so it does not take the competence critic | only the *matched-budget evidence*: outcome-supervised critic >> goal-latent and progress value (search solved 0.48 vs 0.24 goal / 0.21 progress; SFT RSR 0.42-0.43 vs 0.19 goal / 0.18 progress). Caveat from our own data: a generic success critic trained on pi0 + pi1 + demo continuations (B4) is as good or better (solved 0.56, SFT 0.427), so we cannot claim that *policy specificity* matters; the honest claim is "outcome-labelled continuation critic vs task-progress critic" |
| Short bridge then hand back to the frozen policy | **taken** | Back on Track, CoRe, Backtrack, FAVOR, World-in-Loop | - |
| One failure state, many branched corrections | **taken** | Hi-WM, Rollback, Kintsugi, 2610.37196, RoboAware | - |
| Accept a correction only if the frozen policy's continuation succeeds (sim-verified labels) | **taken as a rule** | 2610.05151 (frozen student continuation test), Kintsugi (expert MC), PLD / SimPT (sim success) | the combination with WM + C^pi selection over a large proposal pool, and the measured cost / precision trade-off (3.3x less sim than sim-prefix scoring; hits per verification) |
| Distill search into the VLA instead of planning at test time; no test-time module | **partly taken** | SAILOR (ExIt; distilled DP loses on Square), Rollback, TACO, PLD, WM-DAgger, Hi-WM | the recovery-specific *distill vs plan* comparison at equal compute on LIBERO-Recover with nominal preserved (pi1 nominal 0.785 vs 0.796 reference) is not reported anywhere; we have only online best-of-8 on pi1 (0.466 vs 0.418, p 0.26), not a Back-on-Track monitor + CEM arm |
| SFT on verified data beats imagination-only RL and residual RL at equal sim budget; RL collapses nominal | **partly corroborated by others** | RISE (offline-ratio collapse), PLD (residual RL + SFT works), WMPO / WoVR (RL in WM works) | still a useful controlled result on recovery; frame as a finding, not a contribution |
| WM over-optimism under search, quantified | **open, closest work is passive** | CureWM (passive optimism 80 %), 2610.41970 (selector worse than random), 2610.37196 (headroom overstated), CLWM / RENEW (exploitation in toy envs), Back on Track / SAILOR (qualitative; SAILOR reports robustness at 8x) | **defensible**: on the same 16,384 sim-verified bridges, search-selected picks have imagined C^pi 0.73 vs true-state 0.27 vs real SR 0.17; 80-85 % of the gap is the WM image tokens, the critic adds the rest; random picks show no gap; latent error is not larger on picks (the error is adversarially aligned, not larger). Cross-WM agreement (WM v1 x V-JEPA 2-AC) lowers FP 0.82 -> 0.64-0.70 with only a small solved-rate change (0.37 / 0.40 single WM vs 0.43 agreement, train split). Nobody reports this decomposition |
| Proposal library >> VLA samples / WM planning from noise | **independently found by others** | 2610.37196 (structured displacements create 4.5-16 pt headroom, native samples ~0), DreamSteer (primitives + samples) | cite as agreement; not a contribution |
| Action-free imagined futures as an auxiliary loss (L_future), geometry energy, recursive B_k chains | **not done by others; ours are negative** | FOCA / ThinkWorld / FLARE-implicit (positive, nominal); GeoBoN (geometry helps BoN for WAMs); 2610.05151 ("recursive" = rounds, not chains) | negative results are defensible as analysis (geometry raises precision per verification +36 % but not solved rate; recursion costs budget), not as headline |
| Active Causal Recovery (probe actions before imagining) | **not found** in the WM-recovery literature swept here | nearest: RoboAware (counterfactual branching over skill families), World-in-Loop (rollback at grasp events) | potentially novel, but the sweep did not cover interactive-perception / active-diagnosis robotics; run a targeted search before claiming (ACR-N still running, no result yet) |
| First method results on LIBERO-Recover | **still true** (as of 2026-11-16) | search found no other method numbers; RoboRecover is a competing benchmark | keep, and add RoboRecover later |

## 4. Recommended positioning
- **Do not claim**: first WM-based recovery planner for VLAs; first policy-conditioned value; first search-then-distill;
  first verification by policy continuation. Say instead that RetryVLA builds on Back on Track's insight (plan toward
  states where the frozen policy can finish, scored by a value learned from its own rollouts) and asks a different
  question: *can this be done once, offline, with ground-truth verification, and amortised into the policy?*
- **Lead with** (1) amortisation: distill verified policy-conditioned recoveries into the VLA, no test-time WM / critic /
  monitor, nominal kept, first method numbers on LIBERO-Recover; (2) measurement: how much imagined recovery survives
  verification under search pressure (optimism decomposition, FP / ECE vs selection, hits per verification, cross-WM
  agreement), which explains why test-time WM planning without verification is fragile and why verification is the
  bottleneck; (3) controlled comparisons of the scoring signal and of the update rule at matched sim budget.
- **Recast the critic claim**: "outcome-labelled continuation critic vs goal / progress critics", acknowledging that a
  generic success critic works as well in our data (so "policy competence" is the motivation, not a proven necessity).
- **Title / abstract wording**: avoid "world-model-guided recovery" as the novelty; use something like
  "Verified, policy-conditioned recovery data for VLAs" or "Amortising world-model recovery search into the policy".
- **Must-cite set** (beyond what RELATED_WORK already cites): Back on Track 2610.19579, liveness OPE 2605.11479,
  SAILOR 2506.05294, 2610.05151, Rollback 2610.36068, TACO 2607.02840, DreamAvoid 2605.11750, ForkCast 2610.18650,
  DreamSteer 2607.02865, World-in-Loop (ECCV 2026), 2610.37196, 2610.41970, PLD 2511.00091, 2610.06921.

## 5. Experiments the overlap makes necessary (no GPU run for this note)
1. **Back-on-Track-style test-time arm** with our WM + C^pi + CEM on pi0 / pi1: monitor (C^pi threshold) -> plan h
   steps in the WM -> hand back, on Spatial and Object test RSR, with nominal and latency. Report "plan at test time"
   vs "distill" vs "both" (pi1 + planner). This is the first reviewer question.
2. **SAILOR / Rollback-style unverified distillation**: distill WM-selected bridges *without* sim verification at the
   same count (our imagination-only arm B3.2 is close; make the comparison explicit, and add a WM-verdict-labelled arm).
3. **2610.05151-style acceptance only**: random or library proposals + continuation acceptance, no WM / critic, at the
   same verification budget (our "random-data SFT" row 0.252 is this; report it under that name).
4. **Liveness value vs our C^pi classifier**: train the discounted liveness value of 2605.11479 on the same
   continuations and swap it in as the critic (cheap, offline) so the critic choice is not a reviewer attack surface.
5. Keep the paired-placebo discipline (2610.06921, 2610.37196): pi1 vs pi1 re-run controls on the same seeds.

## 6. Caveats
- Several "venue" fields come from arXiv comments or project pages, not proceedings (marked). Only SAILOR (NeurIPS
  2025), PLD / WMPO (ICLR 2026), FOREWARN (RSS 2025) and liveness OPE (RSS 2026) were seen on proceedings pages.
- Numbers come from abstracts / HTML text; we did not reproduce them. Back on Track's internal ablations (e.g. liveness
  vs progress value) were not fully read; we found no such ablation in the parts we read.
- The sweep is broad but not exhaustive: Semantic Scholar was rate-limited for some citation queries, the arXiv API
  was unavailable, and ICLR 2027 OpenReview pages did not render. Papers posted after ~2026-11-10 may be missing.

