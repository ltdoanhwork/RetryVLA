# CRIT-V: critic variants for RetryVLA (explicit 3D / chain-of-dynamics / implicit) + rich verification records

Pre-registration: `experiments/PLAN.md` section "CRIT-V" (committed 2b3b110, before any result).
Code: `retryvla/critic/{rich.py, variants.py, train_variants.py, eval_variants.py}`, `scripts/crit_rich_replay.py`,
opt-in hooks `retryvla/envs/vec.py` (`record_cfg["rich"]`) and `retryvla/search/compare_arms.py` (`m5b.rich_record`).
Compute: own 1-GPU shell `a3da0634` (user account, cn28), killed when done. No GPU of the Goal campaign (953bb534) or
of BoT (92927808) was used.

## 0. Bối cảnh (why)

C^pi = transformer trên 72 token DINOv2 (pool 6x6, 2 camera) + token proprio + token task -> P(pi hoàn thành | s),
học từ outcome của chính pi. Trong search, C^pi chấm trạng thái cuối do WM tưởng tượng. RESULTS "RQ3 follow-up": trên
các candidate được search chọn, ~80-85 % độ lạc quan (pred - SR) là do WM (ảnh tưởng tượng sai), ~15-20 % là do chính
critic (critic trên TRUE end state vẫn lạc quan +0.05..+0.11, FP 0.61-0.75). Câu hỏi ở đây: thay đổi CRITIC (thêm 3D,
thêm chuỗi suy luận động lực, thêm mục tiêu động lực ngầm) có giảm phần lỗi của critic và cải thiện xếp hạng trong
từng instance (p@1 / p@4, FP@top-4) không, trên trạng thái tưởng tượng và trên trạng thái thật?

## 1. Designs

All variants share the C^pi input (end-state tokens z_H + q_H + task), recipe (width 256, depth 3, dropout 0.1, AdamW
2e-4, wd 0.05, bs 128, 8,000 steps, 50 % WM-imagined end states per batch, pos_weight neg/pos), the SAME batches and
view masks per seed (3 seeds), and the same training data. Only the extra tokens / heads / losses differ.

| kind | what is added | what it tests |
|---|---|---|
| `base` (CONTROL) | nothing (= RecoveryValue retrained by the same script) | reference; isolates "a different critic" from "a better critic" (independence effect on picks chosen by the shipped C^pi) |
| `pc_true` (1a) | 20 point tokens: 16 local tokens (FPS centres, 32-NN groups, PointNet on [local xyz, class one-hot]) + 4 per-class global tokens, from the TRUE labelled cloud of the state | does explicit 3D help when it is exact? On `imag` the cloud is still the TRUE one (imagined image tokens + true 3D) = upper bound of "3D at search time" |
| `pc_pred` (1b) | same 20 tokens PREDICTED from (z, q) by a 2-layer token predictor, trained jointly with MSE to the frozen `pc_true` tokens (same seed) on real and imagined latents | can 3D be recovered from the WM latent for imagined states (RGR-geometry-head analogue)? |
| `cot` (2) | 4 causal chain queries: contact (finger contact, both fingers, grasped, robot touches other objects) -> object state (height, up-axis, off table, penetration) -> relation (gripper-object dist, object-goal contact, object-goal xy / z offset) -> BDDL goal predicates (each atom + fraction); query k attends to state tokens and queries < k; supervised with sim labels; success head = MLP([CLS, embedding of the predicted chain]) | does routing the success estimate through explicit, physically checkable intermediate quantities improve calibration / FP? |
| `cot_nochain` | same architecture, chain loss weight 0 | capacity control for `cot` |
| `impl` (3) | (i) imagination consistency: align(feature(imagined end)) -> stop-grad feature(TRUE end) of the same candidate (cosine); (ii) latent dynamics in the critic's own space: MLP(feature(start state), embedding(50x7 actions)) -> stop-grad feature(true end); (iii) distributional time-to-success head (bins 50 / 100 / 150 / 200 / 250 / >250 env steps + fail, CE) | does an implicitly dynamics-aware, imagination-robust embedding shrink the imagined-vs-true gap without explicit 3D or symbols? |
| `impl_noalign` | (ii) + (iii) only | isolates the imagination-consistency term |

Why this IMPLICIT design (thiết kế tự chọn): the RQ3 diagnosis says the critic's error on search picks is mostly that
imagined tokens point "the way C^pi rewards". A critic whose features are (i) invariant to the WM's imagination error
for the same candidate and (ii) predictable from start state + actions (a TD-MPC / SPR-style self-predictive latent)
should be harder to fool by imagination artefacts; (iii) is the cheapest "distributional value" signal available in
the logs (success_step). TD bootstrapping across states of the same episode was rejected: the logs contain no
continuation states (bridge-only), and bridge-intermediate states are not on pi's own trajectory, so V(s_t) along an
open-loop bridge is not P(pi finishes | s_t) (it would mix the bridge's remaining actions into the target).

### 1c. PointWorld checkpoint (feasibility, documented, not run)
- PointWorld: arXiv 2601.03782 (NVlabs; code `github.com/NVlabs/PointWorld` Apache-2.0; checkpoints
  `huggingface.co/nvidia/PointWorld_models` small-droid / large-droid / large-droid+behavior under the NVIDIA Open
  Model License = permissive for research / derivatives, commercial allowed with attribution terms).
- Inputs: calibrated RGB-D (scene points featurised with frozen DINOv3), robot joint-space actions turned into robot
  point flows via URDF forward kinematics, H = 10 steps x 0.1 s per forward pass; output per-point 3D flow.
- On our cameras: LIBERO gives exact depth / intrinsics / extrinsics and the robot is a Franka Panda (DROID is Franka
  + Robotiq 2F-85: different gripper geometry, URDF swap needed). Our actions are 20 Hz OSC deltas (H = 50 -> 2.5 s =
  25 PointWorld steps = 3 chained forward passes; joint targets must be obtained by running the OSC controller
  open-loop in sim or by IK). Domain gap: DROID / BEHAVIOR scenes vs LIBERO renders at 128-256 px.
- Verdict: licence OK, technically feasible but a project of its own (URDF / gripper swap, OSC -> joint trajectory,
  chained horizons, ~1 B-param inference on 256-512 candidates x 3 passes per instance). It would replace the WM for
  3D imagination rather than be a critic feature; parked as a Phase-2 option. The `pc_true` upper bound below tells
  whether it is worth it: if exact 3D of the end state does not help the critic, a 3D WM cannot help through it.

### 1d. ThinkWorld
ThinkWorld = arXiv 2610.37666 (verified on arxiv.org/abs): Chain-of-Dynamics = discrete motion tokens between the
current and a future state, future DINO features predicted conditioned on the chain, GRPO reward from imagined-vs-
observed agreement. Our `cot` borrows the "reason through intermediate dynamics before the outcome" structure, but
the chain is symbolic sim state (contacts, object state, predicates) of the END state, not motion tokens, and there
is no RL stage (deviation: our logs have no continuation timeline, see section 2).

## 2. Data recorded (part 4)

**Retro-fit (bridge-only replay, `scripts/crit_rich_replay.py`)**: every logged 50-step bridge of the five banks is
re-executed open-loop from its exact failure snapshot (no policy; restore is bit-exact: eef xy replay vs log <= 1.6 mm)
with the rich recorder: symbolic state (RGR geometry 16 + contacts / grasp 5 + up to 3 BDDL goal atoms + fraction =
25 dims) at t = 0, 5, ..., 50 and labelled point clouds (512 points, fp16 xyz in the table frame + uint8 class) at
t = 0, 10, ..., 50, both cameras (agent view + wrist, 96 px depth + segmentation). Outcome labels are the logged
verified outcomes. Region-type objects of interest (LIBERO-Goal drawers, stove, rack, cabinet top) are resolved to their
owning body (point class) and site position (symbolic).

**New verification runs (`m5b.rich_record: true`, compare_arms / goal search, default off)**: per verified candidate
`data/.../<instance>.pt` with arm / slot / policy seed, actions, full proprio, pooled DINOv2 latents every 10 steps
(2 x 384 fp16), symbolic timeline every 5 steps over the WHOLE episode (bridge + pi continuation), point clouds at
t = 0..50 / 10 and every 50 continuation steps, outcome / success step. This is the pi-future chain the `cot` variant
could not have here (Phase 2).

Sizes / replay checks: 33,296 bridges re-simulated (goal_collect 5,376, goal_verif 5,376, collect_v1 4,608,
verif_r1 2,552, verif_m5b 16,384), eef xy replay vs log max 2.7 mm (p99 1.6 mm), prefix-success agreement 100 % in every
bank. Disk: `data/critic3d` 734 MB (budget 15 GB); 45 checkpoints + preds 0.5 GB (`runs/20261124_CRITV_{goal,spatial}`).
Instrumentation smoke (`m5b.rich_record`, 1 Goal entry, pi0 continuation): format `critv_rich_v1` written, then deleted.

## 3. Results (offline, 3 seeds, instance-cluster bootstrap 400 reps; full table `docs/research/critv/critv.md` (copy of runs/_analysis/critv))

Eval = held-out verified banks: Goal `goal_verif` (168 entries x 32, SR 0.203; WM-search picks SR 0.243),
Spatial `verif_m5b` (256 x 64, SR 0.105). `imag` = WM-imagined end state (search-time view), `true` = real end state.
p@k = rank ALL verified candidates of an instance by the critic (search pressure), success rate of the top-k.
d = variant - `base` (retrained control), seed-mean, 95 % CI.

**PRIMARY: Goal, imagined end states**

| kind | AUC | ECE | p@1 | p@4 | FP@top4 | pred - SR (picks) | dAUC [CI] | dFP@top4 [CI] | dp@4 [CI] |
|---|---|---|---|---|---|---|---|---|---|
| C^pi shipped (ref) | 0.639 | 0.361 | 0.208 | 0.213 | 0.783 | +0.373 | - | - | - |
| base (control) | 0.714 | 0.225 | 0.232 | 0.246 | 0.682 | +0.076 | - | - | - |
| pc_true (3D oracle) | 0.702 | 0.218 | 0.220 | 0.235 | 0.679 | +0.067 | -0.012 [-0.027, +0.002] | -0.003 [-0.029, +0.027] | -0.011 [-0.031, +0.007] |
| pc_pred | 0.686 | 0.225 | 0.226 | 0.231 | 0.700 | +0.071 | -0.028 [-0.046, -0.012] | +0.018 [-0.006, +0.044] | -0.015 [-0.031, -0.000] |
| cot | 0.707 | 0.224 | 0.244 | 0.246 | 0.676 | +0.080 | -0.007 [-0.024, +0.009] | -0.006 [-0.030, +0.018] | +0.000 [-0.018, +0.019] |
| cot_nochain | 0.685 | 0.228 | 0.222 | 0.233 | 0.706 | +0.084 | -0.029 [-0.044, -0.017] | +0.024 [-0.003, +0.050] | -0.012 [-0.030, +0.003] |
| impl | 0.711 | 0.221 | 0.244 | 0.235 | 0.708 | +0.073 | -0.003 [-0.017, +0.013] | +0.026 [+0.002, +0.052] | -0.010 [-0.026, +0.004] |
| impl_noalign | 0.705 | 0.211 | 0.214 | 0.238 | 0.680 | +0.042 | -0.009 [-0.024, +0.003] | -0.002 [-0.029, +0.023] | -0.008 [-0.025, +0.010] |

Goal, TRUE end states: base AUC 0.771 / p@4 0.286 / FP@top4 0.540; shipped C^pi 0.718 / 0.284 / 0.611; every variant
dAUC in [-0.024, -0.006] (pc_true -0.013 [-0.026, -0.001]); impl / impl_noalign FP@top4 -0.028 / -0.026 (CI includes 0).

**Spatial (verif_m5b), imagined / true**

| kind | AUC imag | p@4 imag | FP@top4 imag | pred - SR picks | dAUC imag [CI] | dFP@top4 imag [CI] | AUC true | dAUC true [CI] |
|---|---|---|---|---|---|---|---|---|
| C^pi v2 shipped (ref) | 0.764 | 0.194 | 0.791 | +0.433 | - | - | 0.833 | - |
| generic success critic (B2, ref) | **0.828** | 0.211 | **0.638** | **+0.035** | - | - | 0.840 | - |
| base | 0.815 | 0.219 | 0.720 | +0.164 | - | - | 0.844 | - |
| pc_true | 0.817 | 0.234 | 0.711 | +0.140 | +0.003 [-0.004, +0.008] | -0.009 [-0.032, +0.010] | 0.842 | -0.002 [-0.008, +0.002] |
| pc_pred | 0.823 | 0.233 | 0.692 | +0.148 | **+0.008 [+0.002, +0.014]** | **-0.028 [-0.048, -0.010]** | 0.846 | +0.001 [-0.004, +0.008] |
| cot | 0.816 | 0.206 | 0.742 | +0.177 | +0.001 [-0.007, +0.009] | +0.022 [+0.003, +0.042] | 0.852 | +0.008 [+0.002, +0.014] |
| cot_nochain | 0.807 | 0.212 | 0.734 | +0.174 | -0.007 [-0.013, -0.002] | +0.015 [-0.007, +0.033] | 0.838 | -0.006 [-0.012, +0.000] |
| impl | 0.800 | 0.221 | 0.718 | +0.142 | -0.015 [-0.026, -0.005] | -0.002 [-0.025, +0.015] | 0.814 | -0.030 [-0.038, -0.021] |
| impl_noalign | 0.802 | 0.211 | 0.723 | +0.133 | -0.013 [-0.021, -0.005] | +0.003 [-0.020, +0.023] | 0.828 | -0.016 [-0.024, -0.009] |

Chain read-out (`cot`, AUC vs sim labels, true / imag): Goal finger contact 0.95 / 0.89, grasped 0.95 / 0.89,
object-goal contact 0.82 / 0.72, BDDL goal atom 0.86 / 0.83; Spatial grasped 0.96 / 0.86, goal atom 0.95 / 0.91.

## 4. Reading (đọc kết quả)

1. **Pre-registered decision: no variant "clearly wins" or is "promising" on the PRIMARY set** (Goal / imag): every
   dAUC <= 0 (-0.003 .. -0.029), no dFP@top4 <= -0.05. -> no Goal search pilot, nothing handed to the campaign.
2. **The biggest effect is not a variant but retraining**: the retrained control beats the SHIPPED C^pi on imagined
   search picks by a lot (Goal AUC 0.714 vs 0.639, pred - SR on picks +0.08 vs +0.37; Spatial +0.16 vs +0.43). This is
   mostly the independence effect (the picks were selected by the shipped critic itself, so its errors were chosen
   adversarially) plus all of goal_collect used for training (shipped held out the WM val episodes). It confirms the
   RQ3 / xwm reading: over-optimism on picks belongs to the scorer that did the selecting; any fresh critic looks better
   on them. Every variant must therefore be compared with `base`, never with the shipped C^pi.
3. **Explicit 3D does not add information the critic lacks**: even the oracle (`pc_true`, exact end-state cloud) is
   flat (Goal imag -0.012, true -0.013; Spatial +0.003 / -0.002). Pre-registered falsification of H1 for this data size
   is met on Goal. `pc_pred` gives the only positive CI anywhere (Spatial imag dAUC +0.008, FP@top4 -0.028, p@4 +0.014)
   but it is far below the rule and reverses on Goal (-0.028). Consistent with RGR: geometric failure is rare and the
   DINOv2 tokens already encode contact / placement (the `cot` read-out of contacts from the same tokens reaches 0.95).
4. **CoT: the chain is learnable but does not help the outcome**: contacts / grasp / goal atoms are read out at 0.86-0.96
   AUC (0.72-0.91 on imagined states), yet success AUC is unchanged vs `base`. Chain supervision does help its own
   capacity control (`cot` - `cot_nochain`: Goal imag +0.022, Spatial +0.009 / +0.014 true), i.e. it regularises the
   larger model, nothing more. The end-state chain does not carry information about pi's FUTURE (the logs had no
   continuation timeline) - that is the version ThinkWorld motivates, and it needs the new `rich_record` data.
5. **IMPL: imagination-consistency does not shrink the imag-true gap**; it costs AUC on true states (Spatial -0.030),
   i.e. aligning imagined to true features trades away true-state discrimination. Only `impl_noalign` lowers pred - SR
   on Goal picks (+0.042 vs +0.076), not ranking.
6. The generic success critic (B2, trained on ~5x more labels incl. other policies) is the best Spatial scorer on
   imagined picks (AUC 0.828, FP@top4 0.638, pred - SR +0.035): label volume / diversity matters more than any
   architecture change here. No Goal counterpart exists.

**Verdict: "no gain at this design / data size" (not closed).** Weakest assumptions, in order: (a) 5.4-5.9k training
labels with BCE ~0.00 by step 5k (all critics overfit equally, the recipe was fixed by pre-registration) - extra
structure cannot show at this n; (b) the chain describes the end state, not pi's future; (c) 3D tokens from 96-px depth
at 512 points. Next questions: CoT / IMPL on pi-continuation timelines from `m5b.rich_record` runs (Phase 2); critics
with more labels (pool verified candidates of all banks, as B2 did) before adding structure; adversarial review
(idea-loop step 4) still pending before any close.

## 5. Hand-off

None (no variant passed). If the campaign wants a better Goal scorer now, the measured gain is from RETRAINING the
plain C^pi recipe on all goal_collect: `runs/20261124_CRITV_goal/ckpt/base_s{0,1,2}.pt` (CriticV kind `base` =
RecoveryValue-equivalent; not a pre-registered winner, its advantage on goal_verif is largely the independence effect).
