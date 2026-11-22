# Cut-point study: where to cut a failed rollout to create recovery data

Written 2026-11-21 (user request, research + documentation + offline analysis; no GPU, no sim). Intake follows
`.cursor/skills/retryvla-idea-loop` (question, hypotheses, falsifiers, pre-registered rules, section 4.4). Offline numbers:
`scripts/analyze_cut_point.py` -> `runs/_analysis/cut_point/summary.json` (CPU, ~7 s, reads existing artefacts only).

User question (paraphrased): (1) how exactly does LIBERO-Recover(y) build its dataset; (2) in the REAL world, how do
we know where to cut a failed rollout to start recovery data; (3) how does the cut state change the recovery dataset
and the final performance.

## TL;DR

- **LIBERO-Recover does not use onset-precise cuts in its release.** The paper says a VLM (Qwen3.5-27B) localises the
  failure onset and the simulator state is taken at the predicted boundary. In the released v1.0.0 annotations
  (1,408 JSONs, all from ONE generator, `Qwen3-VL-PI-LIBERO-4in1`), snapshots sit on a **fixed step grid**: Spatial
  60 / 120 / 180 of 220-step time-outs, Object 80 / 160 / 240 of 280-step time-outs. That is ~T/4, T/2, 3T/4, the
  same rule we used on Goal. The VLM then labels the stage (pre / mid / post-fail) and the level. The curated benchmark
  scenes are the LATE cuts plus 20 steps: Spatial 180 / 200, Goal 260 / 280, Object 240 / 260, LIBERO-90 360 / 380.
- **L4 is injected in the release.** The paper says "no failures are injected". The release has 181 `l4_injected`
  scenes (17 Spatial, 164 Object): a non-target object is moved `onto` (123) or `beside` (58) the task object by
  2-12 cm, then 20 settle steps and a contact / gap / dz validity check. Our curated split has 0 L4 scenes.
- **The cut decides the level.** In their annotations a later snapshot of the same rollout is more often post-fail
  (Spatial 36 % -> 83 % -> 94 %) and harder (Spatial L3 share 35 % -> 45 % -> 63 %). The level changes inside 65 / 84
  Spatial episodes and 130 / 325 Object episodes. So "level" is a property of (rollout, cut), not of the rollout.
- **Our Goal bank, by cut fraction** (seed-7 entries, 56 per fraction, 16 verifications / entry):
  - Recoverability falls monotonically with the cut: union solved T/4 -> T/2 -> 3T/4 is non-increasing in 96 % of
    the 56 failed rollouts.
  - `wm_diverse`: solved 0.77 / 0.66 / 0.34. Hits / 1k sim steps 1.52 / 0.84 / 0.31, so a late cut costs 4.9x more
    sim per verified sample. `random`: 0.79 / 0.61 / 0.32 solved, 0.82 / 0.52 / 0.26 hits / 1k sim.
  - T/4 states are near-nominal: the entry EEF is 6 mm from the closest D0 state of the same task (median), vs 31 mm at
    T/2 and 3T/4. M0's own chunk sample re-succeeds from 46 % of T/4 entries (any of 6; 27 % per sample) vs 20 % at
    3T/4. Many T/4 entries are therefore **before the failure onset**: they are stochastic failures, not failure states.
  - T/4 data is mostly M0 imitating itself. Only 12 % of T/4 trajectories finish inside the bridge. The rest is a
    50-step bridge plus a 68-77-step M0 continuation. At 3T/4, 30-37 % finish inside the bridge.
  - Bridges get rougher with later cuts: bank bridge ||Δ³a|| is 0.17 / 0.32 / 0.44 (`wm_diverse`), 2.5x from T/4 to 3T/4.
  - The cut composition of the distilled mix (F0, 282 trajectories, mean of 3 seeds): T/4 48 %, T/2 37 %, 3T/4 15 % of
    the recovery anchors, i.e. 22.5 % / 17.4 % / 7.1 % of all anchors. **T/4 alone is 5.5x their whole 4.1 % share.**
  - Per task (n = 8 tasks with data): the Δ vs Original correlates negatively with the T/4 share (Spearman −0.50). But
    the 2 tasks with **no** recovery data also dropped (bowl_on_stove 9.7 -> 6.7, wine_cabinet 9.3 -> 7.7). Part of
    the damage is global, from the mix volume, not specific to the cut. The F0 SFT cannot separate the cut effect from
    the volume effect.
- **Real world**: use the LIBERO-Recover definition (Eq. 5-6) operationally. A cut is a state where (a) the deployed
  policy alone no longer succeeds (plan-invalidating) and (b) some recovery still exists. With no resets, estimate (a)
  with failure / progress monitors and (b) with WM-branched rollouts from the real frame. Section 3 gives the pipeline.
- **Pre-registered experiment CUT-1** (section 4.4): cut rule {fixed fractions, onset by branching, onset + 2 s,
  pre-onset, perturbation-injected, critic-drop} x the same bridge-only 4.1 % share as F1. The primary metric is a cross-cut RSR
  matrix on M0's own held-out failure states. Gated on F1.

## 0. Intake (idea-loop format; written before any CUT-1 result)

- **Research question**: for a fixed generator (search + sim verification) and a fixed data share, does the state we
  cut at (where recovery data starts) change (i) the verified yield per sim step, and (ii) the recovery and nominal
  performance after distillation? Can a cut rule that works without sim resets reach the sim-oracle cut?
- **Hypotheses**:
  - H-early: cuts before the failure onset (pi still succeeds from there) give cheap, high-yield data. That data is
    mostly pi's own continuation on near-nominal states. It duplicates or contradicts the demos, so it adds little
    recovery and can cost nominal SR when it is a large share.
  - H-late: cuts well after the onset (post-fail, after repeated retries) give low yield, long and jerky bridges, and
    states the policy rarely sees at deployment. Per-sample value is high on hard (L3 / L4) states, but there are few
    such samples.
  - H-onset: cuts at the onset (the earliest state where pi alone fails but recovery is still feasible), or a short
    delta after it, give the best RSR per training sample with no nominal loss. Expected size: +5 pt own-failure RSR
    vs fixed fractions at equal anchors.
  - H-inject: perturbation-injected failure states (LIBERO-Recover L4 recipe, FailSafe / AHA style) are a valid
    substitute when natural failures are rare. They help on natural failure states only if the injected states cover
    the same level mix.
- **Falsified if**: (a) onset-cut data is not better than fixed fractions on the cross-cut own-failure RSR matrix
  (Δ < +5 pt, or CI includes 0 at >= 80 % power); or (b) pre-onset data is not worse than onset data on nominal SR or
  RSR. The oracle arm is the sim-branching onset. If even the oracle onset does not beat fixed fractions, the cut is
  not the bottleneck (then: mix volume or generator quality, see the F0 diagnosis).
- **Closest prior work**: LIBERO-Recover 2609.05178 (VLM onset + fixed grid), 2610.05151 (LLM auditor "Focus" =
  earliest correctable state), ProTracer 2609.21369 (onset = earliest unrecoverable deviation), RaC 2509.07953
  (intervention when failure is imminent, then rewind), Sirius 2211.08416 (pre-intervention window down-weighted),
  Back on Track 2610.19579 (liveness-value drop triggers recovery). Details in section 3.
- **Arms, budget, metrics, decision rules**: section 4.4.

## 1. How LIBERO-Recover builds its dataset (code + paper + released files)

Sources:
- `third_party/LIBERO-Recovery`: a starVLA fork, HEAD 7584d1b, origin github.com/liulin815/LIBERO-Recovery. It
  contains **only the evaluation pipeline**. There is no generation, localisation or teleop code.
- arXiv 2609.05178 (HTML v1 / v3 read 2026-11-21): sections 2.4, 3.1 and appendices B, C, D, F.4.
- Released annotations `ataier/LIBERO_Recovery_FailureScenes` (v1.0.0, mirrored in `data/assets/levels/annotations`,
  see `scripts/build_levels.py`).
- Curated scene assets `ataier/LIBERO_Recovery_Assets` (`data/assets/scenes/scenes/selected_bddl_scenes_*`).

### 1.1 Pipeline as described in the paper

| stage | what | how chosen / filtered |
|---|---|---|
| 1 Task execution | run pi0, pi0.5, OpenVLA, GR00T, Wan-Policy, Cosmos-Policy from the ORIGINAL LIBERO init states of 130 tasks (Spatial, Object, Goal, LIBERO-90 + 10); render the failed rollouts to video | "no objects ... manually perturbed, no failures injected" (paper text) |
| 2 Failure localisation | Qwen3.5-27B watches the whole video + instruction and splits it into V_pre / V_fail / V_post; "failure timestamp = start of During-Failure"; the simulator state at the predicted boundary is retrieved | JSON parse; unparseable outputs discarded and reviewed (B.2) |
| 3 Characterisation | the same VLM gives level L1-L4, failure category and "minimum corrective action" | 4 teleoperators review each JSON (accept / revise / flag ambiguous); disagreements go to adjudication (B.3) |
| 4 Fine-tune + re-rollout | models fine-tuned on the teleop data are run again to produce more failures | scenes used for teleop data are excluded from eval |
| teleop | 4 operators with a SpaceMouse; 3,184 recovery episodes from 413 failure scenarios, uniformly sampled over 5 suites; several valid recoveries per scene allowed; the operator does not see the original successful actions; success = original BDDL goal | LeRobot v2.1, 20 Hz, agentview + wrist, 625,731 frames |

Definitions (paper Eq. 4-6): s_f is a failure state iff it is execution-induced, G(s_f, l) = 0, and **pi_theta(s_f)
does not reach G** (plan-invalidating). It must also be **recoverable**: there exists pi_rec that reaches G. Levels:
- L1 retry: s_f ≈ s_expected.
- L2 adapt: the object moved but is still usable.
- L3 object-state recovery: the object dropped / tipped / out of reach.
- L4 environment recovery: a non-target object blocks the task.

Appendix F.4 does exactly our "branching" check qualitatively: save the state every 20 steps of a failed GR00T rollout,
re-run from each, and the pre-fail checkpoints still succeed while the fail / post-fail ones do not. They use it only
as an illustration of RD, not to pick cuts.

### 1.2 What the released files actually show (`runs/_analysis/cut_point/summary.json`, key `liberorecover_annotations`)

| item | Spatial | Object |
|---|---|---|
| generator policies in the release | 1 (`Qwen3-VL-PI-LIBERO-4in1`) | 1 (same) |
| failed rollout length | 220 (= time-out) for all 252 | 280 (= time-out) for all 975 |
| snapshot steps | **60 / 120 / 180** (84 each) | **80 / 160 / 240** (325 each) |
| VLM stage by step (post-fail share) | 36 % / 83 % / 94 % | 71 % / 82 % / 99 % |
| level mix by step | L3 35 % -> 45 % -> 63 % | L1 9 % -> 14 % -> 22 %, L3 0 -> 5 -> 5 % |
| episodes whose level changes across its 3 snapshots | 65 / 84 | 130 / 325 |
| VLM `fail_steps` | median 40; on an 8-step grid (= chunk size 8); identical for the 3 snapshots of an episode; **before the first contact in 207 / 249** | median 40; before first contact in 792 / 948 |
| curated eval scenes (`selected_bddl_scenes_*`) | 168 = 84 episodes x step 180 / 200 | 77 = step 240 (50) / 260 (27) |
| L4 | 17 `l4_injected` | 164 `l4_injected` |

Other suites (curated assets only): Goal 91 scenes at step 260 / 280; LIBERO-10 207 scenes at 400-500; LIBERO-90 337
scenes at 360 / 380.

Reading:
- In the release, **the cut is a fixed step grid, and the eval scenes are the latest grid point (+20)**. The VLM
  output is used to label and filter (stage, level), not to place the cut. The fail-step field is coarse: it comes
  from frames sampled every 8 steps, and it fires before the first contact in ~83 % of rollouts, so it marks "the
  approach is going wrong", not the contact event. A cut at the VLM timestamp would be mostly pre-contact, i.e.
  H-early territory. This may differ from the paper's internal pipeline (the paper describes boundary-based
  retrieval); we report what the files contain.
- Late cuts of time-out rollouts are post-fail and after retries. Finding 3 of the paper (pre > during > post success)
  says these states are harder for every model. That fits H-late: the benchmark measures the hardest end of the cut
  axis.
- The L4 injection recipe (from the JSON `injection` field):
  - strategy `onto` or `beside` a task object; `param_m` 0.02-0.12 m; `knocked_from_m` (how far the object was
    moved); `ooi_moved_m` (side effect on the task object);
  - settle 20 steps; validity = contact, xy gap, dz; seed 20261013;
  - level hard-set to L4 with the reason "constructed: X disturbed onto Y". These are injected on top of natural
    failure snapshots, so they are "natural cut + injected obstruction".

### 1.3 Evaluation protocol (code: `examples/LIBERO/eval_files/eval_libero_custom_scene.py`; ours: `LiberoSim.load_scene`)
- Restore the `.npy` MuJoCo state.
- Apply a uniform ±2 cm xy offset to every free-joint object (hinge / slide joints are untouched), using seeded RNGs.
- Re-anchor the OSC goal to the current EEF.
- Drive the gripper to open or closed with 20 zero-delta steps: 5 closed + 5 open trials.
- At most 520 steps; success = the env `done` signal.
- Output: `manifest.jsonl` with `first_contact_step` (EEF force delta > 2.0) and `first_gripper_change_step`, plus
  `.npz` episodes.
- The ±2 cm noise and the closed-gripper start are **eval-time robustness perturbations, not the failure-generation
  recipe**.

### 1.4 How to replicate it on LIBERO-Goal (or any suite) with our code

1. **Failure generators**: `retryvla/search/goal_audit.py` already rolls M0 (pi0 + LoRA) on the 300 Goal train scenes
   and keeps every action (bit-exact restore by replay, `retryvla/search/goal_instances.py`). To diversify, add
   SmolVLA (`SmolVLAWrapper`) and other checkpoints as extra generators (their stage 1 used 6 models). Store failures
   with T = time-out, as they do.
2. **Snapshots**: on a 20-step grid (their F.4 granularity; the release keeps 3 points ~T/4, T/2, 3T/4 + 20).
   - Write `<task>/episode_xxx/step_XXXX.bddl` + `step_XXXX_sim_state.npy` from `LiberoSim.get_state()`.
   - Copy the BDDL with same-type objects on one line (paper C.1.2; `eval_files/fix_bddl_objects.py`).
   - Then the scenes are directly loadable by both `LiberoSim.load_scene` and the upstream client.
3. **Stage / level labels**, either:
   - (a) a local VLM (Qwen3-VL-8B in the HF cache) with the B.2 / B.3 prompts (only figure images are published; the
     original prompts are Chinese); or
   - (b) **privileged sim predicates** (cheaper, deterministic): task-object displacement / tilt / height drop vs the
     init state; gripper-object contact; BDDL goal-predicate progress (`LiberoSim.progress()`); a non-target object
     within d of the task object (L4).
   Calibrate (b) against (a) on ~50 scenes.
4. **Filters = Eq. 5-6 made operational**:
   - plan-invalidating: K = 8 restores of the deployed pi from s_f, require <= 1 success (`run_jobs` with
     `restore_snapshot` jobs);
   - recoverable: our search finds >= 1 verified bridge (`retryvla/search/compare_arms.py`, oracle or `wm_diverse`).
   The paper uses human reviewers and teleop for this. Search is our certificate.
5. **L4 injection**: copy the JSON recipe. `LiberoSim._free_joint_xy_adrs()` gives the free joints. Move a non-target
   object onto / beside the task object by 2-12 cm, settle 20, check contact / gap / dz, then re-check
   plan-invalidating.
6. **Eval**: `retryvla/eval/eval_recover.py` with noise 0.02, 5 + 5 gripper trials, 520 steps; level tables with
   `scripts/analyze_levels.py`.
7. **Recovery demos**: replace teleop with our verified bridges (search + sim verification, then distill).

**Not replicable**:
- the 6 starVLA generator checkpoints and their exact failure distribution;
- Qwen3.5-27B with their exact prompts;
- the human review / adjudication;
- the 3,184 human SpaceMouse recoveries (we can only use their released set);
- stage 4 (re-rollout of teleop-fine-tuned models);
- the paper's "1.1x human completion time" budget per task (the code uses 520 steps).
The paper's claim of "no injection" cannot be reproduced either: the release injects L4.

## 2. Cut-point strategies in SIM (resets available)

Notation: a failed rollout tau = (s_0, a_0, ..., s_T) of the deployed policy pi. A cut t gives the entry state s_t.
Recoverability by pi: p_pi(t) = P(pi reaches G | s_t). Recoverability by anyone: r(t) = 1 if some bridge + pi
continuation reaches G. Failure onset: t_on = min{t : p_pi(t') <= eps for all t' >= t} (the earliest point after which
pi alone never recovers). The "point of no return": t_nr = min{t : r(t') = 0 for all t' >= t}. Good cuts lie in
[t_on, t_nr). This is the LIBERO-Recover Eq. 5-6 window, and ProTracer's "earliest deviation after which the execution
no longer returns".

| # | rule | what is computed | cost per failed rollout (Goal, T = 300) | pros | cons / failure modes |
|---|---|---|---|---|---|
| S1 | fixed fractions T/4, T/2, 3T/4 (current; LIBERO-Recover release grid) | nothing | 0 | free; covers early / mid / late | for time-outs this is a fixed absolute time (75 / 150 / 225 for all 168 entries), unrelated to the onset; mixes pre-onset (stochastic) and post-no-return (dead) entries |
| S2 | onset by sim branching | grid every g = 20 steps; restore K times, run pi, estimate p_pi(t); find t_on by binary search on the monotone envelope (96 % of our rollouts are monotone at 3 points) | full scan 15 x K x ~250 steps = 30k (K = 8); binary search ~4 probes = 8k; sequential test (stop at 2 successes) ~4-5k. Our search costs ~4.8k sim / entry (16 x ~300) | the oracle for "plan-invalidating"; also yields p_pi(t) for C^pi training for free | non-monotone p_pi (recovery behaviours, 2605.11479); K = 8 gives a coarse estimate (p <= 0.125 is not 0); the onset depends on pi (changes after each distillation round) |
| S2b | onset + delta (t_on + 20 / 40 steps) | as S2 | as S2 | lands just inside the failure (the During-Failure stage): the state has visibly changed, so it is cleaner L2 / L3 | delta too large -> post-fail retries, compounding (paper Finding 3) |
| S2c | last recoverable / first unrecoverable bracket | binary search on r(t) with the search itself (oracle CEM or `wm_diverse`) | 4 x search cost (~19k) | brackets t_nr; tells where data becomes impossible | most expensive; r(t) depends on the proposal pool |
| S3 | critic drop | C^pi(s_t, g) from the existing critic (`data/wm/goal_latest_critic.txt`) along tau; cut at the first t with C^pi < tau_c, or at the largest negative step | ~0 sim (one pass) | cheap; same signal as Back on Track's liveness monitor (2610.19579) | critic error at OOD states (it is trained on the same distribution); our WM / critic over-optimism (imagined C^pi 0.86 vs real 0.16, RQ2) |
| S4 | sim predicates | BDDL sub-goal progress regression (`LiberoSim.progress()` falls), task object dropped (z drop > 2 cm after a grasp), tipped (tilt > 30 deg), stall (EEF speed < eps for > 1 s with the goal unmet), gripper closed on nothing (finger width < w_min), contact anomalies (force spike, like upstream `first_contact_step`) | ~0 (needs the sim state replay once) | precise, interpretable, gives the level directly (L3 = task object state changed) | privileged; only covers the failure types we enumerate; "approach goes wrong" failures (most VLM onsets) have no predicate |
| S5 | perturbation injection | take states from SUCCESSFUL rollouts at phase markers (pre-grasp, after grasp, pre-place), inject: object xy shift 2-12 cm, drop the grasped object (open gripper), knock a non-target object onto / beside (L4 recipe), arm displacement; settle 20; keep if plan-invalidating (S2 test) and recoverable | K restores per candidate + search | controls the level mix; unlimited entries; the only option when natural failures are rare | distribution mismatch with natural failures (AHA / FailSafe-style failures are motion-level); "injected" may be easier or harder than natural ones |
| S6 | VLM onset (LIBERO-Recover stage 2, 2610.05151 auditor) | VLM on the rendered video -> timestamp | VLM calls only | no privileged state; transferable to real | coarse (8-step grid); fires before contact in 83 % of their labels; needs human review |

Recommended sim default: **S2 (binary search, sequential test) for t_on, plus S4 predicates to type the failure**.
Train data at t_on and t_on + delta. Log p_pi(t) and C^pi(t) on the grid so S3 and S6 can be evaluated against the
S2 oracle (detection error |t_hat - t_on|, as in ProTracer's onset accuracy within 0.5 s).

## 3. Cut points in the REAL world (no sim, no resets)

### 3.1 What changes
In sim we can restore s_t and *measure* p_pi(t) and r(t). On a real robot we see each failure state once, cannot
restore it, and have no BDDL predicates. Two consequences:
- The cut has to be **estimated** from the logged rollout (video, proprio, actions, policy internals), not measured.
- The recovery data must be collected **at the moment** (online: human takeover or an autonomous recovery attempt), or
  from a re-staged copy of the state. An offline cut can still be used to label, weight or filter data, and to train
  the detector, but it cannot "go back" to that state.

So the real-world question splits into: (a) a runtime trigger (when to stop pi and start a recovery attempt) and
(b) an offline localiser (where the onset was, to weight / filter what was collected).

### 3.2 Signal families (2024-2026; arXiv IDs opened on arxiv.org 2026-11-21 unless marked "repo")

| family | method (ID) | signal | onset precision | notes for us |
|---|---|---|---|---|
| policy-internal failure probe | SAFE 2506.09937 (NeurIPS 2025) | MLP / LSTM on final-layer VLA features, trained on trajectory-level success labels; functional conformal threshold calibrated on successes | runtime score s_t; trade-off accuracy vs detection time | works for pi0 / pi0-FAST / OpenVLA, LIBERO + real; trajectory labels only, so pre-failure steps of failed runs are mislabelled |
| | Hide-and-Seek 2605.30834 | inter- + intra-trajectory contrastive learning from trajectory labels; localises failure-indicative actions | step-level without step labels | LIBERO, VLABench, real; OpenVLA, pi0, pi0.5 |
| | FailureSpot 2609.04277 | weak supervision from action chunks (inconsistent consecutive chunks, frozen / idle actions, aggressive random motion) -> timestamp-level detector | timestamp-level; onset = first "no meaningful progress" step | the cheapest label source for us: our chunks are logged |
| | Backtrack 2610.32941 (repo) | small failure head on frozen VLA features -> rewind | runtime | already a baseline in our repo |
| action consistency / uncertainty | Sentinel (STAC + VLM) 2410.04640 (CoRL 2024) | cumulative distance between overlapping action-chunk distributions at t and t+k; VLM video QA for slow "no progress" failures | STAC fast for erratic failures; VLM for consistent-but-wrong | needs batch sampling of chunks (we already sample 6-12 per state in the pool) |
| | FIPER 2510.09459 (NeurIPS 2025) | RND OOD score in the policy embedding + action-chunk entropy; both windowed, conformal on successes; alarm when both fire | early prediction, failure-data-free | separates benign OOD from failure; good fit for flow-matching pi0 / SmolVLA |
| progress / value estimators | GVL 2411.04549 (ICLR 2025) | VLM in-context progress on shuffled frames | frame-level progress curve | zero-shot; onset = sustained progress plateau / drop |
| | Robometer 2603.02115 | Qwen3-VL-4B reward model, per-frame progress + success, trained with failed trajectories; LIBERO fine-tuned checkpoint exists | frame-level | directly usable on our LIBERO videos as a non-privileged S4 substitute |
| | VLAC 2509.15937 | pairwise progress delta + done between two frames | step-level delta; detects regression / stagnation | delta < 0 sustained = regression |
| | liveness value 2605.11479; Back on Track 2610.19579 | policy-conditioned discounted liveness value from on-policy outcomes; Back on Track monitors it and triggers WM planning when progress is lost | runtime trigger | the closest analogue of our C^pi; their monitor = our S3 |
| VLM failure analysis / onset localisation | LIBERO-Recover stage 2 2609.05178 | VLM V_pre / V_fail / V_post boundaries | coarse (section 1.2: 8-step grid, pre-contact) | human review needed |
| | ProTracer 2609.21369 | training-free VLM + proprioceptive action boundaries; onset = earliest deviation after which the execution never returns to a valid trajectory; FailTime benchmark | 71 % within 0.5 s on short episodes (their numbers) | proprio makes the timestamp precise; definition = our t_on |
| | AHA 2410.00371, REFLECT 2306.15724, RoboFAC 2505.12224, ViFailback 2512.02787 (repo) | failure detection / explanation / correction advice | episode- or subtask-level | typing (L1-L4) and correction text, not the timestamp |
| | 2610.05151 auditor (repo) | LLM picks 0-3 "earliest supported correctable" entries | entry-level | "Focus mode" = earliest correctable = our t_on |
| human-in-the-loop | HG-DAgger 1810.02890 | expert takes over when the novice enters an unsafe region; labels only during takeover | cut = takeover time | the human is the detector |
| | IWR 2012.06733 | intervention-weighted regression; humans intervene at bottleneck entry | cut = intervention | interventions also mark bottlenecks |
| | Sirius 2211.08416 (RSS 2023) | weighted BC; the 15 robot actions before an intervention are a "pre-intervention" class with weight 0 | cut = intervention; pre-cut window explicitly penalised | the direct answer to "what about the states just before the cut" |
| | RaC 2509.07953 | operator intervenes when failure is imminent, first rewinds to a familiar state (recovery), then corrects (correction); episode ends after the intervention | cut = "imminent", i.e. slightly BEFORE the onset | recovery = return to in-distribution, like our bridge to a C^pi-high state |
| WM branching from a real frame | Ctrl-World 2510.10125 (repo) | multi-view action-conditioned video WM on DROID; ranks policies; tends to be pessimistic on low-level success | imagined rollouts from a real frame | the tool for p_hat_pi(t) / r_hat(t) without resets; known over-optimism of other WMs (CureWM 2610.09134, our RQ2) |

Cross-check: the definitions agree. LIBERO-Recover Eq. 5 ("plan-invalidating"), ProTracer's onset, 2610.05151's
"earliest correctable state" and Back on Track's "loss of liveness" are the same quantity, t_on. They differ only in
how it is estimated (VLM, VLM + proprio, LLM, learned value). Human-intervention methods put the cut slightly *before*
t_on (RaC "imminent", HG-DAgger "unsafe region"), and Sirius shows that the states just before the cut should be
down-weighted, not imitated.

### 3.3 Proposed real-world pipeline (detector -> candidate frames -> WM-branched recoverability -> select)

0. **Log everything** for every deployment rollout: both cameras, proprio, the executed and the sampled action chunks
   (for STAC / ACE), the policy features (for SAFE / FIPER), and the outcome (success detector, e.g. Robometer done,
   or a human label).
1. **Runtime trigger (online)**: alarm = conformal-calibrated OR of {feature probe (SAFE / Hide-and-Seek), action
   consistency + entropy (STAC / FIPER), progress-value plateau or drop (C^pi / liveness, Robometer)}, calibrated on
   successful rollouts to a fixed false-alarm rate. On alarm: pause pi and start the recovery-data collector (human
   takeover as in HG-DAgger / RaC, or an autonomous verified attempt, below).
2. **Offline candidate frames** (for every failed or intervened rollout): t_hat_on from ProTracer-style VLM + proprio
   localisation, intersected with the detector curves. Candidates C = {t_hat_on - 1 s, t_hat_on, t_hat_on + 1 s,
   t_hat_on + 2 s, alarm time, intervention time}.
3. **WM-branched recoverability** for each candidate frame o_t (with history):
   - p_hat_pi(t): roll pi K = 8 times inside an action-conditioned WM (Ctrl-World-like) from o_t; score the imagined
     end states with an outcome model (Robometer success, or C^pi on the imagined latent).
   - r_hat(t): run our search (pool + CEM) inside the WM from o_t; r_hat = best imagined bridge + pi-continuation score.
   - Calibration is mandatory: WM optimism is our main known failure (RQ2: imagined C^pi 0.86 vs real 0.16).
     Use (i) cross-WM agreement (xwm), (ii) an isotonic map from imagined to real outcome fitted on the few states where
     reality is known (the rest of the same real rollout: if pi did fail from t, then p_pi(t) is consistent with 0;
     if a human recovered from t, r(t) = 1), (iii) asymmetric thresholds.
4. **Select**: t* = the earliest candidate with p_hat_pi(t) <= eps (plan-invalidating) and r_hat(t) >= rho
   (recoverable). If no candidate passes, label the rollout "unrecoverable at detection" (feeds the detector, not
   the recovery data). Use t* to (a) keep the recovery data that starts within +-1 s of t*, (b) weight the robot's own
   actions in [t* - 15 steps, t*] to 0 (Sirius), (c) train the detector (onset labels).
5. **Recovery data at t*** (one of):
   - (i) human takeover at the alarm (RaC-style: back to in-distribution, then correct). The cut is the alarm time,
     so step 4 only post-filters it.
   - (ii) autonomous: execute the top-k WM-ranked bridges at the alarm, and keep a bridge iff pi then succeeds (real
     verification, one attempt per failure; failures are cheap to repeat only if the scene is re-staged).
   - (iii) re-staging: reproduce the failure state by hand (object pose from the frame) and collect several
     recoveries. This is the real-world analogue of LIBERO-Recover's restore + teleop, and of their L4 injection.
   - (iv) imagined bridges only (no real verification). Not recommended as labels; at most as low-weight data.

### 3.4 Failure modes of the real-world pipeline
- **Early false alarms** (pre-onset cuts): we collect H-early data (pi would have succeeded), which is the Goal T/4
  failure mode in section 4. Mitigation: conformal calibration on successes, and the p_hat_pi <= eps test.
- **Late detection**: the state is already post-fail or after retries (LIBERO-Recover Finding 3: compounding shift),
  so r(t) is low and human recoveries become long.
- **VLM onsets that are too early**: LIBERO-Recover's released fail steps precede first contact in 83 %.
- **WM optimism** under search, inflating r_hat. Partial observability (occluded contact, grasp slip) hides the onset
  from video. Proprio and force help (ProTracer, upstream `first_contact_step`).
- **The onset moves with pi**: after each distillation round t_on shifts. The detector and C^pi must be re-fitted per
  round, or they will cut at the old onset.
- **Non-monotone p_pi(t)**: pi's own recovery behaviours (2605.11479) make "first unrecoverable" ill-defined. Use the
  monotone envelope and require the drop to persist for >= 1 s.
- **Human timing bias**: intervention timing varies by operator and by fatigue (RaC prescribes rules for this reason).

## 4. How the cut point shapes the dataset and the final performance

### 4.1 Mechanisms (hypotheses, restated with the expected sign)
| cut | what the entry state is | dataset effect | expected performance effect |
|---|---|---|---|
| too early (pre-onset, p_pi(t) high) | near-nominal, pi often still succeeds | cheap: high yield / sim step, short bridges; but the label is mostly pi's own continuation; the bridge "teaches" a deviation from a state where the demo action was already fine | little RSR gain; at a large share it re-imitates pi's own noisier behaviour on nominal states and conflicts with D0 -> nominal SR drops |
| at onset (t_on, or + small delta) | pi alone fails, recovery still feasible | medium yield; bridges do real corrective work (L1 / L2, some L3) | best RSR per anchor; nominal neutral at a small share |
| too late (post-fail, after retries) | scene drifted, objects displaced, often dead (t >= t_nr) | low yield, expensive sim per hit, rougher bridges; heavy L3 / L4 | high value per sample on hard states, but few samples; states rarely visited at deployment if pi is stopped at onset |
| injected | chosen level mix | unlimited, controllable | helps on natural failures only if the injected states overlap their distribution |

### 4.2 Offline evidence on LIBERO-Goal (2610.05151 protocol; `runs/_analysis/cut_point/summary.json`)

Setup reminder: 56 failed M0 rollouts of the seed-7 audit, **all time-outs (T = 300)**, so the fractions are the
absolute steps 75 / 150 / 225 for every entry. 168 entries, 16 sim verifications per entry per arm.

**Search yield by cut**

| metric | arm | T/4 (t = 75) | T/2 (t = 150) | 3T/4 (t = 225) |
|---|---|---|---|---|
| solved entries (Wilson 95 %) | `wm_diverse` | 0.77 [0.64, 0.86] | 0.66 [0.53, 0.77] | 0.34 [0.23, 0.47] |
| | `random` | 0.79 [0.66, 0.87] | 0.61 [0.48, 0.72] | 0.32 [0.21, 0.45] |
| hits / verification | `wm_diverse` | 0.386 | 0.242 | 0.100 |
| | `random` | 0.242 | 0.162 | 0.086 |
| hits / 1k sim steps | `wm_diverse` | **1.52** | 0.84 | **0.31** |
| | `random` | 0.82 | 0.52 | 0.26 |
| sim steps / verification | `wm_diverse` | 255 | 290 | 324 |
| candidate success rate (all 32 verified) | both | 0.314 | 0.202 | 0.093 |
| share of successes that finish inside the bridge | both | 0.17 | 0.42 | 0.45 |
| median steps to success from the entry | both | 95 | 87 | 67 |

Recoverability pattern over the 3 cuts of the same rollout (solved by either arm; 1 = solved), T/4 T/2 3T/4:
`111` 24, `110` 14, `100` 8, `000` 8, `010` 1, `011` 1. **96 % are monotone non-increasing.** In 22 / 56 rollouts the
point of no return t_nr (for this pool) lies between two of our cuts. In 8 / 56 nothing works even at t = 75: these
are whole tasks (e.g. all 5 `wine_bottle_on_top_of_cabinet` failures: 0 hits at every cut), so the failure is set up
before t = 75 or lies beyond the pool.

**Is the entry state a failure state at all?** (seed-8 entries of the WM / critic collection, 32 candidates each)

| metric | T/4 | T/2 | 3T/4 |
|---|---|---|---|
| M0's own chunk sample (`vla_s1.0`, then M0) succeeds, per sample | **0.27** | 0.14 | 0.07 |
| ... at least 1 of 6 such samples succeeds | **0.46** | 0.41 | 0.20 |
| any of 32 candidates succeeds | 0.63 | 0.66 | 0.48 |
| primitives / perturbations succeed, per sample | 0.17 / 0.04 | 0.10 / 0.11 | 0.08 / 0.07 |
| entry EEF distance to closest D0 state of the same task (median, seed-7) | **6 mm** | 31 mm | 31 mm |
| 75th percentile of that distance | 10 mm | 54 mm | 66 mm |

Reading: at T/4, M0 re-succeeds by itself from almost half of the entries, and the robot is 6 mm from a demo state.
These are pre-onset states (stochastic failures), i.e. LIBERO-Recover would call them "pre-fail" and Eq. 5 would
reject them as not plan-invalidating. The per-sample gain of perturbations peaks at T/2, the only cut where "doing
something different from M0" beats M0's own sample. (Distance to the nominal manifold alone does not predict
solvability: Spearman −0.07 with solved.)

**Bank and distilled mix by cut** (`wm_diverse` bank 308 trajectories; `random` 282)

| metric | arm | T/4 | T/2 | 3T/4 |
|---|---|---|---|---|
| trajectories (share) | `wm_diverse` | 153 (50 %) | 111 (36 %) | 44 (14 %) |
| | `random` | 140 (50 %) | 96 (34 %) | 46 (16 %) |
| mean bridge / continuation length (steps) | `wm_diverse` | 47.7 / 68.2 | 43.6 / 75.5 | 43.8 / 81.0 |
| trajectories solved inside the bridge | `wm_diverse` | 12 % | 26 % | 30 % |
| bridge ‖Δ³a‖ (bank seg 0) | `wm_diverse` | **0.17** | 0.32 | **0.44** |
| | `random` | 0.20 | 0.35 | 0.33 |
| pool jerk, all candidates (mean abs Δ²a, `arms.jerk`) | `wm_diverse` | 0.053 | 0.083 | 0.075 |
| F0 mix (282 traj, all segments, mean of seeds 7-9): share of recovery anchors | `wm_diverse` | 48 % | 37 % | 15 % |
| ... share of ALL anchors (with 38,240 D0) | `wm_diverse` | **22.5 %** | 17.4 % | 7.1 % |
| F1 mix (36 traj, bridge only): share of all anchors | `wm_diverse` | 1.9 % | 1.6 % | 0.7 % |

Reference: Goal demo ‖Δ³a‖ = 0.077 (`runs/_analysis/smoothness`). T/4 bridges are 2.2x the demos, 3T/4 bridges 5.7x.

**Per-task effect after F0 SFT** (Original -> `wm` bank, mean of 3 seeds, /10)

| task | Original | `wm` | Δ | bank anchors | T/4 share | audit fails |
|---|---|---|---|---|---|---|
| open_middle_drawer | 9.7 | 5.0 | −4.7 | 5,923 | 0.47 | 5 |
| put_bowl_on_stove | 9.7 | 6.7 | −3.0 | **0** | - | 0 |
| put_bowl_on_plate | 8.7 | 7.0 | −1.7 | 2,862 | 0.46 | 5 |
| wine_on_top_of_cabinet | 9.3 | 7.7 | −1.7 | **0** | - | 5 |
| bowl_on_top_of_cabinet | 9.0 | 7.7 | −1.3 | 1,035 | 0.67 | 1 |
| open_top_drawer_bowl | 8.0 | 7.0 | −1.0 | 1,649 | 0.47 | 2 |
| wine_on_rack | 4.0 | 3.0 | −1.0 | 5,696 | 0.77 | 14 |
| turn_on_stove | 10.0 | 9.0 | −1.0 | 310 | 0.33 | 1 |
| cream_cheese_in_bowl | 7.0 | 8.0 | +1.0 | 12,725 | 0.38 | 18 |
| push_plate_to_stove | 4.0 | 7.0 | +3.0 | 3,693 | 0.36 | 5 |

Spearman over the 8 tasks with data: Δ vs T/4 share −0.50, Δ vs anchors +0.07, Δ vs audit fails +0.29 (n = 8, not
significant). The two tasks that gained have the lowest T/4 shares among the high-failure tasks; but tasks with zero
recovery data also lost 1.7-3.0. **Reading: the F0 drop has a global part (mix volume / interference) that no cut
choice explains, plus a possible cut-composition part.** F0 cannot separate them because every arm used the same cut
mix. Only a share-matched, cut-varied run (4.4) can.

### 4.3 LIBERO-Recover, by snapshot step (existing test runs; paired only on Spatial)

| model | Spatial step 180 | Spatial step 200 | Object step 240 | Object step 260 |
|---|---|---|---|---|
| pi0 | 0.129 (n 280) | 0.107 (n 280) | 0.195 (n 190) | 0.286 (n 70) |
| pi1-nominal | 0.129 | 0.146 | - | - |
| pi1 RetryVLA (3 seeds) | 0.425 (n 840) | 0.412 (n 840) | - | - |
| [Object] competence SFT | - | - | 0.305 | 0.486 |

Spatial has both steps for every episode: moving a late cut by +20 steps changes RSR by at most 2 pt. Once the state is
post-fail, a small delta does not matter. Object 260 is a different subset of episodes (27 of 77; not paired), so
its higher RSR is an episode-selection effect, not a step effect. Together with 1.2 (level and stage change with the
grid point within an episode), this says that **the large effects are between grid points (60 vs 180), not within
+-20 steps of a late cut**. Our curated benchmark samples only the late end.

### 4.4 Pre-registered experiment CUT-1 (written 2026-11-21 / finalised 2026-11-22, BEFORE any CUT-1 result)

- **Question**: at a fixed generator and a fixed data share, does the cut rule change own-failure recovery and
  nominal success?
- **Gate**: run after F1 (2610.05151 share-matched follow-up) has a result, on the same recipe. If F1 shows that the
  4 % bridge-only share is still below Original by > 5 pt, CUT-1 is still run (it tests the cut at that share), but a
  positive CUT-1 is then reported as "relative to fixed fractions", not as "matches theirs".
- **Policy / recipe**: M0 (pi0 + LoRA, 4k), their distillation (2,000 updates from M0, D0 + recovery pooled
  uniformly), bridge-only anchors (`--recovery_segments 0`), **≈ 1,640 recovery anchors per arm (4.1 % of anchors, =
  F1)**, round-robin over entries; 3 SFT seeds (7, 8, 9).
- **Failures**: M0 seed-7 audit (search entries), seed-8 (WM / critic), new seed-9 audit for evaluation only; all on
  the 300 TRAIN scenes. Val scenes only for nominal; test untouched.
- **Arms (cut rules)**, same generator `wm_diverse` (+ `random` as a secondary generator):
  1. `fixed`: T/4, T/2, 3T/4 (current entries).
  2. `onset` (oracle, S2): 20-step grid, sequential branching test (K = 8, stop at 2 successes), t_on = first grid
     point after which p_pi <= 1/8 on the monotone envelope.
  3. `onset_d`: t_on + 40 steps (2 s).
  4. `pre_onset`: t_on − 40 steps (control for H-early).
  5. `inject`: perturbation-injected states from successful M0 rollouts (S5: xy shift / drop / L4 onto-beside), kept
     iff plan-invalidating (same K = 8 test) and recoverable.
  6. `critic` (cheap, real-world proxy, S3): first t with C^pi(s_t) < 0.2 on the logged rollout, no branching.
- **Budget**: equal anchors (primary). Sim spend is reported per arm *including* the onset search (branching steps are
  charged to `onset` / `onset_d` / `pre_onset` / `inject`). Search: 16 verifications per entry, as before.
- **Evaluation**:
  - Own-failure RSR on a pooled held-out set E = union of the states produced by rules 1-5 on the seed-9 failures
    (~56 rollouts x 5 rules ≈ 250 states), 3 trials each (LIBERO-Recover protocol: ±2 cm, open / closed gripper,
    520 steps). Report the full arm x eval-rule matrix (cross-cut transfer) and the pooled diagonal-free mean.
  - Nominal: 100 val scenes, their protocol.
  - LIBERO-Recover RSR (Spatial test, 56 scenes x 10) only if the recipe is ported to SmolVLA / pi1 (secondary).
  - Data metrics: verified yield / 1k sim (incl. detection), bridge ‖Δ³a‖, level mix (S4 predicates).
  - Detector metric: |t_hat − t_on| for `critic` and for a VLM onset (Qwen3-VL-8B on rendered video), vs the S2
    oracle; share of cuts with p_pi > 1/8 (pre-onset false cuts).
- **Statistics**: paired over eval states, state x seed bootstrap (95 %) and sign-flip test; McNemar on solved
  entries for yields. MDE ≈ 5 pt RSR at alpha 0.10, power 0.8 (≈ 250 paired states, per-state difference SD ≈ 0.3).
  Below 80 % power: "inconclusive at n = ...".
- **Decision rules**:
  - "onset helps": RSR(`onset`) − RSR(`fixed`) >= +5 with CI > 0 AND nominal(`onset`) >= Original − 2.
  - "H-early confirmed": `pre_onset` is worse than `onset` by >= 3 RSR (CI > 0), OR nominal(`pre_onset`) <
    nominal(`onset`) − 2.
  - "delta matters": |`onset_d` − `onset`| >= 3 with CI excluding 0; else keep `onset` (cheaper to justify).
  - "injection substitutes": RSR(`inject`) on the natural-state part of E >= RSR(`fixed`) − 2 (non-inferior).
  - "real-world viable": `critic` within 3 pt of `onset` (non-inferior at margin 3) with <= 25 % pre-onset false cuts.
  - "cut is not the bottleneck": `onset` within ±3 of `fixed` and all arms within 3 of each other -> close the cut
    axis for this generator and move to mix volume / generator quality (and the mandatory review, idea-loop step 4).
- **Cost estimate**: onset search ≈ 4-8k sim steps per failed rollout (56 x 3 audits) ≈ 0.7-1.3 M steps; searches
  for 5 rule sets ≈ 5 x 168 x 4.8k ≈ 4 M steps; 6 arms x 3 seeds x 2k SFT updates ≈ 18 x 35 min GPU; eval
  ≈ 18 x (250 x 3 + 100) episodes. ≈ 2-3 GPU-days. Not launched (no GPU used for this study).

## 5. Open questions
1. Is the F0 drop mostly volume (F1 will tell) or partly cut composition? If F1 fixes nominal, does the T/4 subset
   still carry no recovery value at a 4 % share?
2. How much of the Goal failure mass is pre-onset? Our 3-point data says ~46 % of T/4 entries are M0-recoverable; a
   20-step branching scan on the 56 seed-7 rollouts (~0.4 M sim steps, CPU-feasible) would give the full p_pi(t)
   curves and the t_on distribution. This is the cheapest next step.
3. Does the onset move with the policy? After one distillation round, re-measure t_on on the same scenes.
4. The release's `fail_steps` semantics (8-step grid, before contact): VLM boundary or chunk index? Ask upstream; it
   matters for anyone using their temporal labels (LIBERO_10_LL histories).
5. Are late (post-fail) states worth training on at all, if at deployment the detector stops pi at the onset? The
   benchmark samples late states; deployment with a monitor would mostly see onset states. Train / eval cut mismatch
   is itself a variable (the cross-cut matrix in CUT-1 measures it).
6. Real world: how well calibrated is a Ctrl-World-like p_hat_pi(t) on real failure frames? A small study with
   re-staged failures (3.3 step 5 iii) would give real p_pi(t) for ~20 frames.
7. Can C^pi (or the liveness value) be trained without resets, only from logged deployment outcomes (2605.11479),
   well enough to serve as the S3 cut rule?
8. Human interventions are cut points chosen by people (RaC "imminent", Sirius): are they earlier or later than the
   sim t_on on the same failures? A sim study with a scripted "human" (intervene when the S4 predicates fire) could
   bound this.

## References used in this study (IDs opened 2026-11-21)
LIBERO-Recover 2609.05178; 2610.05151; ProTracer 2609.21369; FailureSpot 2609.04277; Hide-and-Seek 2605.30834;
SAFE 2506.09937; Sentinel 2410.04640; FIPER 2510.09459; GVL 2411.04549; Robometer 2603.02115; VLAC 2509.15937;
liveness OPE 2605.11479; Back on Track 2610.19579; HG-DAgger 1810.02890; IWR 2012.06733; Sirius 2211.08416;
RaC 2509.07953. From the repo's verified lists: AHA 2410.00371, REFLECT 2306.15724, RoboFAC 2505.12224,
ViFailback 2512.02787, FailSafe 2510.01642, Ctrl-World 2510.10125, CureWM 2610.09134, Backtrack 2610.32941.
