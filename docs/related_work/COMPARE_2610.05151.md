# RetryVLA vs arXiv 2610.05151 (Recursive Self-Improvement through Local Recovery Supervision)

Paper: Zhang, Liu, Zhang (Beihang), *Recursive Self-Improvement of Visuomotor Policies through Local Recovery
Supervision*, [arXiv 2610.05151](https://arxiv.org/abs/2610.05151) (v1, 2026-10-04). Read in full from the arXiv HTML on
2026-11-17. There is no project page or code link on the abs / HTML page, and none in the paper. The teacher client is
"derived from Agent as Policy" ([2609.12541](https://arxiv.org/abs/2609.12541)). The prompts are **not** released; only
their structure is described (Sec. 3.2-3.6).

Gọi tắt paper này là **LRS** (Local Recovery Supervision). Phần re-implementation trong repo cũng dùng tên `lrs`.

> **Sửa lại overlap review (`WM_RECOVERY_OVERLAP.md` A4).** Review cũ viết LRS "accepts a fix only if the frozen policy
> can then finish". Điều này **không đúng**. Acceptance (Eq. 7) là một bài test *short-horizon* có hai bước:
> **progress + confirmation**. Student đóng băng chạy tối đa H_S = 60 action. Cứ Δ = 10 action, một **observer** (một
> model, chấm theo một spec bằng natural language) kiểm tra hai điều: các quan hệ đã đạt vẫn giữ, và có một
> "early progress" condition trong P_r được thỏa. Nếu có, student chạy thêm C = 10 action và progress đó phải vẫn
> đứng. Paper tự nói rõ: "Acceptance supplies evidence of short-term handoff usability, rather than a guarantee of
> eventual full-task success." RetryVLA thì chấp nhận một bridge chỉ khi π0 chạy tiếp **hoàn thành task** (BDDL goal),
> trong <= 300 step, từ exact snapshot. Hai quy tắc **khác nhau**. Pilot dưới đây đo xem chúng bất đồng bao nhiêu.

## 1. What LRS does, precisely (from the paper)

**Failure collection (audit).** The frozen student M0 is rolled out on training scenes. Example: pi0 on LIBERO-Goal,
300 training scenes, 233 successes / 67 failures. An offline **auditor** (multimodal model) then picks the entry:
- It reads a coarse view of the failed rollout: about 1 frame/s, plus at most 8 "change centers" (motion start / stop,
  direction change, gripper change).
- It names one unresolved event (expected relation, observed discrepancy, evidence that the student does not fix it).
- A dense window follows: about ±0.5 s, up to ~96 frames, plus one extra targeted look.
- It returns 0-3 entries (exactly one primary). Mode Focus = earliest supported state at which a correction is still
  useful. Mode Coverage = also prevention / early-correction entries.
- It may use privileged physical-state summaries offline. The entry state is restored "immediately before a_t".

**Specification.** A spec model writes Γ_r = (g_r, E_r, b_r; P_r, f_r, c_r) as structured natural language. It is
frozen before each stage.
- Teacher side: local goal g_r, 1-3 completion checks E_r (AND, each with its nearest false positive), and the
  relations to preserve b_r.
- Student side: 1-2 early-progress conditions P_r (OR), a completed-task alternative f_r, and confirmation conditions c_r.

**Proposal generator = tool-using LLM teacher** (gpt-6.1-sol, medium reasoning; weights fixed). One persistent tool
session per stage:
- The teacher sees the same RGB views (agent + wrist, 128 -> 224 px) and 8-d robot state as the student. It gets no
  depth, contacts, hidden object state or task oracle (the earlier robomimic-Can teacher *did* see simulator geometry).
- Tools: state / image queries, native actions, end-effector motion helpers (coordinates, units, limits), gripper
  commands, complete, abort. It may compute targets or write a short program.
- A native-action request is 1-40 steps. Each session allows <= 100 robot requests and <= 1,200 s.
- Completion is the teacher's own semantic claim: it cites every check plus the observation records. There is no
  local-success oracle.
- Only executed robot actions become targets. Retreats and failed adjustments are kept in the path.
- There is no fixed primitive library ("motion helpers ... do not select a task-specific recovery script").

**Verification = student handoff test + endpoint extension.**
- After a stage, the teacher endpoint is saved and the frozen student runs from it (pending chunks cleared).
- The observer applies Eq. 7: ReceiptValid ∧ B(V) ∧ A(V) ∧ B(V⁺) ∧ R(V⁺), with (H_S, Δ, C) = (60, 10, 10). If the task
  is already complete, completion must hold in >= 2 observations, then confirmation.
- If the test fails, the student branch is kept only as diagnosis. The system restores the **teacher endpoint**
  (replay of audit prefix + teacher actions, with a physical / controller state check) and runs a new stage. The new
  spec uses the endpoint + the latest student behaviour. The new session gets the previous stage's last 6 commands.
- Limits per attempt: <= 5 stages, <= 300 teacher actions, <= 7,200 s.
- Attempts per entry: 3 early in development, 1 in the final configuration. Student actions are never imitation labels.

**Action quality → training windows.**
- An offline evaluator grades every executed teacher action q ∈ {0, .25, .5, .75, 1}, in the state where it was
  actually taken. It may use privileged sim records and the official task predicates (the teacher's local goal is
  excluded from scoring).
- Maximal runs with q >= η = 0.75 become intervals. Anchors j ∈ [s, e - L_min], with L_min = 5 and H_A = 50.
- Each window has a validity mask over the 50 action positions, applied to targets, noisy action inputs, flow targets,
  attention and loss. The loss is quality-weighted (Eq. 10).

**Training recipe (pi0 study).**
- Init: the public pi0 base, fine-tuned 5k updates on LIBERO-Goal (50 demos / task). M0 = the 3k checkpoint, chosen on
  the same validation scenes that are reported later.
- Data: 27 accepted teacher paths (3,974 actions) → 2,095 actions after the quality cut, in 114 runs → 1,639 windows
  (mean 17.5 valid actions). That is 4.14 % of anchors and 1.49 % of valid targets, pooled with 37,907 original windows.
  No oversampling (≈ "nominal replay" at ~96 % ratio).
- Optimiser: +2,000 updates for both arms. Batch 48, AdamW, lr 2.5e-5 with 200 warmup steps, cosine schedule to
  2.5e-6, seed 7, 3x RTX A6000.
- openpi low-memory LoRA config: about 468M of 3.288B parameters trainable.
- The data were gathered across successive prompt / code revisions. The paper itself says acquisition yield is "not
  a fixed-protocol teacher success rate".

**Benchmark / protocol.**
- Custom LIBERO-Goal split, seed 7: 30 / 10 / 10 scenes per task (train / val / test). Only the 100 **validation**
  scenes are reported; the test split is not evaluated.
- Episodes start from the nominal initial state, so this is nominal task success, not recovery from failure states.
- Max 300 env steps per episode; the policy predicts 50 actions and executes 5 before replanning.

**Numbers.**
- pi0 validation: M0 77, original-only continuation 78, recovery continuation **88 / 100**.
- Paired against continuation: 18 failures fixed, 8 successes lost, exact McNemar p = 0.0755. One seed.
- BC-RNN on robomimic Can: 91 → 102 (continuation) → **112 / 130**, from 26 segments of 9 parent failures.

**Compute.** Not reported in sim steps or LLM tokens. Every LLM wall-clock bound above is a *limit*, not a measured
cost. Sec. 5 lists "audits, replay, failed teacher attempts, student checks, scoring calls" as costs to charge, but no
figure is given.

## 2. Side by side

| | **LRS (2610.05151)** | **RetryVLA (current pi1 pipeline)** |
|---|---|---|
| Failure source | Own failed rollouts of M0 on train scenes; an LLM auditor picks the entry state (Focus / Coverage) | LIBERO-Recover failure states (benchmark scenes x gripper init x ±2 cm noise) where pi0 fails a 520-step pre-check; entry = the given failure state |
| Proposal generator | Closed-loop LLM tool teacher (gpt-6.1-sol): observe → compute targets / code → bounded native or EEF actions → feedback; variable length (<= 300 actions over <= 5 stages) | Open-loop 50-step bridges: pool of 256 = SmolVLA chunk samples (2 noise scales) + smooth perturbations + 8 hand primitives (+ random variants); 2 latent-CEM rounds (768 imagined bridges) |
| Scorer / selection | None before the sim (the teacher judges its own local completion) | WM v1 (DINOv2 latent) imagines the end state; C^pi(z_H, g) − μ·roughness; diverse top-8 |
| Verification | Frozen student from the teacher endpoint: progress + confirmation in 60 + 10 actions (Eq. 7) by a model observer; on failure, restore the endpoint and extend the teacher path (<= 5 stages) | Bit-exact replay from the failure snapshot: bridge, then π0 for <= 300 steps; accept iff the **task succeeds** (BDDL) |
| Imitation targets | Teacher actions only, quality windows (q >= 0.75) with validity mask + quality weight | Bridge (segment 0) **and** π0's successful continuation (segment 1), <= 4 per instance, round-robin over instances |
| Training | pi0 LoRA-ish (468M trainable), +2k updates, recovery 4 % of anchors, no oversampling | SmolVLA full fine-tune, init pi1-nominal, 6k steps, bs 32, lr 5e-5, 50 % recovery / 50 % nominal |
| Nominal handling | Original demos pooled (96 %); evaluation is nominal-start success only | 50 % LIBERO demos replay; nominal SR reported (pi1 0.785-0.808 vs pi1-nominal 0.796, eps 2 pt) |
| Benchmark | LIBERO-Goal custom split, 100 val episodes (nominal starts), 1 seed; robomimic Can 130 | LIBERO-Recover Spatial test (56 scenes x 10 = 560 recovery episodes, 520 steps) + nominal Spatial 500; Object held-out; 3 seeds on the main rows |
| Recursion | Defined (audit → collect → update), only one round evaluated | Rounds tried (M6 pi2: no gain over extra training); not claimed |
| Cost | LLM sessions (unreported) + sim (unreported) | Per instance <= 8 x 350 sim steps (mean 2,509 measured) + 768 WM rollouts; WM / critic training amortised |
| Result | +10 pt nominal (78 → 88, p 0.076) | RSR pi0 0.118 → pi1 0.418 ± 0.021 (3 seeds), nominal preserved |

### Equivalent (same idea in both)
1. Exact sim restore at a failure state, generating corrections offline, letting the **frozen** deployed policy run from
   the endpoint as the acceptance signal, and distilling into the *same* policy with original data mixed in. Only the
   student is deployed (no test-time module).
2. Failed student branches are never imitation labels; only accepted correction actions (+ in our case the successful
   continuation) are.
3. Matched-optimisation controls ("original-data continuation" = our pi1-nominal / demo-only SFT at the same steps).

### Different
1. **Acceptance strength.** LRS uses short-horizon progress, judged by an LLM against a spec. We use full task success
   by the frozen π, in ground truth. Ours is a strictly stronger label but needs more sim per test (<= 300 vs <= 70
   steps).
2. **Where proposals come from.** LRS uses one closed-loop expert-like agent that reasons per instance. We use many
   cheap open-loop candidates selected by a learned model (WM + C^pi). These are opposite ends of the
   "smart generator vs cheap generator + selector" axis.
3. **Endpoint extension.** LRS makes the teacher go further when the student cannot continue. We have no extension;
   the closest is RGR's executed 2-bridge chains (negative at equal sim budget).
4. **Targets.** Teacher-only, quality-masked vs bridge + continuation. Our continuation segment is pi0's own behaviour
   from a competent state; LRS deliberately excludes student actions.
5. **Evaluation.** LRS reports nominal-start success on a validation set that was also used for checkpoint selection.
   We report recovery from held-out failure states plus nominal. LRS never measures recovery-from-failure, even though
   it cites LIBERO-RECOVER as the right target.
6. **Teacher dependence.** LRS's yield hangs on a frontier LLM (gpt-6.1-sol) and on revised prompts. Ours hangs on a WM
   + critic trained on ~10^4 sim rollouts.

### Which of our claims it undermines / leaves intact
- **Undermined (already marked in the overlap review):** "verify a correction by the frozen policy's own
  continuation" as a novelty. LRS has a weaker (short-horizon, LLM-judged) version of the same rule, plus endpoint
  extension. Our *full-task, ground-truth* continuation rule is a difference in strength, not in kind. We must cite it
  and must not claim the handoff idea.
- **Undermined in part:** "distill verified recoveries into the same VLA with nominal data". LRS does this too, with
  a matched-steps control. Still ours: the nominal-preservation constraint (eps 2 pt over 3 seeds), the
  recovery-ratio Pareto (P3), and evaluation on failure states.
- **Intact:** (a) WM + C^pi *selection* over a large proposal pool, at equal sim budget, with its quantified
  efficiency (3.1x hits / sim vs random). (b) The imagined-vs-true C^pi optimism analysis (RQ3, RGR, xwm). (c) The
  competence- vs goal- / progress-critic results. (d) LIBERO-Recover RSR numbers (LRS has none). (e) Recovery-specific
  evidence: LRS's +10 pt is nominal success after training on corrections, a different outcome.
- **New question LRS raises for us:** is a closed-loop reasoning teacher a better *proposal source* than our open-loop
  pool? And is our C^pi useful *on top of* such a teacher? This is what arms (a) / (d) below test.

## 3. Faithful re-implementation in our stack (arm design; pre-registered in `experiments/PLAN.md` "LRS")

Code: `retryvla/baselines/lrs_teacher.py`, config `configs/exp/lrs_pilot.yaml`. Arms:

| arm | proposal | selection | acceptance | sim budget / instance |
|---|---|---|---|---|
| (a) `lrs` | LLM tool teacher (their interface), 1 attempt (their final setting), <= 5 stages / 300 actions | none | **their** handoff test (60 / 10 / 10, progress + confirmation) | charged: teacher steps + student tests (+ full check, reported separately) |
| (a3) `lrs3` | same, up to 3 independent attempts (their early setting) | none | their rule | <= 2,800 |
| (b) `single` = pi1 generator | 256 pool + 2 CEM, WM + C^pi, diverse top-8 | WM + C^pi | ours (full task, 300 steps) | 8 x 350 = 2,800 |
| (c) `random` (no WM, no critic) | same 256 pool | uniform random 8 | ours | 2,800 |
| (d) `lrs3_cpi` | the same 3 stage-1 teacher paths as (a3) | C^pi on the **real** teacher endpoint decides which attempt to hand-off-test / extend first | their rule | <= 2,800 |

Every accepted path also gets our full-continuation check, reported as "full-verified". So each arm can be scored
under both acceptance rules.

**Deviations (unavoidable, documented):**
1. **Teacher model.** There is no API LLM in this environment (no OpenAI / Anthropic / Gemini key; only local
   weights). The teacher is **Qwen3-VL-8B-Instruct** (local HF transformers, bf16, temperature 0.7), not gpt-6.1-sol.
   This is far weaker, so the absolute yield is a lower bound on what LRS would get with a frontier model. A secondary
   variant `lrs_geo` also gives the teacher the 3-D positions of the two task objects (the access level of LRS's own
   robomimic-Can teacher).
2. **No auditor.** LIBERO-Recover already provides the failure state, and all arms start there (same entry for
   everyone). LRS's auditor would choose an earlier pre-action state from a full rollout, which we don't have for
   benchmark states.
3. **Spec + teacher in one model.** The teacher writes its own local goal + completion checks as the first part of
   each stage (one model rather than a separate spec model).
4. **Observer = programmatic, privileged predicates** instead of an LLM observer. The spec for Spatial tasks is fixed
   (all tasks are "pick up the black bowl ... place it on the plate"):
   - P = {possession newly established (finger contact + bowl lifted >= 2 cm); bowl-plate xy distance reduced by
     >= 4 cm while held}.
   - f = task success (BDDL) at 2 consecutive checks.
   - b = possession achieved by the teacher must not be lost unless the bowl is placed.
   - Checks every Δ = 10 actions (= our n_action_steps), up to H_S = 60, then C = 10 confirmation actions.
   This observer is *more* reliable than an LLM observer, so it is an upper bound on their acceptance precision.
5. **LLM turn cap.** <= 12 LLM turns per stage instead of <= 100 requests / 1,200 s, for compute. Each turn may issue
   several commands.
6. **Training view.** No LLM action-quality grader. For (a) we train on the whole accepted teacher path (the
   "whole-path imitation" control listed in their Sec. 5). We use our SFT recipe (6k steps, r 0.5), optionally
   teacher-segment only (`train.recovery_segments=[recovery]`, closer to theirs).

## 4. Pilot results

See `experiments/RESULTS.md` section "LRS". The numbers are copied there once the pilot is done.
