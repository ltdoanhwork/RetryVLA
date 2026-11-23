# RetryVLA novelty vs Kintsugi-VLA (arXiv 2609.31048)

Written 2026-11-23 (research + design + pre-registration only; no GPU job launched). Source: arXiv HTML v1 of
2609.31048 read in full (only version; 25 Sep 2026; Snegirev, Semenyakina, Maliukov, Altamirano Cabrera, Tsetserukou,
Skoltech; comment field empty). **Code: none found.** The GitHub repo `Artifex2002/project_Kintsugi` has the same
name but is an unrelated adversarial-attack project on SmolVLA / LIBERO. Web search and a secondary listing
(lee-jun-young98.github.io) also report no code link. Pre-registration: `experiments/PLAN.md`, section "KIN-1".

**Tóm tắt (VI).** Kintsugi đo *expert-relative recoverability* p_rec^E(s) bằng restore-and-branch, dùng nó để
định vị một frontier, rồi lấy start state ở các độ sâu cố định 0.15 / 0.50 / 0.85 trong cửa sổ trước frontier.
Recovery demo do PPO expert có privileged state sinh ra. Tín hiệu này là của expert, không phải của VLA đích. Ba đại
lượng khác nhau là p_rec^E(s) (expert), p_rec^{pi_b}(s) (VLA nền bị đóng băng) và Delta p^pi(s; D) (lợi ích học
thực tế). Chính paper cũng cho thấy chúng lệch nhau: teacher tự mâu thuẫn (Spearman 0.10-0.69 giữa ba seed), và
teacher đạt 68.9 % trên Protocol G trong khi SmolVLA đã fine-tune chỉ đạt 28.8 %. Đề xuất của user về cơ bản đúng,
với 5 chỗ cần sửa hoặc làm rõ (mục 1.3). Nhưng phần "student-aware" không còn trống. Có ba nhóm prior work:
(i) 2610.05151 có frozen-student handoff verification; (ii) RE-0 (2609.32416) có *paired counterfactual advantage*
của correction so với student từ cùng state, rồi distill có trọng số; (iii) TimelyDAgger (2609.33157) đã định nghĩa
"learning utility" U = J(Update(pi, D)) và có proxy TASR (rho 0.72 với downstream success). Vì vậy novelty của
RetryVLA không thể là "student-aware" chung chung. Novelty phải là: **ước lượng lợi ích học Delta p cho từng (state,
bridge) candidate, được calibrate bằng micro-fine-tune / datamodel ngay trên VLA đích, dùng WM để giảm chi phí
verify, và chứng minh bằng đo đạc rằng p^E, p^{pi_b} và handoff gain không thay được Delta p**. Thí nghiệm KIN-1 có
falsifier rõ: nếu rho(p^E, Delta p) >= 0.6 thì luận điểm gap sai trên stack của mình.

## 1. Verified summary of Kintsugi-VLA

### 1.1 What the paper does (section references are to the HTML v1)

| item | Kintsugi-VLA (exact) | ref |
|---|---|---|
| question | "given a failed simulator rollout, from which states should recovery demonstrations be collected?" | Sec. I |
| quantity | p_rec^E(s; c, H, R, nu) = Pr_xi[Y(s, pi_E, c, H, R, xi) = 1]. They call it "expert-relative rather than an intrinsic property of the physical state" | Sec. III-A, Eq. 1 |
| estimator | adaptive restore-and-branch MC, ladder 5 -> 15 -> 30 trials, pointwise Wilson intervals (z = 1.96); no time-uniform coverage. 0/30 gives U = 0.1135, so U < 0.1 cannot be certified | Sec. III-B, Alg. 1 |
| frontier | j* = first index of the maximal terminal suffix with U_i < tau (tau = 0.2; sensitivity checks at 0.1 / 0.5); also tracks crossings, separated rises, recovery islands | Sec. III-C, Eq. 3 |
| selection | window [t_dev, t_f]. t_dev = earlier of corridor departure and progress plateau (0 if neither fires). Depth d = (t - t_dev) / (t_f - t_dev). **Targeted = fixed depths 0.15, 0.50, 0.85**; uniform = uniform in the same window; random = task-defined perturbations of valid states. "local recovery-probability values and detected recovery islands are retained for analysis but do not directly alter the selected depths" | Sec. III-D, Eq. 4 |
| matching | difficulty matching = equal demo counts across *expert-recoverability* bins; frame-budget matching = common frame allowance | Sec. III-E |
| data | each successful privileged-expert continuation (full, to task completion) becomes a demo in the VLA observation space | Sec. III-E |
| task / sim | Isaac Lab, simulated Franka, one cube lift-and-transport task; success = cube height > 0.06 m and goal distance < 0.05 m (no release / hold needed); 50 Hz | Sec. IV-A |
| expert | PPO (rsl_rl), seed-42 checkpoint, **privileged 43-D simulator observation**; nominal success 223 / 224 (99.55 %) | Sec. IV-A, V-A |
| student | SmolVLA, two 224x224 RGB views (scene + wrist) + instruction + **8-D state** (EE pos, quaternion, previous gripper action), 7-D actions | Sec. IV-A |
| source failures | 200 failed episodes: **33 from the PPO teacher + 167 from the base VLA** | Sec. IV-B |
| variants | V0 nominal-only, V1 random-perturbation, V2 uniform-window, V3 targeted; at least 3 training seeds | Sec. IV-C |
| Protocol A | 300 frozen in-distribution recovery states x 5 attempts, paired state IDs / seeds | Sec. IV-C |
| Protocol B | from nominal init, disturbed track and clean track | Sec. IV-C |
| Protocol G | 4 clutter seeds x 2 physics profiles = 8 conditions; pool of 3,016 states x 15 calibration continuations = 45,240 attempts; 90 frozen states (30 hard [0.2, 0.5), 30 medium, 30 easy by *teacher* calibration); states with p < 0.2 are excluded | Sec. IV-D, Table I |
| historical analysis | 127 failed episodes, tau = 0.2. 77 (60.6 %) have a stable frontier, 75.6 % are non-monotone, 70.9 % have islands. Frontier depth median 0.856 (5-95 %: 0.752-0.960). 2,825 branch launches, **17,779 s branching time** | Sec. V-B |
| teacher audit | 3 PPO seeds on 256 shared states x 5: probe recovery 49.2 / 54.8 / 53.8 %; **Spearman 0.693 / 0.210 / 0.103**; binary agreement 0.79-0.86; mean abs. difference 0.21-0.39 | Sec. V-C, Table II |
| teacher, Protocol G | 31 / 32 nominal, **310 / 450 = 68.89 %** recovery (state-bootstrap CI 62.2-75.3 %); hard / med / easy 40.0 / 68.7 / 98.0 % | Sec. V-D, Table III |
| efficiency | to reach q = 34 %, targeted needs **30.6 % fewer demos** (R = 1.44); at q = 37 %, **33.7 % fewer frames** (R = 1.51). "do not include the simulator compute required for restore-and-branch recoverability estimation" | Sec. V-G, Table V |
| diagnosis | oracle diagnosis 34.6 -> 37.4 %; predicted diagnosis 34.3 % | Sec. V-H |

**Table IV (closed-loop SmolVLA success, %, aggregate over seeds; no dispersion reported):**

| variant | A difficulty | A frames | B disturbed | B clean | G recovery |
|---|---|---|---|---|---|
| V0 nominal-only | 12.5 | 12.5 | 55.7 | 76.8 | 9.6 |
| V1 random | 25.9 | 28.8 | 58.6 | 75.8 | 22.1 |
| V2 uniform-window | 28.8 | 31.7 | 59.5 | 75.8 | 25.0 |
| V3 targeted | **34.6** | **38.4** | **63.4** | 74.7 | **28.8** |

Their own limitations (Sec. VI) include: dependence on the expert ("Cross-expert robustness of the VLA benefit ...
remains untested"); no per-seed dispersion and no significance test ("Statistical precision, significance, and
clean-task non-inferiority therefore remain unresolved"); one task; "Bin separation establishes teacher-level
difficulty ordering but not ... direct transfer to VLA recovery difficulty"; "The current selector uses fixed
normalized depths and does not directly exploit local recovery-probability values or detected recovery islands.
Island-aware selection and joint optimization of recovery points and measurement budget remain open"; "total
savings remain unmeasured after including restoration, recoverability estimation, unsuccessful continuations, and
training". Future work (iii) is "jointly optimize recovery-state selection and measurement budgets".

### 1.2 The user's claims, checked one by one

| # | claim | verdict | evidence |
|---|---|---|---|
| a | Privileged-PPO-expert recovery demos improve SmolVLA | **correct** | Sec. IV-A, Table IV (V0 12.5 -> V3 34.6) |
| b | difficulty-matched: nominal-only 12.5, random 25.9, uniform-window 28.8, targeted 34.6 % | **correct numbers**, but "random" needs a qualifier (1.3-i) | Table IV, column A-difficulty |
| c | selection by p_rec^E(s) = P(success \| s, pi_E) | **correct**, but the full estimand is conditioned on (c, H, R, nu) | Eq. 1 |
| d | privileged 43-D state vs 2 RGB views + 8-D robot state | **correct** | Sec. IV-A |
| e | shifted-condition expert 68.89 % vs targeted SmolVLA 28.8 % | **numbers correct**, comparison needs care (1.3-ii) | Table III, Table IV col. G, Sec. V-F |
| f | fixed normalized depths 0.15 / 0.50 / 0.85 | **correct** | Sec. III-D |
| g | local recoverability / islands used for analysis, not for selection; acknowledged in Limitations | **correct** (quote in 1.1), but recoverability still sets the window and the matching bins (1.3-iii) | Sec. III-D, VI |
| h | "not to optimize downstream VLA improvement" (acknowledged) | **partly**: Limitations admit fixed depths and an open "joint optimization of recovery points and measurement budget". They never say the selector is not optimized for VLA improvement; that is our inference. Their closest admission is "not ... direct transfer to VLA recovery difficulty" (about bins) | Sec. VI |
| i | no student-aware verification at data generation (expert-verified only) | **correct for verification**, with a nuance (1.3-iv) | Sec. III-E, IV-B |
| j | student-handoff verification already exists in 2610.05151 | **correct**, and it is weaker than full-task success. RE-0 (2609.32416) and TimelyDAgger (2609.33157) also overlap (1.4) | `COMPARE_2610.05151.md` |
| k | ~30 % demo saving excludes sim cost | **correct** (30.6 % demos, 33.7 % frames, explicit exclusion quote) | Sec. V-G |

### 1.3 Corrections and nuances the paper text forces

- **(i) "random" (V1) is not random trajectory times.** V1 uses "task-defined perturbations to valid states rather
  than sampling recorded trajectory times". The fair "same window, different positions" control is V2 (uniform).
  The headline gain is V3 - V2 = +5.8 pt (difficulty) / +6.7 pt (frames). V3 - V0 = +22.1 pt.
- **(ii) 68.89 % vs 28.8 % is not a clean paired gap.** The teacher number is on the 90-state Protocol G set that was
  *balanced by teacher calibration* and excludes states with teacher p < 0.2, so the set favours the teacher. The
  paper says the VLA score "remains well below the privileged teacher's ... score, which is evaluated separately".
  It does not state that the episodes are paired. Also, 28.8 % appears twice in Table IV: V3 on Protocol G and V2 on
  A-difficulty. Cite it as "V3, Protocol G". A cleaner in-distribution reference for the teacher is the probe
  recovery of 49-55 % (Table II), so even the teacher is far from perfect off-nominal.
- **(iii) The selection is not blind to recoverability.** The window end t_f comes from p^E, and difficulty matching
  uses p^E bins. What is fixed is the *position inside the window*. Phrase it as "expert recoverability defines the
  window and the difficulty strata; the selected depths are fixed", not as "recoverability is used only for
  analysis".
- **(iv) Student-aware sourcing exists.** 167 / 200 source failures are the base VLA's own failures. So the *states*
  are partly student-induced (DAgger-like), but recoverability, acceptance and labels are all the expert's. The gap
  is in measurement, acceptance and selection, not in where failures come from.
- **(v) The evaluation strata are also expert-relative.** Protocol A difficulty matching and the Protocol G bins use
  teacher recoverability. The VLA benefit is never broken down by *student* recoverability or by realised learning
  benefit. There is also no per-seed dispersion and no significance test (their own words).
- Minor: the task is a single Isaac Lab cube lift-and-transport task. LIBERO-Recover is cited only in related work;
  they report no LIBERO numbers. The source corpus comes from "the base VLA" (unspecified checkpoint).

### 1.4 Prior work that already occupies "student-aware" (found 2026-11-23; arXiv abs pages opened, titles / dates match)

| paper | what it already does | what it does not do |
|---|---|---|
| 2610.05151 Recursive Self-Improvement via Local Recovery Supervision (2026-10-04) | frozen-student handoff test (60 + 10 actions, LLM-judged progress), then distill | no learning-benefit estimate, no selection among candidates, no WM |
| **RE-0** 2609.32416 (2026-09-26), Code-as-Policy agents | restore snapshot, run student baseline vs teacher correction under shared randomness, **admit only if the LCB of the paired advantage is > 0**, return control to the student, distill weighted by measured local benefit; bound: per-round gain >= verified intervention gain - error terms. Ablation: verified admission matters, weighting does not | not VLA / visuomotor; benefit is *execution-time* advantage, not post-training Delta p; no WM, no proxy calibration |
| **TimelyDAgger** 2609.33157 (2026-09-27) | defines **learning utility U_B(Q, T) = J(Update(pi, D_ID ∪ D_E))** for expert takeover timing; TASR proxy without retraining (Spearman 0.72 with success over 15 settings); fixed-timing study 42 % -> 78 % | dataset-level and timing-level; expert completes the episode; no per-candidate estimate, no restore-and-branch, no bridge selection |
| TILDE 2610.25376 (2026-10-18) | on-policy distillation at student-induced states + teacher recoveries, "retains only confirmed teacher recoveries" | no learning-benefit selection |
| RAIL 2611.12251 (2026-11-09) | fixes the teacher-student imitation gap by training the privileged teacher in a shared representation | changes the teacher, not data selection |
| CUPID 2506.19121, DataMIL 2505.09603, DemInf 2502.08623 | influence functions / datamodels / mutual information rank demos by effect on closed-loop return | nominal demos, not recovery data; no restore-and-branch candidates |
| STAIR 2606.15587 (RELATED_WORK N1, added by another agent) | real robot: a near-perfect expert gives poor learner data because recovery-state coverage is missing | expert-side fix, no per-candidate learning benefit |
| ADVISOR 2007.12173; Swamy et al. 2208.02225; Impossibly Good Experts (ICLR 2023) | the "imitation gap" of privileged teachers (theory + adaptive insubordination) | classic grounding for gap (1) |
| 2606.10229 What Demonstration Curation Metrics Do to Your Policy | detection AUROC 0.804 yields the *worst* curated policy (13.3 %). Curation must be judged by the policy it produces | supports evaluating selection by realised Delta p |

Consequence: "student-aware" alone, "verify the student can take over" (2610.05151, RE-0) and "learning utility" as
a concept (TimelyDAgger) are **not novel**. The open slot is narrower. See section 3.

## 2. Gap statement (paper-ready, English)

> Kintsugi-VLA [2609.31048] shows that simulator restore-and-branch can turn failed rollouts into useful recovery
> data. It selects recovery start states by *expert-relative* recoverability,
> p^E_rec(s) = Pr[success | s, pi_E], the probability that a privileged expert completes the task from s.
> Expert-relative recoverability is not necessarily aligned with the target VLA's downstream learning and execution.
> For a recovery candidate x = (s, a_{1:H}), three quantities can disagree:
> (i) the expert's recoverability p^E_rec(s);
> (ii) the frozen base policy's recoverability p^{pi_b}_rec(s), i.e. whether the VLA can already finish from s, or
> take over after a short bridge, C^{pi_b}(s_H); and
> (iii) the learning benefit Delta p^{pi}(s; D) = p^{pi_{D}}_rec(s) - p^{pi_b}_rec(s), the gain in the VLA's own
> recovery success after training on recovery data D that contains x.
> A state can be easy for a 43-D privileged expert yet unlearnable from two RGB views. A state can be hopeless for the
> expert yet fixable by a short bridge after which the VLA itself succeeds. And a state that the VLA already recovers
> from teaches nothing. Kintsugi's own audit reflects this. Equally competent experts rank the same states with
> Spearman correlations as low as 0.10. A privileged teacher recovers 68.9 % of shifted-condition states, while the
> SmolVLA trained on its targeted data recovers 28.8 %. Recent student-aware methods test whether the frozen
> student can take over after a correction [2610.05151], or whether the correction beats the student from the same
> state [RE-0, 2609.32416]. These are execution-time criteria. None of them estimates, per candidate, what the policy
> will *learn* from it. We study recovery-data selection by an estimate of Delta p^{pi}. The estimate is calibrated
> against realised post-training gains on the target VLA and made affordable by world-model pre-ranking, so that
> only a small fraction of candidates needs to be verified in the simulator.

Wording rules: do not claim "first student-aware recovery data" or "first to verify student takeover". Do not
present 68.89 vs 28.8 as a paired gap. Cite TimelyDAgger for "learning utility" as a dataset-level objective. Cite
RE-0 for paired counterfactual admission. Also say that Kintsugi's targeted depths beat uniform by +5.8 pt without
any student signal. Our claim is that the student signal adds to that, and this must be shown (KIN-1).

## 3. RetryVLA mechanism that addresses the gap (and what makes it new)

### 3.1 Objects
- Source: the base policy's own failures. Phase 1 turns them into data: a LIBERO-Recover failure instance, and the
  states along pi_b's failed continuation, snapshots every 40 steps.
- Candidate x = (s, b): start state s plus a bridge b = a_{1:H} (H <= 50). Bridges come from four sources: the
  expert prefix (first H steps of pi_E), SmolVLA chunk samples, perturbations and primitives. The WM v1 + C^pi
  ranker pre-scores a pool of 256 per state; this is the existing Phase-1 engine.
- Verification (sim, ground truth): restore s, execute b, then let frozen pi_b continue to the BDDL goal (520-step
  cap). Accept if it succeeds. This gives Ĉ(s_H) from 2-4 continuation seeds.

### 3.2 Learning-utility estimate Û(x) ≈ Delta p^pi
Proxies, all computable without retraining, are defined before any data exists:

| proxy | definition | expected sign | already in repo |
|---|---|---|---|
| P1 learnability | flow-matching loss of b under pi_b at s (mean over 8 noise / time draws) | inverted-U: too low = already known, too high = off-manifold | SmolVLA loss in `retryvla/train/sft.py` |
| P2 manifold distance | distance of b to pi_b's sampled chunks at s | negative | `search/arms.py::manifold_dist` |
| P3 smoothness | jerk of b | negative (SMOOTH-V) | `search/arms.py::jerk` |
| P4 handoff gain | G = Ĉ(s_H) - p̂^{pi_b}(s) (RE-0-style paired advantage) | positive | verification logs |
| P5 need | 1 - p̂^{pi_b}(s) | positive | continuation logs |
| P6 gradient alignment | cosine between the FM-loss gradient of x and the mean gradient of a probe set of verified recoveries from *other* failures (TracIn-style first-order influence; LoRA / last-layer projection) | positive | new, about 0.5 day |
| P7 coverage | descriptor distance to already-selected items (greedy diversity) | positive (set level) | `search/arms.py::descriptor` |
| p^E (reference) | expert MC recoverability | the gap claim predicts weak correlation | new (expert) |

**Calibration (the new part).** On 30 calibration groups (group = verified items from one start state), we train
M = 16 micro-fine-tunes from pi_b. Each uses a random half of the groups, 600 steps, with nominal replay. We then
evaluate each micro-model on all 30 start states (10 trials each). The realised per-group benefit is
Deltâ_g = mean success(included) - mean success(excluded). This is a DataMIL-style inclusion / exclusion datamodel
on the *target VLA*. Û is then a ridge / monotone combination of P1-P6, fitted with leave-group-out CV. The
primary Û is chosen by CV Spearman before the main SFTs. Cost about 11 GPU-h. An optional upgrade is CUPID-style
influence, if P6 is weak.

**Selection.** Among *verified* candidates, greedily pick the top Û with P7 diversity, up to budget B.
Unverified candidates are never selected. WM pre-ranking decides which candidates are worth verifying: about 8 of
256 per state. The sim cost of every arm is reported, including the p^E / p^pi measurement.

### 3.3 Tie to the phase plan
- **Phase 1** (WM turns the policy's own failures into verified recovery data) supplies the candidate pool and the
  cheap pre-ranking. That is what makes per-candidate verification affordable; see `VERIFICATION_COST.md`. A
  verification costs about 6 s, while a WM score costs about 0.3 ms.
- **Phase 2** (train without learning the failure) supplies the recipe: bridge-only labels (segment 0), a small
  recovery share and nominal replay, so the selected items teach the correction and not pi_b's own continuation.
  P1 / P6 are defined on exactly these labels.

### 3.4 Novelty delta, precisely

| | Kintsugi | 2610.05151 | RE-0 | TimelyDAgger | Back on Track | **RetryVLA (KIN-1 mechanism)** |
|---|---|---|---|---|---|---|
| signal used to choose data | p^E frontier + fixed depths | none (one LLM path) | paired execution advantage | takeover timing (FTA), TASR proxy | n/a (test time) | **calibrated estimate of post-training Delta p, per candidate** |
| student in the loop at generation | source states only | handoff test | baseline + handoff | gate on student features | value of pi_b | handoff verification + learnability / influence proxies |
| realised learning benefit measured per state | no | no | no (per round) | dataset-level only | no | **yes (datamodel micro-FTs + in-situ eval)** |
| candidate generator | expert | LLM tool teacher | LLM teacher | expert | WM CEM | WM-ranked pool incl. expert prefix |
| sim cost of measurement reported | no (excluded) | no | partly | no | n/a | **yes, charged to each arm** |
| benchmark | 1 Isaac task | LIBERO-Goal nominal | code-as-policy | 11 tasks | LIBERO-Goal + Franka | LIBERO-Recover Spatial (+ Goal) |

**Back on Track** (2610.19579) is a test-time planner with a policy-conditioned value. It produces no training data
and runs no selection, so it overlaps only through the "plan to states favourable for pi_b" objective of the bridge
generator, which RetryVLA already acknowledges (`WM_RECOVERY_OVERLAP.md`).

Direction B from the user ("student-aware recovery bridging": WM finds a short bridge and we verify that pi_b can
take over) is, on its own, **2610.05151 / RE-0 with a WM proposer**. It is novel only through the WM-ranking cost
reduction, which is an efficiency claim. It must beat continuation verification at equal sim budget; that is the
`HAND-E` vs `RLU-full` contrast in KIN-1. Direction A (selection by learning utility) carries the novelty. Lead with
A and use B as the generator.

## 4. What still overlaps (honest list)
1. Restore-and-branch MC recoverability with Wilson intervals: Kintsugi (and our C^pi labels). Not ours.
2. Turning the base policy's own failures into recovery data: Kintsugi (167 / 200 VLA failures), 2610.05151, RE-0,
   DAgger family. Not ours.
3. Frozen-student takeover verification: 2610.05151 (proxy) and RE-0 (paired). Ours is full-task ground truth;
   that is a difference of degree.
4. Distilling verified local benefit with "selection matters more than weighting": RE-0 found exactly that.
   If RLU-E does not beat HAND-E, our result *replicates* RE-0 rather than extending it.
5. "Learning utility" as the objective: TimelyDAgger (dataset level). Influence / datamodel curation: CUPID,
   DataMIL. Our extension is to recovery candidates from restored failure states, at the candidate level.
6. WM pre-ranking before expensive verification: V-GPS / RoboMonkey / FOREWARN (test time). Ours is for data
   generation (`VERIFICATION_COST.md` section 1c).
7. Our expert is not Kintsugi's: no PPO teacher in LIBERO. We use a scripted privileged controller plus a sim-oracle
   fallback (deviation D-K1). A reviewer can say "Kintsugi with a weak expert". We mitigate with expert-competence
   gates (nominal >= 90 %) and by reporting p^E curves next to theirs (frontier depth median 0.856).

## 5. Experiment
Pre-registered as **KIN-1** in `experiments/PLAN.md` (2026-11-23). It covers the arms, gates, decision rules, power
and compute. Main-table row "Kintsugi-style": `experiments/RESULTS.md`, appended note "Main table: Kintsugi-style
row (plan)".

## References (opened on arxiv.org/abs on 2026-11-23; title + date match)
2609.31048 Kintsugi-VLA (v1 2026-09-25) · 2610.05151 (2026-10-04) · 2609.32416 RE-0 (2026-09-26) · 2609.33157
TimelyDAgger (2026-09-27) · 2610.25376 TILDE (2026-10-18) · 2611.12251 RAIL (2026-11-09) · 2506.19121 CUPID
(CoRL 2025, PMLR v305) · 2505.09603 DataMIL (ICLR 2026 proceedings) · 2502.08623 DemInf · 2007.12173 ADVISOR ·
2208.02225 Swamy et al. · 2606.10229 curation metrics · 2610.19579 Back on Track · 2606.23617 RECALL.
