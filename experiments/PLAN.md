# RetryVLA research plan (agreed 2026-10-22)

## Title / question
**RetryVLA: Policy-Competence-Guided Recovery Search and Distillation for Vision-Language-Action Models**

> Given a pretrained VLA policy and a compatible world model, can we search for recovery actions that return the robot
> to states from which the ORIGINAL policy can successfully resume, and distill these actions into the policy to improve
> failure recovery without sacrificing nominal task performance?

## Two contributions, evaluated independently

| Module | Input -> output | Claim |
|---|---|---|
| **M-A Policy-Aware Recovery Planner** | frozen VLA pi + WM + failure state s_f -> recovery bridge a_{0:H-1} | finds states from which pi resumes successfully, more efficiently than alternatives |
| **M-B Recovery-Preserving Distillation** | verified bridges + nominal data -> pi' | RSR(pi') up, NSR(pi') >= NSR(pi) - eps |

Planner is VLA-architecture independent; distillation is the VLA upgrade. Either can stand if the other is weak.

## Method
1. **Competence critic** C^pi(s, g) = P(success | s, g, pi), trained ONLY on real continuation rollouts of that pi
   (labels = pi actually finished from s). WM-imagined states are inputs, never the source of labels.
2. **Search** on the WM: s_hat_H = W(s_f, a), maximize J(a) = C^pi(s_hat_H, g) - lam * Cost(a) - beta * U_W(a).
   The planner only needs a bridge for pi to take over, not the whole task.
3. **Verify** every candidate in sim: execute bridge, then pi; keep only robust successes (>= 2 of 3 seeds).
   Anything not verifiable by sim must NOT be called verified.
4. **Distill**: L = L_rec + alpha * L_nominal (+ beta * L_preserve, e.g. distillation from pi on nominal states;
   designed for flow-matching action heads, not a KL on actions).

**Constraint (pre-registered):** NSR(pi') >= NSR(pi) - eps, **eps = 2 percentage points**, tested over >= 3 seeds.

## Claims policy
- Say "compatible with VLA / WM architectures that share an observation/action interface"; claim generality only after
  >= 2 VLA backbones OR >= 2 world models.
- Novelty = recovery-specific objective (bridge to hand control back) + policy-grounded calibration + reliable search
  (uncertainty + verification) + efficient acquisition without large-scale RL. NOT the WM architecture.
- Do not claim: QD / diversity as the cause, data balancing, multi-round self-improvement, WM == true dynamics, SOTA.
- Spatial test split is a DEV set from M7 on. Final numbers come from held-out splits frozen before evaluation.

## Evidence so far (details: experiments/RESULTS.md; final status 2026-10-26, all queued jobs done)

**Review checklist (2026-10-26).** Status legend: **proven** = significant, replicated or >= 3 seeds; **partial** =
directional / 1 seed / dev split only; **open** = not shown or contradicted.
- [x] Q1 policy-grounded competence critic beats goal latent and progress / success supervision -> proven (search + SFT).
- [x] Q3 distillation needed (vs demo-only, online best-of-N, imagination-only, residual RL) -> proven on RSR.
- [x] M-B constraint on Spatial: pi1 nominal 0.785 vs pi1-nominal 0.796 (-1.1 pt, 3 seeds) -> pass; P3: r 0.5 best,
      r 0.7 fails eps (0.766), L_preserve not needed (within seed noise).
- [~] Q2 why a WM: acquisition efficiency (hits / sim) replicated; WM data > random data and WM-as-planner NOT shown.
- [ ] Better critic -> better policy: B4 / Cpi-pool win in search, but success-data SFT = pi1 (0.427 / 0.796).
- [ ] Held-out (Object) eps-test with >= 3 seeds; P4 matched-sim-budget distillation; generality (2nd VLA / WM).

| Claim | Spatial (dev) | Object (held-out, frozen recipe) | Status |
|---|---|---|---|
| competence > latent-goal in search | 32 / 2 failures, p = 7e-8 | 66 / 8, p = 2e-12 | **proven** (replicated) |
| competence > goal after SFT | +0.230 [+0.161, +0.298], 3 seeds | +0.112 [-0.004, +0.227], 1 seed | **partial** (proven on Spatial dev, directional on Object) |
| policy-grounding (C^pi) > progress critic (P2 / B2) | search 0.480 vs 0.213 solved (progress < random); SFT n=60: 0.430 vs 0.177, p < 1e-4 | - | **proven** (Spatial) |
| B4 success critic wins because of pi0 labels | B4 - B4-noPi0 +0.119, p = 0.004; Cpi-pool = B4 (p = 0.82) | - | **proven** (search level) |
| better critic -> better distilled policy | success-data SFT 0.427 vs pi1 0.418 (+0.008, p 0.74) | - | **open** (search gain does not transfer to SFT gain) |
| WM finds more recoveries per sim than random | 3.1x hits / sim | 2.2x | **proven** (replicated) |
| WM ~ real-dynamics prefix at ~1/3 transitions | +0.033 [-0.049, +0.115] | - | **partial** (non-inferiority margin ~3 pts, cost estimated) |
| WM plans bridges without the proposal library (P1 / M11) | wm_plan_v2 0.398 vs library 0.458 solved (-0.64 hits / inst, p = 0.02) | - | **open** (planner below library) |
| distillation >> demo-only (same init + steps) | +0.292 [+0.206, +0.383] | +0.181 [+0.092, +0.277] | **proven** (replicated) |
| online best-of-8 cannot replace distillation | pi0 +0.000; pi1 +0.027 n.s. | - | **proven** (Spatial, 1 seed each) |
| imagination-only / residual RL cannot replace verified distillation | RISE-inspired 0.068-0.113 (< pi0); residual RL 0.295, nominal 0.022 | - | **proven** (Spatial, 1 seed each) |
| WM data > random data at EQUAL trajectory count | +0.167 | -0.008 (n.s.) | **open** (not replicated -> reframed as acquisition efficiency) |
| nominal preserved (eps 2 pt vs pi1-nominal) | pi1 0.785 vs 0.796 (-0.011, 3 seeds): pass; all competence-data SFTs pass, goal / progress / random fail | 0.942 vs 0.944, 1 seed | **proven** on Spatial / **partial** on Object (needs 3-seed eps-test) |
| recovery-nominal Pareto (P3) | r 0.1 / 0.3 / 0.5 / 0.7: RSR 0.323 / 0.375 / 0.418 / 0.436, nominal 0.788 / 0.814 / 0.785 / 0.766 (r 0.7 fails eps); L_preserve 0.429 / 0.806 | - | **partial** (r 0.5 best within eps; L_preserve not needed; sweep points 1 seed) |

Open weakness raised by the user (2026-10-22): the current "search" mostly RE-RANKS a proposal library (VLA samples +
hand primitives); the WM does not generate behaviour. M-A must show the WM can PLAN bridges itself.

## PoC plan (to prove the approach)

| # | Experiment | Question | Status |
|---|---|---|---|
| P1 | **WM as planner** (M11): wm_no_prim (no primitives), wm_plan (broad noise + 5-round elite CEM in imagination), wm_plan_v2 (WM / critic retrained with broad-action data) on the 122 paired M5b failures | does the WM plan bridges without a hand-made library? is WM accuracy off-distribution the bottleneck? | **done** (M11: no, planner < library; broad-action WM data does not close the gap) |
| P2 | **Strong progress / task-success critic baseline**: same architecture + data, label = task progress or reward-to-go (policy-agnostic), same search budget | is the gain from policy-grounding (C^pi) and not from "any learned value"? | **done** (B2 / B3 / B4: yes; progress < random, B4 wins via pi0 labels) |
| P3 | **Recovery-nominal Pareto frontier**: vary recovery ratio (0.1-0.7) and nominal replay; + L_preserve variant; 3 seeds; report NSR >= NSR(pi) - 2 pts | M-B constraint, forgetting | **done** (seed 0 sweep: r 0.5 best within eps, r 0.7 fails, L_preserve not needed) |
| P4 | **Matched-sim-budget distillation** on Object (use the whole bank each arm buys with the same sim budget) + 3 seeds | does acquisition efficiency translate into a better policy? | todo |
| P5 | Cost accounting end-to-end: WM data collection + WM / critic training amortized; measured transitions for all arms; wall-clock of best-of-N | efficiency claim | partial |

## Scale plan (after PoC)
1. **Generality of M-A**: second VLA backbone (e.g. a different SmolVLA checkpoint / another LIBERO-trained VLA) and/or
   a second WM (pixel / video WM with an adapter to the critic interface).
2. **Suites**: LIBERO-Goal and LIBERO-10 with the frozen recipe (Setting A), plus LIBERO-Recover level breakdown (L1-L4).
3. **Iterative extension** (only after M-A and M-B are solid): pi_k -> C^{pi_k} -> search -> verified distillation ->
   pi_{k+1}, with the critic re-grounded on pi_k continuations each round (M6 tried this with a stale critic: no gain).
4. **Mechanistic analysis**: action-distribution of verified bridges vs pi0 proposals (coverage), which behaviours appear
   only after SFT; motivates a learned recovery-proposal head only if a coverage bottleneck is shown.
5. Baselines for the paper: random search, oracle sim CEM, real-dynamics prefix, latent-goal (DINO-WM style), progress
   critic, online best-of-N, demo-only, residual RL at matched transitions (systems-level comparison).

## Compute rules
Only GPUs in scripts/shells/shells.json:allow_only (49b70230 = this machine; 8a2808f9 shared, light jobs only).

---

## Unified framing: action-grounded + action-free recovery (added 2026-11-01)

Paper 02 (action-grounded, verified recovery = the M-A / M-B work above) and Paper 03 (action-free recovery futures)
are merged into ONE RetryVLA paper. Everything above stays valid; this section adds the new questions, the design
and the ablation. Results go to `experiments/RESULTS.md` (not here). Related work for this framing:
`docs/related_work/RELATED_WORK.md` sections F-J.

**Question.** A world model generates recovery futures. How do we turn them into recovery behaviour that a VLA can use,
and which signal actually helps?

| RQ | Question | Status (2026-11-01) |
|---|---|---|
| **RQ1** | Does choosing the recovery target with C^pi beat goal distance and progress? | **evidence exists** (= Q1 above): competence SFT 0.42 vs goal 0.19 vs progress 0.18 |
| **RQ2** | Can imagined futures WITHOUT actions help when used as a C^pi-filtered auxiliary loss instead of action labels? | new, pending |
| **RQ3** | How optimistic is the WM + critic verifier when the search exploits it, and does an uncertainty penalty U(B) reduce that? | new, pending |

### Design: C^pi at the centre
C^pi(s, g) (trained only on real continuations of the frozen pi) does two jobs:
1. **Selects targets.** Score imagined end states with `J = C_pi - lambda * U_ens - mu * cost`, where U_ens is the
   disagreement across the 4 heads of the WM / critic ensemble (`retryvla/search/scorers.py`; lambda in {0, 0.5, 1, 2}).
2. **Routes samples at training time.**
   - top candidate **verified** in sim (bridge + pi continuation succeeds) -> **action-grounded SFT** (flow-matching loss);
   - top candidate **unverified but high C^pi** -> only the **action-free `L_future`** loss: a small head on SmolVLA
     hidden states (same hook point as `build_preserve`) predicts the pooled DINOv2 latent of the target future state
     z_{t+H}; cosine / L2, weight `beta_f`, recovery samples only. No pseudo-action labels.
   - `MixDataset` returns `future_latent` and `has_action`; `has_action = 0` samples do not contribute to the flow-matching loss.
   - 50% nominal demo replay as before.

```
failure state -> latent WM imagines K futures -> score C_pi - lambda*U(B) -> top candidates
   top -> sim verify -> verified -> action-grounded SFT --\
   top -> unverified, high C_pi -> L_future (action-free) --+--> pi_new  <-- nominal demos 50%
```

### Supporting work (plan parts 0-3)
- **L1-L4 breakdown.** LIBERO-Recover levels are not released for our scenes (`level = "unk"`). Check upstream
  `liulin815/LIBERO-Recovery` first; otherwise hand-label 20 Spatial + Object scenes with written criteria
  (`configs/env/levels_*.yaml`), plus a small 2-annotator agreement check. Per-level RSR in `retryvla/eval/metrics.py`,
  recomputed from existing eval files (pi0, demo-only, pi1 x 3 seeds, goal, progress); no new rollouts.
- **Backtrack baseline (no WM)**, after Landeg & Tambe 2610.32941: retrace the EE to the pose k steps earlier (or a
  safe pose above, gripper open), then pi0 continues; single-try and retry-up-to-N within the same step budget;
  plus **backtrack-distill** (SFT on successful backtrack segments).
- **Verifier optimism curve (RQ3).** For every candidate log predicted C^pi after imagination and the real sim outcome
  (mostly in existing `verif_*` logs). Report false positives (WM / critic says success, sim says fail), ECE and
  precision@k as the CEM budget grows (8, 32, 128, 512 candidates). Distill the best-lambda arm (3 seeds) only if
  verified yield or precision clearly improves.

### 6-arm ablation (same data budget, 3 seeds, 50% nominal replay)
| # | Arm | Recovery signal |
|---|---|---|
| 1 | demo-only | none |
| 2 | action-grounded (= current pi1) | verified bridges + continuations, SFT |
| 3 | action-free, C^pi-filtered | top-K imagined futures by C^pi, `L_future` only |
| 4 | action-free, unfiltered | top by goal distance or random, `L_future` only |
| 5 | hybrid = 2 + 3 | verified -> SFT, unverified high-C^pi -> `L_future` |
| 6 | backtrack-distill | SFT on successful Backtrack segments |

Report: RSR (overall and per L1-L4), nominal SR (eps = 2 pt as pre-registered above), McNemar vs pi1, sim cost.
Arm 3 vs 4 isolates the role of C^pi filtering for action-free data; arm 5 vs 2 is the RQ2 test; arm 6 vs 2 tests
whether the WM beats "back up and retry" when both are distilled.

### Go / no-go
- If **hybrid is not better than action-grounded (p >= 0.1) on L3-L4**: RQ2 is reported as an explained negative
  result ("imagination only helps when it is verified"). This is still an analysis contribution.
- If **backtrack-distill ~= pi1**: revisit the claim about the WM's role before going further.
- 3D WMs (PointWorld / PointAction style), memory and representation steering (2603.05487) only if RQ2 is positive.

### Claim boundaries (in addition to "Claims policy" above)
- RQ1 claims rest on the existing Spatial 3-seed + Object search evidence; per-level claims only after the L1-L4 labels
  have documented criteria and an agreement check.
- Do not call unverified / action-free samples "verified" or "recovery demonstrations"; they are auxiliary targets.
- RQ2: claim a benefit of action-free futures only if hybrid > action-grounded (3 seeds, McNemar) without failing eps;
  "C^pi filtering matters" only from arm 3 vs arm 4.
- RQ3: optimism numbers hold for OUR latent DINOv2 WM + critic on LIBERO-Recover; do not generalise to pixel / video WMs.
  The uncertainty penalty is claimed only for the lambda chosen on train / pilot failures, then frozen for test.
- Do not claim that the WM beats a no-WM recovery baseline unless arm 2 (or 5) beats arm 6.
- Risk: `L_future` may not reach the action head. Back it with a probe (do embeddings after training separate
  high- vs low-C^pi states?) before interpreting a null result.

Compute for this round: eval / search (MuJoCo, CPU-heavy) on shell `0097c39c`, SFT on `dedc84ce` (see
`scripts/shells/shells.json`; the compute rules above predate this).

## Real-world extension: verification (user decision, 2026-11-02)
Verification will follow a Ctrl-World-like approach ([arXiv 2510.10125](https://arxiv.org/abs/2510.10125)). Once
real data is available, a world model trained on the real dataset will act as the simulator / verifier, in place
of MuJoCo sim verification. Its verdicts are WM verdicts, not "sim-verified"; see RQ3 for optimism caveats.
Baselines on LIBERO-Recover (sim) are surveyed in `docs/related_work/LIBERO_RECOVER_BASELINES.md` (2026-11-02).

## Recursive Policy-Conditioned Geometric Recovery (RGR), design note (user request 2026-11-08)
Results go to `experiments/RESULTS.md` ("RGR"), ops to `experiments/NOTES.md`. Code: `retryvla/geom/` (features,
head, exec), `retryvla/search/rgr_core.py`, `retryvla/search/rgr_search.py`, `configs/exp/rgr_sweep.yaml`,
`scripts/geom_labels.py`, `scripts/analyze_rgr.py`. Default search / eval paths are unchanged (new modules only;
`SimPool` worker gained an opt-in `bridge()` command).

**Formulation.** B_0 = {s : C^pi(s, g) >= tau} (the frozen VLA's competence region, tau = 0.5).
B_{k+1} = B_k U {s : exists bridge b with T_hat(s, b) in B_k and GeomValid(s, b)}. A recovery is a CHAIN of short
bridges b_1..b_K (H = 50 env steps each) from the failure state through intermediate recoverable states into B_0,
instead of one bridge. Bridge score J(b) = C^pi(s_H, g) - lambda E_geom(b) - 1[not GeomValid] - mu R(b)
(R = action roughness, mu = 1 as every earlier arm; the plan's beta U_WM term is not used because the WM's own
ensemble heads are identical (U == 0, NOTES RQ3) and the bootstrap U_ens did not help (RQ3)). Chain value of a first
bridge: Q(b_1) = max(J(b_1), gamma max_{b_2} J(b_1, b_2)), gamma = 0.9 (imagined 2-step backward lookahead).
E_geom terms (`retryvla/geom/features.py`, each in [0, 1], object-state terms counted only as a WORSENING vs the
segment start): off_table (an object falls off), penetrate (contact penetration > 1 cm), tilt (manipulated object
tips over), disturb (goal object pushed > 5 cm = rigid-body / collision consistency), align (gripper-object distance,
weight 0.25), infeasible (object lifted without finger contact = floating). GeomValid = no off_table and the other
hard terms < 0.5.
Two sources of geometry: (a) ORACLE: privileged MuJoCo poses / contacts / gripper of the TRUE state after each
executed segment (upper bound, labelled "oracle" everywhere); (b) OBSERVATION-ONLY: a small geometry head on the WM v1
latent (`retryvla/geom/head.py`, labels from sim replays of the collect_v1 TRAIN bridges, trained on real + WM-imagined
latents), applied to imagined states and to observed states. Head accuracy is reported (val episodes, real and
imagined latents, vs copy-frame-0 baseline).
**Forward geometric correction.** The first bridges of the chain are EXECUTED in sim, the real state is re-encoded;
if the latent discrepancy to the WM prediction (normalized MSE > 0.40; WM v1 h10 mean is ~0.32) or the head's
object-position discrepancy (> 3 cm, geo arms) is large, the next bridge is re-planned from the observed state (fresh
SmolVLA samples at the real observation + primitives + the lookahead set) instead of continuing the imagined plan; if
the observed state is already in B_0 the chain hands over to pi right there. Geo arms drop chains whose executed
segment is geom-invalid (head on the observed state; oracle: true state) and execute the next first bridge.
**Verification unchanged.** Every final chain is re-executed bit-exactly from the ORIGINAL failure snapshot
(prefix = concatenated bridges, then the frozen pi), exactly as all earlier verified data; only these are written.

**2x2 (+ oracle), paired** (same instances / proposal pools / continuation seeds as the rq3 / vj / xwm sweeps;
M5b train shards 0-3, never test):

| | single bridge | recursive (K = 2, forward correction) |
|---|---|---|
| no geometry | `single` (= wm_unc_l0 / M5b pipeline) | `rec` |
| geometry head (obs-only) | `single_geo` | `rec_geo` |
| geometry oracle (privileged) | - | `rec_geo_oracle` (upper bound) |

Fairness: imagination <= ~768 WM bridge rollouts per arm and instance (single: 256 pool + 2 x 256 CEM; recursive:
256 pool + 128 CEM + 12 x 16 lookahead + re-plans, logged per instance); sim budget = 8 x (H + 300) worst-case env
steps per arm, segment executions included (a 2-bridge chain costs 400, so recursive arms verify ~6 chains vs 8
single bridges). Optional ablations: `rec_geo_nofc` (chain chosen fully in imagination), depth K = 3.

**Metrics.** Search (train instances, paired McNemar on solved + sign-flip on hits): verified recovery yield = solved
rate and hits per verification / per 1k sim steps; FP rate (imagined C^pi >= 0.5 but sim fails) imagined vs actual,
and for recursive arms the imagined vs observed C^pi of s_1 (B_0 membership agreement); geometric validity of the
verified chains (TRUE geometry at every segment boundary, measurement only); #bridges per solved chain; sim cost
(verification + execution steps); re-plan rate and latent discrepancy. Per official level (train has L1 / L2 / L3,
no L4: train scenes 4 L1 / 34 L2 / 74 L3, pilot 20 L2 / 40 L3; the 256 M5b pi0 failures are 11 L1 / 83 L2 / 162 L3)
-> L3 ("object-state recovery") is the key level. Distillation (only if a variant is clearly better than `single` in
verified yield, esp. on L3): recovery dataset from that arm, 1-seed SFT with the standard config (pi1 recipe, 6k steps,
r 0.5), test RSR overall + per level (Spatial test: 1 L1 / 11 L2 / 16 L3 official, no L4; Object test L1-L2 only)
and nominal (<= 2 pt drop vs pi1-nominal 0.796).

**Claim boundaries.** Oracle numbers are upper bounds (privileged state), never "observation-only". The geometry head
is trained on TRAIN bridges only and only its measured accuracy may be quoted. A search-level gain is a claim about
acquisition (verified yield per sim), not about the policy, unless the distilled policy improves test RSR beyond the
~4 pt noise level (1 seed = directional only). Recursive gains may come from closed-loop re-planning (re-observing
after the first bridge) rather than from geometry: the 2x2 separates the two; do not attribute a `rec` gain to
"geometric recoverability". L4 cannot be evaluated (no L4 scenes in our splits). Do not claim the recursive set B_k is
computed: it is approximated by K <= 3 search with a learned C^pi and WM.

## Active Causal Recovery (ACR) PoC, design note (user request 2026-11-09; written BEFORE any ACR result)
"Act Before You Imagine". Results -> RESULTS "ACR", ops -> NOTES, independent review -> `experiments/reviews/ACR_review.md`
(skill `retryvla-idea-review`; the idea may not be closed before that review is answered).
Code: `retryvla/causal/` (hypotheses, probes, families, belief, causes), `retryvla/search/acr_search.py`,
`configs/exp/acr_poc.yaml`, `scripts/acr_select_instances.py`, `scripts/analyze_acr.py`, `scripts/acr_chain.py`.
Default search / eval paths unchanged (SimPool workers gain opt-in `set_hidden` / `probe` / `run_family` commands).

**Question.** RGR / xwm / RQ3: imagined recoveries are over-optimistic (FP 0.67-0.83) and geometric failures are rare
(2-5 %). New question: when several physical explanations fit the SAME observation, does a short safe probe, chosen for
its value to the recovery DECISION, select a better recovery than passive re-observation, a random probe, an
information-maximising probe, or simply one more recovery trial? Kill test: if knowing the cause (oracle) does not help,
probing cannot help either.

**Real vs planning environment (no sim-oracle leak).** Real env = the failure snapshot + hidden parameters h* (Tier B)
or the true state (Tier A); probes and recovery trials run there and are the scarce resource (env steps). The agent's
planning model never sees h*: Tier B = the same snapshot simulated under particles h in H (types x magnitudes that are
NOT the true magnitudes); Tier A = population statistics cross-fitted on OTHER episodes (no per-instance planning sim,
because without hidden parameters the planning sim would equal the real env).

**Recovery families F** (open-loop prefix <= 100 steps, then the frozen pi continuation 300 steps, as every search arm):
`vla` (SmolVLA chunk samples), `regrasp_slow`, `regrasp_deep` (descend deeper, long close, fast lift), `push_goal`
(push the object along object->goal), `reposition` (retreat + reposition primitive), `lift_wait` (open, raise, hold).
Object-relative families use the object pose with simulated perception noise (sigma 2.5 cm = the RGR geometry head's
median error); 8 jittered variants per family (fixed seeds).
**Outcome table (paired evaluation).** For every (state, h*) every family's 8 variants are executed ONCE in the real env
from the failure snapshot (48 rollouts) -> O[f, j]. Every arm's trial allocation is scored by look-up in the SAME
table (trial j of family f = variant j), so arms are perfectly paired and any budget B can be read off. Probe side
effects: for `evi_probe` the allocated trials are ALSO re-run from the post-probe snapshot (reported separately;
primary = table, honest check = post-probe).
**Allocator (identical for every arm; arms differ only in the belief b).** Greedy: add the trial that maximises
U(n, b) = sum_h b(h) [1 - prod_f (1 - V(h, f))^{n_f}] until B trials (V = planning success rate of family f under h).
Point-mass belief -> concentrates; diffuse belief -> hedges across families. no_probe = prior belief.

**Tier B (controlled hidden physics; run first).** 25 train-split M5b/RGR states x 5 h* types: `nominal`, `slippery`
(object geom friction x[0.1, 0.3], object geom priority raised so its friction governs finger + table contacts),
`sticky` (object sliding friction x[4, 8]), `heavy` (object mass x[4, 8]), `weak_grip` (gripper actuator gain /
force range x[0.25, 0.4]). True magnitudes are drawn per case from those ranges; the planning particles are the 2 range
ENDS per type + nominal (9 particles), so the planner never has the exact h*. Pixel-identical first render is a unit
test. Magnitude ranges are fixed on 3 separate CALIBRATION states (not in the 25) only to avoid "nothing works"
(every type must leave >= 1 family with planning success > 0 on >= 2 / 3 calibration states); then frozen.
**Tier A (natural, no injection).** <= 60 RGR train instances: up to 30 that `single` left unsolved (cause priority
held / blocked / near_goal over free) + others for cross-fitting. Cause label (privileged, at the snapshot,
`retryvla/causal/causes.py`): `held` (finger contact, gripper closed), `blocked` (object elevated > 3 cm or another
free object within 9 cm), `near_goal` (object-goal xy < 8 cm), `free`. V(cause, f) and the response -> cause classifier
are fitted on the other episode groups (5-fold group cross-fitting, leakage unit = failure episode).

**Probes** (<= 30 env steps, reversible where possible, executed in the real env): `hold` (zero action = passive),
`squeeze` (close / open in place), `lift2` (close, raise 2 cm, lower, open), `nudge_{+x,-x,+y,-y}` (gripper closed,
1 cm lateral, return), `touch_grasp` (descend 2 cm, close, raise 2 cm, lower, open), `yaw_wiggle`. Response features
(observation-only): finger qpos trace summary, commanded vs achieved eef displacement, end gripper width; perception
upper bound adds the privileged object displacement (labelled "oracle perception"). Safety: Risk(a) = planning P(any of
off_table / tilt / disturb from `geom_terms`); probes with Risk > 0.2 are not allowed; real-env abort if object
displacement > 3 cm or tilt.
**Belief / EVI.** Tier B: likelihood = Gaussian kernel (per-feature scale = std over particles + floor) between the
real response and each particle's planning-sim response; b0 uniform. EVI(a) = sum_h b0(h) U(n(b_a,h), h) -
sum_h b0(h) U(n(b0), h), where b_a,h = posterior after the response particle h would produce. Objective EVI - lam
Risk - beta Cost (lam 1, beta 0.001 / step); probe only if the best objective > 0, else no probe. Info gain = expected
entropy reduction of b over the same simulated responses. Tier A: classifier posterior instead of the kernel.

**Arms** (paired, same outcome table): `no_probe` (b0), `passive` (hold), `random_probe`, `infogain_probe`,
`evi_probe`, `extra_trial` (b0 and B + 1 trials: one FULL extra trial, more generous than the probe's step cost),
`oracle_cause` (b = point mass on the nearest particle to h* / the true cause), plus `single` (Tier A only, from RGR).
Budgets B in {2, 4, 8}; B = 4 primary (pre-registered), others reported.

**Pre-registered gates.**
- G0 (decision relevance; Tier B): argmax_f V(h, f) differs across the 5 types on >= 30 % of states AND
  `oracle_cause` beats `no_probe` by >= 10 pt solved at B = 4 (McNemar p < 0.10, unit = (state, h*) case).
  If G0 fails the idea is NOT closed: the review skill runs on that failure (families too coarse? hypotheses wrong?
  B too large?) before a decision.
- G1 (information): `evi_probe` > `passive`, `random_probe`, `extra_trial` on solved (B = 4), decision-change rate > 0
  with positive mean gain on the changed decisions.
- G2 (decision relevance of the probe choice): `evi_probe` solved >= `infogain_probe` at <= its probe cost.
Metrics: solved / hits per 1k real steps (probe steps charged), probe cost, planning-sim steps, posterior accuracy
(MAP type), decision-change rate + conditional gain, probe safety violations, post-probe re-run solved; paired McNemar
+ sign-flip, per type / per cause / per level. Train split only, no SFT, no test.

**Claim boundaries.** Tier B is a controlled hidden-physics benchmark built on LIBERO-Recover states, not the official
benchmark; the planner is a simulator with the right STRUCTURE (parameter class) but not the true parameters (a
system-identification setting, like M1's oracle scorer is a sim scorer). Object-relative families use simulated
perception noise, not the observation-only head. Tier A is the only natural-failure evidence. No policy claim without
distillation + test RSR.
**Calibration outcome (2026-11-09 16:05 UTC, 3 calibration states only, before any evaluation state was run;
`runs/_analysis/acr/calib_grid.csv`).** Note the definition change made during code checks (before calibration):
`slippery` lowers the FINGER friction (priority raised), not the object's: with low object friction the bowl slid
onto the plate by itself (visible, and solved the task) -> not a hidden cause. Pooled sub-goal rates (3 variants x 3
states, scripted prefix only): nominal grasp-lift 0.22 / push 0.22. Applying the rule: slippery qualifies at
0.001-0.03 (grasp 0, push kept) -> [0.001, 0.03]; weak_grip 0.003-0.1 -> [0.003, 0.1]; heavy only at 3.0 (>= 10 kills
everything, 1.0 does not suppress grasp) -> [2, 4] around the single qualifying grid point (deviation: the rule gives a
single point; particles need two ends); `anchored` (planar joint friction) had no outcome effect at any magnitude up
to 100 N -> dropped. Tier B therefore has 4 cases per state (nominal + 3 hidden types), 25 states = 100 cases, 7
planning particles. The calibration also shows the scripted families are weak (~1 / 3 sub-goal success at nominal
with 2.5 cm perception noise); success is still counted after the pi continuation.

## ACR-N: "Act Before You Imagine" on NATURAL failures, design note (user pivot 2026-11-16; written BEFORE any ACR-N result)
Supersedes the 2026-11-09 ACR plan for the next round (ACR Tier B was stopped at 2 / 25 states on the user's request;
its code is reused). Results -> RESULTS "ACR-N", ops -> NOTES, adversarial review afterwards (`retryvla-idea-review`).
**Status 2026-11-17: G0 / G1 FAILED at 2800 and 1400 -> independent review `docs/reviews/ACRN_REVIEW.md`: PIVOT, not
close.** Probe-to-identify-a-discrete-cause is closed as tested (oracle cause 0.192 vs no_probe 0.219 on U; bias-corrected
MI ~0.12 bits; power ~0.5 for +8 pt). Kept findings: a cross-fitted fixed `grasp_slow` beats the allocator (U 0.274, S
1.000) and hindsight 0.52 is mostly noise (split-half ceiling ~0.21). Next: F1 selector shoot-out (offline), F1b probe
features in a learned option-outcome predictor, F2 observation-only pose + fresh `single` (~2 GPU-h) before any distill.
Code: `retryvla/causal/{oracle,families,probes,belief}.py`, `retryvla/search/acr_search.py` (tier `N`),
`configs/exp/acrn.yaml`, `scripts/acrn_select_instances.py`, `scripts/analyze_acrn.py`, `scripts/acrn_chain.py`.

**Question.** RGR showed that pose geometry (even oracle) does not raise the solved rate, and every imagined-recovery
variant is bottlenecked by WM over-optimism. Do the failures that `single` cannot solve have a hidden PHYSICAL cause
(grasp state, slip, contact / blockage, movability, policy OOD) that (a) changes which recovery works and (b) can be
established cheaply by a short safe probe, so that choosing the probe by decision value and recovering conditioned on
the resulting belief solves more of them at the SAME sim budget?

**Instances (train only, deterministic, `scripts/acrn_select_instances.py`, seed 20261116).** Primary set U = ALL
RGR train instances that `single` left unsolved (73 of 127; no cherry-picking), ordered so L3 and instances with
finger/object contact at the failure snapshot come first (if compute runs out, the first N are the priority list).
Control set S = 24 `single`-solved instances (stratified by task) -> used for cross-fitting the decision table /
belief model and for a no-harm check. Selection-on-failure caveat: on U, `single` = 0 by construction, so absolute
rates on U are biased upwards for anything re-run with new seeds; only PAIRED comparisons inside the new outcome table
are tests (arm 1b below is the honest control).

**Failure-cause taxonomy (oracle, privileged MuJoCo; `retryvla/causal/oracle.py`).** Measured at the failure snapshot:
contact census (finger-object, gripper-other body, object-object, object-fixture other than the table, object-goal),
gripper aperture + actuator force, object up-axis, and two INTERVENTIONAL oracle tests run from the exact snapshot
(never charged, never visible to non-oracle arms): lift test (close, raise eef 3 cm: object lift / eef lift) and
movability test (external horizontal force 2 x m g on the object for 20 steps, robot frozen: object displacement).
Cause = first match:
1. `tipped`: object up-axis z < 0.8.
2. `held_ok`: finger-object contact, gripper closed, lift ratio >= 0.6 (secure grasp; failure is transport / place).
3. `held_slip`: finger-object contact but lift ratio < 0.6 or only one finger (insecure / slipping grasp).
4. `stuck`: object touches a fixture / other object (not table, not the gripper) or sits > 3 cm above the table, AND
   movability displacement < 1 cm (blocked / jammed).
5. `obstructed`: gripper (fingers / hand) in contact with a non-target body (plate, fixture, other object).
6. `contact_free`: object touches a fixture / other object / goal but is movable (cluttered / on the rim).
7. `free`: object only on the table, not touched (approach / localisation / policy-OOD failure).
Distribution reported on U and S; pre-registered merge: causes with < 4 instances in U+S are merged into the nearest
group (tipped -> contact_free, held_ok/held_slip -> held, stuck/obstructed -> blocked) for the decision table only.
Policy-OOD is reported as an outcome attribute (pi-only success rate from the failure state = `vla` option row), not
as a cause.

**Probe library (<= 45 env steps, bounded; `retryvla/causal/probes.py`).** `hold` (15 steps zero action = passive
re-observation), `squeeze` (close / restore), `lift2` (close, raise ~2.4 cm, lower, restore), `touch_grasp`,
`yaw_wiggle` (wrist micro-rotation), `nudge_{px,mx,py,my}` (~1.2 cm lateral, return), `release` (open, rise 2 cm).
Observation-available response (inference-legal): proprio trace (finger opening min / end / std while closed,
achieved eef displacement / lift, return error, commanded-minus-achieved path = lag) + VISUAL response: DINOv2 latent
change (WM v1 encoder, agent-view and wrist separately) start -> peak and start -> end, + C^pi of the post-probe obs.
Tactile (finger-object contact fraction) and oracle perception (privileged object displacement / lift) are reported as
separate upper-bound feature sets. Safety: real-env abort if the object moves > 3 cm; Risk(a) = cross-fitted rate of
the probe worsening the state (off_table / tilt / goal disturbed, `geom_terms`); probes with Risk > 0.2 are excluded.
Probe cost = its env steps, charged to the same sim budget as recovery trials.

**Recovery options (cause-conditioned option set, `retryvla/causal/families.py`).** Every option = open-loop prefix
then the frozen pi continuation (300 steps), executed in the real sim from the failure snapshot:
`vla` (pi only, new seed), `wm_vla` (top-C^pi WM v1 imagined SmolVLA samples / perturbations, = the `single` scorer
without CEM), `wm_prim` (top-C^pi WM v1 imagined proposal primitives: open / retreat / reposition / regrasp),
`hover` (re-approach from a pre-grasp pose), `grasp_slow`, `grasp_firm` (regrasp variants), `push_goal`, `retreat`
(back off), `place_goal` (keep the grasp, carry above the goal, lower, release: for a secure grasp), `pull_free`
(push the object 5 cm toward the robot out of clutter / off a fixture, back off). 8 variants each (fixed seeds,
2.5 cm simulated perception noise for object-relative options) -> OUTCOME TABLE O[option, variant] per instance
(80 sim trials). Every arm's choices are scored by look-up in the same table (perfectly paired).

**Decision table and belief (cross-fitted, 5-fold GROUP folds by failure episode).** V[c, o] = smoothed success rate
of option o on training-fold instances of cause c (Beta prior centred on the option's marginal rate, strength 2).
Belief: b0(c | failure obs) = multinomial logistic regression on observation-only failure features (proprio at the
failure: gripper opening, eef height; C^pi of the failure obs; RGR geometry-head predictions of finger contact /
gripper-object distance / object height from the real latent). After probe a with response r:
b_a(c) ∝ b0(c) p(r | c, a), p = diagonal Gaussian per cause (pooled-variance shrinkage) fitted on training folds.
Sequential allocation (identical for every arm; arms differ only in the belief): greedily append the trial that
maximises the belief-weighted P(>= 1 success) per expected step cost until the step budget is used; executed in that
order by table look-up until the first success or until the budget is exhausted (actual table steps charged).
EVI(a) = E_{c ~ b0} E_{r ~ train responses of cause c} [V_c(alloc(b_a(r), S - cost_a))] - E_{c~b0}[V_c(alloc(b0, S))];
cost enters through the reduced budget; a* = argmax EVI(a) - lam Risk(a) (lam = 1); probe only if the objective > 0.

**Arms (paired, same instances, same step budget S including probe steps).**
(1) `single` (historical RGR picks; = 0 on U by construction) and (1b) `no_probe` = allocation with b0 (honest control),
plus (1c) `wm_only` = all budget on `wm_vla` (fresh single-like); (2) `passive` = `hold` probe; (3) `random_probe`
(uniform over safe probes, fixed seed); (4) `evi_probe`; (4b) `infogain_probe` (max expected entropy reduction, same
budget accounting); (5) `oracle_cause` (point-mass belief on the true cause, charged the EVI arm's probe cost);
(6) `oracle_cause_free` (no probe cost) -> 5 vs 6 separates information value from budget; (7) `hindsight_best`
(best single option per instance in hindsight: option-set ceiling, not an arm); (8) `extra_budget` (b0, S + 45 steps).
Budgets: S = 2800 env steps (= `single`'s 8 x 350, primary), S = 1400 secondary.

**Metrics.** solved rate (U primary; U+S; per level; per cause), verified recovery yield (hits per 1k steps), probe
cost, steps to first success, decision-change rate (allocation's first option differs from no_probe) and the gain on
changed decisions, cause-ID accuracy of b0 vs b_a (MAP, cross-fitted), option-choice entropy across causes, probe
safety violations; McNemar (exact) vs `no_probe` and vs `wm_only` (and vs `single` on U+S). Honest check: the
allocated trials of `evi_probe` (and `random_probe`) are re-run from the TRUE post-probe state (probe side effects in).

**Pre-registered decision rules.**
- G0 (actionable causes): `oracle_cause` beats `no_probe` by >= 8 pt solved on U at S = 2800 with exact McNemar
  p < 0.10. If G0 FAILS: do not close; diagnose (i) coverage: `hindsight_best` on U (option set too weak if < 30 %),
  (ii) cause -> option relevance: fraction of instances whose oracle-belief first option differs from the b0 first
  option, mutual information between cause and best option, per-cause V table spread, (iii) whether the bottleneck is
  the pi continuation (prefix sub-goal reached but pi fails: `place_goal` / grasp prefix success vs final success);
  report what would need to be true.
- G1 (information): if G0 passes, `evi_probe` beats `no_probe`, `passive`, `random_probe` and `extra_budget` on U
  (>= 5 pt, p < 0.10 vs `no_probe`) with a positive mean gain on changed decisions; else analyse the belief update
  (cause-ID accuracy with / without probe, proprio vs tactile vs oracle-perception features).
- G2 (go to distillation): G1 passes AND the post-probe honest re-run keeps >= 80 % of the table gain AND
  `evi_probe` >= `single` on U+S (no harm on S). Then generate a probe-then-recover dataset and run 1-seed SFT + test
  RSR / nominal (deployable form discussed in RESULTS). Otherwise no distillation.

**Claim boundaries.** Causes are defined from privileged sim; oracle arms are upper bounds. Probe responses are
from the same simulator (no sensor noise beyond rendering). Scripted options use simulated perception noise, not the
observation-only head. U is selected on `single` failure. No policy claim without distillation + test RSR.

## ACR-N follow-ups F1 / F1b / F2, pre-registration (2026-11-17, from docs/reviews/ACRN_REVIEW.md section 5; written BEFORE any F1 / F1b / F2 result)
**Status 2026-11-18:** F1 "selection matters" FAILED ((d) - (a) = 0.000, 5/5), cause not reopened, F1b probing closed as tested; F2 FAILED (head-pose grasp_slow 0.233 vs fresh single 0.164 on U, +6.8 pt, p 0.33; S 0.917) -> no distillation; only lead: localisation-gated re-grasp (needs new failures + pre-registration). RESULTS "ACR-N".
Same 97 instances (U 73 single-unsolved, S 24), same outcome table (data/acr/n1/N), same executor (`analyze_acrn.execute`:
a trial starts only if used + worst-case cost <= budget, actual table steps charged, stop at the first success), same
5 episode folds; budgets 2800 (primary) and 1400. Code: `scripts/acrn_f1.py` (offline), `retryvla/search/acr_search.py`
tier N modes `latents` / `f2`, `scripts/analyze_acrn_f2.py`.

**F1 selector shoot-out (offline, 0 sim).** Every model is fitted on the training folds only.
- (a) `fixed`: the single option whose fixed policy (variants in order) has the highest solved rate on the training
  folds at that budget.
- (b) cause table with strong shrinkage: `cause_s20` = oracle-cause V table, Beta strength 20 toward the option
  marginal, same sequential allocator; `cause_tie` = the top-2 options by training-fold fixed solved rate, oracle
  cause only breaks the tie (pick the one with the higher per-cause rate), then alternate.
- (c) `learned`: per-option L2 logistic regression (C = 0.3, standardised) predicting per-trial success from failure
  observations: obs_x (proprio, C^pi, RGR geometry-head predictions) + DINOv2 failure-latent PCA-16 (PCA fitted on the
  training folds); sequential ranking by p_o * r[o, k] / cost_o, r[o, k] = training-fold failure decay (P(variant k
  succeeds | earlier variants of o failed) / P(variant 0 succeeds), Beta-smoothed).
- (d) `learned_sub`: (c) + outcome-informed sequencing: after a failed trial of option o with sub-goal outcome s
  (scripted options: lift >= 3 cm / obj-goal xy < 4 cm / neither; WM / vla options: "na"), multiply every other
  option o' by m[o, s, o'] = P(o' trial succeeds | instance has a failed o-trial with outcome s) / P(o' succeeds),
  shrunk toward 1 with strength 20 (training folds). Caveat: s uses privileged geometry after the prefix
  (upper bound on outcome-informed selection; a proprio proxy would be weaker).
- Ceilings: split-half oracle (choose by the instance's own variants 0-3, evaluate 4 trials on variants 4-7), hindsight.
- **Primary contrast: (d) vs (a) on U at 2800.** "Selection matters" iff (d) - (a) >= +5 pt, exact McNemar p < 0.10
  AND (d) on S >= 0.95; else conclude "a fixed scripted prior is the result; selection is noise-limited at n 73".
  Cause is reopened only if (b) >= (c) on BOTH budgets (best of cause_s20 / cause_tie, U).
**F1b probe features.** Same as (c) with probe-response features of ALL probes added (feature sets proprio, obs =
proprio + visual, tactile, oracle; each set: concatenated responses of the 11 probes, PCA-16 within fold). Metric:
pooled cross-fitted AUC of the predicted per-option success probability over (instance, option, variant) cells,
cluster bootstrap over instances (2000 resamples) for the AUC difference vs (c). Probing continues iff AUC +0.05
(bootstrap 90 % CI lower bound > 0) AND solved +5 pt vs (c) at either budget (U); else probing is closed as tested and
the AUC ladder obs < proprio < tactile < oracle is reported as the finding. (A real probe would be charged its steps;
F1b does not charge them, i.e. it is optimistic for probing.)
**F2 deployability (sim, train only).** Re-run `grasp_slow`, `hover`, `grasp_firm` x 8 variants on all 97 instances
with object / goal positions from the RGR geometry head on the real failure latent (observation-only; no sim GT pose,
no added noise: perception_sigma 0; eef from proprio; same variant seeds / continuation seeds as the table), plus a
fresh full `single` (RGR rgr_search arm `single`, 256 pool + 2 CEM rounds + diverse top-8 verified, new proposal and
continuation seeds: eval.seed + 1000, rgr.seed_offset 1000) on all 127 RGR train instances (reported on U / S).
Primary: head-pose fixed `grasp_slow` x 6 at 2800 on U. **Go to a distillation PROPOSAL** (not launched) iff head-pose
grasp_slow-fixed keeps >= 80 % of the GT-pose cross-fitted fixed gain on U (solved >= 0.22) AND beats fresh `single`
on U by >= 8 pt (exact McNemar p < 0.10); else report the library gain as privileged-pose-only and diagnose.
Also reported: head-pose cross-fitted fixed choice among the 3 options, head pose error at the failure state, S.

## LRS: re-implementation of arXiv 2610.05151 (local recovery supervision) vs RetryVLA, design note (user request 2026-11-17; written BEFORE any LRS result)
Comparison + deviations: `docs/related_work/COMPARE_2610.05151.md`. Code `retryvla/baselines/lrs_teacher.py`,
`configs/exp/lrs_pilot.yaml`, analysis `scripts/analyze_lrs.py`. TRAIN split only (M5b / RGR instances).
- Research question: at an equal simulator budget on the same pi0 train failures, does an LLM tool-teacher with
  frozen-student handoff acceptance (LRS) produce as many verified recoveries as our WM + C^pi generator (`single`), and
  does C^pi help LRS?
- Hypothesis: with a local 8B VLM teacher, LRS's closed-loop reasoning is weaker than a large cheap pool + WM / C^pi
  selection (expected: solved_full single 0.4-0.6 vs lrs3 0.1-0.3). Their short-horizon handoff rule over-accepts
  relative to full-task continuation (precision < 0.6). C^pi ranking of real teacher endpoints cuts sim per accepted
  path.
- Falsified if: lrs3 solved_full >= single - 0.05 (then the teacher is a competitive generator and must be scaled /
  distilled), or precision of their rule >= 0.8.
- Closest prior work: 2610.05151 (LRS), Agent as Policy 2609.12541 (teacher client), Kintsugi-VLA 2609.31048, MAGMA-GEN
  2609.20056 (re-execution validated corrections), Code as Policies.
- Arms: (a) `lrs` 1 attempt (their final protocol), (a3) `lrs3` <= 3 attempts, (d) `lrs3_cpi` same stage-1 paths,
  hand-off order by C^pi(real endpoint), secondary `lrs_geo` (teacher also sees object 3-D positions = their robomimic
  teacher access); controls (b) `single` (RGR rgr_sweep rows, identical instances / seeds, 8 verifications) and (c)
  `random` (8 uniform picks from the same 256 pool, paired continuation seeds, new run). No oracle arm is needed for
  the generator question: the acceptance rule is itself measured against the ground-truth full-continuation oracle.
- Instances + split: pilot = RGR shard 0 of 4 (M5b select_seed 2026, n ~ 31 pi0 failures, train); scale = all 4 shards
  (127). Budget per instance: <= 2,800 sim steps (= 8 x (50 + 300), RGR budget), charged for teacher actions + student
  hand-off tests + extensions; the final full-continuation check of accepted LRS paths is reported both charged and
  uncharged. LLM: <= 12 turns / stage, <= 5 stages, <= 300 teacher actions / attempt.
- Primary metric + test: solved_full (>= 1 path whose replay + frozen pi0 <= 300 steps reaches task success), paired
  exact McNemar vs `single`. Secondary: their-rule accept rate, precision P(full | accepted) with Wilson CI, full hits /
  1k sim steps, LLM seconds / instance, per level (L1-L3).
- Decision rules (pilot): (R1) lrs3 solved_full >= single - 0.05 -> scale to 127 + 1-seed SFT. (R2) single - lrs3 >= 0.15
  with McNemar p < 0.10 -> "our generator wins at this teacher strength"; run `lrs_geo` on the pilot to bound teacher
  strength before any claim. (R3) precision < 0.6 (Wilson upper < 0.75) -> report "short-horizon handoff over-accepts".
  (R4) lrs3_cpi: same solved_full with >= 20 % fewer sim steps per accepted path, or >= +1 solved instance per 10 ->
  "C^pi helps LRS". Anything else -> inconclusive at n ~ 31; scale to 127 if LLM time <= 6 h.
- SFT (only if >= 15 full-verified LRS paths exist): pi1 recipe (init pi1-nominal, 6k steps, bs 32, r 0.5,
  round-robin) on LRS paths vs `single` trajectories of the SAME instances at the same trajectory count, 1 seed
  ("seed 0, preliminary"), Spatial test RSR + nominal; LRS within 3 pt of single -> 3 seeds.
- Minimum detectable effect at n 31 (alpha 0.10, power 0.8, ~35 % discordant): ~0.25; at n 127: ~0.12. The pilot is
  powered only for large gaps.

## T2Mem (test-time memory for fail -> recovery), status 2026-11-16
Design + gates: NOTES "T2Mem PoC"; results: RESULTS "T2Mem"; review: experiments/reviews/T2MEM_review.md.
Status: approach A (memory policy, repeat-breaking) closed as tested; RoboTTT-style failure-as-context parked (reopen
iff a history-vs-state probe gains >= 0.03 AUC or the train-split a2 repeat rate is >= 20 % with CI lower bound > 10 %).
Approach B (per-instance memory in search) closed, negative. The imagination recalibrator is inconclusive with realistic
labels: not used until more instances + an online n = 127 check.

## Next-stage phase plan (user request 2026-11-17; PLAN only, no experiment launched)
Full plan: `docs/plans/PHASE_PLAN_2026-11.md` (Vietnamese prose). Summary:
- **Phase 1 = WM-agnostic recovery data engine**: `WorldModel` / `Critic` / `Verifier` API (adapters WM v1, V-JEPA 2-AC,
  Cosmos-Predict2.5-2B robot/action-cond as tier-2 rescorer); levers aimed at the measured bottleneck (WM over-optimism
  on search-selected states): L1 verified-negative WM fine-tuning, L2 critic-on-imagination / recalibrator, L3
  pessimistic selection-aware scoring with a real heterogeneous ensemble, L4 option library + WM (gated on ACR-N F2),
  L5 learned proposal, L6 multi-fidelity + budgeted verification; metrics solved / hits / unique-solved per 1k sim, FP /
  gap / ECE vs N; n = 256 primary (MDE ~7 pt). Loop pi_k -> C^{pi_k} (refit) -> WM_k -> search -> bank -> students, with an
  "extra steps" control and stopping rules on pilot.
- **Part B**: LIBERO-Recover (2609.05178) harvests NATURAL failures of 6 policies with VLM localisation / levels and no
  injection; our suite generator does the same (automatic sim-predicate failure detection, rule-based L1-L4 calibrated
  against official labels, kappa >= 0.6), plus a separately reported `injected` generator; first target LIBERO-Goal
  (train on our scenes, test on the 91 official Goal scenes).
- **Phase 2**: three students of the same verified bank: S1 SmolVLA (SFT control; RECAP-style advantage conditioning,
  flow-DPO on paired success / fail bridges, ExIt rounds, DSRL / GRPO-lite RL baseline, WiSE-FT nominal guard; KI is
  moot while only the action expert is trained), S2 the WM + critic itself (evaluated by FP / calibration / verified
  yield of the NEXT search round), S3 Diffusion Policy (LIBERO and robomimic / robosuite head-to-head with SAILOR).
- Baselines: Back-on-Track-style test-time arm, SAILOR-style, no-WM sim-only search at equal sim, unverified WM data,
  teleop Combined, scripted regrasp, Kintsugi frontier. Milestones N0-N9 with gates (G-P1, G-loop, G-DP); each
  milestone gets its own pre-registered intake section here before it runs.

## 2610.05151 comparison: RetryVLA on THEIR protocol, design note (user request 2026-11-17 10:14 UTC; written BEFORE any result)
Full doc: `docs/related_work/COMPARE_2610.05151.md` (extraction, side-by-side, deviations D1-D4, compute).
- Research question: on 2610.05151's own protocol (pi0, custom LIBERO-Goal split, 100 val scenes, 300 steps, replan 5,
  seed 7), does RetryVLA data (WM + C^pi search, full-task sim verification by M0's continuation) give a post-training
  gain over "Original continuation" at least as large as their LLM-teacher data (78 -> 88)?
- Hypothesis: yes, Delta >= +5 pt (our Spatial gain over demo-only is +29 pt RSR, but their metric is nominal-start
  success where the headroom is only ~22 failures / 100).
- Falsified if: G0 passes and mean Delta (3 seeds) < +5 or CI includes 0.
- Closest prior work: 2610.05151, Kintsugi-VLA 2609.31048, Back on Track 2610.19579, MAGMA-GEN 2609.20056.
- Arms: Original (M0 + 2k on D0), RetryVLA (M0 + 2k on D0 + verified bank, their mix), no-WM sim search at equal
  verifications, RetryVLA with our mix (ratio 0.5). Their method itself: reported numbers only (no code, no LLM API).
- Instances: failures of M0 on the 300 train scenes only; val (100) for evaluation only; test (100) untouched.
- **G0 reproduction gate**: M0 (best of 1k-5k on val, as theirs) in [67, 87] and Original in [68, 88]; otherwise
  debug interface, at most one logged recipe change, no claims.
- Decision: "matches" mean Delta >= +5 with bootstrap CI > 0; "beats" mean Delta >= +10 with CI lower bound > +5;
  else "below". Cross-paper 88 vs ours reported only with deviations D1-D3. Power: n 100 paired, +10 pt ~ p 0.08 per
  seed -> 3 seeds.
- MDE (alpha 0.10, power 0.8, 3 seeds pooled, ~25 % discordant): ~7 pt.
- Status 2026-11-17: building infra (split, converter, Pi0Wrapper, eval_goal, pi0 LoRA SFT); heavy jobs queued
  behind ACR-N F2 (GPUs 1-3).
- **Status 2026-11-18: G0 PASS** (M0 4k = 85, Original = 82 / 100 val; RESULTS "G0").
- **Stage-2 pre-registration (2026-11-18 17:30 UTC, written BEFORE any audit / search result):**
  - Audit: M0 (4k) rolls out once per train scene with policy seeds 7 and 8 (300 + 300 rollouts, their protocol).
    Seed-7 failures = search instances. Seed-8 failures = WM / critic training data only.
  - Entries per failed rollout of length T: t in {T/4, T/2, 3T/4}, rounded down to a multiple of 5. An entry is
    restored bit-exactly by replaying M0's recorded action prefix from the init state.
  - WM / critic data (seed-8 entries): 32 candidates per entry (12 pi0 chunk samples at noise 1.0 / 1.5, 12
    perturbations, primitives), H = 50, then M0 continues for <= 300 steps. The outcome is the C^pi label.
    DINOv2-small 6x6 tokens every 5 steps.
  - The Goal WM and C^pi are trained with the Spatial v1 recipes (`m2_train_wm`, `m2_train_critic`, same
    hyper-parameters).
  - Search on seed-7 entries, 16 sim verifications per entry per arm, paired continuation seeds:
    - `random`: 16 random members of the 256 pool.
    - `wm_diverse`: 256 pool + 2 CEM rounds in imagination, J = C^pi - mu R, diverse top-16 (= the pi1 generator).
  - Bank: verified bridge + M0 continuation, <= 4 per entry.
  - Distillation, their recipe: 2,000 updates from M0, D0 + bank pooled uniformly (no oversampling), warmup 200,
    cosine. 3 seeds (7, 8, 9) for Original, RetryVLA(`wm_diverse`) and RetryVLA(`random`). Same trajectory count
    per arm (`random` bank subsampled to the `wm_diverse` count, round-robin over entries).
  - Primary: Δ = RetryVLA(`wm_diverse`) − Original on the 100 val scenes, scene x seed bootstrap. Decision rules as
    above ("matches" / "beats" / "below"). Secondary: `wm_diverse` vs `random` (solved entries, McNemar p < 0.10;
    SFT Δ, bootstrap).
  - Test split untouched until the recipe is frozen. The cross-paper reference stays their 88.

## Agentic recovery data vs search + verify, and jerkiness of searched recoveries (user request 2026-11-20; review + offline analysis only, written BEFORE any SMOOTH-V result)
Full doc: `docs/reviews/AGENTIC_SMOOTHNESS_REVIEW.md` (smoothness numbers, literature with verified IDs, reviewer Q&A,
ranked merge designs). Analysis: `scripts/analyze_smoothness.py`, `scripts/smoothness_policy_proxy.py` ->
`runs/_analysis/smoothness/`.
- Finding (CPU, 0 sim): pi1-bank bridges have mean ||Δ³a|| 4.7x the Spatial demos (Goal 3.4x), but pi0's own
  continuations are 4.2-4.3x with replan seams 2.9-4.4x; jerk is not associated with success in the pool, and 81 % of
  solved instances have a smooth (primitive / VLA s1.0) solution. Smoothing shifts the open-loop path ~4 mm -> must be
  re-verified. Offline proxy (same obs + noise): pi1 chunks are ~2.4x jerkier than pi1-nominal at failure states
  (Δd3 +0.41 [+0.31, +0.51], n 40), so the jerk is distilled into the policy.
- Novelty: agentic failure-driven data engines are taken (2610.05151, 2611.17394 Agentic-Teacher, 2610.01178,
  2610.29161); so is policy-compatibility filtering (2611.17394 FM-loss filter). Open: C^pi as selector / acceptance
  for any generator (agent arm already pre-registered as LRS `lrs3_cpi`).
- Next (pre-registration draft in the doc, section 5): **SMOOTH-V** = `smooth_select` / `smooth_reverify` /
  `smooth_bridge_only` vs pi1 at equal trajectory count, 3 SFT seeds, gate G0 re-verify yield >= 70 %, closed-loop
  action logging added to eval. Rules: "helps" ΔRSR >= +3 (CI > 0) or nominal >= +2 with RSR non-inferior; "free"
  |ΔRSR| < 3 and closed-loop d3 -25 %; "hurts" ΔRSR <= -3 (CI < 0). MDE ~4-5 pt. Not launched (no GPU used; the
  2610.05151 chain and its GPUs untouched). Then merge (i) agent-written option library ranked by WM + C^pi on ACR-N U.
- Status 2026-11-20: intake only; adversarial review (idea-loop step 4) required after the SMOOTH-V PoC.

## Cut-point study: where to cut failed rollouts for recovery data (user request 2026-11-21; research + offline analysis only, written BEFORE any CUT-1 result)
Full doc: `docs/research/CUT_POINT_STUDY.md` (LIBERO-Recover generation recipe + replication on Goal, sim / real-world
cut-point taxonomy, real-world detector -> WM-branched recoverability pipeline, offline numbers, CUT-1
pre-registration, open questions). Analysis: `scripts/analyze_cut_point.py` -> `runs/_analysis/cut_point/summary.json`
(CPU, no sim).
- Findings: the LIBERO-Recover release cuts on a fixed step grid (~T/4, T/2, 3T/4 of time-outs; eval scenes = the late
  point + 20) and injects its L4 scenes (181 `l4_injected`), despite the paper text; the level depends on the cut.
  On Goal (wm_diverse), T/4 -> T/2 -> 3T/4: solved 0.77 / 0.66 / 0.34, hits / 1k sim 1.52 / 0.84 / 0.31, bridge
  ||Δ³a|| 0.17 / 0.32 / 0.44; T/4 entries are largely pre-onset (M0 re-succeeds from 46 %, EEF 6 mm from D0) and are
  48 % of the F0 recovery anchors (22.5 % of the whole mix); tasks with no recovery data also lost, so part of the F0
  drop is volume, not cut.
- Pre-registered CUT-1 (doc section 4.4): cut rule {fixed, onset (sim branching oracle), onset + 2 s, pre-onset,
  injected, critic drop} at the F1 bridge-only 4.1 % share, 3 SFT seeds, pooled seed-9 own-failure RSR (cross-cut
  matrix) + nominal val. "Onset helps" iff ΔRSR vs fixed >= +5 (CI > 0) with nominal >= Original − 2; MDE ~5 pt.
  Gated on F1. Cheapest next step: 20-step branching scan of the 56 seed-7 rollouts (p_pi(t) curves, t_on).
- Status 2026-11-22: intake + offline analysis only; not launched; adversarial review (idea-loop step 4) after CUT-1.

## KIN-1: expert vs student recoverability vs learning utility for recovery-data selection (Kintsugi-VLA 2609.31048), design note (user request 2026-11-23; written BEFORE any KIN-1 result)

Background and verified facts: `docs/research/NOVELTY_VS_KINTSUGI.md`. Status 2026-11-23: pre-registered only;
nothing launched. Queue: on 953bb534 after F1 and after the Back on Track evals; coordinate with both agents first.

- **Research question**: when we pick recovery data from the base policy's own failures, which signal predicts and
  maximises the VLA's realised gain: expert recoverability p^E (Kintsugi), frozen-student recoverability / handoff
  (2610.05151, RE-0), or an estimate of the learning benefit Delta p^pi?
- **Hypotheses**:
  - H1 (gap): across states, Spearman(p^E, realised Delta p) is weak (<= 0.3).
  - H2: a proxy Û, calibrated on micro-fine-tunes, predicts Delta p better, by Delta rho >= 0.2.
  - H3: at equal data size and recipe, RLU-selected data gives >= +5 pt held-out RSR over the Kintsugi-style
    selection, and >= +3 pt over uniform-among-verified handoff selection.
- **Falsified if**: rho(p^E, Delta p) >= 0.6 (expert recoverability already aligned, so the gap claim is dropped), OR
  RLU-E is within ±3 pt of HAND-E (then selection beyond verification adds nothing, which replicates RE-0's
  "admission, not weighting" result).
- **Closest prior work**: Kintsugi-VLA 2609.31048, 2610.05151, RE-0 2609.32416, TimelyDAgger 2609.33157, CUPID
  2506.19121, DataMIL 2505.09603 (added to RELATED_WORK via the novelty doc).
- **Policy / recipe (Phase 2, frozen before results)**: pi_b = pi1-nominal (RESULTS row 1, the init of every M5+ SFT).
  SmolVLA SFT 6k steps, bs 32, lr 5e-5, **bridge-only labels** (segment 0), recovery share 0.2, nominal replay 0.8,
  equal B = 200 bridges per arm (round-robin over source instances), SFT seeds 0 / 1 / 2. Deviation trigger: if F1
  shows that bridge-only at a small share is harmful on Goal, keep the recipe anyway and report it with that
  caveat. Any other change goes to NOTES as a deviation.
- **Instances / split** (train only; test untouched for selection): Spatial train has 56 groups. Group-stratified,
  seed 0: **fold A** = 36 groups, used for sources; **fold B** = 20 groups, used for held-out eval.
  - N = 100 pi_b-failure instances in fold A (scene x gripper mode, precheck 2 rollouts).
  - Candidate start states per instance: the failure snapshot plus pi_b's failed continuation, logged every
    40 steps.
  - **Kintsugi window**: t_dev = 0 (the snapshot is already off-nominal, their default); t_f = p^E frontier
    (tau 0.2, Wilson, ladder 5 -> 15 -> 30). Start states at depths 0.15 / 0.50 / 0.85, plus 3 uniform draws.
    That makes 7 start states per instance.
  - Optional replication on LIBERO-Goal (pi0 LoRA, 2610.05151 protocol) only if Spatial is positive.
- **Expert pi_E (deviation D-K1 from PPO)**:
  - Primary: a scripted privileged pick-place controller for the Spatial family. It reads object / plate poses from
    the sim and outputs OSC delta-EEF + gripper actions (7-D, the SmolVLA action space). It can build on the eef
    command executor in `retryvla/baselines/lrs_teacher.py`. Stochasticity nu = action noise (sigma 0.05) plus
    jittered approach height / grasp offset.
  - Gate E0: nominal success >= 90 % on 50 train inits, and recovery from the failure snapshots reported.
  - Fallback if E0 fails: an oracle sim-CEM closed loop with a privileged progress objective (expensive; log the
    deviation).
  - A small PPO teacher is not cheap in CPU-bound LIBERO (~10^7 steps / task), so it is skipped.
- **Measurements on all 700 start states**:
  - p^E: MC ladder, state-only, no rendering, CPU.
  - p^{pi_b}: 8 continuations, stop early at 2 successes.
  - Candidate pool per state:
    - expert prefix (first 50 steps of pi_E);
    - top-8 of 256 WM v1 + C^pi ranked pool bridges;
    - every candidate verified: bridge, then pi_b continues; 2 seeds, plus 2 more on accepted candidates.
  - Proxies P1-P7 (doc section 3.2).
- **Realised Delta p (proxy validation, oracle arm first)**:
  - Calibration: 30 groups (group = verified items of one start state). M = 16 micro-FTs from pi_b, each on a
    random half of the groups (600 steps, same recipe). Every micro-model is evaluated on the 30 start states,
    10 trials each.
  - Deltâ_g = success(included) - success(excluded).
  - Validation: 20 fresh groups, 8 more micro-FTs.
  - Û = ridge / monotone fit of P1-P6, leave-group-out CV on the calibration groups. Primary Û frozen by CV
    Spearman before any main SFT.
  - Also: rho of p^E, p^{pi_b}, G = Ĉ(s_H) - p^{pi_b}(s) and Û with Delta p (Spearman matrix + bootstrap CI over
    groups).
- **Arms** (equal B, same recipe, 3 seeds each):
  - Primary:
    - C0: nominal replay only, same steps.
    - K-U: uniform window, expert bridges (Kintsugi V2).
    - **K-T: Kintsugi-style targeted depths, expert bridges** (main contrast).
    - HAND-E: expert bridges accepted by verified handoff, uniform among accepted (2610.05151-style, with
      ground-truth handoff).
    - RLU-E: top-Û among verified expert bridges + P7 diversity.
    - **RLU-full: top-Û among all verified candidates (expert + WM pool)** = RetryVLA.
  - Secondary, if budget allows:
    - K-T-full: Kintsugi-faithful labels = full expert continuation (the main-table row).
    - K-pi: depths on the *student* frontier p^{pi_b} (links to CUT-1 `onset`).
    - GAIN-E: rank by handoff gain G (RE-0-style).
- **Primary metric**: held-out own-failure RSR on fold B.
  - Eval states: 20 groups' instances plus their pi_b-continuation states at 0 / 120 / 240 steps (~240 states),
    3 trials each, LIBERO-Recover protocol (±2 cm, open / closed, 520 steps).
  - Paired over states; group x seed bootstrap (95 %) and sign-flip test.
  - Report per level L1-L4 where labelled.
  - Secondary: in-situ RSR on the fold-A source states (realised Delta p at full scale for the selected items, a
    second proxy check); Spatial test RSR (dev split) for C0 / K-T / best arm only; nominal Spatial 10 x 50.
  - Budget-normalised: RSR gain per sim-hour, **including** the p^E / p^pi measurement and verification (Kintsugi
    excluded this).
- **Decision rules**:
  - Gap supported: rho(p^E, Delta) <= 0.3 AND rho(Û, Delta) - rho(p^E, Delta) >= 0.2 (CI > 0), on the
    50 calibration + validation groups. Gap falsified: rho(p^E, Delta) >= 0.6. In between: "partly aligned"; report
    the matrix, no gap claim.
  - Kintsugi replicated in our stack: K-T - K-U >= +3 (CI > 0). If not, report "depth targeting does not transfer
    to LIBERO-Recover with this expert" (it is not a failure of RLU).
  - RLU beats Kintsugi: RLU-full - K-T >= +5 with CI > 0, AND nominal(RLU-full) >= nominal(C0) - 2.
  - Beyond handoff: RLU-E - HAND-E >= +3 with CI > 0. If within ±3: no selection claim; the mechanism reduces to
    verified admission (RE-0), and we report that.
  - WM generator matters: RLU-full - RLU-E >= +3 with CI > 0 at equal or lower sim cost.
  - Inconclusive: power < 80 % at the observed variance. Then write "inconclusive at n = ...", never "negative".
- **Power**:
  - ~240 paired eval states, per-state difference SD ~0.3 (CUT-1 assumption), alpha 0.10 two-sided, power 0.8:
    nominal MDE ≈ 2.49 x 0.3 / sqrt(240) ≈ 4.8 pt. With a design effect for 20 correlated groups (~1.5-2x variance),
    MDE ≈ 6-7 pt. Kintsugi's own +5.8 pt is therefore only ~60-70 % powered. If K-T vs K-U is null, call it
    inconclusive, not a refutation.
  - Proxy rho on 50 groups: detectable rho ≈ 0.35 (Fisher z, alpha 0.10, power 0.8).
- **Compute** (unit costs from NOTES / VERIFICATION_COST: ~6 s / verification amortised, SFT 6k ≈ 1.3 GPU-h, RSR
  eval ≈ 1 GPU-h / 560 episodes, nominal 0.4 GPU-h):

  | item | GPU-h |
  |---|---|
  | expert engineering + gate E0 | ~1.5 days of engineering, <1 GPU-h |
  | p^E (~700 states x ~12 rollouts, state-only, CPU) | ~0, plus ~3 CPU-h |
  | p^{pi_b} (700 x ~8) | ~9 |
  | candidate verification (700 x 9 x 2, plus re-seeds) | ~25 |
  | proxies (FM loss, gradients) | ~2 |
  | calibration + validation micro-FTs (24 x (8 min + 300 eval episodes)) | ~16 |
  | 6 primary arms x 3 seeds: SFT 1.3 + held-out eval 1.3 + in-situ 0.5 + nominal 0.4 | ~63 |
  | secondary arms (3 x 3) + Spatial test for 3 arms | ~35 |

  Primary total ≈ **115 GPU-h ≈ 2.5 days on 953bb534's 2 A100s** (primary only), ~3.3 days with the secondary arms.
- **Gating order**:
  1. E0.
  2. Measurement + calibration (≈ 52 GPU-h, about 1 day). The gap verdict comes out here and can be written up even
     if the SFT arms are null.
  3. Only if rho(p^E, Delta) < 0.6: main arms. Otherwise run only C0 / K-U / K-T for the main-table row.
- Review: idea-loop step 4 (`retryvla-idea-review`) after the measurement stage and again after the arms.

## Back on Track + ForkCast-style test-time arms (user request 2026-11-23; coordinator re-prioritisation 08:05 UTC: LIBERO-Goal protocol FIRST; written BEFORE any result)
Repro doc: `docs/related_work/BACK_ON_TRACK_REPRO.md` (method extraction of 2610.19579 + 2605.11479, every deviation,
fairness note). ForkCast note: `docs/related_work/notes/2610.18650.md` section 9. Code: `retryvla/baselines/bot/`
(collect, liveness, train), `retryvla/policy/bot.py` (`BackOnTrackPolicy`, `ForkCastPolicy`), opt-in hooks in
`eval/{rollout,eval_goal,eval_recover,eval_nominal}.py` (default paths unchanged). Chain `scripts/bot_goal_chain.sh`.
- Research question: on the 2610.05151 LIBERO-Goal protocol (pi0 3B M0 4k, 100 val scenes, seed 7, <= 300 steps,
  replan 5), how much does test-time WM planning (Back on Track) or test-time WM selection (ForkCast-style) add to M0,
  and how does that compare with distillation (Original / RetryVLA banks / F1) at its inference cost?
- Hypothesis: small gain at best on this protocol (M0 already 85 / 100; nominal starts, not their perturbed starts;
  our Goal WM is trained on failure-state bridges, not on M0 rollouts + play data): BoT(C^pi) Δ vs M0 in [-5, +5].
  Back on Track reports +8.9 pt on perturbed LIBERO-Goal with pi0.5.
- Arms (Goal, all paired on the same 100 val scenes and pi0 seeds; base trajectories are identical until a trigger):
  1. M0 (existing, 85 / 100), Original s7-s9 (79.3), F0 RetryVLA `rec_wm` s7-s9 (68.0), F1 (other agent, when done).
  2. **M0 + BoT (C^pi)**: monitor + planner value = 1 - 2 C^pi (our Goal critic, `goal_latest_critic`), labelled variant.
  3. **M0 + BoT (liveness)**: their value, trained per their recipe on 900 on-policy M0 rollouts of the 300 TRAIN
     inits (policy seeds 8 / 9 / 10).
  4. **M0 + ForkCast-style (progress)**: Commit + 4 library primitives (retreat20, regrasp, reposition_0 / 180),
     WM v1 (Goal) imagination to a common 50-step horizon, ForkCast progress head (pooled tokens, LN, 2-layer MLP;
     label t / T on the same successful M0 rollouts), argmax with Commit winning ties, first decision at step 60.
  5. **M0 + ForkCast-style (C^pi)**: scorer swap only.
  6. Complementarity: Original s7 + BoT and RetryVLA `rec_wm` s7 + BoT (C^pi and liveness; the value stays
     conditioned on M0, as their "filter on top of Filtered-BC"); F1-best + BoT when F1 lands.
  Fixed hyper-parameters (theirs, NO tuning on val): ensemble 5, W = 3, eta_t = sigmoid(t_c 60, sigma 10) -> 0,
  H = 10, h = 5, CEM 256 x 5 (= their 1,280 plans), 32 elites (unstated in the paper; ours), PCA 98 %, s = 0.6,
  gripper = nominal, gamma 0.993. V-JEPA 2-AC: not on Goal (no Goal-trained adapter); WM v1 (Goal) only.
- Secondary (after Goal, same shell): LIBERO-Recover Spatial test (dev, 560 eps) pi0 + BoT(liveness / C^pi),
  pi1 s0-s2 + BoT(liveness on each pi1's own train rollouts), Object test pi0 + BoT (Object pi1 checkpoint was deleted
  in a disk cleanup -> no Object pi1 + BoT), nominal Spatial for pi0 / pi1 + BoT. Same decision rules.
- Metrics: Goal SR (/100) with Wilson CI; rescued / broken vs the base policy (their Table I metric); trigger rate on
  base-failed vs base-succeeded scenes; inference cost per episode (VLA / monitor / planner GPU wall time, #WM rows,
  #triggers) and batch-1 latency (`bot_bench`); RSR overall + L1-L3 on Spatial.
- Stats: McNemar exact (paired per scene) vs the base policy; for multi-seed policies scene x seed bootstrap (10k) of
  Δ; Wilson CIs for rates. n = 100 val scenes; with M0 at 85 only 15 failures can be rescued.
- MDE: one-arm-vs-M0 McNemar at n = 100 with ~10-15 discordant pairs (alpha 0.10, power 0.8): ~8 pt. Everything
  below that is "inconclusive at n = 100", not "no effect".
- Decision rules:
  - "BoT helps on this protocol": Δ(M0 + BoT − M0) >= +3 with McNemar p < 0.10 (rescued > broken).
    "BoT hurts": Δ <= −3, p < 0.10. Otherwise inconclusive.
  - "Distillation matches test-time planning at lower inference cost": |SR(best distilled) − SR(M0 + best BoT)| <= 3 pt
    (bootstrap CI within ±8) AND inference-cost ratio (BoT per-episode GPU time / plain policy) >= 10x. If the ratio is
    < 10x we only state the measured ratio.
  - "Test-time beats distillation": M0 + BoT − best distilled >= +3 with CI > 0. "Distillation beats": the reverse.
  - "Complementary": policy + BoT beats both the policy and M0 + BoT by >= 3 pt (CI > 0).
  - ForkCast scorer: "C^pi better than progress as a test-time selector" iff Δ(fc_cpi − fc_prog) >= +3 with McNemar
    p < 0.10.
- Compute: one RetryVLA 1-GPU shell 92927808 (`vla_RetryVLA_DoanhLT_bot1g`, user account, coordinator OK); 953bb534
  stays with the Goal-improvement campaign. Goal chain ≈ 1 h collection + 0.3 h training + 8 evals x ~0.2-0.4 h.
  Spatial secondary ≈ 15-20 GPU-h. Shell is killed when the queue is done.

## Goal campaign: how far can RetryVLA data improve pi0 on the 2610.05151 protocol (user priority 2026-11-23 08:05 UTC; written BEFORE any stage-1b / 2 / 3 / 4 / 5 result)
Reference numbers (val, 100 scenes): M0 85; Original 82 / 77 / 79 (79.3); F0 (47 % recovery anchors) wm 68.0,
random 67.7; paper Original 78 -> Recovery 88.
Interim F1 (4 % bridge-only anchors, before this note): wm 58 / 69 / -, random 49 / - / -, i.e. worse than F0.
Artifact check D1 (done before this note): orig_s7 cast to bf16 = 83 vs 82 fp32, so the bf16 final ckpt is not the
cause. f1_wm_s8 (no resume, 69) and f1_wm_s7 (2 resumes, 58) are both low, so the resume path is not the main cause
either (it stays a logged deviation). F1 is analysed when complete (rules in RESULTS "main comparison").
Compute: 953bb534 (2 GPUs) + second shell 8f473147 (vla_RetryVLA_DoanhLT_2g_b, requested 2026-11-23 08:25; queue > 12 h).
Common: their distillation recipe (M0 + 2k, warmup 200, cosine, D0 + bank pooled uniformly) unless a stage varies it.
3 SFT seeds (7, 8, 9). Bootstrap = scene (or entry) x seed. Test split untouched until stage 5.

**Stage 1b, own-failure RSR (diagnostic, idea-loop step 3).** Does recovery data teach recovery at all, while it costs
nominal?
- Eval set E8 = the 168 seed-8 fixed entries (M0 failures of the seed-8 audit; used for WM / C^pi training, never in
  any SFT bank), 1 trial each. Restore by replaying M0's actions, then the policy acts alone for <= 300 steps.
  `retryvla.eval.eval_goal_entries`.
- Policies: M0, Original s7-9, rec_wm s7-9, rec_rand s7-9, F1 wm / random s7-9 (when done).
- Reading:
  - "data teaches recovery" iff mean RSR_E8(rec_wm) − RSR_E8(Original) >= +5 (CI > 0).
  - Else: "the bank does not transfer to held-out own failures"; go to the generator / cut / label levers
    (stages 2-4) with RSR_E8 as the co-primary metric.

**Stage 2, CUT-1 lite (onset cut; docs/research/CUT_POINT_STUDY.md 4.4, reduced to the `onset` / `pre_onset` rules).**
- Onset scan (`retryvla.search.goal_onset`) on the 56 failed M0 rollouts of audit seed 7, and on seed 8 for
  evaluation:
  - backward 20-step grid, K = 8 M0 continuations, need 2;
  - t_last = last point with >= 2 / 8, t_on = t_last + 20; never-recoverable rollouts excluded.
- Bank: seed-7 onset entries, wm_diverse search (existing Goal WM / C^pi, 256 pool + 2 CEM, 16 verifications), then
  36 bridges round-robin, bridge-only (= F1 share, ~4 %). Arm `onset` vs F1 `wm` (fixed T/4-T/2-3T/4) at equal data,
  3 seeds.
- Eval: nominal val (their protocol) + RSR on E8 ∪ E8on (seed-8 onset states, 1 trial).
- Rules:
  - "onset helps": RSR(onset) − RSR(F1 wm) >= +5 (CI > 0) AND nominal(onset) >= nominal(F1 wm) − 2.
  - "nominal restored": nominal(onset) >= Original − 2.
  - "cut is not the bottleneck": |Δ RSR| < 3 and |Δ nominal| < 3.
  - pre_onset is a secondary 1-seed arm only if onset helps.

**Stage 3, SMOOTH-V on Goal (docs/reviews/AGENTIC_SMOOTHNESS_REVIEW.md 5).** On the stage-2 winner bank:
- `smooth_select` (lowest-jerk verified bridge per entry first);
- `smooth_reverify` (Savitzky-Golay (11, 3) on the bridge + 5-step handoff blend, replay from the snapshot, keep
  only re-verified successes, refill from the same entry).
- Gate G0: re-verify yield >= 70 %.
- Rule: "smoothness helps" iff nominal >= +2 or RSR >= +3 vs the unsmoothed bank (CI > 0, the other within −2).

**Stage 4, training levers (Phase 2) on the best bank so far, each vs that bank's default:**
- (a) share sweep 1 / 2 / 4 / 8 % (9 / 18 / 36 / 72 bridges), 1 seed first, the 2 best x 3 seeds;
- (b) loss masking = their validity mask: chunk targets beyond the bridge end are masked (no M0-continuation
  actions as targets), noisy action tokens too;
- (c) segment choice: bridge + continuation vs bridge only at equal anchors;
- (d) nominal replay / preservation: action-matching L2 to M0's own chunk on D0 states (weight 0.5) vs none.
- Selection rule: maximise nominal val subject to RSR_E8 >= best-so-far − 2; ties -> fewer recovery anchors.
- Pre-registered success: nominal >= Original + 5 (CI > 0) = "matches theirs"; nominal >= Original and
  RSR_E8 >= +10 = "recovery without nominal cost".

**Stage 5, final:**
- Frozen best recipe x 3 seeds plus Original x 3 seeds on the 100 TEST scenes (once), vs the paper's Recovery 88
  (cross-paper reference only).
- RSR on a fresh seed-9 audit's failure entries (held out from everything), both cut rules.

**Cost / ETA (2 shells, 4 GPUs):**

| stage | cost | done by (UTC) |
|---|---|---|
| F1 | - | 11-23 ~11:30 |
| 1b | ~15 GPU-min per policy | 11-23 ~14:00 |
| 2 onset scans | ~2-4 h per audit seed on 1 GPU | 11-23 ~13:00 |
| 2 search | ~1.5 h | - |
| 2 SFT 3 seeds | ~3 h on 3 GPUs | 2 total 11-23 ~20:00 |
| 3 | ~8 h | 11-24 ~06:00 |
| 4 | ~20 runs x 2.8 h / 4 GPUs ≈ 14 h | 11-24 ~22:00 |
| 5 | ~6 h | 11-25 ~06:00 |

Every stage gets its RESULTS entry + reading + decision against these rules before the next stage's code / config
is changed (deviations logged in NOTES).

## CRIT-V: critic variants (explicit 3D / chain-of-dynamics / implicit) + rich verification records, design note (user request 2026-11-23; written BEFORE any CRIT-V result)
Status 2026-11-24: offline done, no variant passes (RESULTS "CRIT-V"); no pilot / hand-off; review pending (not closed).
Doc: `docs/research/CRITIC_VARIANTS.md`. Code: `retryvla/critic/{rich.py, variants.py, train_variants.py, eval_variants.py}`,
`scripts/crit_rich_replay.py`, opt-in `m5b.rich_record` in `retryvla/search/compare_arms.py` (default path unchanged).
Compute: own 1-GPU shell `a3da0634` (user account); never 953bb534 (Goal campaign priority) / 92927808 (BoT).
- Research question: does adding explicit 3D (point clouds), an explicit chain of sub-goal dynamics, or implicit
  dynamics objectives to C^pi reduce the CRITIC share of search over-optimism (~15-20 %; RESULTS "RQ3 follow-up") and
  improve within-instance ranking of verified candidates, on imagined and on true end states?
- Hypotheses (expected direction / size):
  - H1 3D-true (point-cloud tokens from sim depth + segmentation, TRUE end state): AUC(true) +0.01..0.03 over the
    retrained C^pi control; mostly on contact / placement failures. 3D-pred (3D tokens predicted from the WM latent,
    head trained on sim labels like the RGR geometry head): smaller or no gain on imagined states (the WM latent is the
    bottleneck, RGR head AUC 0.86 real -> 0.61 imagined).
  - H2 CoT (end state -> chain of sub-goal tokens: contact / grasp -> object displacement & lift -> BDDL predicate
    progress -> success conditioned on the chain): better calibration (ECE) and FP@top-4 on search-selected picks,
    because the success head must route through physically checkable intermediate quantities. Control: same
    architecture with chain loss weight 0 (separates supervision from capacity).
  - H3 IMPL (imagination-consistency: critic embedding of the WM-imagined end state regressed onto the stop-grad
    embedding of the TRUE end state of the same candidate + start->end latent-dynamics prediction in the critic's own
    space + distributional time-to-success head): smaller imagined-vs-true gap (C(a) - C(b)) on search picks.
- Falsified if: no variant beats the retrained C^pi control by the decision rule below on goal_verif imagined end
  states, AND the 3D-true oracle does not beat it on true end states (then 3D information is not what the critic lacks).
- Deviation from the request, decided before results: the existing banks store only bridge latents, no continuation.
  Re-running every verification with the policy (Goal search took ~9 GPU-h) is out of budget, so the rich replay is
  BRIDGE-ONLY (open-loop prefix from the exact snapshot, bit-exact, no policy); the chain is therefore the end-state
  sub-goal chain + start->end deltas, not pi's future timeline. pi-continuation timelines (predicates every 5 steps,
  point clouds every 50) are recorded by the new opt-in `m5b.rich_record` for future verification runs (Phase 2).
- Closest prior work to check: PointWorld (CVPR 2026, point-world.github.io), ThinkWorld (chain-of-dynamics, arXiv
  id to verify), DreamerV3 / TD-MPC2 / SPR (self-predictive latents), distributional RL (C51), 3D value / affordance
  critics (e.g. 3D Diffuser Actor features); goes to RELATED_WORK.
- Arms (all critics: same end-state input z_H (72 DINOv2 tokens, 2 cams) + q_H + task, same training data, same steps,
  50 % imagined end states, 3 seeds): (0) C^pi-ret = RecoveryValue recipe retrained by the same script (CONTROL);
  (0') shipped C^pi (Goal `goal_latest_critic`, Spatial C^pi v2) and generic success critic (Spatial B2 `wm_success`;
  none exists for Goal) as references; (1a) 3D-true; (1b) 3D-pred; (2) CoT; (2') CoT-nochain (lambda 0);
  (3) IMPL; (3') IMPL-noalign (only the distributional + dynamics heads).
- Data / split (train split scenes only, no test data): Goal: train goal_collect (seed-8 entries, 5,376), eval
  goal_verif (seed-7 entries, 168 x 32 = random 16 + wm_diverse 16). Spatial: train collect_v1 + verif_r1 (C^pi v2
  data), eval verif_m5b (256 x 64, random / wm_diverse / wm_noise / wm_qd). Model selection: none (fixed steps).
- Metrics (eval banks, imagined end states = each suite's WM open-loop rollout of the logged bridge; true = logged
  latents): AUC, ECE (10 bins), p@1 / p@4 (per instance, rank all verified candidates by the critic, success rate of
  the top-k, mean over instances), FP@top-4 = P(fail | in the per-instance top-4 and pred >= 0.5), mean pred - SR on
  search picks (wm_diverse). Stats: instance-cluster bootstrap (2,000) of the paired difference vs C^pi-ret, seed-mean.
- PRIMARY: goal_verif, imagined end states, variant vs C^pi-ret.
- Decision rule "clearly wins" (-> small Goal search pilot + hand-off to the campaign): dAUC >= +0.03 with 95 % CI > 0
  AND dFP@top-4 <= -0.05 AND dp@4 >= 0, all seed-mean on the PRIMARY set, AND Spatial verif_m5b imagined dAUC >= -0.01.
  "Promising": dAUC >= +0.015 with CI > 0 or dFP@top-4 <= -0.05 with CI < 0 -> document, no pilot, follow-up listed.
  Otherwise "no gain at this design / data size" (idea-loop diagnosis + review before any close).
- Pilot (only if "clearly wins"): goal_search config, seed-7 TRAIN entries, first 48 entries, arms wm_diverse (C^pi)
  vs wm_diverse (variant), 16 verifications each, paired continuation seeds; report solved, hits / verification;
  McNemar. On a3da0634 only.
- MDE: with 168 Goal instances, instance-bootstrap SE of a paired AUC difference is ~0.01 -> MDE ~0.025-0.03 (alpha
  0.05 two-sided, power 0.8); smaller effects are "inconclusive at n = 168".
- Storage budget <= 15 GB new data (fp16 points, 512 pts / frame, 6 frames / bridge -> ~0.7 GB for ~33k bridges).

## RetryVLA v2 re-orientation: pointer (2026-11-23; plan doc only, no rules added here)
User re-oriented the project to "From Offline Recovery Data Generation to Recovery-Aware Policy Learning". Full plan
(Vietnamese prose): `docs/plans/RETRYVLA_V2_PLAN.md`; related work verified on 2 sources: `docs/related_work/RELATED_WORK.md`
section P (CARF 2609.21982, RePO-VLA 2605.09410 = NeurIPS 2026, RECAP 2511.14759, FAR 2607.01111 = CoRL 2026, CrossVLA
2605.21854, Failure-to-Success Bridging WIP). Summary: Algorithm 1 (natural failures -> CUT-1 localisation -> WM search
-> C^pi ranking -> sim verification) outputs D_N / D_R (verified bridges from s_f) / D_P (verified matched-state
preventive pairs a+ > a- at the pre-failure checkpoint s-) / D_F (negatives, never BC); Algorithm 2 trains
lambda_N L_N + lambda_R L_R + lambda_P L_P (bounded hinge on matched pairs, not DPO) + lambda_A L_A (velocity anchoring
to the frozen base on nominal states), optional gate g(h). Stages P2-0 .. P2-5 on LIBERO-Goal (2610.05151 protocol),
P2-3 (+ verified pre-failure positives) decisive. **No P2 stage is pre-registered yet**: each needs its own intake here
(owned by the Goal-campaign agent) before it runs; the doc's thresholds are proposals, not decision rules.

## Goal campaign G2 (curated onset bank + continuation) and P2-3 (prevention pairs D_P), pre-registration (coordinator 2026-11-24 12:40 UTC; written BEFORE any G2 / P2-3 result)
Inputs:
- F1 diagnosis (RESULTS 2026-11-24): data effect, no pipeline bug. Bridge-only open-loop targets hurt (D3 63);
  the same 36 trajectories with continuation anchors are harmless (D5 80 vs Original 82).
- Stage 1b: F0 data lifts own-failure RSR +13.5.
- Onset scans: seed 7 gives 52 onset + pre-onset entries, seed 8 gives 53.
Common recipe: their distillation (M0 + 2k, warmup 200, cosine, D0 + bank pooled uniformly, no oversampling),
bf16 / resume path (D2 = 84, validated), 3 seeds (7, 8, 9), 100 val scenes.

**G2 bank (curated):** seed-7 onset cut states (52), compare_arms `goal_search_onset_cur.yaml`:
- pool = 128 pi0 samples at noise 1.0 + 8 fixed + random primitives; no perturbation family; candidates with
  > 1 gripper sign change dropped;
- WM v1 Goal + C^pi, `wm_diverse` + `random`, 16 verifications, <= 4 verified trajectories per entry;
- trajectories = bridge + M0 continuation (segments 0 + 1).

**Arms (anchors counted over the whole trajectory):**
- `G2` = curated wm_diverse bank, 36 trajectories round-robin (≈ D5's share ≈ 4.4 k anchors ≈ 10 %);
- `G2r` = curated random bank, same count;
- `G2-72` = 72 trajectories (if the bank has them), 1 seed first;
- controls: Original, D5 (old fixed-cut uncurated bank, same count, 1 seed; seeds 8 / 9 added).

**P2-3 D_P (prevention pairs, `retryvla.search.goal_prevent`):**
- At each pre-onset state s^- of seeds 7 (bank) and 8 (held-out eval, never trained): a- = M0's recorded next 50
  actions.
- Candidates = 24 pi0 samples at noise 1.0 + 8 fixed primitives (<= 1 gripper flip), 1 screening trial each, K = 3
  for screen hits and for a-.
- Pair iff p(a+) >= 2/3 and p(a+) − p(a-) >= 1/3; <= 2 positives per s^-.
- D_P trajectory = a+ + M0 continuation.

**P2-3 arms (equal trajectory count N = 36 + n_P):**
- Policy A = D0 + curated D_R (G2 bank) with N trajectories;
- Policy B = D0 + G2 bank (36) + D_P (n_P);
- n_P = min(36, available). If n_P < 15: underpowered, run B at 1 seed only and report "inconclusive".
- The hinge L_P (v2 plan Algorithm 2) is NOT used in this first test; D_P enters as flow-matching positives
  (= v2 ablation "s"). Documented choice: the cheapest decisive test of "does prevention data help beyond
  recovery data".

**Failure-event detector E-v1** (`retryvla/eval/goal_events.py`, frozen at this commit): grasp_lost, displaced
(collateral), tipped, off_table, stall. Evaluated offline by bit-exact replay of the saved val actions
(`eval_goal --save_actions`). Not validated against hand labels yet (v2 plan κ check pending), so its metrics are
secondary.

**Metrics:**
- (a) nominal SR (primary), Δ vs Original, scene x seed bootstrap;
- (b) failure incidence (E-v1);
- (c) first-attempt success;
- (f) fixed / broken vs Original and vs M0, per task;
- (g) RSR_E8 (168 held-out seed-8 fixed entries) and RSR on the 53 seed-8 onset entries (E8on).

**Decision rules:**
- **G2:**
  - "curation fixes the nominal cost": nominal(G2) >= Original − 2 (CI lower bound > −5) AND RSR_E8(G2) −
    RSR_E8(Original) >= +5 (CI > 0).
  - "matches theirs": nominal(G2) − Original >= +5 with CI > 0.
  - "beats F0": RSR_E8(G2) >= F0 wm (0.282) − 3 AND nominal(G2) >= F0 + 8.
- **P2-3 (decisive), B vs A at equal trajectory count:**
  - "prevention helps": nominal(B) − nominal(A) >= +3 (CI > 0) OR failure incidence(B) <= incidence(A) − 5 pt
    (CI < 0), with RSR_E8 non-inferior (−3).
  - "prevention data = more recovery data": |Δ nominal| < 3 and |Δ incidence| < 5.
  - vs paper: Δ(B − Original) is compared with their +10 (78 -> 88), cross-paper only.
- MDE ≈ 7 pt nominal at n = 100 x 3 seeds; smaller effects = "inconclusive at n = 100".

**Compute (953bb534 + 8f473147, 4 GPUs):**
- G2 search 3 shards, ~1 h; D_P 2 seeds x 2 shards ~2-3 h, in parallel.
- Then SFT waves of 4 runs x ~2.9 h (incl. eval + events): G2 x3, G2r x3, D5 s8 / s9, A x3, B x3 ≈ 14 runs
  ≈ 4 waves ≈ 12 h.
- ETA: G2 table ~11-25 04:00 UTC, P2-3 table ~11-25 10:00 UTC.

## LRS-G: full reproduction of the 2610.05151 Recovery arm on the Goal protocol, pre-registration (user request 2026-11-24 13:41 UTC; written BEFORE any LRS-G result)
Goal: run THEIR pipeline end to end (audit -> LLM tool teacher -> 60 + 10 LLM-judged handoff -> action-quality
windows -> M0 + 2k with their recipe, 3 seeds) on our G0-validated Goal stack (M0 85, Original 79.3) and compare with
their Recovery 88 and our Original. Code `retryvla/baselines/lrs_goal.py` (collection), `scripts/lrs/` (grading,
dataset, chain), trainer opt-in `pi0_sft --masked_recovery` (quality weights + validity mask). Compute: one
agent-requested 2-GPU user-account shell (killed when done); never 953bb534 / 8f473147.
- Research question: does their LLM-teacher recovery data, at their volume / share and recipe, lift nominal val SR of
  our pi0 M0 the way it did theirs (+10 vs Original), and how precise is their LLM-judged handoff vs sim ground truth?
- Hypothesis: with a local 8B VLM teacher the accepted volume is below theirs and the effect is small
  (expected Δ vs Original in [-3, +4]); their short-horizon LLM judge over-accepts (precision of "accept" vs full-task
  sim success < 0.6). The 4 % share + teacher-only windows should NOT reproduce our F0 nominal cost (-11).
- Pipeline as implemented (deviations D-L1..D-L8 vs the paper, logged before results):
  1. Failures: M0 audit seed 7 on the 300 train scenes (already run: 244 / 56; theirs 233 / 67). D-L1 auditor: no LLM
     auditor; entries = the 168 fixed entries (3 per failed rollout at t = 75 / 150 / 225 = their max of 3 per event),
     IDENTICAL to our goal_search entries (paired with the WM banks).
  2. Teacher: Qwen3-VL-8B-Instruct (local, bf16; D-L2: theirs gpt-6.1-sol), same views (agentview + wrist 128 px) and
     8-d state, tools move_to / move_by (<= 40 steps), gripper, done, abort; frozen spec (local goal, completion
     checks, relations to preserve, 1-2 student progress conditions + completed-task alternative) written by the
     teacher in its first reply of each stage (D-L3: theirs a separate spec LLM). <= 5 stages, <= 300 teacher actions,
     <= 8 replies per stage (D-L4: theirs <= 100 requests / 1,200 s), extensions see the last 6 commands and the
     student summary. 1 attempt per entry (their final protocol).
  3. Acceptance = their Eq. 7: frozen M0 (execute 5 / replan) runs <= H_S = 60 actions from the teacher endpoint, an
     LLM observer (same Qwen3-VL-8B, greedy; D-L5) checks the progress conditions every Delta = 10 actions, the first
     supported advance must survive C = 10 more actions with relations preserved. Fail -> restore the teacher endpoint,
     extend. EXTRA (not in the paper): every handoff test, accepted or rejected, is continued to <= 300 student steps
     and scored by the BDDL success check (sim ground truth) -> judge precision / recall.
  4. Quality: offline LLM evaluator (Qwen3-VL-8B, privileged sim summaries before / after) grades every teacher
     COMMAND q in {0, .25, .5, .75, 1} (D-L6: theirs every action); windows = maximal runs with q >= 0.75, anchors with
     >= L_min = 5 valid targets, targets masked after the run end, loss weighted by q (D-L7: no masking of noisy action
     tokens / attention). Teacher actions only.
  5. Training: their recipe = the Original recipe (M0 4k + 2,000 updates, bs 48, warmup 200, cosine 2.5e-5 -> 2.5e-6,
     D0 + windows pooled uniformly, no oversampling), seeds 7 / 8 / 9 (paired with orig_s7..s9), 100 val scenes.
  6. D-L8 volume: if the seed-7 entries give < 800 anchors (half of their 1,639), a second attempt is run on the
     entries without an accepted path (their early protocol allowed 3); logged, no other change.
- Arms: `lrs` (above) vs Original (existing 3 seeds, 79.3) vs paper (Original 78 / Recovery 88, cross-paper only);
  context arms already measured: F0 `wm` 68.0, F1 / D-series.
- Primary metric + test: nominal val SR mean over 3 seeds; Δ vs Original, paired scene x seed bootstrap (10k);
  McNemar per seed. Secondary: per task; E-v1 failure incidence; first-attempt success; RSR_E8 / E8on (same scripts as
  the campaign); judge precision P(sim success | accept) and recall P(accept | sim success) with Wilson CIs; accepted
  paths / entry, anchors, share, LLM calls / s per entry.
- Decision rules:
  - "reproduces their gain": Δ >= +5 with CI lower bound > 0 (their +10 at 1 seed);
  - "harmless, no gain": Δ in [-3, +5) or CI spans 0 -> "their data does not reproduce +10 on our stack at this
    teacher strength" (not a verdict on a stronger teacher);
  - "hurts": Δ <= -3 with CI upper bound < 0;
  - judge: precision < 0.6 (Wilson upper < 0.75) -> "LLM-judged short-horizon handoff over-accepts";
  - LRS vs our banks (cross-arm, same entries): solved entries (>= 1 accepted path with sim success) vs goal_search
    wm_diverse solved 0.589, McNemar on the 168 paired entries.
- MDE ≈ 7 pt nominal at n = 100 x 3 seeds; below that = "inconclusive at n = 100".
- By-product: side-by-side demos (LRS vs WM bridge, same entries / seeds), docs/research/RECOVERY_DEMOS.md.
- Addendum (2026-11-24 14:35 UTC, after the collection started, BEFORE any training / eval result; first 25 entries
  already give ~1.1k anchors): **volume matching.** Primary arm `lrs` = their volume: whole accepted paths are
  subsampled (seed 0, round-robin over tasks) until the window anchors reach ≈ 1,639 (their count, ≈ 4.1 % of
  anchors); `lrs_all` (every window that passes the filter) is a secondary arm, 1 seed (seed 7) if GPU time allows.
  Also logged: D-L4b (a stage that hits the 12-reply cap with >= 1 executed action is handed off like a claim; their
  sessions allow <= 100 requests), D-L3b (empty `student_progress` -> the spec's completion checks are used).

## Goal campaign additions: CRITIC SWAP, G2-phys, FC labelling, AUX heads (user / coordinator 2026-11-24 15:00 UTC; written BEFORE any of these results)
State at writing:
- G2 curated onset search CLOSED: wm_diverse 132 trajectories / 43 entries, random 115 / 44. It ran with the SHIPPED
  C^pi (`goal_latest_critic`).
- D_P CLOSED: 82 positive trajectories at 44 / 51 s^-, mean p(a-) 0.11.
- G2 wm s7-9 SFT running. The goal_prevent broadcast bug ((50,7) vs (20,7), a- shorter than H) was fixed before the
  D_P run (commit 44868bd).

**0. Critic swap.**
- All NEW WM search / ranking uses the CRIT-V retrained `base` critic (`runs/20261124_CRITV_goal/ckpt/base_s0.pt`,
  link `data/wm/goal_critic_base.txt`; WMCriticRanker now loads CriticV checkpoints).
- Deviation: the G2 bank, D_P and all runs launched before this note used the shipped C^pi. D_P uses no critic
  (pure sim verification), so only G2's wm_diverse picks are affected.
- Optional ablation (only if GPUs idle): re-run the G2 wm_diverse search on 18 entries (shard 0) with `base`.
  Metric: hits / verification, McNemar on solved entries.

**(b) G2-phys (physical-plausibility filter, `retryvla.search.goal_phys`).**
- Every verified G2 trajectory is replayed bit-exactly with privileged geometry. Reject if any of:
  - penetration > 1 cm;
  - off_table;
  - object jump > 3 cm / step without finger contact;
  - bridge d3 > 2 x median D0 d3 (50-step windows);
  - seam || a_H − a_{H−1} || > 3 x median D0 || Δa ||;
  - gripper flips > 1;
  - replay mismatch.
- Reported: survivors, reasons, per-task balance.
- Arm `G2phys_wm` = survivors of the wm_diverse bank, 36 trajectories round-robin (= G2's count), 3 seeds.
- Rule "phys filter helps": nominal(G2phys) − nominal(G2) >= +3 (CI > 0) OR failure incidence −5 pt (CI < 0),
  with RSR_E8 within −3. If < 36 survive: use all survivors and match G2 to the same count in a 1-seed check.

**(a) FC labelling (`retryvla.search.goal_fc`).**
- Failed audit rollouts (seeds 7, 8) are replayed with detector E-v1 and read with the onset scan.
- FC = [t_last, t_end): t_end = first physical event after t_last + 5 if it falls within t_on + 40, else t_on.
- s^- = t_last; s^f = t_end.
- D_F = M0's actions inside FC only (negatives; never whole episodes); used in the hinge L_P and as critic hard
  negatives later.
- Reported: event-confirms-onset rate, event types, FC length, false-alarm events before t_last.
- Use: if the FC end differs from t_on by >= 20 steps for >= 30 % of the rollouts, D_R entries at s^f are rebuilt
  for a later bank (new search with the `base` critic). Otherwise s^f = t_on stays (no rebuild).

**(c) AUX heads (pi0 distillation).**
- A small MLP head on the action-expert output tokens (mean over the 50 action tokens; aux OFF at inference, no extra
  inputs).
- Targets over the next chunk at k = 0, 10, 20, 30, 40, from privileged sim state:
  - grasped (finger contact and lift >= 2 cm), BCE;
  - finger-object contact, BCE;
  - object lift (cm), smooth-L1;
  - object-goal xy distance, smooth-L1;
  i.e. 4 x 5 values (a chain-of-dynamics summary).
- D0 targets from the hdf5 flat states (set_state + sim_geom, no stepping); bank / D_P targets from the goal_phys
  replays.
- Loss = FM + 0.1 x aux (aux normalised per target).
- Arms on the best bank so far (G2 or G2phys, chosen by the G2 rules): `X` vs `X+aux`, 3 seeds.
- Rule "aux helps": nominal +3 (CI > 0) or failure incidence −5 pt (CI < 0), with RSR_E8 within −3.
- 3D point-flow targets are deferred (needs object point sampling; only if the aux rule is met).

**Queue order (4 GPUs; no extra shell, the one extra already requested is 8f473147):**
1. G2 wm s7-9 (running);
2. P2-3 A / B s7-9 (decisive);
3. G2phys s7-9;
4. G2 random s7-9;
5. D5 s8 / s9;
6. AUX x 3 (after its code + targets);
7. critic-swap ablation.

Each run ≈ 3.3 h incl. evals. ETAs: G2 ~11-24 19:30; P2-3 ~11-25 03:00; G2phys ~11-25 06:30; AUX ~11-25 14:00 UTC.

## V2R: RetryVLA v2 training + eval track on LIBERO-Recover Spatial, arms R0-R4 (user request 2026-11-24 15:03 UTC; written BEFORE any V2R result)
Mirror of the Goal campaign (G2 / P2-3 / critic swap / G2-phys / FC / AUX, sections above) on LIBERO-Recover Spatial
with SmolVLA. Owner: the V2R agent. Never uses 953bb534 / 8f473147 / 1123d1ea (Goal agents). Code shared through
common modules (CriticV loader in `WMCriticRanker`, `vec.events_scan`, the detector `retryvla.eval.goal_events.detect`,
the goal_phys metrics); all new code is opt-in, default paths unchanged.
- Research question: on LIBERO-Recover, do the v2 data-side levers (better critic for ranking, physical-plausibility
  curation, verified prevention pairs D_P, auxiliary structured heads) improve recovery (RSR) and prevention (failure
  incidence / first-attempt success on nominal episodes) of the distilled SmolVLA over pi1, without nominal cost?
- Common recipe (= pi1): init pi1-nominal (`runs/20261013-080655_M1_sft_nominal_only/checkpoints/006000`), 6k steps,
  bs 32, lr 5e-5, warmup 300, 50 % nominal replay, recovery segments bridge + continuation, 208 recovery trajectories
  round-robin over instances, train.seed 0 / 1 / 2. Only the final checkpoint is saved (save_every 6000; quota).
- Eval (each SFT): Spatial test = dev split, 56 scenes x 10 (5 close + 5 open, eval.seed 7, 520 steps), RSR overall +
  official L1 (1 scene) / L2 (11) / L3 (16) (no L4 scenes); nominal LIBERO-Spatial 10 x 50 (280 steps). New opt-in
  `eval.save_actions` stores every episode's actions (outcomes unchanged: recording only reads state) -> closed-loop
  action jerk (mean ||Δ³a||, dims 0-5), completion steps, and the failure-event detector.
- Failure-event detector **E-v1s** = Goal E-v1 (`goal_events.detect`, same thresholds: grasp_lost, displaced,
  tipped, off_table, stall), run offline by bit-exact replay of the saved nominal-episode actions from the same
  `load_nominal` init (new `vec.events_scan_load`, reuses events_scan's sampler). Replay check: success must match
  the logged outcome, mismatching episodes excluded and counted. Frozen at the commit of this section's code; not
  validated against hand labels (secondary metrics, like Goal).
- Arms (all 3 seeds unless stated):
  - **R0** = pi1 (existing 3 seeds: M5c s0, M8 seed_pi1 s1 / s2; RSR 0.418 ± 0.021, nominal 0.785). Its nominal evals
    are re-run with save_actions (same eval seed) for the event metrics only; RSR / nominal SR of R0 stay the
    registered ones (the re-run's SR is reported as a reproducibility check).
  - **R1 critic swap**: re-search the M5b TRAIN failure instances (same selection: select_seed 2026, n_select 288,
    trials 2-9, 8 shards; pi0 precheck read from the M5b precheck.csv; pool 256 = SmolVLA samples + perturbations +
    primitives, 2 latent-CEM rounds, diverse top-16 verifications, pi0 continuation, <= 4 trajectories / instance,
    same per-slot continuation seeds as M5b) with wm_diverse ranked by the CRIT-V retrained `base` critic
    (`runs/20261124_CRITV_spatial/ckpt/base_s0.pt`, trained on collect_v1 + verif_r1 only, so M5b instances are
    held out for it) instead of the shipped C^pi v2 -> bank -> 208 round-robin -> SFT. Search-level control = the M5b
    wm_diverse run on the same instances / slot seeds (historical; bf16 non-determinism caveat, ~72-78 % pick
    agreement on reruns): solved, hits / verification, McNemar on solved instances.
  - **R2 = R1 + physical-plausibility curation** at equal count (208): every verified R1 trajectory is replayed
    bit-exactly from its snapshot with privileged geometry (Spatial port of `goal_phys`, `retryvla.search.rec_phys`);
    reject if replay mismatch, penetration > 1 cm, any object off table, object jump > 3 cm / step without finger
    contact (impulse), bridge d3 > 2 x median D0 d3 (D0 = lerobot/libero Spatial demos, 50-step windows), seam
    ||a_H − a_{H−1}|| > 3 x median D0 ||Δa||, gripper flips > 1 in the bridge. Survivors are selected SMOOTH-V
    smooth-first: round-robin over instances, inside an instance lowest bridge d3 first. If < 208 survive: all
    survivors and R1 is re-matched to the same count in a 1-seed check.
  - **R3 = R2 + D_P prevention pairs** at equal total count: pre-failure states s^- come from the source episodes of
    the TRAIN failure groups (`data/assets/scenes_full/assets_times/bddl_scenes_spatial/<task>/<episode>/step_XXXX`,
    20-step grid 0..200; the benchmark scenes are step_0180 / 0200), loaded exactly as recorded (noise 0, gripper as
    recorded). Onset scan with pi0 (= the D_R continuation policy): backward from step 160 in 20-step steps, K = 8
    pi0 rollouts (<= 520 steps) per grid point, t_last = first (latest) point with >= 2 / 8 successes; never
    recoverable episodes dropped. s^- = t_last if it has >= 1 failing rollout. a- = the first 50 actions of the first
    failing pi0 rollout from s^- (pi0's own mistake). Candidates = 24 SmolVLA samples (noise 1.0) + 8 fixed
    primitives, <= 1 gripper flip; screen 1 trial, hits get 2 more; a- gets K = 3 trials (same continuation seeds);
    pair iff p(a+) >= 2/3 and p(a+) − p(a-) >= 1/3, <= 2 positives per s^- (`retryvla.search.rec_prevent`). D_P
    trajectory = a+ (segment 0) + pi0 continuation (segment 1); the a- and losing candidates are kept in
    pairs.jsonl (D_F, never BC). R3 = (208 − n_P) R2-selected trajectories + n_P D_P trajectories, n_P =
    min(#D_P, 70). If n_P < 20: R3 at 1 seed only, "inconclusive (underpowered)".
  - **R4 = best of R1-R3 + auxiliary structured heads** (aux OFF at inference): an MLP on the action-expert hidden
    states (input of action_out_proj, mean over the 50-token chunk, same hook as L_future), targets from the
    privileged replays at k = 0 / 10 / 20 / 30 / 40 steps ahead of the frame: chain-of-dynamics = grasped (finger
    contact and lift >= 2 cm), finger-object contact (BCE), object lift (cm), object-goal xy distance (cm)
    (smooth-L1); object point-flow summary = 3-D displacement of the manipulated object and of the goal object
    between the frame and k = 10..40 (rigid bodies: their point flow is summarised by the body translation; smooth-L1).
    Loss = FM + 0.1 x aux (standardised targets). Deviation from Goal (decided now): aux targets exist only for
    recovery / D_P frames (replayed); nominal demo frames are masked (lerobot/libero has no sim states). The head is
    saved separately (`aux_head.pt`); the evaluated policy checkpoint is the plain SmolVLA. Best of R1-R3 = highest
    mean RSR among arms that pass the nominal guard; within 2 pt -> the simpler arm (R1 < R2 < R3).
  - Optional **R5** = R3 + D_F hinge on matched-state pairs: only if R3 "prevention helps" (below); otherwise not run.
- Metrics: RSR mean ± sd over seeds + official per level; ΔRSR vs R0 (scene x seed bootstrap 10k CI, paired
  sign-flip p over scenes, McNemar n01 / n10 pooled over seed pairs); nominal SR (guard: mean >= pi1-nominal
  0.796 − 2 pt); per-task nominal regression vs pi0 (0.656) and pi1-nominal (tasks dropping > 6 pt = 3 / 50
  episodes, per seed mean); E-v1s failure incidence and first-attempt success on the 500 nominal episodes; completion
  steps (successful episodes); closed-loop action jerk (nominal + RSR episodes); data funnel per bank (instances ->
  solved -> verified -> survivors -> selected); recovery anchor share.
- Decision rules (each arm vs its direct control; R1 vs R0, R2 vs R1, R3 vs R2, R4 vs best(R1-R3)):
  - R1 "critic swap helps": ΔRSR >= +4 pt with CI > 0 and nominal guard; "hurts": ΔRSR <= −4 with CI < 0;
    otherwise "no detectable effect (MDE ≈ 5 pt)". Search-level: "better ranking" iff solved +5 pt with McNemar
    p < 0.10 vs M5b wm_diverse.
  - R2 "curation helps": ΔRSR >= +3 (CI > 0), or nominal +3 (CI > 0), or closed-loop jerk −15 % with RSR within
    −3 and nominal within −2.
  - R3 "prevention helps": nominal +3 (CI > 0) or E-v1s incidence −5 pt (CI < 0) or first-attempt success +3
    (CI > 0), with RSR non-inferior (>= R2 − 3); "prevention data = more recovery data": all |Δ| below these.
  - R4 "aux helps": ΔRSR +3 (CI > 0) or nominal +3 (CI > 0) or incidence −5 pt (CI < 0), others within −3 / −2.
  - Any arm with nominal below the guard is reported as failing M-B regardless of RSR.
- Power: per-seed RSR sd ≈ 0.02 (pi1), scene-cluster bootstrap CI half-width for 3 vs 3 seeds ≈ 0.035-0.04 ->
  MDE ≈ 5 pt RSR (alpha 0.05, power 0.8); nominal MDE ≈ 4-5 pt. Below that: "inconclusive at n = 560 x 3".
- Visual check (user rule, `.cursor/skills/retryvla-visual-check`): every new bank (R1 bank, R2 survivors, D_P)
  gets 6-8 stratified renders (2 worst-jerk, worst handoff jump, most flips, per task, random) -> contact sheets +
  MP4 + action strip in `runs/_analysis/visual/recover_<bank>/`, LOOKED AT before its distillation; 2-3 sentences
  per bank in NOTES. A bank that visibly fails (e.g. objects through the gripper) is filtered before SFT (logged).
- Held-out at the end: the frozen winning recipe on LIBERO-Object (26 test scenes x 10, nominal Object 10 x 50),
  vs Object pi1 (M10 competence SFT); 3 seeds if compute allows, else 1 seed labelled preliminary. If the winner
  uses the critic swap, an Object `base` critic is retrained with the CRIT-V recipe on Object train data (deviation
  logged before the run).
- Compute: own 2-GPU user-account shells `ea7f9ee3` (vla_RetryVLA_DoanhLT_v2rec) + `f72a5ddd` (_v2rec_b, requested
  15:57 UTC because one shell leaves R3 / R4 undone by 11-25 morning). Remote tmux chain `rv_v2r_chain` per shell
  (resumable: every task skip-if-marker), local watchdog `rv_v2r_watch` (45-min stall / session-gone relaunch <= 3,
  final analyze -> RESULTS / NOTES / REGISTRY -> commit -> det shell kill). Quota budget <= 20 GB new (banks ~4 GB,
  12 checkpoints x 1.2 GB, verification logs slim).
- Review: docs/research/REVIEW_2026-11-25_RECOVER.md by 2026-11-25 05:00 UTC; adversarial review (idea-loop step 4)
  before any close / scale.
