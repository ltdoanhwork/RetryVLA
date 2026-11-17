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
