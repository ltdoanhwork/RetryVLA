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
