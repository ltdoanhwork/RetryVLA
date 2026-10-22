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

## Evidence so far (details: experiments/RESULTS.md)

| Claim | Spatial (dev) | Object (held-out, frozen recipe) | Status |
|---|---|---|---|
| competence > latent-goal in search | 32 / 2 failures, p = 7e-8 | 66 / 8, p = 2e-12 | **strong, replicated** |
| competence > goal after SFT | +0.230 [+0.161, +0.298], 3 seeds | +0.112 [-0.004, +0.227], 1 seed | strong / directional |
| WM finds more recoveries per sim than random | 3.1x hits / sim | 2.2x | replicated |
| WM ~ real-dynamics prefix at ~1/3 transitions | +0.033 [-0.049, +0.115] | - | non-inferiority margin ~3 pts, cost estimated |
| distillation >> demo-only (same init + steps) | +0.292 [+0.206, +0.383] | +0.181 [+0.092, +0.277] | **strong, replicated** |
| online best-of-8 cannot replace distillation | pi0 +0.000; pi1 +0.027 n.s. | - | supported |
| WM data > random data at EQUAL trajectory count | +0.167 | -0.008 (n.s.) | **not replicated -> reframe as acquisition efficiency** |
| nominal preserved | 0.77-0.81 vs pi1-nom 0.80 | 0.942 vs 0.944 | 1 seed on Object, needs eps-test |

Open weakness raised by the user (2026-10-22): the current "search" mostly RE-RANKS a proposal library (VLA samples +
hand primitives); the WM does not generate behaviour. M-A must show the WM can PLAN bridges itself.

## PoC plan (to prove the approach)

| # | Experiment | Question | Status |
|---|---|---|---|
| P1 | **WM as planner** (M11): wm_no_prim (no primitives), wm_plan (broad noise + 5-round elite CEM in imagination), wm_plan_v2 (WM / critic retrained with broad-action data) on the 122 paired M5b failures | does the WM plan bridges without a hand-made library? is WM accuracy off-distribution the bottleneck? | **running** (DAG m11) |
| P2 | **Strong progress / task-success critic baseline**: same architecture + data, label = task progress or reward-to-go (policy-agnostic), same search budget | is the gain from policy-grounding (C^pi) and not from "any learned value"? | todo |
| P3 | **Recovery-nominal Pareto frontier**: vary recovery ratio (0.1-0.7) and nominal replay; + L_preserve variant; 3 seeds; report NSR >= NSR(pi) - 2 pts | M-B constraint, forgetting | todo |
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
