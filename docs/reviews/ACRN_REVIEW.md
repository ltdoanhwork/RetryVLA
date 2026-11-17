# ACR-N independent review (2026-11-17, reviewer: fresh-context agent)

Scope: "Act Before You Imagine: Active Causal Recovery on natural failures" (ACR-N). Evidence read: PLAN "ACR-N" design
note, RESULTS "ACR-N" (still the IN-PROGRESS stub at review time), NOTES ACR-N ops log, `runs/_analysis/acrn/acrn.md`
+ `decision.json` + `rows_b{2800,1400}.jsonl`, `scripts/analyze_acrn.py`, `retryvla/causal/families.py`,
`docs/related_work/WM_RECOVERY_OVERLAP.md`. The ACR agent's own "weakest assumptions" report was **not** in
RESULTS / NOTES when this review was written (RESULTS still says "no numbers until the chain finishes"), so the
diagnosis below is the reviewer's own. The honest post-probe re-run (`data/acr/n1/postprobe_N.jsonl`) does not exist
either (moot because G1 failed, but G2 could not have been evaluated).

Reviewer-run checks (offline, CPU only, 0 sim, on the existing outcome table; scripts in `docs/reviews/acrn_checks/`,
run with the `retryvla` conda python): c1 split-half / fixed option / round-robin / kNN selector, c2 cross-fitted best
fixed option + per-cause ceiling, c3 MI permutation null + McNemar power, c4 hits per 1k env steps per option.

## 0. Summary
- **Hypothesis as tested:** single-unsolved failures have a hidden physical cause that changes which recovery option
  works; knowing it (oracle) or inferring it with a cheap EVI-chosen probe raises solved rate at equal env-step budget.
- **What the data actually say (reviewer's reading):**
  1. Cause knowledge is nearly useless *for this option library*: oracle_cause 0.192 vs no_probe 0.219 on U at 2800
     (McNemar 3/5, p 0.73); even an in-sample (optimistic) "best fixed option per cause" ceiling is 0.315 vs 0.274 for
     the best single option without any cause, i.e. at most ~+4 pt. Bias-corrected MI(cause; best option) is
     ~0.12 bits (observed 0.494, permutation-null mean 0.370, p 0.024), not 0.49.
  2. **The elaborate decision table loses to "always regrasp slowly".** A cross-fitted fixed-option policy
     (`grasp_slow` x 6, chosen on training folds; it was chosen in all 5 folds) gets **0.274 on U and 1.000 on S** at
     2800 vs no_probe 0.219 / 0.917 (McNemar 4/0, p 0.125). Round-robin over all options 0.192; kNN-instance selector
     0.219-0.233. The allocator's machinery is not where the 22 % comes from.
  3. **hindsight_best 0.521 is mostly a max-over-noise ceiling, not a reachable target.** Of the 39 covered U
     instances, 17 have only 1-2 successes in 80 trials. Split-half test (choose the option with the instance's own
     variants 0-3, evaluate on variants 4-7): 0.205 vs fixed-marginal-option 0.164 vs hindsight-on-4-7 0.493. Even
     perfect per-instance knowledge from 4 real trials recovers only ~4 of the ~33 pt gap. There is real
     instance x option structure (P(two variants of the same option both succeed) 0.0078 vs 0.0013 under
     independence), but most of the 0.22 -> 0.52 gap is lucky draws that no selector can predict.
  4. Probes: EVI chose "none" for 93/97 (all 97 at 1400). Correct behaviour given (1): if cause is worth ~0 bits of
     decision value, a probe that resolves cause has EVI ~0. The probe library was never the binding constraint.
  5. Power: at the observed discordance (8/73 = 0.11), exact McNemar power for the pre-registered +8 pt is
     ~0.45-0.58 (alpha 0.10). G0 failing at n = 73 is **inconclusive for +8 pt**, but the point estimate is
     negative at 2800 and the in-sample ceiling says the effect cannot be large with this option set.
- **Verdict: PIVOT (redesign), do not close the broader direction; close "probe-to-identify-a-discrete-cause" as the
  mechanism.** The defensible finding is about option *selection* and the option *library*, and the right "probe" is
  the failed recovery attempt itself (sequential, outcome-informed selection), not a separate 15-45-step probe.
- **Evidence that would change the verdict:** (a) a cause-conditioned selector with a properly regularised per-cause
  ranking beating the cross-fitted fixed `grasp_slow` by >= 5 pt on U at n >= 150 -> reopen cause-conditioning;
  (b) any observation / probe-response feature set raising a learned option-outcome predictor's cross-fitted AUC
  over options by >= 0.05 vs failure-obs-only and turning that into >= +5 pt solved -> reopen active probing.

## 1. Nearest prior work (arXiv IDs verified on the abs page + a second source)
| # | paper (venue / year) | arXiv / link (verified via) | what it does | how it differs from us | implication |
|---|---|---|---|---|---|
| 1 | PhysCaP (2026) | [2608.21031](https://arxiv.org/abs/2608.21031) (abs v2 + HTML + project page physcap.github.io) | code-as-policy agent with Planner deciding when to explore / stop; proprioceptive mass / stiffness probes; includes a LIBERO task; VLAs fail on hidden properties | pre-task property identification on tasks *designed* so the hidden property decides success; no failure recovery, no EVI | **Highest overlap for "act before you imagine" on LIBERO.** Shows probing pays when the task is built so hidden state matters; our natural failures are not of that kind (cause MI ~0.12 bits). Any ACR claim must cite it and state the regime difference. |
| 2 | Poke and Strike (2025) | [2509.00178](https://arxiv.org/abs/2509.00178) (abs + HTML + project page) | RL exploration policy rewarded by the *sensitivity of a privileged task policy* to property errors; uncertainty-based switch to execution | decision-relevant information gathering (an EVI-like reward) learned end-to-end; pre-task, single dynamic task | Our EVI is the same principle; its result says the probe must target what the *downstream policy* is sensitive to, not a taxonomy. Supports dropping the discrete cause. |
| 3 | Interactive learning of physical object properties (2024) | [2404.07344](https://arxiv.org/abs/2404.07344) (HTML v2 + HTML) | expected-information-gain exploratory action selection over a Bayesian network of properties | identification objective, not task value | EIG-only probing (our `infogain_probe`) is known to over-explore; matches our infogain degrading S (0.833). |
| 4 | Seeing is Believing (2025) | [2504.03245](https://arxiv.org/abs/2504.03245) (abs + HTML) | VLM uncertainty -> symbolic belief -> belief-space planning with information-gathering skills | symbolic, mobile manipulation, no recovery | Belief-space planning works when predicates gate skill preconditions; our causes do not gate options strongly. |
| 5 | Interactive Perception survey (T-RO 2017) | [1604.03670](https://arxiv.org/abs/1604.03670) (abs + DOI 10.1109/TRO.2017.2721939) | defines IP: forceful interaction creates signal | background | Cite for framing; "probes as actions" is not novel. |
| 6 | FailSafe (2025, IROS 2026) | [2510.01642](https://arxiv.org/abs/2510.01642) (abs + HTML v4 + project page) | generates failures + executable corrections; VLM detects and outputs recovery deltas for VLAs (ManiSkill) | injected failures, VLM predicts the correction directly (no cause -> option table) | Cause-free direct correction prediction is the strong baseline to the "learned selector" pivot. |
| 7 | RoboFAC (2025) | [2505.12224](https://arxiv.org/abs/2505.12224) (abs + HTML + project page) | failure taxonomy, analysis (type / location / explanation) and high/low-level correction; +29 % relative real-world | VLM supervisor, categorised failure types | Their taxonomy helps a *language* correction; our result says taxonomy -> option mapping carries little decision value for low-level recovery in LIBERO. A useful negative counterpoint. |
| 8 | AHA (2024, ICLR 2025) | [2410.00371](https://arxiv.org/abs/2410.00371) (abs + aha-vlm.github.io) | failure detection + free-form reasoning, plugged into planners / reward | perturbation-generated failures; reasoning, not recovery control | Cause explanation != actionable cause; cite for "diagnosis" line. |
| 9 | REFLECT (CoRL 2023) | [2306.15724](https://arxiv.org/abs/2306.15724) (abs + GitHub real-stanford/reflect) | hierarchical multisensory summary -> LLM failure explanation -> correction plan | planning-level failures | Diagnose-then-recover works for *plan* errors; ours are execution-level. |
| 10 | ViFailback (CVPR 2026) | [2512.02787](https://arxiv.org/abs/2512.02787) (abs + CVF open-access PDF) | real-world failure diagnosis (4 types incl. gripper state / 6D pose) + visual correction guidance for a VLA | VLM guidance, real robot | Their "gripper state / pose" types ~ our held_slip / free; again guidance, not EVI. |
| 11 | ReCoVLA (2026) | [2606.09630](https://arxiv.org/abs/2606.09630) (abs + HTML; also listed in our LIBERO_RECOVER_BASELINES) | VLM infers failure mode + recovery stage -> compiles reward -> trains residual recovery policy; frozen VLA | cause-conditioned *training* of recovery policies, not test-time option selection | Closest "cause-conditioned recovery" design; if we keep causes, it is as a training-time reward selector, not a test-time probe target. |
| 12 | RecoveryChaining (2024) | [2410.13979](https://arxiv.org/abs/2410.13979) (abs + HTML) | HRL recovery policy with nominal controllers as options; learns *which* controller to switch to via sim verification | learned option selection from failure states | **Direct prior for the pivot** (learned option selector with verified outcome labels). Novelty must come from VLA-scale / WM-vs-scripted comparison, not from "learn which option". |
| 13 | SRSA (ICLR 2025) | [2503.04538](https://arxiv.org/abs/2503.04538) (HTML + abs) | predicts zero-shot transfer success of every library skill, retrieves top-k, then verifies | skill retrieval for new tasks | Same "outcome predictor over a library, verify top-k" recipe as our proposed F1. |
| 14 | FAR: Failure-Aware Retry (2026) | [2607.01111](https://arxiv.org/abs/2607.01111) (abs v2 + HTML) | test-time retry that learns from the failed attempt (failure-contrastive preference adaptation) + perturbations; continual improvement | adapts the policy from the failed attempt | Supports "the failed attempt is the probe" pivot; it is an overlap risk for that pivot too. |
| 15 | RoboMonkey (2025) | [2506.17811](https://arxiv.org/abs/2506.17811) (abs + HTML) | sample-then-verify test-time scaling for VLAs | verifier over sampled actions | Baseline framing for "selection is the bottleneck". |
| 16 | Back on Track / SAILOR | [2610.19579](https://arxiv.org/abs/2610.19579), [2506.05294](https://arxiv.org/abs/2506.05294) (verified in WM_RECOVERY_OVERLAP.md) | WM + policy-conditioned value search for recovery; WM search + distill | WM-based | Our `wm_single` option has ~1/3 of `grasp_slow`'s hits/1k on U (0.07 vs 0.20; equal on S) -- a concrete data point against WM-only recovery on hard failures (with the selection caveat in 2.3). |

Also seen, single-source or weaker relevance (not relied on): ReCoVLA second source is arXiv HTML only; Tactile
Curiosity / TacEx (2609.40134, abs only), Contextual-bandit controller selection (Rigter et al., RA-L 2020, PDF only).

**Positioning.** "Probe then recover" (IP, PhysCaP, Poke and Strike), "diagnose then recover" (AHA, REFLECT,
RoboFAC, ViFailback, ReCoVLA) and "learn which recovery option to run" (RecoveryChaining, SRSA) all exist. What no
paper we found measures is the thing ACR-N *did* measure: on natural (not injected) VLA failures in LIBERO, with
privileged ground-truth causes, a paired outcome table of 10 recovery families x 8 variants, **how much decision value
cause knowledge and probes actually carry** (answer: ~0 at this option set, MI ~0.12 bits after bias correction) and
**how far a fixed scripted prior beats WM-search recovery on hard failures**. That negative / quantitative result is
defensible and publishable as an analysis; "ACR as a method" is not, at these numbers.

## 2. Steelman / strawman
**Steelman (strongest reading in favour).** A cause-free, verified, multi-family recovery library already solves
22-27 % of failures that the project's WM-guided `single` search solved 0 % of, at the same 2800-step budget, with
no harm on S (fixed `grasp_slow`: S 24/24). Information-seeking is not refuted in general: the probes were proprio /
latent-only, the causes were a hand-written taxonomy, and the option set had nothing that a cause could switch
*into* that was not also the best default. G0 at +8 pt had < 60 % power. Real instance x option structure exists
(within-option success correlation ~6x independence), so a richer, instance-level signal (not a 6-way cause) could
still be worth something.

**Strawman (most deflationary reading).** Nothing in ACR-N's mechanism did anything. The 22 % is "try scripted
regrasps instead of WM search", and the probability-weighted allocator is beaten by a constant. Oracle cause *hurts*
(-3 pt; on changed decisions -0.07, n 27) because the per-cause V table is fitted on ~8-46 instances x 8 variants of
~0.05-0.15 success rate: the posterior moves weight to whichever option got lucky in that cause's training folds.
hindsight_best 0.52 is a max over 10 noisy Bernoulli streams, not a target. Probes never fire because they cannot
change decisions. The "cause" framing is a post-hoc narrative over a selection problem dominated by noise.

**Reviewer's position:** the strawman is closer to the truth for the *mechanism*; the steelman is right that the
*library result* is real and is the project's strongest positive number on single-unsolved failures to date
(with the caveats in 2.3 / 2.5).

## 3. Counter-questions
### 3.1 Construct validity
1. **Is "cause" the right latent?** Bias-corrected MI(cause; best option) ~0.12 bits (c3: 0.494 obs vs 0.370
   permutation null; p 0.024). `best` is argmax over 8-trial rates, so "none" (34/97 instances, incl. 17 of the 46
   contact_free) and 1-success ties dominate it. Answer: recompute MI on instances with >= 3 successes, or against the ranking
   of options by the posterior mean, not argmax. Cost: 0 GPU.
2. **Does `hindsight_best` measure a reachable ceiling?** No: split-half (c1) shows 0.493 hindsight on 4 trials vs
   0.205 for the best possible *data-driven* choice (instance's own other 4 trials). Report a "split-half oracle"
   ceiling (~0.20-0.25) next to hindsight. Cost: 0.
3. **Are the causes actionable by construction?** Most option prefixes are cause-agnostic re-grasps; only `place_goal`
   (held_ok, n 3 in U) and `pull_free` (stuck/contact) are cause-specific, and both are weak (P(final | sub-goal)
   0.215 for place_goal: the pi continuation, not the prefix, fails). The taxonomy predicts *what is wrong*, not
   *what will fix it*. Answer: per-cause, is there an option whose per-cause rate exceeds its marginal by > 2 SE?
   (c2 in-sample per-cause best: held_slip -> grasp_slow 0.58, obstructed -> hover 0.50, both n <= 12.) Cost: 0.
4. **Selection-on-failure + new seeds.** U is `single`=0 by construction; any re-run with new seeds gains by
   regression to the mean. The 0.219 vs `single` 0.000 McNemar (16/0) is therefore not a fair test of "the library
   beats single"; only `wm_only` (fresh seeds of the single-like scorer, 0.137) is. Library vs wm_only: 12/6, p 0.24.

### 3.2 Strongest form
1. **Decision table too noisy?** Yes, decisively. Per cause ~8-46 instances x 8 variants with Beta(strength 2) shrink
   toward the marginal; oracle_cause changes 37 % of first decisions for -0.07 per change. Strongest variant:
   heavy hierarchical shrinkage (strength 20-50) or "use cause only to break ties among top-2 marginal options".
   Cost: 0 (offline).
2. **Learned selector over options from failure obs** (the user's question). Tested crudely by the reviewer: kNN on the
   10 obs features (K 10) 0.233 at 2800, 0.164 at 1400 (vs no_probe 0.068, oracle_cause 0.110, fixed grasp_slow
   0.096 cross-fitted). Weak at 2800, interesting at 1400 (n.s.). A proper outcome predictor (logistic / GBM per
   option on obs + head + RGR geometry + latent, trained on U+S folds, ranking by P(success)/cost) was never run.
   Cost: 0 GPU (offline on the table), < 1 h.
3. **Outcome-informed sequential selection** (the failed recovery attempt as the probe). Trials are executed until
   success; each failure gives a full 300-400-step response (achieved lift, sub-goal reached, O_sub) that is far more
   informative than a 15-step `hold`. The allocator updates only via V (cause-level). Strongest variant: condition
   the next option on the *sub-goal outcome* of the last trial (e.g. lift reached but final failed -> place_goal;
   no lift -> hover). `O_sub` is already in the data. Cost: 0.
4. **Budget regime.** At 1400 oracle_cause 0.110 vs no_probe 0.068 (3/0, p 0.25) and on S oracle 0.79 vs 0.54
   (6/0, p 0.03): cause helps when only ~3 trials fit. Is cause value a small-budget phenomenon (less room for
   brute-force)? Test at 700 / 1050 / 1400 / 2100 on the same table. Cost: 0.
5. **Option library.** Only 6 trials of a scripted option fit in 2800 (worst-case 440 steps each). Shorter prefixes /
   early-abort on sub-goal failure (`O_sub` lift < 1 cm after 60 steps) would double trials per budget. Cost: new sim.

### 3.3 Confounds and fairness
1. **Perception noise vs privileged geometry.** Scripted options read obj / goal positions from privileged sim with
   2.5 cm noise (families.py). That is *not* observation-only; WM options are. The library's advantage over
   `wm_single` (hits/1k on U 0.20 vs 0.07; on S 1.22 vs 1.28) may partly be privileged localisation. Answer: rerun
   grasp_slow / hover with object pose from the RGR geometry head on the real latent. Cost: ~1-2 GPU-h for U.
2. **Charging.** `execute` skips trials whose worst-case cost would exceed the budget, and the planner uses mean
   failed-trial cost; fixed-option arms use the same executor (fair). OK.
3. **wm_single is a reduced `single`** (fresh 256 pool + 2 CEM rounds, diverse top-8, no verification ranking beyond
   C^pi). The "library beats WM" reading needs `single`'s full pipeline on new seeds as the comparator, not 0.000.
4. **S degradation.** no_probe 0.917, passive / infogain 0.833, oracle_cause 0.833 at 2800; 0.542 for no_probe at
   1400 vs wm_only 0.833. The allocator under-uses `wm_single` on S where it is the best option per step (1.28 / 1k):
   a learned cause-free selector would have to learn "easy states -> WM, hard -> regrasp". That is itself evidence
   that state-dependent selection matters, just not via cause.
5. **Cross-fitting folds by failure episode** (5 folds, 97 instances): fine; but option prior / V include S
   instances, which shifts V toward options that succeed on easy states (wm_single). Fit V on U-train only
   (c2: Utrain best-fixed picks hover / grasp_slow, U 0.233, S 0.833).

### 3.4 Statistical power
1. Exact McNemar, alpha 0.10 two-sided, observed discordance ~0.11-0.15 at n 73: power 0.44-0.58 for +8 pt, 0.18-0.24
   for +5 pt, 0.80 only at ~+10-15 pt (c3). G0 / G1 as pre-registered could only detect effects >= ~+10 pt.
2. n 150 gives 0.79 for +8 pt at q 0.15; n 250 gives 0.60 for +5 pt. Extending U requires more natural failures
   (other suites / test-free train scenes).
3. The 1400 result (oracle +4 pt, 3/0) has n01 + n10 = 3: uninformative either way.
4. Multiplicity: 12 arms x 2 budgets x 3 sets; the only "significant" effects are vs `single` (biased) and
   hindsight. Pre-register a single primary contrast in the follow-up.

### 3.5 Alternative mechanisms
1. **Retry diversity, not selection:** fixed grasp_slow beats every mixture; mixtures waste trials on low-rate
   options. Gains come from *more* trials of the best default.
2. **Luck / regression:** within-option correlation is small in absolute terms (0.0078); most solved U instances are
   1-2/80. Distinguish "recoverable by policy" from "occasionally lucky" before claiming coverage.
3. **pi continuation is the bottleneck:** grasp prefixes reach lift >= 3 cm 31 % of the time but finish only 21-25 %
   of those; place_goal reaches goal-xy 12 % but finishes 22 % of those. Better *prefixes* (or longer scripted
   segments) may matter more than better *selection*.
4. **Privileged pose** (3.3.1) could explain scripted > WM.

### 3.6 What would flip the verdict
- Toward "continue ACR (probing)": a learned outcome predictor where adding probe-response features raises
  cross-fitted option-ranking AUC by >= 0.05 and solved by >= 5 pt over the obs-only predictor (F1b).
- Toward "close everything": the learned selector and sub-goal-conditioned sequencing do not beat fixed
  grasp_slow (F1), and grasp_slow's advantage vanishes with head-predicted pose (F2).
- Toward "distill": F2 keeps >= 80 % of grasp_slow's U gain with observation-only pose.

## 4. Audit
| check | status | note |
|---|---|---|
| oracle / upper bound run | yes | oracle_cause, oracle_cause_free, hindsight, coverage; but hindsight is a noisy-max ceiling, add split-half ceiling |
| powered for the effect that matters | **no** | ~0.45-0.58 for +8 pt; write "inconclusive at n 73 for <= +8 pt; point estimate negative at 2800" |
| strongest variant tested | **no** | no regularised cause table, no learned option-outcome predictor, no outcome-informed sequencing, no fixed-option / round-robin baselines |
| budgets / seeds / instances equal | yes | perfectly paired table look-up; same executor |
| no leakage | yes (train only) | scripted options use privileged pose + noise: not leakage but not deployable |
| deviations documented | partly | movability test shortened (NOTES); postprobe re-run and agent's diagnosis / weakest assumptions **missing** from RESULTS |

## 5. Decisive follow-ups (all 0 GPU unless stated; train split; pre-register in NOTES first)
| rank | experiment | could flip because | cost | decision rule |
|---|---|---|---|---|
| **F1** | **Selector shoot-out on the existing table** (U, 2800 and 1400, 5 episode folds, identical executor): (a) fixed best option chosen on training folds; (b) cause table with strong shrinkage (strength 20) / cause as tie-break; (c) learned per-option outcome predictor (L2 logistic, features = failure obs + head geometry + C^pi + DINOv2 latent PCA-16), ranking P(success)/cost with failure-decay; (d) (c) + sub-goal-outcome conditioning after each failed trial (`O_sub`); ceilings: split-half oracle, hindsight. | decides whether *any* state-conditioning beats a constant, and whether cause beats learned features | ~1 h CPU | primary contrast (d) vs (a) on U at 2800. "Selection matters" iff (d) - (a) >= +5 pt, exact McNemar p < 0.10 AND S >= 0.95; else conclude "a fixed scripted prior is the result; selection is noise-limited at n 73". Cause is reopened only if (b) >= (c) on both budgets. |
| F1b | Same as F1(c) + probe-response features (proprio / visual / tactile / oracle sets; all probes, not EVI-chosen) | the only remaining test of "probes carry decision value" | ~0.5 h CPU | probes continue iff option-ranking AUC +0.05 (cluster bootstrap CI > 0) and solved +5 pt vs F1(c) at either budget; else close probing as tested (report the AUC ladder obs < proprio < tactile < oracle as the finding). |
| F2 | **Deployability of the library gain:** rerun `grasp_slow`, `hover`, `grasp_firm` x 8 on U+S with object / goal pose from the RGR geometry head (observation-only), plus a fresh full `single` run on new seeds as the honest comparator | separates "scripted re-grasp" from "privileged localisation" and from regression to the mean | ~2 GPU-h (97 states x 24 trials ~ 2.3k trials) | Go to a distillation pilot (recover-by-regrasp dataset -> 1-seed SFT, nominal check) iff head-pose grasp_slow-fixed keeps >= 80 % of its U gain (>= 0.22) AND beats fresh `single` on U by >= 8 pt (p < 0.10). Else report the library gain as privileged-only. |

Larger, only if F1 says selection matters: extend U to n >= 150 (power 0.8 for +8 pt) by adding natural failures from
other train scenes; add early-abort prefixes to double trials per budget.

## 6. Counter-questions for the ACR agent (must be answered before RESULTS records a verdict)
1. Where is your diagnosis / weakest-assumption list? RESULTS still says "IN PROGRESS"; write it with the numbers.
2. Why does the allocator under-weight `grasp_slow` x 6 relative to the cross-fitted fixed policy (no_probe 0.219 vs
   0.274; 217 of 465 trials are grasp_slow)? Is it the S-inflated `wm_*` prior or the per-trial decay V[c,o,k]?
3. Report hindsight from 4 held-out variants given the instance's other 4 (split-half ceiling) next to hindsight.
4. Was the postprobe honest re-run executed? If not, say so (G2 unevaluable).
5. How much of grasp_slow's U advantage over wm_single survives observation-only pose (F2)?
6. Is the 1400 / S result (oracle 0.79 vs 0.54, 6/0) a budget-regime effect or an allocator bug at small budgets
   (no_probe on S 0.542 vs wm_only 0.833)?

## 7. Recommendation
**Pivot (redesign), not close.** Close the specific mechanism "probe to identify a discrete physical cause, then
pick an option by cause" as tested (oracle cause does not help, the EVI rule correctly refuses to probe, MI is ~0.12
bits after bias correction; power caveat noted). Keep and sharpen the two real findings: (i) a cause-free scripted
re-grasp prior solves ~27 % of `single`-unsolved failures with no harm on S, beating WM-search recovery 3x per step on
hard states (pending F2's deployability check); (ii) selection is noise-limited: the 0.22 -> 0.52 gap is mostly not
learnable (split-half ceiling ~0.21). Next step is F1 (free, decides whether any selector beats a constant), then F2
(decides whether the library gain is deployable and worth distilling). "Active" information should re-enter only as
the outcome of failed recovery trials (F1d) or via probe features in a learned predictor (F1b), not as a separate probe
gated by a cause taxonomy.

## 8. Responses (filled by the main agent)
| question | response |
|---|---|
| | |

## 9. Final verdict (main agent, after responses + follow-ups)
