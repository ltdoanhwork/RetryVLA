# T2Mem for RetryVLA: independent review (2026-11-16, reviewer: fresh-context agent)

Scope: PoC 0 (`scripts/poc_mem_repeat.py`, `runs/_analysis/t2mem/poc0_repeat_touch_*.md`), PoC 1 (`scripts/t2mem_cache.py`,
`scripts/t2mem_poc1.py`, `runs/_analysis/t2mem/poc1.{md,json}`), NOTES section "T2Mem PoC" (2026-11-16), RESULTS
"RQ3 follow-up" / "Cross-WM disagreement" / "RGR" / "B2 / P2". **Follow-up F1 (`poc1_followup.md`) was NOT present when
this review was finished**; Section 2.2 / 3 lists two design problems in F1 that limit what any F1 result can mean.

All numbers marked "[R]" were recomputed by the reviewer on CPU from `poc1.json` and `cache_verif_m5b.pt` (scripts
`/tmp/t2m_review_stats{,2,3}.py`, not committed; 0 sim, 0 GPU). Everything else is quoted from the repo files.

## 0. Summary
- **Hypothesis as tested.** (A) A memory policy trained on "failed attempt -> verified recovery" episodes helps only if
  the policy repeats a failed grasp within an episode. PoC 0 measured that repeat rate. (B) A per-instance, T2Mem-style
  fast-weight memory of VERIFIED outcomes (self-supervised h_wm -> h_true writes, plus an outcome/label channel) re-ranks
  a FIXED pool of 64 sim-verified candidates over R = 4 rounds x K = 4 picks. The hope: it stops re-picking the
  over-optimistic modes of WM v1 + C^pi v2. PoC 1 is an offline replay on 256 verif_m5b TRAIN instances, 4 folds by
  failure episode.
- **Verdict.**
  - **A: close as tested** (the "repeat-breaking" version). It is *untested* in the form that recent prior work
    (RoboTTT DAgger Distillation) actually shows to work. Park it; do not spend GPU on it now.
  - **B: redesign, not close.**
    - The negative is powered (Section 2.4) and diagnosed: in this pool, neighbours of failed picks are *more* likely
      to succeed (0.22 vs 0.11 [R]). So "avoid what failed" is the wrong prior.
    - The write rule has a plausible implementation flaw: its target is absolute (residual vs the RAW critic), not a
      residual on the current, recalibrated prediction. The F1 follow-up also freezes a dead latent read-out
      (Section 2.2, Q2.2a/b).
    - The closing rules (oracle / strongest variant / top-3 questions) are not yet met. Two CPU-only replays can settle
      it (follow-ups 1 and 3).
  - **Recalibrator lead: continue, as a SEPARATE idea** ("imagination-aware verifier recalibration"). It is not a memory
    result.
    - It equals plain cross-fold logistic regression on [l_wm, h_wm]: 2.316 vs 2.327 hits r2-4 [R].
    - It was trained with all 64 labels per instance. With only the 16 labels an online search would actually observe,
      the gain shrinks from +0.47 to +0.18 hits r2-4 (p 0.02) [R]. Prequential online learning gives +0.10..+0.20 [R].
    - It needs an online CEM check before any claim: Goodhart risk, since it will be optimised against.
- **Evidence that would change the verdict.**
  - B -> continue: a fixed-write memory (residual target on the current score, live latent read-out) beats the frozen
    recalibrator by >= +0.10 hits r2-4 (about the cluster-adjusted MDE, 0.135) with episode-clustered sign-flip
    p < 0.10, in the K = 4 replay or the K = 1 x 16 replay.
  - B -> close: the same fixed-write memory is <= recalibrator in both replays (the CI upper bound below +0.10).
  - Recalibrator -> close: in an online paired search (63 train instances, 16 verifications), C^pi v2 + recalibrator
    does not beat C^pi v2 on hits / verification (sign-flip p >= 0.10), or its FP on its own picks goes back to >= 0.8.
  - A -> reopen: on TRAIN-split failures, a2_failed_grasp episodes with >= 2 attempts and a repeat are >= 20 % of a2
    episodes (already 15-26 % on the test-split diagnostic [R]), AND a history-conditioned recovery probe beats a
    state-only one (Q2.5c).

## 1. Nearest prior work (all arXiv IDs verified by opening arxiv.org/abs/<id>, title matched, 2026-11-16)
| # | paper (venue / year) | arXiv (verified) | what it does | how it differs from us | implication for design / claims |
|---|---|---|---|---|---|
| 1 | T^2Mem: Learning Test-Time Memory for Robotics (Liu, Huang, Hong, Du, Cao, Fei-Fei, Wu; 2026) | [2609.36720](https://arxiv.org/abs/2609.36720) | Puts TTT fast weights inside a pretrained VLA. Online self-supervised writes encode the observation history. Action supervision shapes the memory; memory and policy are trained in alternation. RoboMME: 17.9 % -> 56.8 %. | Its target regime is *memory-dependent* tasks: the information needed is no longer in the current observation. Neither A (failure state is mostly observable in LIBERO) nor B (a 4-12-point labelled regression per instance) is in that regime. B also has no action loss to shape the memory. | Citing T^2Mem as motivation for B is a stretch: B is few-shot surrogate regression (rows 9-12), not history compression. Claim novelty only for "TTT memory of verification outcomes in a WM-guided search". |
| 2 | RoboTTT: Context Scaling for Robot Policies (Jiang et al., NVIDIA / Stanford / UT; 2026) | [2607.15275](https://arxiv.org/abs/2607.15275) | TTT fast-weight VLA with 8K-step context. **"DAgger Distillation": failures as context, corrections as targets.** +33 % vs +9 % for standard DAgger on the same data; most of the gain is "recovering after wrong actions". | This is approach A almost exactly, but its premise differs: it does not require the policy to *repeat* a failure. The failure context conditions *which* correction to make. | PoC 0's kill rule (repeat rate) tests a narrower mechanism than RoboTTT's. Approach A is closed only for "repeat-breaking"; the "failure-as-context" version is untested here and has a strong prior result. |
| 3 | VITA: Zero-Shot Value Functions via Test-Time Adaptation of VLMs (Ziakas & Russo, ICLR 2026) | [2506.10085](https://arxiv.org/abs/2506.10085) | A value function whose lightweight module takes a gradient step on a meta-learned self-supervised loss at test time, sequentially over a trajectory. | Closest to B: TTT applied to a *value / verifier*, not a policy. But VITA adapts along one trajectory and has no verified labels at test time. | Supports B's t2m_ss variant as a reasonable design (meta-learned SS inner loss for a critic). It also shows the meta-learned *initialisation* matters a lot, which matches our static-control finding. |
| 4 | Learning to (Learn at Test Time): RNNs with Expressive Hidden States (Sun et al., 2024) | [2407.04620](https://arxiv.org/abs/2407.04620) | TTT layers: the hidden state is a model updated by self-supervised GD; inner loss on learned key / value projections. | Same inner-loop machinery as FastMem (`t2mem_poc1.py` L136-200). But TTT layers are trained on long sequences with many writes; ours does 3 writes of 4 items. | With 3 writes, a linear fast weight is at best a few-shot linear regressor. The meta-learned W0 / readout (= the static control) is expected to dominate. |
| 5 | Titans: Learning to Memorize at Test Time (Behrouz et al., 2024) | [2501.00663](https://arxiv.org/abs/2501.00663) | Neural long-term memory updated at test time by a "surprise" (gradient) signal, with momentum and forgetting. | Surprise-weighted writes plus forgetting. Our write is a plain one-step delta rule with an absolute target. | Suggests the fix in Q2.2b: write the *surprise* (y - current prediction), not y - raw-critic prediction. |
| 6 | Linear Transformers Are Secretly Fast Weight Programmers (Schlag, Irie, Schmidhuber, 2021) | [2102.11174](https://arxiv.org/abs/2102.11174) | Shows linear attention = fast-weight programming. Proposes the delta rule, which writes v - W k (the residual on the *current* readout). | FastMem's GD step on mean of squared (kW - v) is a delta rule, but v_label is computed against the raw critic, not the current recalibrated score. | Direct design guidance for follow-up 1: the delta target should be the residual on the current prediction. |
| 7 | MemoryVLA (Shi et al., 2025) | [2508.19236](https://arxiv.org/abs/2508.19236) | Perceptual-cognitive memory bank in a VLA for long-horizon manipulation. | Memory of the episode history, for non-Markov tasks. | Same regime caveat as row 1: LIBERO-Recover Spatial failure states are close to Markov. Memory policies need a task where history matters. |
| 8 | MemER: Scaling Up Memory for Robot Control via Experience Retrieval (2025) | [2510.20328](https://arxiv.org/abs/2510.20328) | Retrieves relevant past keyframes into a VLA's context (explicit memory). | Explicit retrieval memory, not fast weights. | This is the "explicit memory" baseline class. knn_label / knn_latent are our analogue; they gained only +0.04..+0.07 (n.s.). |
| 9 | SAM2Act (Fang et al., 2025) | [2501.18564](https://arxiv.org/abs/2501.18564) | Memory architecture (SAM2-style memory bank) for memory-dependent manipulation (MemoryBench). | Memory of past observations within an episode. | Same as rows 7 / 8. Approach A would need MemoryBench-like non-Markov failures to show a gain. |
| 10 | Scaling Laws for Reward Model Overoptimization (Gao, Schulman, Hilton, 2022) | [2210.10760](https://arxiv.org/abs/2210.10760) | Proxy-reward optimisation (best-of-n, RL) improves the gold reward, then degrades it, with predictable scaling. | Our WM + C^pi optimism (FP ~0.8, RESULTS RQ3: gap grows with N 8 -> 512) is exactly this proxy-gold gap. | The recalibrator is "retrain the proxy on the optimiser's own distribution". Expect it to be re-exploited once CEM optimises against it. Gate it online. |
| 11 | Reward Model Ensembles Help Mitigate Overoptimization (Coste et al., 2023) | [2310.02743](https://arxiv.org/abs/2310.02743) | Conservative ensemble objectives (worst-case / uncertainty-weighted) reduce overoptimisation. | xWM "min" (RESULTS) is this idea; it cut FP 0.82 -> 0.70, n.s. | Memory of verified outcomes is a *complement* (it uses real labels); ensembles use none. Compare against xWM-min, not only against static. |
| 12 | When to Trust Your Model: MBPO (Janner et al., 2019) | [1906.08253](https://arxiv.org/abs/1906.08253) | Model exploitation; short model rollouts branched from real states. | Our CEM optimises long (50-step) open-loop imagined bridges. | Exploitation grows with horizon and optimisation pressure. A verifier that learns from verified failures attacks the symptom; RGR / RQ3 say the cause is the imagined end state. |
| 13 | PETS: Deep RL in a Handful of Trials using Probabilistic Dynamics Models (Chua et al., 2018) | [1805.12114](https://arxiv.org/abs/1805.12114) | CEM planning through a probabilistic ensemble dynamics model. | Same CEM-over-model structure. There the model is refit on real outcomes between iterations. | The "refit on real data between CEM rounds" baseline is the natural non-TTT version of B. Its cheapest instance is refitting the recalibrator online. |
| 14 | Practical Bayesian Optimization of ML Algorithms (Snoek, Larochelle, Adams, 2012) | [1206.2944](https://arxiv.org/abs/1206.2944) | GP-surrogate BO: the surrogate is updated after every expensive evaluation; acquisition trades exploit vs explore. | B is BO over a finite candidate set, with a strong but biased prior mean (C^pi on imagination). | The missing baseline is a GP / BO surrogate with prior mean = recalibrated C^pi and kernel on h_wm, plus UCB / EI. knn_label is a crude version of it. |
| 15 | Few-Shot Bayesian Optimization with Deep Kernel Surrogates (Wistuba & Grabocka, 2021) | [2101.07667](https://arxiv.org/abs/2101.07667) | Meta-learns a deep-kernel GP across tasks, then adapts with few evaluations. | The cleanest "meta-learned prior + per-task few-shot update" design. t2m_label is an amortised, linear version of it. | Their result, and ours, is that the meta-learned prior carries most of the gain. Report memory vs prior separately (which the static control did, correctly). |
| 16 | Transformers Can Do Bayesian Inference (PFNs; Mueller et al., 2021) / Neural Processes (Garnelo et al., 2018) | [2112.10510](https://arxiv.org/abs/2112.10510) / [1807.01622](https://arxiv.org/abs/1807.01622) | Amortised posterior predictive given a context set of (x, y) pairs. | An in-context surrogate is a stronger "memory" than one GD step on a linear fast weight. | If B is redesigned, a small set-transformer / PFN over the verified picks is the strongest-form memory to test. |
| 17 | RoboMonkey (Kwok et al., 2025) / V-GPS "Steering Your Generalists" (Nakamoto et al., 2024) | [2506.17811](https://arxiv.org/abs/2506.17811) / [2410.13816](https://arxiv.org/abs/2410.13816) | Test-time sample-and-verify for VLAs, with learned verifiers / value functions re-ranking action samples. | Their verifiers are static at test time and never learn from the instance's own verified outcomes. | Positioning: "a verifier that adapts within an instance from real verification outcomes" would be new relative to these, *if* it worked. Currently it does not beat the static recalibrator. |
| 18 | Reflexion (Shinn et al., 2023) / Algorithm Distillation (Laskin et al., 2022) / RACER (Dai et al., 2024) | [2303.11366](https://arxiv.org/abs/2303.11366) / [2210.14215](https://arxiv.org/abs/2210.14215) / [2409.14674](https://arxiv.org/abs/2409.14674) | Learning from failed attempts across trials (verbal memory; in-context RL distilled from learning histories; failure-recovery data with rich language corrections). | Multi-trial retry with memory of the failed trial. Our search *is* multi-trial (16 verifications), but the trials are open-loop re-rankings of a fixed pool. | AD / Reflexion gains come from many trials with informative feedback. We have 3 writes of binary outcomes on a pool where 37 % of instances have zero successes [R]. |
| 19 | Hyperband (Li et al., 2016) | [1603.06560](https://arxiv.org/abs/1603.06560) | Successive halving: reallocate evaluation budget away from configurations that look bad early. | This is about cross-instance budget allocation, not within-instance ranking. | The most useful information in the round-1 outcome is *instance difficulty*: 12.5 vs 1.6 remaining successes after an r1 hit vs an r1 miss [R]. A Hyperband-style allocation across instances is a better use of "memory" for dataset yield (follow-up 5). |

**Positioning.**
- *Not new:*
  - "Learn from failed attempts with a fast-weight / sequence policy" (RoboTTT DAgger Distillation, AD, Reflexion).
  - "Retrain the verifier on the optimiser's distribution" (overoptimisation literature; also the project's own B4 /
    Cpi-pool critics).
  - "Update a surrogate after each expensive evaluation" (BO, FSBO, PETS).
- *Would be new:* an instance-level test-time-adapted verifier inside a WM-guided recovery search for VLAs, shown to
  close part of the proxy-gold gap *beyond* a cross-instance recalibrated verifier. That is exactly the comparison that
  failed (gate 3).
- *Defensible today:*
  - A negative, mechanistic statement: within-instance outcome memory adds nothing over a recalibrated verifier at 16
    verifications, because failed and successful candidates are co-located in critic-embedding space. So "remember
    what failed" pushes the search away from the good region.
  - A positive, modest statement: verifier recalibration on imagined end states recovers about 89 % of the
    oracle_true gain (2.327 vs 2.387 hits r2-4) offline. This needs the realistic-label and online checks.

## 2. Counter-questions (>= 3 per category; each: why it matters / what answers it / cost)

Reference numbers [R]:
- Per-instance hits r2-4, sd of paired differences: t2m_label - t2m_label_static 0.729; t2m_label - static 1.300;
  t2m_label_static - static 1.463.
- n = 256 instances in 54 failure-episode clusters (folds 56 / 58 / 82 / 60).
- Successes per instance: 37 % of instances have 0 of 64; median 2; mean 6.75; base rate 0.105.
- Static top-4 FP 0.806. Mean sigmoid(l_wm) is 0.428 vs a real success rate of 0.105.
- Label-oracle ceiling (perfect ranking after the same r1): 4.22 hits r2-4, vs 2.387 oracle_true and 1.859 static.

### 2.1 Construct validity
- **Q2.1a. Is "re-rank a fixed 64-pool" a valid proxy for "CEM stops re-picking over-optimistic modes"?**
  - *Why it matters.* The pool is 16 random + 16 wm_diverse + 16 wm_noise + 16 wm_qd candidates per instance
    (cache `arm`). In CEM, memory would act on *proposal generation* (refit elites, shift the mean away from failed
    modes), and later rounds would sample *near* earlier elites. That makes repeated modes much more frequent than in
    a pool that contains only 0.6 % of near-duplicate action pairs (cos > 0.99 [R]).
  - *What answers it.* An online paired CEM run with a memory-penalised objective (follow-up 6). An offline proxy:
    count how often CEM rounds 2-3 in existing M11 / xWM logs pick candidates within the h_wm radius of an
    already-failed verified pick.
  - *Cost.* Proxy 0 GPU-h (logs). Online about 2 GPU-h + CPU sims.
- **Q2.1b. Does the metric "hits r2-4" match the RetryVLA objective (recovery *data* yield and solved rate)?**
  - *Why it matters.* Solved@16 is near-saturated by the pool: static 0.551, every learned method 0.570-0.578, ceiling
    0.629 (instances with >= 1 success [R]). Most of the gain is extra hits on instances that already have one (gain of
    t2m_label_static: +0.94 on r1-hit instances vs +0.15 on r1-miss instances [R]). Extra hits on already-solved
    instances add *redundant* recoveries, which RESULTS B2 / M5 found worth little for SFT.
  - *What answers it.* Report distinct-recovery yield (hits deduplicated by end-state cluster) and solved@k for k = 4,
    8, 12, 16, per method.
  - *Cost.* CPU minutes.
- **Q2.1c. Did PoC 0 test the mechanism approach A would exploit?**
  - *Why it matters.* RoboTTT's DAgger Distillation ([2607.15275](https://arxiv.org/abs/2607.15275)) gets its gain from
    "failures as context". The memory conditions *which* correction to make; no repeat is needed. PoC 0 only counts
    *repeats of the same pose*. Also, 27 / 73 failed episodes have 0 finger-bowl contact segments and 21 / 73 have
    exactly 1 [R]: a repeat cannot happen there by construction, yet those episodes sit in the denominator.
  - *What answers it.* A history-sufficiency probe: does adding the pre-failure trajectory (or the failed-attempt
    segment) to a probe predicting which verified recovery succeeds beat state-only features? (Q2.5c.)
  - *Cost.* About 0.5 GPU-h if the verif logs keep pre-failure latents; otherwise needs resim (CPU).
- **Q2.1d. Is the PoC 0 attempt definition sound?**
  - *Why it matters.*
    - Repeat = EEF position relative to the bowl at contact onset within 1-3 cm (no orientation, approach direction,
      or gripper aperture). Results swing 0-15 % with tol 1-3 cm and merge gap 10 / 20.
    - The md header still describes the close-command definition ("close held >= 5 steps ... within 0.1 m"), but
      `attempts_touch` ignores `min_hold` / `near` (cosmetic, but misleading in the archived output).
    - Restricted to a2_failed_grasp episodes (the only mode where repeat-breaking could help), the rate is 15-26 % at
      tol 2-3 cm (8 / 34 at 2 cm, gap 10) [R]: on the threshold.
    - Within-episode repeat rate 23-41 % vs cross-episode null 0-14 % (ratio 3-9x): when retries happen, they loop.
  - *What answers it.* Report the a2-only rate with a bootstrap CI (n = 34 is small: 95 % CI roughly 11-41 %), plus an
    orientation-aware repeat (gripper yaw within 15 deg).
  - *Cost.* CPU minutes.

### 2.2 Strongest form
- **Q2.2a. The write target is the residual vs the RAW critic, not vs the current (recalibrated) prediction.**
  - *Why it matters.* In `FastMem.write` (L166-176), v_label = lab_scale * (y - sigmoid(l_wm)): l_wm is the raw C^pi
    logit, mean p 0.43 vs a 0.105 success rate. So almost every written label is a large negative constant.
  - Meanwhile the learned W0 label column *is* the recalibrator (the only path that learns in the static control).
    The delta step moves k·W toward that absolute target, so each write partly *replaces* the recalibrated readout
    around the written keys instead of adding the instance's surprise on top of it.
  - This is a sufficient explanation for t2m_label (2.234) < t2m_label_static (2.327). The deficit is largest on
    r1-hit instances (-0.19 hits [R]), where writing "failure" near the good region hurts most.
  - A Titans / delta-rule-correct design writes v = y - sigmoid(current score) into a separate memory W_mem
    (init 0) whose readout is added to the frozen recalibrated logit.
  - *What answers it.* Follow-up 1.
  - *Cost.* About 0.5-1 GPU-h (or CPU).
- **Q2.2b. In F1, can the memory use its latent channel at all?**
  - *Why it matters.*
    - Stage A is the static (no-write) model. There o is initialised to 0 and W0[:, :dk] = 0, so both gradients are
      exactly 0 and o stays 0.
    - Stage B trains only `k., v., log_eta, lab_scale`, so o is frozen at 0. The self-supervised (h_true - h_wm)
      channel therefore has **zero effect on scores** in both mem_on_recal and mem_on_recal_ss.
    - In mem_on_recal_ss (lab_scale = 0), the only effect of a write is to pull the recalibrator's label column toward
      0 around the written keys, i.e. to *erase* calibration. It is expected to be <= recal_r1 by construction.
    - The F1 rule ("memory adds value iff mem_on_recal - recal_r1 > 0, p < 0.10") therefore tests only the label
      channel, with the Q2.2a target flaw. **A NO from F1 must not be reported as "test-time memory is negative".**
  - *What answers it.* Init o with small random weights (or unfreeze o and W0[:, :dk] in Stage B); use the residual
    target from Q2.2a.
  - *Cost.* Included in follow-up 1.
- **Q2.2b2. Is one GD step per round on 4 items, with a linear fast weight and dk = 64, the strongest memory?**
  - *Why it matters.* After 3 writes of 4 points, a linear W is a 12-point linear regressor. TTT / Titans gains need
    many writes. The strongest few-shot surrogate is an in-context set model (PFN / NP), a GP with a deep-kernel prior
    (FSBO), or at least more rounds with smaller K.
  - *What answers it.* Replay with K = 1, R = 16 (same 16 verifications, 15 writes), and K = 2, R = 8; add a GP baseline
    (kernel on h_wm, prior mean = recalibrated logit, both signs of residual).
  - *Cost.* CPU / GPU < 1 h (follow-up 3).
- **Q2.2c. Is a per-instance memory the right decision unit?**
  - *Why it matters.* The static control *is* a cross-instance memory (trained on other folds), and it won. Instances
    come in 54 failure-episode clusters of about 4.7 consecutive steps, so a campaign-level memory written *across
    instances of the same failure episode / task* could carry far more information than 12 labels per instance.
  - *What answers it.* A prequential replay over the instance stream with episode-level writes. The reviewer's quick
    prequential *recalibrator* (labels = only the 16 picks per instance, refit every 10 instances) gives
    +0.10..+0.20 hits r2-4 over static over the full stream and +0.22..+0.24 on the second half [R], vs +0.47 for the
    fold-trained version.
  - *Cost.* CPU minutes.
- **Q2.2d. Is the 16-verification budget the regime where memory matters?**
  - *Why it matters.* RESULTS RQ3: the WM share of the over-optimism grows with imagination budget (gap +0.01 at N = 8
    to +0.33 at N = 512). Memory of outcomes should matter more when there are more verifications and stronger
    optimisation.
  - *What answers it.* Replay with R = 8 x K = 4 (32 of 64; the pool allows it), with the gate-3 comparison.
  - *Cost.* CPU minutes.

### 2.3 Confounds and fairness
- **Q2.3a. Label access is not realistic for the recalibrator (and for meta-training).**
  - *Why it matters.* The cross-fold models learn from all 64 labels per training instance (about 12k labels,
    including 16 *unselected random-arm* labels per instance, which are the best-calibrated part of the data: RESULTS
    RQ3, random arm AUC 0.83). An online search observes only its own 16 selection-biased picks.
  - Retrained on only the static top-16 picks, the logistic recalibrator gives +0.18 hits r2-4 (p 0.02), not +0.46
    [R]: about 40 % of the reported gain.
  - *What answers it.* Re-run gate 3 and F1 with meta-training restricted to picked candidates (the rounds' own picks).
  - *Cost.* About 1 GPU-h.
- **Q2.3b. Is the sign-flip test valid with 256 instances in 54 failure-episode clusters?**
  - *Why it matters.* Instances from the same failure episode (`__step` suffix) are correlated. Gate 3's p 0.047 at
    instance level becomes p 0.40 with the test done on cluster means [R]. Gate 1 (t2m_label vs static) survives
    (p 5e-5).
  - Conclusion: "t2m_label is significantly *worse* than its control" is not established; "it is not better" is.
  - *What answers it.* Re-run every gate with episode-clustered sign flips (flip whole clusters).
  - *Cost.* CPU seconds.
- **Q2.3c. Do the explicit baselines have a fair hypothesis class?**
  - *Why it matters.* knn_label uses the residual y - sigmoid(l_wm) with lam in {1..32} > 0 only. Since nearly every
    residual is negative and neighbours of failed picks are *more* likely to succeed (Q2.5a), knn_label is forced to
    push the wrong way: round-2 delta -0.086 hits vs static [R].
  - A fair explicit baseline would also allow a positive-only (exploit-near-success) memory and a residual vs the
    recalibrated prediction. Also, knn / ttt_plain hyper-parameters were grid-searched; the t2m models were not
    (lr / epochs / dk fixed).
  - *What answers it.* Add knn_pos (successes only), knn on the recalibrated residual, and a 3-point grid for t2m.
  - *Cost.* CPU minutes / about 1 GPU-h.
- **Q2.3d. Seeds and determinism.**
  - *Why it matters.* t2m_* per-instance hits are means over 3 meta-training seeds; the knn / static arms are
    deterministic. Averaging lowers the variance of the t2m arms, which helps their p-values but not their means.
  - *What answers it.* Report per-seed gate-3 differences; the effect sign should be consistent across seeds.
  - *Cost.* Free (rerun logging).

### 2.4 Statistical power
- **Q2.4a. MDE for the decisive comparison (gate 3: t2m_label - t2m_label_static, hits r2-4).**
  - Paired-difference sd 0.729, n = 256. MDE at two-sided alpha 0.10, power 0.8:
    (1.645 + 0.842) x 0.729 / sqrt(256) = **0.113 hits r2-4 = 0.0094 / verification**.
  - Cluster-robust (54 episodes): SE 0.054, MDE **0.135 hits = 0.011 / verification** [R].
  - The pre-registered effect of interest (+0.03 / verification = 0.36 hits) is 2.7-3.2x the MDE, so gate 3 is
    **well powered**. The observed -0.092 hits has a cluster-robust 95 % upper bound of about +0.014 hits: any memory
    gain >= +0.02 hits r2-4 is excluded for *this* implementation.
- **Q2.4b. MDE vs static (gate 1).**
  - sd 1.300, so MDE 0.202 hits (0.017 / verification); cluster-robust 0.219 (0.018 / verification).
  - The +0.03 / verification threshold is detectable; gate 1 passed at p < 1e-4 under both.
- **Q2.4c. Is the denominator diluted?**
  - 95 of 256 instances (37 %) have 0 successes in the pool, so every method ties there. On the 161 informative
    instances the gate-3 difference is -0.147 hits (MDE 0.18) [R].
  - Same conclusion. Power is not the reason the idea failed.
- **Q2.4d. Power for an online follow-up.**
  - At n = 63 (xWM / RQ3 instance set), the MDE on hits / instance with sd about 1.3 is about 0.41 hits. Only effects
    of the size of the full recalibrator gain (+0.47) are detectable.
  - Use n = 127 (M5b shards 0-3) for the recalibrator online check: MDE about 0.29.

### 2.5 Alternative mechanisms
- **Q2.5a. Failures and successes are co-located; "avoid what failed" is the wrong prior.**
  - After static r1, on instances with >= 1 remaining success, candidates in the nearest quartile (h_wm cosine) to a
    failed pick succeed at **0.221 vs 0.106** in the farthest quartile (base 0.155). In action space: 0.226 vs 0.135.
  - Within the static top-12 the most-similar 4 still beat the least-similar 4 (0.252 vs 0.214) [R]. After an r1
    success, near-success candidates are at 0.311 vs 0.129.
  - So in critic-embedding space, failure is not a separate "over-optimistic mode": it is outcome noise around the
    good region. Any memory that pushes away from failures loses.
  - *What answers it.* The asymmetric memory variants in follow-up 3. If none beats the recalibrator, B's premise is
    falsified on this data.
  - *Cost.* CPU.
- **Q2.5b. Is the recalibrator just learning arm identity / candidate source?**
  - *Why it matters.* Arms differ hugely in calibration (RESULTS RQ3: wm_diverse C(a) 0.727 vs SR 0.171; wm_noise
    0.491 vs 0.076).
  - Answered by the reviewer [R]: an arm-only recalibrator (l_wm x arm one-hot) gives +0.10 hits r2-4 (p 0.13). Plain
    logistic on [l_wm, h_wm] gives +0.43..+0.46 (C = 0.01..0.1), equal to t2m_label_static (+0.467, difference
    p 0.41-0.86).
  - So the lead is "retrain the critic head on imagined end states with real labels", not arm priors. That is the same
    family as B4 / Cpi-pool (RESULTS B2 / P2: more pi0 labels and a better trainer gave +0.06..+0.08 solved online).
  - *Remaining question.* Does a recalibrated C^pi v2 differ from simply training C^pi on WM-imagined latents (the
    RISE-like B3.2 imagination critic)?
  - *Cost.* Compare against existing B3.2 / B4 critics offline on this cache: about 0.5 GPU-h.
- **Q2.5c. For approach A: is the failure state already Markov-sufficient?**
  - *Why it matters.* If the current observation already reveals the failure (bowl displaced, gripper empty), history
    adds nothing, and a memory policy cannot beat the existing state-conditioned recovery SFT, whatever the repeat
    rate. This, not the repeat rate, is the right kill criterion for A.
  - *What answers it.* Probe: predict "which verified recovery arm / bridge succeeds" from (failure-state latent) vs
    (failure-state + pre-failure segment latents) on verif_m5b train instances.
  - *Cost.* About 0.5 GPU-h if latents exist, else CPU resim.
- **Q2.5d. Is the instance-level signal "difficulty", not "mode"?**
  - After an r1 hit, the instance has 12.5 remaining successes; after an r1 miss, 1.6, and 62 % of r1-miss instances
    have no success at all [R].
  - What an outcome memory mostly learns is "keep exploiting here / this instance is hopeless". That is a
    *budget-allocation* signal across instances (Hyperband), not a re-ranking signal within an instance.
  - Static r2-4 yield: 3.93 hits / 12 verifications on r1-hit instances vs 0.46 on r1-miss instances [R].
  - *What answers it.* Follow-up 5.
  - *Cost.* CPU minutes.

### 2.6 What result would flip the verdict?
- **Q2.6a. B -> continue.** A fixed memory (residual target on the current recalibrated score, live latent read-out,
  optionally asymmetric) beats the frozen recalibrator by >= +0.10 hits r2-4 with episode-clustered sign-flip p < 0.10,
  in the K = 4 x R = 4 or the K = 1 x R = 16 replay. It must also not lose on solved@16.
- **Q2.6b. B -> close.** The same comparisons give a cluster-robust 95 % upper bound < +0.10 hits in both replays, and a
  GP / kNN-pos baseline is also <= the recalibrator. Then write: "per-instance outcome memory is negative at n = 256
  (offline, fixed pool, 16-32 verifications); failure neighbours are not worse than average (Q2.5a)".
- **Q2.6c. Recalibrator lead -> close.** With realistic labels (picks only, prequential), it is < +0.10 hits r2-4 offline,
  or the online paired CEM check (n = 127, 16 verifications) gives no gain in hits / verification (sign-flip p >= 0.10)
  with FP on its own picks >= 0.8 (re-exploited).
- **Q2.6d. A -> reopen.** The history-sufficiency probe (Q2.5c) shows history adds >= 0.03 AUC over state-only for
  predicting the successful recovery, or train-split a2 repeat rate >= 20 % with CI lower bound > 10 %.

## 3. Audit
| check | status | note |
|---|---|---|
| oracle / upper bound run | **partial** | `oracle_true` (C^pi on the TRUE end state) was run: 2.387 hits r2-4. It is the ceiling for *latent-only* correction, **not** for label-using memory: the label-oracle ceiling (perfect ranking after the same r1) is 4.22 hits r2-4 [R]. The recalibrator closes 89 % of the static -> oracle_true gap but only 20 % of the static -> label-oracle gap. Failure to approach either is now diagnosed (Q2.5a co-location; Q2.2a write target). |
| powered for the effect that matters | **yes** (for this implementation) | Gate-3 MDE 0.113 hits r2-4 (0.0094 / verification), cluster-robust 0.135 (0.011 / verification), vs the pre-registered effect of interest of 0.36 hits. The negative is powered. But the instance-level p 0.047 for "worse than control" does not survive clustering (p 0.40). |
| strongest variant tested | **no** | Untested: (i) residual-on-current-score write + live latent read-out (Q2.2a/b); (ii) more writes (K = 1 x 16) / larger budget (R = 8); (iii) asymmetric / exploit-near-success memory and a GP / PFN surrogate; (iv) memory acting on CEM *proposals*, not on a fixed pool; (v) campaign-level memory across instances of the same failure episode. (i)-(iii) and (v) are CPU-cheap. For A: the RoboTTT-style failure-as-context policy was never tested; only the repeat premise was. |
| budgets / seeds / instances equal across arms | **yes, with a caveat** | Same 256 instances, same 16 verifications, identical round 1, same folds. Meta-learned arms are averaged over 3 seeds, deterministic arms are not (Q2.3d). Learned arms (and the static control) use all 64 labels of training-fold instances; an online search would see only its picks (Q2.3a: the recalibrator gain falls from +0.47 to +0.18). knn grid excludes lam <= 0 and positive-only writes (Q2.3c). |
| no leakage (test split, expert demos of test scenes) | **OK for PoC 1; minor for PoC 0** | PoC 1 is train-only (verif_m5b), folds by failure episode, C^pi v2 trained on collect_v1 + verif_r1 (pilot instances), WM v1 on collect_v1. Worth a one-line check that no verif_m5b failure state appears in the critic's training set. PoC 0 uses Spatial *test* (dev) episodes; nothing was tuned, but a *design decision* (drop A) was taken on test-split data. Repeat on train-split failures before citing it in a paper. |
| deviations from pre-registration documented | **mostly** | PoC 0: the rule was written for the close-command attempt; the switch to touch segments after seeing the chatter is documented in NOTES (OK). The archived md header still describes the close definition (fix). The T2Mem design + gates live in the docstring + NOTES only, with **no PLAN.md entry** (the idea-loop skill asks for PLAN). The F1 rule is pre-registered in NOTES before running (OK), but F1's structural limits (Q2.2b) are not stated there. |

## 4. Ranked follow-ups (cheap, train split only)
| rank | experiment | could flip because | cost (GPU-h) | decision rule |
|---|---|---|---|---|
| 1 | **F1' fixed write.** Separate W_mem (init 0) on top of the FROZEN recalibrator. Inner target v_label = y - sigmoid(current score); latent read-out o initialised small random (or trained in Stage B); 3 seeds. Same replay (K = 4, R = 4). Arms: mem_fixed, mem_fixed_pos (writes only y = 1), mem_fixed_ss. | Q2.2a/b: the tested write overwrites the recalibrator and F1 cannot use its latent channel. This is the only cheap test of B in its intended form. | 0.5-1 (or CPU) | Memory adds value iff mem_fixed - recal >= +0.10 hits r2-4 AND episode-clustered sign-flip p < 0.10. Otherwise B is negative at K = 4. |
| 2 | **Realistic-label recalibrator.** Train the recalibrator (logistic on [l_wm, h_wm], and the t2m static control) only on the picks a search would verify. Prequential over the instance stream, by episode order; 3 orders. | Q2.3a: the recalibrator lead may be about 40 % of its reported size. This decides whether an online check is worth GPU. | 0 (CPU, about 2 min) | Lead continues iff prequential gain >= +0.15 hits r2-4 on the second half of the stream with clustered p < 0.10 (reviewer's quick run: +0.22..+0.24 second half, so likely PASS). |
| 3 | **More writes / bigger budget / stronger surrogates.** Replay K = 1 x R = 16 and K = 4 x R = 8 (32 verifications). Arms: recal, recal + F1' memory, recal + GP (kernel on h_wm, both residual signs), knn_pos. | Q2.2b2 / Q2.2d: the memory had only 3 writes; the regime where TTT helps was never tested. | < 1 (mostly CPU) | Same as rank 1 in either regime. If both fail, close B (with Q2.5a as the mechanism). |
| 4 | **Episode-clustered re-analysis + per-seed reporting** of all PoC 1 gates and F1. | Q2.3b: the "worse than control" claim rests on instance-level p 0.047 (cluster p 0.40). | 0 (seconds) | Report both. Only cluster-robust p is used for decisions from now on. |
| 5 | **Instance-level budget reallocation** (Hyperband-style) using the r1 outcome. Offline on the same pool: fixed total of 16 x 256 verifications; continue r1-hit instances, cap r1-miss instances at 8. Metrics: distinct hits / verification and solved rate. | Q2.5d: the information in verified outcomes is mostly instance difficulty (12.5 vs 1.6 remaining successes). This may be the real "test-time memory" win for *data construction*. | 0 (CPU) | Positive iff distinct hits / verification +20 % at <= 2 pt solved-rate loss. Then pre-register an online version. |
| 6 | **Online recalibrator check** (only if rank 2 passes). C^pi v2 vs C^pi v2 + recalibrator inside the M5b / wm_unc_l0 CEM pipeline, n = 127 train instances, 16 verifications, paired seeds. Report FP on own picks. | Q2.6c: offline re-ranking cannot show re-exploitation (Gao et al. 2210.10760). | about 2 + CPU sims | Keep iff hits / verification sign-flip p < 0.10 and solved McNemar n01 > n10; else close the lead. |
| 7 | **Approach A sufficiency probe + train-split PoC 0.** Probe history vs state-only for successful-recovery prediction on verif_m5b; rerun `poc_mem_repeat.py --by touch` on train-split failure resims, reporting the a2-only rate with a CI. | Q2.1c / Q2.5c: A's kill criterion tested the wrong mechanism, on test-split data. | about 0.5 + CPU resim | Reopen A only per Q2.6d. |

## 5. Responses (filled by the main agent; every question answered)
| question | response (answered by / follow-up run / out of scope because) |
|---|---|
| Q2.1a | |
| Q2.1b | |
| Q2.1c | |
| Q2.1d | |
| Q2.2a | |
| Q2.2b | |
| Q2.2b2 | |
| Q2.2c | |
| Q2.2d | |
| Q2.3a | |
| Q2.3b | |
| Q2.3c | |
| Q2.3d | |
| Q2.4a-d | |
| Q2.5a | |
| Q2.5b | |
| Q2.5c | |
| Q2.5d | |
| Q2.6a-d | |

**Reviewer's top-3 counter-questions (closing rule 4):**
1. **Q2.2a/b.** Does the memory write the *right* thing? Today it writes a residual vs the raw critic, which overwrites
   the recalibrator. In F1 its latent read-out is frozen at 0, so F1 tests only the label channel.
2. **Q2.5a.** Failed picks' neighbours succeed *more* often (0.22 vs 0.11). Is "avoid what failed" a wrong premise
   for this search, so that B can only win via exploit-near-success / difficulty signals?
3. **Q2.3a.** Does the recalibrator lead survive realistic label access? Picks-only gives +0.18, not +0.47.
   Does it survive online CEM optimisation against it?

## 6. Final verdict (main agent, after responses + follow-ups)

Reviewer's recommendation (to be confirmed or overridden by the main agent after Section 5):
- **Approach A (memory policy, repeat-breaking): close as tested; park the RoboTTT-style variant.**
  - Only 8-15 % of 73 failed test-split episodes repeat a failed grasp. 27 / 73 have no bowl contact at all.
  - The a2-only rate (15-26 %) is on the threshold and its CI is wide (n = 34).
  - The mechanism prior work validates (failure-as-context) was not tested, and it requires a non-Markov failure
    signal that LIBERO-Recover Spatial may not have (Q2.5c).
- **Approach B (per-instance test-time memory in recovery search): redesign. Not closable yet under the hard rules**
  (strongest variant untested; top-3 questions open).
  - The current negative is real and powered *for this implementation*: memory over its static control is -0.092 hits
    r2-4, cluster-robust 95 % CI upper bound about +0.014, MDE 0.135.
  - It is explained by (i) a write target that conflicts with the learned prior and (ii) the data fact that failure
    neighbours are not worse.
  - Run follow-ups 1, 3, 4 (all < 1 GPU-h, mostly CPU). If they fail, close B with the Q2.5a mechanism as the stated
    reason.
- **Recalibrator lead: continue as its own idea** ("imagination-aware verifier recalibration"; prior-work family:
  overoptimisation / B4 / Cpi-pool critics).
  - Offline: +0.47 hits r2-4 (fold-trained), +0.18 picks-only, +0.10..+0.24 prequential.
  - Gate it with follow-up 2 (CPU), then follow-up 6 (online, n = 127).
  - Do not call it a memory result, and do not distill from it until the online check passes.
- **F1** (`poc1_followup.md`) had not appeared when this review was written. Whatever it shows, interpret it under
  Q2.2b: a NO is not evidence against test-time memory in general.
