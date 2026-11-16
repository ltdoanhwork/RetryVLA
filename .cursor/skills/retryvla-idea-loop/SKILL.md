---
name: retryvla-idea-loop
description: >-
  End-to-end loop for testing a RetryVLA research idea seriously: intake with pre-registered decision rules in
  experiments/PLAN.md, an oracle-first paired PoC (equal budgets, train/pilot only, 3 seeds, nominal check, per-level
  L1-L4, McNemar / bootstrap), a mandatory diagnosis of negative results, a mandatory adversarial research review by a
  separate agent, and recording in RESULTS / NOTES / RELATED_WORK plus commit. Use whenever a new RetryVLA idea is
  proposed, designed, piloted, evaluated or about to be closed / pivoted / scaled, or when the user asks to evaluate,
  test or decide on a research idea, hypothesis or PoC.
---

# RetryVLA idea loop (never close after a shallow attempt)

User rule (2026-11-16): research ideas are tested SERIOUSLY. A negative PoC is a statement about one implementation
at one budget on one instance set, not about the idea. An idea is closed only after steps 1-5 below are all done.

Related skills: `retryvla-experiment` (configs, splits, registry), `retryvla-eval` (RSR / nominal protocol),
`retryvla-search` (CEM, scorers, proposals), `retryvla-ops` (env, tmux), `retryvla-multi-gpu-shells` (borrowed GPUs),
`retryvla-idea-review` (reviewer prompt, template, hard closing rules used in step 4).

```
Idea loop progress:
- [ ] 1. Intake: dated PLAN.md section with pre-registered decision rules (BEFORE any result), committed
- [ ] 2. Serious PoC: oracle arm first, paired, equal budget, train/pilot only, 3 seeds, nominal, L1-L4, stats
- [ ] 3. If negative / null: diagnosis checklist + "what must be true" + weakest assumptions
- [ ] 4. Adversarial research review by a separate agent (positive OR negative); all questions answered
- [ ] 5. Record (RESULTS / NOTES / RELATED_WORK / PLAN status) and commit + push
```

## 1. Intake (before running anything)

Write a dated section in `experiments/PLAN.md`, e.g. `## <IDEA> design note (user request YYYY-MM-DD; written BEFORE any result)`:

```markdown
- Research question: <one sentence>
- Hypothesis: <mechanism + expected direction + rough effect size>
- Falsified if: <the concrete observation that would refute it, incl. the oracle arm>
- Closest prior work to check: <2-5 papers / methods; add to docs/related_work/RELATED_WORK.md>
- Arms: control (current best, e.g. `single` / pi1), method, oracle / upper bound, cheap ablation(s)
- Instances + split: <train/pilot shards, n>; budget per arm: <sim steps / verifications / SFT steps>
- Primary metric + test: <e.g. solved rate, McNemar vs control; RSR scene x seed bootstrap>
- Decision rules: continue if ..., distill / test-eval if ..., redesign if ..., inconclusive if power < 80 %
- Minimum detectable effect at this n (alpha 0.10, power 0.8): <number>
```

Commit it (step 5 conventions) before the first result exists. Any later change is a logged deviation in NOTES.

## 2. Serious PoC

1. **Oracle / upper-bound arm first** (privileged sim state, oracle scorer, ground-truth labels). If even the oracle
   does not help, that is a finding to DIAGNOSE (step 3: which assumption breaks?), not a reason to close.
   If the oracle helps and the method does not, the gap is the research question.
2. **Paired** comparisons: same instances, same protocol seeds, same snapshots for every arm.
3. **Equal budgets**: sim steps / verifications / proposals / trajectory count / SFT steps
   (`train.match_trajectory_count_of`). Report budget-normalised metrics (hits / 1k sim) next to solved rate; a method
   that spends budget elsewhere (e.g. executing bridges) must be charged for it.
4. **Splits**: tune on train / pilot only; Spatial test only for the frozen recipe; never use test-scene expert demos
   (`assert_allowed()`; see `retryvla-experiment`).
5. **3 seeds** (SFT seeds or search seeds) before any claim; 1-seed numbers are labelled "seed 0, preliminary".
6. **Nominal preservation**: any fine-tuned policy reports nominal SR; drop <= 2 pt vs reference (mean over seeds).
7. **Per-level reporting**: LIBERO-Recover L1-L4 (`experiments/levels/`, `scripts/analyze_levels.py`) next to overall.
8. **Statistics**: McNemar (n01 / n10, p) for paired binary outcomes; scene x seed bootstrap CI + sign-flip for RSR
   deltas; Wilson CI for rates. State n and power; below ~80 % power write "inconclusive at n = ...", not "negative".
9. Pilot first (`search.max_instances`, 1 shard), then the pre-registered n. Pre-register every follow-up run in NOTES
   (one-line hypothesis + decision rule) before launching.

## 3. Negative or null result: required diagnosis

Answer each item with a number or file, in RESULTS (idea section, "Reading"):

- [ ] **Is the bottleneck the component tested?** Compare oracle vs method vs control. If the oracle of the component
      does not help, the bottleneck is elsewhere (name it).
- [ ] **Budget-split artifact?** Did the method spend budget on something the control did not (execution, extra
      models, fewer verifications)? Compare at equal sim steps and per-verification precision.
- [ ] **Option set too weak?** Were good candidates even in the proposal pool (oracle coverage, best-of-pool success)?
      A selection method cannot beat a pool with no solutions.
- [ ] **Verifier / WM over-optimism?** Predicted vs realised success of the picks (mean pred vs SR, FP at pred >= 0.5,
      ECE). Large gaps mean the scorer is being exploited.
- [ ] **Continuation policy competence?** Does the frozen pi actually finish from the states the method reaches
      (C^pi on real end states, success of continuation from oracle-good states)?
- [ ] Power / n, implementation bugs (replay check, default-path equivalence test), deviations from the pre-registration.

Then write:
- **What would need to be true for the idea to work** (e.g. "imagined end-state C^pi within 0.1 of real").
- **Weakest assumptions** (ranked list) and which follow-up tests each.
- **The next question** the negative result points to (see the example below).

## 4. Mandatory adversarial research review (after EVERY PoC, positive or negative)

Freeze evidence (commit code, configs, analysis), then launch a separate Task (`subagent_type: generalPurpose`,
fresh context) using the prompt and `template.md` from `retryvla-idea-review`. The reviewer must, in this order:

1. **Web research**: the closest 2025-2026 papers (>= 8 sources, arXiv IDs verified by opening arxiv.org/abs/<id>),
   nearest-prior-work table, explicit overlap / novelty risks for our claim.
2. **Counter-questions**: steelman (strongest reading in favour) and strawman (weakest / most deflationary reading) of
   the result; specific counter-questions citing our numbers and files.
3. **1-3 decisive follow-up experiments** (cheap, train split, could flip the verdict), with cost and decision rule.
4. **Only then** a recommendation: continue / pivot (redesign) / close, with justification and the evidence that
   would change it.

Output: `experiments/reviews/<IDEA>_review.md`. The main agent answers every counter-question in its "Responses"
section, runs the decisive follow-ups that are affordable, and re-checks the hard closing rules in
`retryvla-idea-review`. **No "close" / "negative" / "no distill" verdict is written before this review exists and
its questions are answered.** A positive result gets the same review before scaling, distilling or test-eval.

## 5. Recording and commit

- `experiments/RESULTS.md`: idea section (design pointer, tables, per-level, stats, decision applied, Reading, verdict).
- `experiments/NOTES.md`: ops log (shells, incidents, deviations, pre-registered follow-ups).
- `docs/related_work/RELATED_WORK.md`: papers found in intake + review (verified IDs, how we differ).
- `experiments/PLAN.md`: status line on the idea section (continue / pivot / closed + date + review link).
- Runs: `python scripts/register_run.py runs/<id> --note "..."` -> `experiments/REGISTRY.md`.

Commit with explicit paths (author is the repo config, ltdoanhwork <lethiendoanh.work@gmail.com>; the script adds
no tool trailers, never pass `Co-authored-by` / Cursor trailers; never add `.env` or secrets):

```bash
scripts/git_commit.sh "<IDEA>: <what changed>" experiments/RESULTS.md experiments/NOTES.md ...
env -u LD_LIBRARY_PATH git push -q origin main \
  || { env -u LD_LIBRARY_PATH git pull -q --rebase origin main && env -u LD_LIBRARY_PATH git push -q origin main; }
```

Do not call `scripts/git_commit.sh` without paths when unrelated files are dirty (it would `git add -A`).

## Example: turning negatives into the next question (project history)

| PoC | Result | Diagnosis | Next question |
|---|---|---|---|
| RQ2 action-free imagined futures (2026-11-01/03) | arm 3 (C^pi-filtered futures) test RSR 0.125 vs pi1 0.418, 3 seeds | WM over-optimism: filtered futures had mean C^pi 0.86 but only 0.16 sim success; the targets teach states pi cannot finish from | Can imagined end states be trusted at all? -> measure imagined vs real C^pi gap per method |
| RGR recursive geometric recovery (2026-11-08) | no arm beats `single` (solved 0.425; rec_geo 0.386, McNemar p >= 0.23), even with privileged geometry (rec_geo_oracle 0.378) | Geometry oracle = head, geometric failures are rare (2-5 %): geometry is not the bottleneck. Forward correction shows imagined C^pi ~0.41 vs observed 0.15-0.16: WM imagination is | Would an independent WM remove the exploitation? -> xwm |
| xwm cross-WM agreement, WM v1 x V-JEPA (2026-11-09) | min-agreement best on hits / sim (+12-17 %) and FP, but n.s. (n = 60, McNemar p 0.29) | Errors are correlated: when both WMs predict success the pick still fails 58-74 %; also under-powered | Not "close": verify at n = 127; and target the shared failure (imagined end states), e.g. cheap real verification / active probing (ACR) |

Pattern: each negative named the failing assumption with a number (pred vs real), checked whether the oracle of the
tested component helped, and produced a sharper hypothesis instead of a closed idea.
