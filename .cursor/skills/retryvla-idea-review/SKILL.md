---
name: retryvla-idea-review
description: >-
  Mandatory independent review before a RetryVLA research idea is closed, scaled up or distilled. Spawns a separate
  fresh-context agent that does web literature research (verified arXiv IDs), asks adversarial counter-questions,
  audits the kill criteria / statistical power / fairness of the PoC, and returns a continue / redesign / close verdict
  with the evidence still required. Use after any RetryVLA PoC or negative result (e.g. RGR, xwm, ACR), whenever an
  agent is about to write "negative -> close" / "no distill", or when the user asks to stress-test, red-team, or
  "research more before closing" an idea.
---

# RetryVLA idea review (red-team before closing)

A PoC that comes out negative is a claim about one implementation, at one budget, on one set of instances. It is not yet a claim about the idea. This skill makes sure an idea is closed only after its strongest version has been tested and an outsider has tried to break the conclusion.

## When to use it

Use it whenever any of these happens:
- a RetryVLA PoC finished (positive or negative)
- a pre-registered gate failed
- an agent is about to write "close", "drop", "no distill" or "scale up" in RESULTS / PLAN

## Workflow

```
Review progress:
- [ ] 1. Freeze the evidence (commit; list result files)
- [ ] 2. Spawn the reviewer (separate Task, fresh context, prompt below)
- [ ] 3. Reviewer writes experiments/reviews/<IDEA>_review.md (template.md)
- [ ] 4. Main agent answers EVERY counter-question in the review file (section "Responses")
- [ ] 5. Run the cheap follow-ups that could flip the verdict (<= ~2 GPU-hours each, train split only)
- [ ] 6. Re-check the closing rules below; write the final verdict in RESULTS + PLAN
```

**Step 1: freeze the evidence.** Commit the code, the config and the analysis output. Then collect the paths the reviewer needs:
- the design note (the PLAN section)
- the RESULTS section
- the analysis markdown / decision json
- the code entry points
- NOTES entries about incidents or deviations

**Step 2: spawn the reviewer.** Use the Task tool with `subagent_type: generalPurpose`. Do not reuse the implementing agent's context: the point is independence. Fill in this prompt:

```
You are an independent reviewer for the RetryVLA project (repo /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA).
Idea under review: <IDEA name + one-sentence hypothesis>.
Read: <design note path + section>, <RESULTS section>, <analysis md / json>, <code paths>, <NOTES section>,
and .cursor/skills/retryvla-idea-review/template.md (your output format).
Tasks:
1. Literature (web search + fetch). At least 8 sources relevant to THIS hypothesis. Verify every arXiv ID by opening
   arxiv.org/abs/<id> (title must match; IDs quoted by the user or in docs have been wrong before). Build the
   nearest-prior-work table: what each does, how it differs, what it implies for our design or claims.
2. Counter-questions. At least 3 in EACH category of template.md, specific to this PoC (cite numbers / files),
   each with: why it matters, what result would answer it, cost estimate.
3. Audit the conclusion: was the hypothesis tested in its STRONGEST form? Was the oracle / upper bound run? Is the
   sample size powered (compute the minimum detectable effect at alpha 0.10, power 0.8 for the paired test used)?
   Are budgets / seeds / instances equal across arms? Any leakage (test split, expert demos of test scenes)?
4. Ranked follow-ups: cheap experiments (<= ~2 GPU-hours, train split) that could FLIP the verdict, ranked by
   (probability of flipping x importance) / cost.
5. Verdict: continue / redesign / close, with the exact evidence that would change it.
Write the review to experiments/reviews/<IDEA>_review.md following template.md. Do not edit any other file.
Do not run GPU jobs. Report the file path and the verdict in your final message.
```

**Step 4: answer the questions.** For each counter-question, the main agent writes one of:
- "answered by <file / number>"
- "follow-up run: <run id>, result"
- "out of scope because <reason>"

Leaving a question unanswered blocks closing.

**Step 5: run the follow-ups.** Run the ones ranked highest. Use the existing infrastructure and claims, following the `retryvla-ops` and `retryvla-multi-gpu-shells` skills. Pre-register each one in NOTES before running it, with a one-line hypothesis and a decision rule.

## Closing rules (hard)

An idea may NOT be marked closed / negative while any of these holds:
1. The oracle or upper-bound arm was not run, or it was run and it helped, but the reasons the method failed to reach it are not diagnosed.
2. Power is below about 80 % for the effect size that would matter. In that case, write "inconclusive at n = ...", not "negative".
3. The strongest variant named by the review is untested. This includes a larger budget, a different decision unit, or a better proposal source.
4. Any of the reviewer's top-3 counter-questions is unanswered.
5. A deviation from the pre-registration is undocumented.

"Redesign" is a valid outcome. In that case, write the new design note before running anything new.

## Output locations

- Review: `experiments/reviews/<IDEA>_review.md` (template: [template.md](template.md))
- Responses + final verdict: the same file (section "Responses"), plus a one-paragraph summary in `experiments/RESULTS.md` (idea section) and `experiments/PLAN.md` (status line)
