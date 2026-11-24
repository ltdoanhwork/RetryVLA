---
name: retryvla-run-lessons
description: >-
  RetryVLA-specific run and research lessons learned the hard way: resumable chains with DONE / summary.json / CLOSED
  markers and a 45-min stall watchdog, a mandatory final "analyze + register + commit" step, paired designs and
  selection effects (regression to the mean, fresh-seed comparators), pre-registration and power, WM over-optimism
  measurement, degenerate ensemble heads, cut-point and data-mix-share pitfalls (T/4 pre-onset cuts, 47 % vs 4 %
  recovery anchors), bridge jerk, protocol reproduction gates (2610.05151 G0), ruling out bf16 / code-path
  regressions with control runs, LIBERO-Goal / LIBERO-Recover specifics, where results live, and commit conventions.
  Use when writing or launching a RetryVLA chain / watchdog, interpreting or reporting a RetryVLA comparison,
  designing a follow-up, reproducing another paper's protocol, building a distillation mix, or committing results.
---

# RetryVLA run + research lessons

Adds only what the other skills do not cover. Read with:
- `determined-gpu-cluster-ops` (personal skill): det CLI, tmux on the remote side, EDQUOT-safe writes, borrowing
  rules, `\r` / argv traps, watchdog loop pattern. Not repeated here.
- `retryvla-ops` (env, sim conventions), `retryvla-multi-gpu-shells` (shells.json, launch_remote / remote_ctl),
  `retryvla-experiment` (configs, splits, registry), `retryvla-eval` (RSR / nominal protocol), `retryvla-search`
  (CEM, scorers), `retryvla-idea-loop` + `retryvla-idea-review` (pre-registration, PoC, diagnosis, review),
  `flow-diagram` (diagrams).

## 1. Chains and watchdogs

Existing patterns to copy: `scripts/goal_f1_chain2.py` (shared locked queue + per-GPU lanes), `scripts/bot_watch.py`
(local watchdog, <= 3 relaunches, re-asserts claims, analysis at CHAIN_DONE), `scripts/xwm_chain.py` (owner back-off),
`scripts/rgr_chain.py` (phase-2 decision + `det shell kill`), `scripts/goal_diag_codepath.sh` (marker-guarded steps).

Every stage is skip-if-done, keyed on a marker that is written LAST:
```bash
[ -f $O/DONE ] || torchrun ... -m retryvla.train.pi0_sft --out $O --resume_every 250 ...
[ -f $O/ckpt_02000/eval_val/summary.json ] || python -m retryvla.eval.eval_goal --ckpt $O/ckpt_02000 ...
```
- Search parts: `data/recovery/<x>/s<k>/p<n>/CLOSED`; per-instance `done.jsonl`. A part without `CLOSED` is redone on
  relaunch (RQ3 sweep: check the datasets, write `CLOSED` by hand only if complete).
- Resumed searches must keep pools paired: `m5b.paired_rng` draws every instance's proposals in order and skips done
  ones only after drawing. Without it, resumed instances get different pools than the control.
- Stall watchdog: no log write for 45 min, or session / shell gone -> kill the exact `rv_` session, relaunch, resume.
- The chain's last stage is ALWAYS `analyze -> append RESULTS / NOTES -> register_run -> commit + push -> release
  claims -> det shell kill` (unless `scripts/shells/keep_<id>`). Lessons:
  - BoT chain DONE 11-23 11:00, analysed 11-24 08:50; F1 table auto-appended with "reading below when written" and no
    reading; G0 finished 08:56 and GPUs 1-3 idle until 17:15 because the agent turn ended with nothing queued.
  - Pending-work audit (11-08) found finished outputs (true-end diag) never folded into the summary.
  - At the start of every session: `remote_ctl.sh status`, list `runs/*/DONE` / `summary.json` newer than the last
    RESULTS entry, and analyse + commit anything finished before launching new work.
- Release claims by session name only (slot-key release killed a sibling's `rv_af_dg`; another chain relaunched the
  same eval 100+ times).

## 2. Comparisons: paired, fresh, powered

| symptom | cause | fix |
|---|---|---|
| Method "solves 22 % of U vs single 0 %" | U = instances the historical `single` run failed: selection on the comparator's noise -> regression to the mean (fresh `single` re-solves 16 % of them) | Re-run the comparator on fresh seeds on the same set (`rgr.seed_offset 1000`; NOT `eval.seed`, it also seeds the LIBERO-Recover object noise and changes instances). Decide only vs the fresh run |
| Control numbers differ from an earlier sweep on the same pools | bf16 GPU non-determinism in WM rollouts flips CEM picks (72-78 % slot agreement) | Only in-run controls in paired tests; never mix runs |
| "Untriggered" test-time episodes differ from base | batched bf16 non-bit-identical inference | Report the placebo flip floor (~3-6 %) next to rescued / broken |
| Point estimate negative, p 0.3, n = 73 | power ~0.5 for +8 pt | Write "inconclusive at n = 73 (MDE ...)", compute MDE in the pre-registration, pick n for 80 % power |
| Gate threshold looks great (0-3 cm vs > 6 cm split) | chosen on the evaluation instances | New instances + new pre-registration before using it |

- Pre-register (PLAN / NOTES, committed) BEFORE any result, including follow-ups and diagnoses (D1, D2 ...). Later
  changes are logged deviations. Clarifications that see no data are fine but must say so.
- Report rescued / broken (McNemar n01 / n10) next to the SR delta; Wilson CI for rates; scene x seed bootstrap for RSR.

## 3. World model / critic pitfalls

- **WM over-optimism is the recurring bottleneck.** Always measure predicted vs realised on the picks: mean imagined
  C^pi vs sim SR (RQ2: 0.86 vs 0.16; RGR: ~0.41 vs 0.15), FP rate at pred >= 0.5, ECE, and the WM share of the gap
  (critic on the TRUE end state vs on the imagined one: ~80-85 % is the WM). Two different WMs share errors (xwm: both
  agree on success, still fail 58-74 %).
- **Degenerate ensembles.** All LatentWM checkpoints had 4 BIT-IDENTICAL `z_heads` / `q_heads` (zero init + identical
  gradients) -> U_WM == 0 and every `lambda_unc` arm was a no-op. Check `max |W_e - W_0| > 0` and that U has spread
  before reporting any uncertainty result; use bootstrap heads (`retryvla/wm/boot_heads.py`). U also collapses when
  C^pi saturates at 1.0 on optimised picks.
- A critic that ranks bridges offline is not an online monitor: Goal C^pi (val AUC 0.66, trained on failure-entry
  bridge ends) fired on 67 % of successes inside BoT (51 vs 85). Liveness trained on nominal on-policy rollouts did
  better. Progress-head selectors reward "looks later in the task" (ForkCast progress 32).

## 4. Data / distillation pitfalls

- **Cut point.** Fixed T/4 cuts are mostly pre-onset: entry EEF 6 mm from D0, M0 re-succeeds from 46 % of them, only
  12 % of T/4 trajectories finish inside the bridge (rest = 50-step bridge + M0 continuation). Level and difficulty
  are properties of (rollout, cut). Use onset-by-branching (earliest state where pi alone fails but recovery exists)
  and report solved rate per cut fraction. See `docs/research/CUT_POINT_STUDY.md`.
- **Mix share.** F0 bank = 282 trajectories = 47 % of anchors (bridges 14.1k + M0 continuations 22.4k frames) vs
  2610.05151's 4.1 % -> Original 79.3 -> 68.0, damage also on tasks with no recovery data. Always report recovery
  anchor share, continuation share, and bridge-only vs bridge + continuation, per task. Match share before comparing
  generators (`wm` ≈ `random` under a flooded mix says nothing about the generator).
- **Continuations.** Continuation segments re-imitate M0's imperfect behaviour from off-nominal states; count them as
  self-imitation, not recovery data. Bridge action-chunks near the bridge end still contain continuation actions
  unless masked.
- **Bridge jerk.** Bank bridges have ||Δ³a|| 4.7x demos (Spatial; Goal 3.4x), rougher at later cuts (0.17 -> 0.44);
  pi1 chunks are 2.4x jerkier than pi1-nominal: jerk is distilled. Report jerk of banks and student; smoothing moves
  the path ~4 mm, so re-verify smoothed bridges. `scripts/analyze_smoothness.py`.
- Nominal check for every SFT (drop <= 2 pt, 3 seeds); per-task forgetting (`scripts/analyze_data_funnel.py`).

## 5. Reproducing another paper / ruling out our own regressions

- **Reproduction gate first** (2610.05151 G0): reproduce their baseline numbers on their protocol (M0 85 vs paper 77,
  Original 82 vs 78) and pre-register pass / fail before any comparison. Doc: `docs/related_work/COMPARE_2610.05151.md`.
- Build their data from their records: replay-built D0 was dropped because only 82 % of demo replays succeed in our sim.
- **When a new arm is unexpectedly worse, rule out the code path with a control before reading the result.** F1
  (59.0) < F0 (68.0) < Original (79.3) after changing the trainer (single GPU accum 3, compact bf16 resume, forced
  restart, bf16 final): D1 cast Original to bf16 (83 vs 82: weights not the cause); D2 runs the Original recipe
  through the NEW path (`scripts/goal_diag_codepath.sh`). Any infra change (precision, resume format, accum, DDP vs
  single) needs such a default-recipe control and must be logged as a deviation.
- Deviations table in the repro doc (e.g. `BACK_ON_TRACK_REPRO.md` D1-D15), written before the full run.

## 6. Benchmark specifics

LIBERO-Goal (2610.05151 protocol, `retryvla/train/pi0_sft.py`, `retryvla.eval.eval_goal`):
- pi0 3B + LoRA (451.5 M trainable), global bs 48; 100 val scenes, <= 300 steps, replan 5; eval 6-15 min / 100 ep.
- M0 audit on 300 train scenes: 56 failures / seed (cream_cheese, wine_rack dominate), 3 tasks have 0 failures ->
  C^pi undefined there. Entries have no M0 pre-check (T/4 entries may be solvable).
- Checkpoints: `trainable.pt` (LoRA merge on load); keep the M0 init `runs/goal_rsi/m0_s7/ckpt_04000`.
LIBERO-Recover (Spatial / Object): official `load_scene` noise, `snapshot()` / `restore_snapshot()` for branching,
`recover.instruction` source stated; the curated split has 0 L4 scenes; report L1-L4 (`scripts/analyze_levels.py`).

## 7. Where things live

| what | where |
|---|---|
| plans + pre-registrations | `experiments/PLAN.md`, `docs/plans/` |
| results (tables, decisions, readings) | `experiments/RESULTS.md`; raw `runs/<ts>_<M>_<tag>/summary.json`, `runs/goal_rsi/*.json`, `runs/_analysis/` |
| ops log, incidents, cleanup rounds, deviations | `experiments/NOTES.md` |
| run registry | `experiments/REGISTRY.md` via `scripts/register_run.py` |
| reviews | `experiments/reviews/`, `docs/reviews/` |
| research notes | `docs/research/` (CUT_POINT_STUDY, DATA_FUNNEL_AND_FORGETTING, VERIFICATION_COST, ...) |
| related work | `docs/related_work/RELATED_WORK.md`, `COMPARE_*.md` |
| shells, claims | `scripts/shells/shells.json`, `scripts/shells/*` |

## 8. Commit conventions

```bash
scripts/git_commit.sh "<AREA>: <what changed + key numbers>" experiments/RESULTS.md experiments/NOTES.md <paths...>
env -u LD_LIBRARY_PATH git push -q origin main \
  || { env -u LD_LIBRARY_PATH git pull -q --rebase origin main && env -u LD_LIBRARY_PATH git push -q origin main; }
```
- Author = repo config ltdoanhwork <lethiendoanh.work@gmail.com>; the script adds no trailers. Never add
  `Co-authored-by` / Cursor trailers.
- ALWAYS pass explicit paths: with no paths the script runs `git add -A` and sweeps in other agents' dirty files.
  Parallel agents share the tree; commit only your own files. Never commit `.env`, tokens, `MUJOCO_LOG.TXT`, data, runs.
- Message style: area prefix + verdict + key numbers (e.g. `2610.05151 F1 share-matched distillation: results table`).
- If EDQUOT blocks git object writes, wait for headroom; don't delete things to force the commit.
