# V2R review, 2026-11-25: RetryVLA v2 levers on LIBERO-Recover Spatial (SmolVLA)

Pre-registration: `experiments/PLAN.md` section "V2R" (commit a547a63, written 2026-11-24 15:59 UTC before any V2R
result). Deviation D-V1 (R2 filter) is logged in `experiments/NOTES.md` (2026-11-25 03:35 UTC, before any R2
result). Status at writing: **2026-11-25 ~03:50 UTC. No R1-R4 policy has finished training or eval yet.** The tables
in the auto section at the end are refreshed by the chain after each stage.

## 0. Headline status (honest)

- **Lost night.** R1 search finished on 11-24 at about 19:00 UTC. The training / eval chain was **not** started in the
  same agent turn, so both 2-GPU shells (`ea7f9ee3`, `f72a5ddd`) sat idle until 11-25 03:10 UTC. That is about 8 h x
  4 GPUs. The chain and watchdog have been running since 03:10 UTC.
- **Done so far:**
  - R1 search (critic swap) on the 256 M5b train failure instances;
  - the R1 bank plausibility scan (R2 input);
  - visual checks of the R1 bank and of the R2 selection;
  - smoke tests of every new code path.
- **Running:**
  - R1 SFT x 3 seeds (about 55 min each);
  - D_P generation, 2 shards over 56 train source episodes (5-12 min per episode). The first 7 episodes give 6 verified
    positives at 5 / 7 s^- (t_last 40-160);
  - R0 nominal re-evals with saved actions (event metrics);
  - R2 SFT s0 (since 03:39 UTC; s1 / s2 follow when the R1 SFTs finish).
- **Can be said now (search level only):**
  - the retrained `base` critic does **not** find recoveries for more instances: solved 0.570 vs 0.547 for the
    shipped C^pi v2 on the same instances and slot seeds, McNemar 25 / 19, p 0.45;
  - verified hits per verification: 0.178 vs 0.171.
- **Strongest data finding of the night:** searched recovery bridges are far from demo-like.
  - Bridge jerk (median ||Δ³a||) is 0.446 vs 0.098 in the LIBERO-Spatial demos (4.6x).
  - Only 5 / 436 trajectories pass the pre-registered plausibility filter.
  - 11 % do not even replay to success from their rebuilt snapshot.


## 0b. Update 2026-11-25 08:55 UTC

- **Second incident.** From about 04:15 to 08:35 UTC the chain launched nothing new. Tasks from a restarted chain
  became zombies, and the chain counted them as alive.
  - Only f72a5ddd GPU1 kept working; it trained R2 s0-s2.
  - Fixed and restarted at 08:39 UTC. The chain now runs 3 sim jobs per shell.
- **Trained so far:** R1 s0-s2 and R2 s0-s2.
- **Evaluating since 08:39:** R1 s0-s2 RSR + nominal (RSR about 45 min, nominal about 20 min + event replay).
- **D_P closed** (`data/recovery/v2r_dp`):

  | stage | count |
  |---|---|
  | train source episodes | 56 |
  | with t_last (>= 2 / 8 pi0 successes) | 52 |
  | s^- with a pi0 failure | 48 |
  | s^- with >= 1 verified pair | 42 |
  | positive trajectories | 78 (n_P = 70) |

- **R3 SFT s0-s2** since 08:43 UTC: 138 R2-selected + 70 D_P = 208 trajectories.
- **D_P visual check** (full fps, `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/visual/recover_v2r_dp/`):
  - Most late pairs (s^- at t140-160) are the "last centimetre" of the task: the bowl is already over the plate, a+ =
    5-24 steps of lowering / release, and pi0's own a- fails there.
  - Early pairs (t060-080) are 50-step reaches / regrasps.
  - The bank is smoother than R1: bridge d3 median 0.24 vs 0.45, no chatter.
  - The "FAIL" banner on these sheets is a renderer artefact: there is no success_step in the D_P meta.

  ![D_P late pair](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/visual/recover_v2r_dp/v2r_dp_sh0_bank0_ep16_sheet.png)
  ![D_P early pair](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/visual/recover_v2r_dp/v2r_dp_sh1_bank0_ep1_sheet.png)

- **R0 event baselines** (E-v1s on 500 nominal episodes; 96 % replay-checked):

  | seed | SR | failure incidence | first-attempt success |
  |---|---|---|---|
  | s0 | 0.804 | 0.068 | 0.784 |
  | s1 | 0.770 | 0.088 | 0.746 |

  Dominant events: grasp_lost, tipped, displaced.
- **New ETAs (UTC):**

  | item | ETA |
  |---|---|
  | R1 table | ~09:40 |
  | R2 | ~10:30 |
  | R3 | ~11:30 |
  | R4 (aux) SFT / evals | ~12:30 / ~13:30 |


## 1. What ran

| step | what | where / when | output |
|---|---|---|---|
| R1 search | compare_arms, wm_diverse only, ranked by the CRIT-V `base` critic (`runs/20261124_CRITV_spatial/ckpt/base_s0.pt`); same 256 instances, pool 256, 16 verifications, pi0 continuation, <= 4 trajectories / instance | ea7f9ee3, 4 lanes, 11-24 16:01 to ~19:00 UTC | `data/recovery/v2r_r1` (436 trajectories, 6.0 GB), runs `*_V2R_v2r_r1_search_s{0..7}of8` |
| R1 plausibility scan | `retryvla.search.rec_phys`: bit-exact replay of every trajectory with privileged geometry | f72a5ddd, 03:10 to 03:25 UTC | `data/recovery/v2r_r1/phys_wm_diverse{,_hard}.jsonl`, `phys/` (aux targets) |
| visual checks | `scripts/v2r/visual_bank.py` (wraps the retryvla-visual-check renderer); 8 stratified renders per bank | 03:11 / 03:45 UTC | `runs/_analysis/visual/recover_v2r_r1/`, `recover_v2r_r2/`, `recover_v2r_r2h/` |
| R1 SFT s0 / s1 / s2 | pi1 recipe (init pi1-nominal, 6k steps, 50 % nominal, 208 trajectories round-robin) | ea7f9ee3 GPU 0 / 1, f72a5ddd GPU 0, from 03:13 UTC | `runs/*_V2R_v2r_r1_s{0,1,2}` |
| D_P | `retryvla.search.rec_prevent`, 2 shards x 28 train source episodes | ea7f9ee3, from 03:10 UTC | `data/recovery/v2r_dp/sh{0,1}` |
| R0 event re-evals | nominal 10 x 50 with saved actions for R0 s0-2 and pi1-nominal, then E-v1s replay | f72a5ddd, from 03:11 UTC | `runs/*_V2R_v2r_r0_s*_nominal`, `runs/*_V2R_v2r_pi1nom_nominal` |

## 2. R1 search: critic swap

Same 256 instances, same pools and slot seeds as M5b. The control is the historical M5b run (bf16 nondeterminism
caveat).

| scorer | solved | hits / verification | verified trajectories | mean predicted value on picks |
|---|---|---|---|---|
| shipped C^pi v2 (M5b wm_diverse, behind pi1) | 0.547 | 0.171 | 423 | 0.735 |
| CRIT-V `base` (R1) | 0.570 | 0.178 | 436 | 0.837 |

- McNemar on solved instances: 25 solved only by R1 vs 19 only by M5b, p 0.45.
- Pre-registered "better ranking" rule (+5 pt, p < 0.10): **not met**.
- The base critic had imagined AUC 0.815 vs 0.764 offline, but that does not turn into more found recoveries. It is
  also more optimistic in absolute terms on its own picks: it predicts 0.84 where the realised hit rate is 0.18.
  This is the RQ3 pattern again: whichever critic does the selecting looks over-optimistic on its own picks.

## 3. R2 input: physical plausibility of the R1 bank

Thresholds: Spatial D0 = 432 lerobot/libero Spatial demos. Demo d3 median 0.098, demo ||Δa|| median 0.082.

| rule (pre-registered) | rejects (of 436) |
|---|---|
| bridge d3 > 2 x demo median (0.195) | 362 |
| seam > 3 x demo median ||Δa|| (0.246) | 331 |
| gripper flips in bridge > 1 | 50 |
| replay mismatch (open-loop replay from the rebuilt snapshot does not end in success) | 49 |
| object jump > 3 cm / step without finger contact | 24 |
| penetration > 1 cm | 7 |
| object off table | 2 |
| **kept (all rules)** | **5 trajectories / 4 instances** |

Deviation D-V1, decided before any R2 result: with 5 survivors, R2 would test nothing.
- **R2 keeps the hard physical rejects** (replay mismatch, penetration, off table, impulse, gripper flips > 1). That
  leaves 307 trajectories over 118 instances.
- **Jerk and seam become the SMOOTH-V smooth-first ranking**, not rejects. Selection: 208 trajectories, 118
  instances, bridge d3 median 0.374 vs 0.446 for the whole bank (−16 %).
- Reading:
  - The bridges are intrinsically jerky. The dominant source is gripper chatter and the noisy SmolVLA chunk samples
    at noise 1.0 / perturbations.
  - Demo-relative thresholds from the Goal campaign do not transfer.
  - The 11 % replay mismatch should be investigated (snapshot rebuild vs search-time snapshot) before any claim that
    rests on replayed geometry.

## 4. Visual checks (looked at before distillation)

R1 bank (8 stratified renders, 5 opened):

![R1 worst-jerk](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/visual/recover_v2r_r1/v2r_r1_s2_wm_diverse_ep19_sheet.png)
![R1 worst-seam](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/visual/recover_v2r_r1/v2r_r1_s7_wm_diverse_ep16_sheet.png)
![R1 per-task](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/visual/recover_v2r_r1/v2r_r1_s7_wm_diverse_ep42_sheet.png)

MP4s sit next to each sheet (same name, `.mp4`). Index:
`/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/visual/recover_v2r_r1/index.md`.

Observations on the R1 bank:
- The successes are genuine: the bowl is visibly placed on the plate at the end.
- The worst bridges are gripper chatter (11-14 open / close flips in 50 steps) with little end-effector motion.
- One handoff has a 1.79 action jump.
- The pi0 continuations themselves chatter (15-53 flips per trajectory).
- Same character as the M5b bank behind pi1, so the bank was distilled as R1.

R2 selection (hard filter + smooth-first; 8 renders, 4 opened):

![R2 per-task, regrasp primitive](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/visual/recover_v2r_r2h/v2r_r1_s3_wm_diverse_ep17_sheet.png)
![R2 worst-jerk](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/visual/recover_v2r_r2h/v2r_r1_s0_wm_diverse_ep16_sheet.png)

Index: `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/visual/recover_v2r_r2h/index.md`.

Observations on the R2 selection:
- Chatter is gone (<= 1 flip).
- The remaining bridges are either pi0-like reaches or clean regrasp primitives (lift about 6 cm, lower, close), then
  pi0 takes over.
- Worst-jerk bridges (d3 1.33) still appear, because instances that have nothing smoother are filled round-robin.

D_P bank: rendered by the chain when generation closes (`runs/_analysis/visual/recover_v2r_dp/`). The R3 SFT waits
45 min for the agent's look, then proceeds and writes `PROCEEDED_WITHOUT_LOOK` (logged deviation).

## 5. Reference (R0 = pi1, registered)

| arm | seeds | test RSR | per seed | official L1 (1 scene) | L2 (11) | L3 (16) | nominal |
|---|---|---|---|---|---|---|---|
| pi0 | 1 | 0.118 | - | 0.800 | 0.109 | 0.100 | 0.656 |
| pi1-nominal (init) | 1 | 0.138 | - | 0.700 | 0.091 | 0.119 | 0.796 |
| demo-only | 3 | 0.136 ± 0.015 | | | 0.106 | 0.106 | 0.815 |
| **R0 = pi1** | 3 | **0.418 ± 0.021** | 0.439 / 0.396 / 0.420 | 0.833 | 0.448 | 0.383 | 0.785 ± 0.021 |

Decision rules, from the pre-registration:
- **R1 vs R0**: ΔRSR >= +4 with CI > 0.
- **R2 vs R1**: ΔRSR >= +3, or nominal +3, or closed-loop jerk −15 % with RSR within −3.
- **R3 vs R2**: nominal +3, or E-v1s incidence −5 pt, or first-attempt +3, with RSR >= R2 − 3.
- **R4 vs best(R1-R3)**: ΔRSR +3, or nominal +3, or incidence −5, with the other metrics within −3 / −2.
- **Nominal guard**: mean >= 0.776.
- **MDE**: about 5 pt RSR.

## 6. Queue and ETAs (2 shells x 2 GPUs; each GPU runs 1 SFT + 1 sim job)

| item | ETA (UTC, 11-25) |
|---|---|
| R1 SFT x 3 done | ~04:15 |
| R1 RSR + nominal evals (6) | ~06:00 |
| R2 SFT x 3 (as SFT slots free) | ~05:15 |
| R2 evals | ~07:00 |
| D_P closed (measured ~5-12 min / source episode, 28 per shard) | ~06:30 |
| R3 SFT | from ~06:45 |
| R3 evals | ~09:30 |
| R4 (needs R1-R3 evals) SFT | ~10:30 |
| R4 evals | ~12:15 |
| Object held-out | not queued yet; after R4, if the winner is clear |

Each stage completion appends an auto table to RESULTS and NOTES and refreshes section 7 below, via the agent-box
watchdog `rv_v2r_watch` (it also commits). The watchdog kills a shell after 2 h idle and both shells at ALL_DONE.

## 7. Auto results (refreshed by the chain)

<!-- V2R_AUTO_BEGIN -->
(no stage complete yet)
<!-- V2R_AUTO_END -->

## 8. Open questions for the user

1. **Plausibility filter.** R2 now uses hard physical rejects plus smooth-first ranking (D-V1). Do you prefer a
   strict variant, which would need smoothed and re-verified bridges (SMOOTH-V `smooth_reverify`)? A strict filter
   at demo-relative thresholds leaves almost nothing.
2. **Replay mismatch (11 %).** Is it worth a quick investigation (search snapshot vs rebuilt snapshot) before R4's
   aux targets, which use the same replays?
3. **Object held-out.** Run only after R4, with the frozen winner, at 1 seed (preliminary) or 3 seeds (about 9 GPU-h)?
