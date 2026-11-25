---
name: retryvla-visual-check
description: >-
  Mandatory visual check of RetryVLA generated data and method comparisons: look at generated data with your own
  eyes, not just numbers. Renders full-frame-rate MP4 (every step, 20 Hz, H.264) + preview GIF + contact-sheet PNG (caption bar: task, entry, segment bridge /
  continuation / teacher / handoff, gripper, success; EE-xyz + gripper action strip) from RecoveryWriter banks, LRS-G
  teacher records, CRIT-V rich records or any action array, replayed bit-exactly in sim (CPU) when frames are not
  stored, and side-by-side videos on the same entry. Use after EVERY data-generation stage (recovery banks, D_P / D_F,
  smoothed banks, LRS teacher windows) and BEFORE distillation, when comparing methods (ours vs LRS, random vs WM,
  raw vs smooth), when reporting results, and when the user says visual check, xem bằng mắt, so sánh bằng mắt,
  video, demo, data trông thế nào, jerk, giật, smooth, mượt.
---

# RetryVLA visual check

User rule (verbatim intent): **look at generated data with your own eyes, not just numbers; always check visually.**
Metrics (SR, jerk, C^pi) hide gripper chatter, detours, contacts through objects, frozen arms and handoff jumps.

Script: `.cursor/skills/retryvla-visual-check/scripts/render_traj.py` (run from the repo root after
`source env/activate.sh`). CPU only by default (MuJoCo OSMesa / llvmpipe, ~1-2 s per sim step at 256 px incl.
render; physics is identical to EGL). Never put render jobs on GPUs of shells `953bb534` / `8f473147`.

## When (mandatory)

| Trigger | What to render | Gate |
|---|---|---|
| Any data-generation stage finishes (recovery bank, D_P / D_F, smoothed bank, LRS teacher windows, mixes) | `--sample <bank> --n 8` per bank (stratified, below) | BEFORE distillation / SFT launch |
| Comparing methods (ours vs LRS, random vs WM, raw vs smooth, pi0 vs pi1) | `--side-by-side A B` on the SAME entry, same start frame, 2-3 entries | before writing the comparison |
| Reporting a result in chat / RESULTS / slides | embed the PNG / GIF, link the MP4 | every report |

Stratified sample (what `--sample` picks, 6-8 per bank): 2 worst-jerk, 1 worst boundary (handoff) jump, 1 most
gripper flips (if > 1), 1 per task, random fill. For banks not in RecoveryWriter format, pick the same strata by hand.

## Workflow

```
Visual check:
- [ ] 1. Render: python .cursor/skills/retryvla-visual-check/scripts/render_traj.py --sample <bank_root> --n 8 \
         --out runs/_analysis/visual_check/<stage>/<bank>   (add --replay on 1-2 episodes to check determinism)
- [ ] 2. LOOK: Read every *_sheet.png (Read tool on the PNG), and at least one GIF frame for anything suspicious
- [ ] 3. Write 2-3 sentences of qualitative observations per bank (template below), in the chat and in
         experiments/NOTES.md next to the stage's entry
- [ ] 4. If anything looks wrong: fix / filter the bank before distillation, re-render
- [ ] 5. Report: markdown images with absolute paths + MP4 links
```

Never write "bank looks fine" without having opened the sheets in this turn. Looking means: Read the PNG.

### Rendering rule: never judge smoothness from a subsampled GIF

The user once saw every demo as "giật giật" (jerky) because the GIFs kept every 4th to 6th step at 300-400 ms per
frame with 48-64 colours, i.e. about 2.5 fps playback of 20 Hz data. That choppiness is a RENDERING artefact and says
nothing about the motion. Rules:
- **MP4 is the artefact for smoothness**: every sim step = one frame, played at the native control rate (LIBERO
  20 Hz, or `meta/info.json` fps of the bank), H.264 yuv420p, CRF ~23, 256 px per panel. `render_traj.py` does this
  by default (`write_mp4`), as does `scripts/demos/render_goal_demos.py`.
- **GIFs are previews only**: every frame, or every 2nd frame for clips > 150 steps, with the matching real-time duration
  (50 / 100 ms at 20 Hz), >= 128 colours (`write_gif` asserts stride <= 2). Never stride more than 2. Never slow it down.
- **Do frame-by-frame checks on consecutive frames**: `scripts/demos/render_fullfps.py` writes `*_consec_sheet.png`
  (12 consecutive steps around the clip's max |d3 a|, plus 12 at the window start). Evenly spaced tiles cannot show jerk.
- **Report jerk numbers next to every clip**: mean ||d3 a[:6]|| (also d2), gripper flips per 100 steps, and the EE
  jerk ||d3 ee_xyz|| in mm (what the camera actually sees), always next to a human-demo and an M0-nominal reference
  clip of the same task (`render_fullfps.py` panels H / N). Then you can tell real data jerk (bridge, teacher, chatter)
  from playback artefacts.

### What to look for

- **Jerk / giật**: zig-zag EE curves in the strip, alternating action directions, saw-tooth z. Compare with a demo.
- **Gripper chatter**: many open / close flips in the black gripper trace (a regrasp primitive has exactly 1).
- **Weird contacts**: gripper through the table / object, object pushed off the table, drawer hit sideways, object
  dragged instead of lifted, object teleporting between tiles.
- **Unnatural detours**: big loops away from the target, long idle (flat EE curves, arm frozen), oscillation in place.
- **Handoff jumps**: discontinuity at the dashed segment line (bridge -> continuation, teacher -> handoff); look at
  `handoff|da|` in the header and the frames on both sides of the line.
- **Outcome sanity**: SUCCESS banner but the task visibly not done (or the reverse) means a checker / label issue.

### Observation template (2-3 sentences)

> `<bank>` (n=8, 2 worst-jerk, 1 worst-jump, 1 flips, 4 per-task): bridges mostly <behaviour>; <k>/8 show
> <problem> (ep <i>: <frame / time>); handoff <smooth / jump of X at t=50>. Verdict: OK to distill / filter <rule> /
> regenerate.

## Script usage

```bash
R=.cursor/skills/retryvla-visual-check/scripts/render_traj.py
# one bank episode (stored PNG frames; --replay = re-simulate from the snapshot + state check)
python $R bank:data/recovery/m5b/s0/p0/wm_diverse#12 --replay --out runs/_analysis/visual_check/x
# stratified sample of a bank (+ index.md with the sheets)
python $R --sample data/recovery/goal_search/s0/p0/wm_diverse --n 8 --out runs/_analysis/visual_check/goal_search
# side by side, same entry: ours (bank) vs LRS teacher + handoff
python $R --side-by-side bank:data/recovery/goal_search/s0/p0/wm_diverse#4 \
  lrs:data/recovery/lrs_goal#open_the_middle_drawer_of_the_cabinet__demo_9__s7__t075 --out runs/_analysis/visual_check/x
# raw vs smoothed / any action array: acts:<file.npy | file.npz:key>@<instance_id>
python $R --side-by-side acts:raw.npz:a@<iid> acts:smooth.npz:a@<iid> --out ...
# CRIT-V rich record (no actions stored -> bridge k from data/wm/<same dir>/<iid>.pt, replayed)
python $R rich:data/critic3d/goal_verif/<iid>.pt#3 --out ...
# M0 alone from a LIBERO-Goal entry (audit actions[t:])
python $R audit:<goal_entry_id> --out ...
```

Sources:
- `bank:<root>#<ep>`: RecoveryWriter LeRobot v3 dataset (`data/`, `meta/`, `retryvla/trajectories.jsonl`): every
  `data/recovery/<run>/s*/p0/<arm>`, D_P / D_F `bank0`, smoothed banks, window datasets. Segment 0 = bridge, 1 =
  continuation.
- `lrs:<root>#<iid>`: LRS-G records `a*/l*/rows.jsonl` + `actions/<iid>.npz` (`path` + `student_<k>`), segments
  teacher st<k> / handoff (judge window `n_test`) / continuation; replayed.
- `rich:<pt>#<k>`, `audit:<iid>`, `acts:...@<iid>` as above.
- Instance ids: LIBERO-Goal entry ids (`<task>__demo_<d>__s<seed>__t075 | __on295 | __pre280`, looked up in
  `data/goal_rsi/*/entries*.jsonl`, restored exactly like `goal_instances`) and LIBERO-Recover ids
  (`<scene_tag>__close<k> | __open<k>`, restored like `compare_arms` with eval.seed 7, noise 0.02, settle 20).

Options: `--res` panel size (<= 384), `--tiles` contact-sheet frames (8), `--no_mp4`, `--no_gif`, `--fps` (20; banks use
their own fps), `--gpu` (EGL, own GPU only), `--sim_res` camera size for replays. Outputs per trajectory: `<name>.mp4`
(every step at the control rate), `<name>.gif` (preview, <= 224 px / panel, stride <= 2, real-time),
`<name>_sheet.png`, `<name>.json` (jerk = mean ||Δ²a||, jerk3 = mean ||Δ³a||, max ||Δa||, gripper flips and flips per
100 steps, boundary jumps, replay `replay_state_maxdiff`, logged vs replayed success step).

Full-frame-rate reference set for LIBERO-Goal (human demo | M0 nominal | LRS v2 teacher + handoff | WM bridge + M0, on the
episode clock, jerk numbers in the captions):
`python -m scripts.demos.render_fullfps --entry <goal entry id>` -> `runs/_analysis/visual/fullfps/<entry>/`
(`quad.mp4`, `sbs_L_vs_W.mp4`, single clips, `*_consec_sheet.png`, `stats.json`). Side-by-side prints the start-frame
difference: > ~15/255
means the two panels do not start from the same state (not a paired comparison).

Checks the script does for you: bank `--replay` must give `replay_state_maxdiff` 0.0 and the logged success step
(verified on Spatial m5b and Goal goal_search); if not, the snapshot / protocol differs, stop and investigate.

## Reporting format (chat answers)

Embed images with absolute paths and link videos:

```markdown
![goal_search wm_diverse ep4](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/visual_check/x/goal_search_s0_wm_diverse_ep4_sheet.png)
![side by side](/abs/path/sbs_....gif) · [MP4](/abs/path/sbs_....mp4)
```

Put the 2-3 observation sentences right under the images. Keep committed examples small (sheet PNG, preview GIF, and
MP4s < 3 MB under `docs/related_work/figs/demos/fullfps/`); everything else stays under `runs/`.

## Examples (committed, `examples/`)

- `examples/recover_m5b_wm_diverse_ep12_sheet.png` / `.gif`: LIBERO-Recover, m5b wm_diverse bridge + pi0
  continuation, replay bit-exact (state diff 0.0, success @ 83 as logged). Observation: the 50-step bridge moves
  the EE only a few cm, while the gripper command chatters (11 flips in total, 3 open/close pulses inside the bridge,
  more right after the handoff). The roughness is in the gripper channel, not in the arm path.
- `examples/goal_wm_vs_lrs_sbs_sheet.png` / `.gif`: LIBERO-Goal, same entry `open_the_middle_drawer..demo_9__s7__t075`,
  WM bridge + M0 (success @ 182) vs LRS teacher + handoff (judge ACCEPT, sim FAIL). Observation: the WM bridge is
  almost static (EE flat for 50 steps, so M0 does all the work afterwards: a "no-op bridge"). The LRS teacher keeps
  the gripper closed from step ~20 to ~230 without opening the drawer, and there is a 1.84 action jump at the
  teacher -> handoff boundary. The judge accepted it anyway.
- Full-fps reference set (`docs/related_work/figs/demos/fullfps/`, `jerk_table.csv`): human demos |d3a| 0.06-0.09,
  EE jerk 0.4-0.7 mm. M0 nominal is already 2-2.5x human (|d3a| 0.16-0.21) because the Goal M0 (pi0, execute 5 / replan) re-plans every 5 steps: |Δa|
  at chunk boundaries is 3-5x the within-chunk value (autocorrelation 0.6-0.7 at lag 5, also visible in EE accel), a
  real ~4 Hz shudder. Use M0 nominal, not the human demo, as the bar for "M0-like" motion.

## Related skills

`retryvla-run-lessons` (bridge jerk numbers, distillation pitfalls), `retryvla-experiment`, `retryvla-idea-loop`
(run this before distillation and before any comparison verdict), `retryvla-ops` (env, rendering).
