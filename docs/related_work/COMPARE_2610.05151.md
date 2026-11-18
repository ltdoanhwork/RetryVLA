# RetryVLA vs arXiv 2610.05151 (Recursive Self-Improvement via Local Recovery Supervision)

Written 2026-11-17. Source: full arXiv HTML v1 (read in full). Code / data / checkpoints: **none released**
(arXiv abs + HTML have no code link; web search 2026-11-17 found no GitHub / HF / project page).

## 1. What the paper actually does (extracted precisely)

**Correction to our overlap review (WM_RECOVERY_OVERLAP A4):** acceptance is NOT "the frozen policy finishes the
task". It is a short-horizon handoff test: the frozen student runs from the teacher endpoint for at most
H_S = 60 actions, an observer checks every Delta = 10 actions for one of 1-2 LLM-written "early progress" conditions
(OR a completed-task alternative), then C = 10 more actions of confirmation; protected relations must hold
(Eq. 7). The observer is an LLM judging structured natural-language checks, not a sim success oracle. The paper says
so itself: "short-term handoff usability, rather than a guarantee of eventual full-task success".

| item | 2610.05151 |
|---|---|
| failure source | natural failures of the current student: audit M0 on 300 TRAIN init scenes of LIBERO-Goal (233 succ / 67 fail) |
| entry (where to start) | offline LLM auditor: coarse view (~1 frame/s + <= 8 motion / gripper change centres), then a dense window (+-0.5 s, ~96 frames), one extra inspection; Focus mode (earliest supported correctable state); 0-3 entries / event; may use privileged physical summaries offline |
| spec | LLM writes a frozen local spec Gamma_r = (local goal, 1-3 AND completion checks, relations to preserve; student: 1-2 OR progress conditions + completed-task alternative + confirmation) |
| proposal generator | fixed multimodal LLM (gpt-6.1-sol, medium reasoning) as a closed-loop tool teacher (client from "Agent as Policy" 2609.12541): state / image queries, native actions (1-40 steps per request), eef motion helpers with units / limits, gripper, complete, abort; can compute targets or write short programs; same RGB views + 8-d robot state as the student, no hidden object state / depth / contacts. No primitive library, no search, no sampling of the policy |
| budgets | <= 5 stages, <= 300 teacher actions, <= 7,200 s per attempt; per session <= 100 robot requests, <= 1,200 s. Extensions see the last 6 commands; retries changed from 3 to 1 per entry during development |
| verification | restore teacher endpoint (replay audit prefix + teacher actions, verify physics + controller state), frozen student test (60 + 10 actions, LLM observer); fail -> restore endpoint, LLM writes a new spec, teacher EXTENDS the path. Student actions are never labels |
| scorer / selection | none (one path per attempt); after acceptance an offline LLM evaluator (with privileged sim records, BDDL predicates) grades every teacher action q in {0, .25, .5, .75, 1} |
| data | 27 accepted paths, 3,974 actions -> windows of q >= 0.75, L_min 5: 2,095 actions, 114 runs, 1,639 windows (5-50, mean 17.5), 28,649 valid targets = 4.14 % of anchors / 1.49 % of targets next to 37,907 original windows |
| training | pi0 (openpi, public base) M0 = 3,000 of 5,000 updates (val SR 24/73/77/75/76 at 1k-5k, chosen ON the reported val scenes); post-train 2,000 updates from M0, AdamW restart, bs 48, seed 7, warmup 200 to 2.5e-5, cosine to 2.5e-6, betas (0.9, 0.95), eps 1e-8, wd 1e-10, clip 1, no EMA; openpi low-mem config (LoRA + other trainable modules, ~468 M of 3.288 B params); loss = quality-weighted, validity-masked flow matching (mask also on noisy action tokens / attention); uniform shuffle, no recovery oversampling |
| nominal handling | original demos always in the mix (96 % of anchors); control "Original" = same 2,000 updates on original data only |
| benchmark | custom LIBERO-Goal split: 10 tasks x 50 demos, seed-7 split 30 / 10 / 10 scenes per task (scene = task + XML + initial physical state); 7-d actions 20 Hz; agentview + wrist 128x128 -> pad + bilinear to 224; 8-d state; obs index i paired with action i+1 (replay-checked) |
| eval | 100 VALIDATION scenes (10 / task), restore XML + init state, policy seed 7, <= 300 actions, chunk 50, execute 5 then replan, native success check. Reserved test never evaluated. NO separate recovery metric |
| numbers | M0 77, Original 78, Recovery 88 / 100; paired 18 fixed / 8 broken, McNemar p 0.0755; per task: 5 up, 4 same, 1 down. BC-RNN robomimic Can: 91 -> 102 (orig) -> 112 / 130 with 26 segments (2,547 labels; that teacher saw sim geometry) |
| compute | not reported (3 x RTX A6000 for training; LLM / collection cost not given) |
| caveats they state | 1 seed; val also used to pick M0; collection protocol changed during collection (yield not a fixed-protocol rate); single round only |

## 2. Side by side

| axis | 2610.05151 | RetryVLA (current pi1 pipeline) |
|---|---|---|
| policy | pi0 3.3 B (LoRA-style, 468 M trainable) | SmolVLA 450 M (`HuggingFaceVLA/smolvla_libero`) |
| failure source | natural failures of M0 on train init scenes (67) | LIBERO-Recover failure states (Spatial train scenes x 5 close + 5 open gripper x +-2 cm noise), pi0 fails them |
| entry | LLM auditor picks 0-3 pre-failure states | given by the benchmark (the failure snapshot itself) |
| proposals | ONE closed-loop LLM tool-teacher path per attempt, extended on failure | 256 open-loop 50-step bridges: SmolVLA chunk samples + perturbations + primitive library, + 2 latent-CEM rounds |
| scorer / selection | none; LLM spec + LLM observer | WM v1 (DINOv2 latent) imagination + C^pi(s, g) (learned from real continuations of pi), diverse top-8/16 |
| verification | frozen student 60 + 10 actions, LLM-judged progress (proxy) | bit-exact snapshot, bridge, frozen pi continues to the BDDL goal (300 steps); ground-truth success only |
| labels | teacher actions only, LLM-graded, quality-weighted windows | bridge + pi continuation (segment 0 / 1), unweighted |
| data size | 1,639 windows (1.5 % of targets) | 208 trajectories (round robin over instances) |
| training | 2,000 updates from M0, uniform mix (no oversampling) | 6,000 steps from pi1-nominal, recovery ratio 0.5, bs 32, lr 5e-5 |
| nominal | measured only (Original control) | constraint NSR >= ref - 2 pt (pi1 0.785 vs 0.796, 3 seeds) |
| benchmark / metric | 100 nominal LIBERO-Goal val scenes, 1 trial each | test RSR (56 held-out failure scenes x 10 trials) + nominal Spatial 500 |
| statistics | 1 seed, McNemar p 0.076 | 3 SFT seeds, scene x seed bootstrap |
| cost driver | LLM wall-clock (up to 2 h / attempt), few sim steps | sim verifications (~2.5k env steps / instance) + WM / critic training |

**Equivalent:** restore-from-sim-state branching; the frozen policy's own continuation as the acceptance signal; distill
into the same policy with original data kept; "Original continuation at the same number of updates" as the control
(= our pi1-nominal / demo-only control).

**Different:** where proposals come from (LLM tool teacher vs sampled pool + WM / C^pi ranking); acceptance strength
(short-horizon LLM-judged progress vs full-task ground-truth success); labels (teacher only vs bridge + continuation);
benchmark (nominal Goal scenes vs held-out failure states, i.e. they never measure recovery directly).

**Our claims it undermines:** "accept a correction only if the deployed policy's continuation succeeds" as a novelty
(theirs is a weaker, proxy version but the idea is the same); "self-improvement by distilling verified recoveries into
the same VLA" (they also run the outer loop, though only one round). **Left intact:** WM + C^pi selection over a large
pool and its sim-efficiency; ground-truth full-task verification; the optimism-under-search measurements; recovery
measured on held-out failure states with nominal preserved (they only report nominal-start success); 3-seed statistics.

## 3. PRIORITY (user 2026-11-17 10:14 UTC): RetryVLA on THEIR protocol

Goal: numbers directly comparable to their Table 1 (M0 77, Original 78, Recovery 88 / 100).

### 3.1 What is released / reconstructable
- Code, split file, checkpoints, recovery data: **none**. Everything below is rebuilt from the paper text.
- Their eval matches openpi's LIBERO example almost exactly (libero_goal max 300 steps, replan every 5 of the 50-step
  chunk, pad + resize to 224, seed 7), adapted to restore their own scene states. We copy that.
- Data: original LIBERO-Goal hdf5 (`yifengzhu-hf/LIBERO-datasets`, downloaded): 10 tasks x 50 demos, 128x128
  agentview + wrist, `init_state` per demo = their "scene". 300 train demos x ~126 steps ~= their 37,907 windows.
- Base model: `lerobot/pi0_base` (PyTorch port of the openpi pi0 base, downloaded). lerobot 0.5.1 has `use_peft`.

### 3.2 Deviations we cannot avoid (logged before any result)
- D1 split: their seed-7 assignment procedure is unknown. Ours: per task `np.random.default_rng(7).permutation(50)`,
  first 30 train, next 10 val, last 10 test (`data/splits/goal_rsi_v1.json`). Same sizes, different scenes.
- D2 framework: lerobot PyTorch pi0 instead of openpi JAX; LoRA via peft approximating openpi `*_low_mem` (LoRA r16
  on Gemma-2B, r32 on the 300M expert, SigLIP + projections trainable, ~470 M trainable). Same optimizer numbers.
- D3 eval sim: our `LiberoSim` (robosuite / MuJoCo from hf-libero), render at 128 to match the training images.
  We keep openpi's 10 no-op settle steps after the restore (not stated in the paper).
- D4 teacher: not needed for the priority track (we run OUR method on their protocol and compare with their
  reported numbers; their method is not re-run because the LLM teacher is not available, see section 4).

### 3.2b As implemented (2026-11-18, before any G0 number)
- Split `data/splits/goal_rsi_v1.json` (`scripts/goal_rsi_build.py`): per task `default_rng(7).permutation(50)` ->
  30 / 10 / 10 demo ids; a scene = task + that demo's `init_state`.
- D0 = the ORIGINAL hdf5 records of the 300 train demos, aligned as in the paper (post-action obs i paired with action
  i + 1, last frame dropped), images flipped H + W to our sim convention (flipped hdf5 frame vs our render at the same
  state: MAE 5-7 / 255; unflipped 73), state = `ee_states` (pos + axis-angle) + 2 gripper qpos (= our sim state to
  < 2 mm). 38,240 windows (theirs 37,907). We do NOT replay the demos in our sim: replay drifts (82 % of replays
  succeed), which would mislabel 18 % of the data. `data/goal_rsi/d0_train` (LeRobotDataset, 128 px, fps 20).
- pi0: `lerobot/pi0_base` + peft LoRA r16 (LLM attention + MLP) and r32 (expert), SigLIP + projector + state / action /
  time projections trainable: **451.5 M trainable** of 4.06 B in the lerobot port (theirs ~468 M of 3.288 B).
  `retryvla/policy/pi0_lora.py`, trainer `retryvla/train/pi0_sft.py` (DDP, bf16 autocast, gradient checkpointing).
  M0 schedule = openpi defaults (warmup 1,000, cosine 2.5e-5 -> 2.5e-6 over 30k steps; the paper gives no M0
  schedule). Continuation: warmup 200, cosine to 2.5e-6 at 2,000 (paper). Continuation re-injects a fresh LoRA on
  the merged M0 weights (functionally identical start, AdamW restarted as theirs).
- Eval `retryvla/eval/eval_goal.py`: restore the init state, 10 no-op steps, <= 300 actions, chunk 50 / execute 5,
  policy noise seeded from 7, 128 px renders.
- Chain `scripts/goal_g0_chain.py` (tmux `rv_goal_g0`, local 8b889199, train DDP on GPUs 1-3 with 16 / GPU, eval
  co-located on GPU 2): M0 5k + val eval of every 1k ckpt -> selection -> Original 2k -> val -> `runs/goal_rsi/g0.json`.

### 3.3 What we must build
1. `goal_rsi_v1` split + converter hdf5 -> LeRobotDataset (300 train demos; image flip = our sim convention; state
   = eef pos + axis-angle + 2 gripper qpos; their alignment obs[i] -> action[i+1], checked by replay).
2. `Pi0Wrapper` with the same `predict_chunk` interface as `SmolVLAWrapper` (so `run_jobs`, search, verification
   work unchanged) + `eval_goal` (their protocol on any scene list).
3. pi0 SFT entry point (LoRA, bs 48, their schedules): M0 = 5,000 updates from pi0_base, checkpoints every 1,000;
   continuation = 2,000 updates from M0 (Original / Recovery mixes).
4. RetryVLA on Goal: audit M0 on the 300 train scenes (1 rollout, snapshots every 20 steps) -> failed rollouts;
   C^pi for M0 (real continuations of M0 from train-scene states, our critic code); WM v1 retrained on Goal
   (DINOv2 latent, our wm code; data = train demos + M0 rollouts + random bridges); entries = snapshots of failed
   rollouts (rule below); search = our pool (pi0 chunk samples + perturbations + primitives) + WM / C^pi + 2 CEM
   rounds, diverse top-16, sim verification = bridge + M0 continuation to the BDDL goal; bank = bridge + continuation.
5. Distill with THEIR recipe (2,000 updates from M0, D0 + recovery windows shuffled uniformly, no oversampling) as the
   primary comparison; our recipe (recovery ratio 0.5) as secondary.

### 3.4 Compute (shell 8b889199, GPUs 1-3 only; measured numbers go to NOTES)
- pi0 3.3 B bf16 inference ~8 GB; LoRA training at bs 48 with gradient checkpointing ~40-60 GB, one A100.
- Estimates (to be replaced by measurements): train ~1 s / update -> M0 5k ~1.5 h, each 2k continuation ~35 min;
  eval 100 episodes ~10-15 min with 8 sim workers; audit 300 rollouts ~30 min; C^pi data (~3k continuations) ~1.5 h;
  WM training ~3-5 h; search on ~70 failures x 2-3 entries ~3-4 h. Total ~1.5-2 GPU-days incl. 3 seeds.
- GPUs 1-3 are held by ACR-N F2 (`acrn_f2_chain`, ETA ~13:30-14:00 UTC); node CPU load ~46 / 32. Only CPU-light prep
  and short smoke tests run before that; heavy jobs are queued behind it (claims in `shells.json`).
- Cheaper first step: SmolVLA on their protocol. `smolvla_libero` saw ALL Goal demos (incl. our val scenes), so it is
  only an interface smoke test, never a baseline. A clean SmolVLA track = `smolvla_base` + the same 300 train demos.

### 3.5 Decision rules (pre-registered in experiments/PLAN.md, 2026-11-17)
- **G0 reproduction gate** (before any RetryVLA number counts): our M0 (best of 1k-5k on val, same selection as
  theirs) in [67, 87] / 100 (77 +- 10 ~ binomial 95 % half-width + split slack) AND Original (M0 + 2k on D0) in
  [68, 88]. Fail -> interface debug (image flip, state, alignment, normalisation), then at most one recipe change
  (train-expert-only) logged as a deviation; no RetryVLA claim on their protocol until G0 passes.
- **Main comparison** (3 post-training seeds; paired on the 100 val scenes): Delta = Recovery_RetryVLA - Original.
  "matches theirs" if mean Delta >= +5 and the scene x seed bootstrap CI excludes 0; "beats theirs" if mean
  Delta >= +10 (their gain) with CI lower bound > +5; "below" otherwise. Cross-paper numbers (88 vs ours) are
  reported with D1-D3 attached, never as a paired test.
- Equal-sim-budget no-WM arm (random pool + sim verification, same verifications) decides whether WM / C^pi matter
  on their protocol: McNemar on solved entries, p < 0.10.
- Power: at n = 100 paired scenes a +10 pt gain with ~25 discordant pairs gives p ~0.08 (theirs 0.076); hence 3 seeds.

## 4. Secondary: their method on our stack (only if cheap)
No LLM API in this environment (no OpenAI / Anthropic / Gemini keys). Local open models exist (Qwen3-VL-8B-Instruct,
Qwen3-8B, Qwen3.8-27B in the HF cache; transformers 5.3 supports Qwen3-VL; the local vLLM env is 0.7.2, too old
for Qwen3-VL). A faithful re-run would need a Qwen3-VL-8B tool teacher (far weaker than gpt-6.1-sol) plus an LLM
observer; closest cheaper substitute = our primitive library executed closed-loop as the "teacher" with their
60 + 10 handoff test. Arms (a) theirs, (b) ours pi1, (c) ours without WM at equal sim, (d) theirs + C^pi ranking are
designed but deferred until the priority track has passed G0.

## 5. Status
See experiments/PLAN.md section "2610.05151 comparison" and experiments/NOTES.md for live status / ETA.
