// RetryVLA progress deck, part 1: setup, data, M0, M1 (oracle search)
const { HEX, card, text, bullets, img, badge, stat, caption } = require("./lib");
const V = require("./videos");

module.exports = function part1(pres, C) {
  // ---------- 1. Title ----------
  pres.addSection({ title: "Intro" });
  let s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Intro" });
  s.addText("RetryVLA", { placeholder: "title" });
  s.addText("World-model-guided synthetic recovery learning for VLAs\nProgress M0–M5 · LIBERO-Recover Spatial · Oct 2026",
    { placeholder: "body" });
  s.addNotes("Progress update. Question: can a world model find recoveries from real VLA failure states, and can those be distilled into the VLA so it recovers without search at test time?");

  // ---------- 2. Question + pipeline ----------
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Intro" });
  s.addText("The question we are answering", { placeholder: "title" });
  card(s, C, 0.6, 1.35, 12.1, 1.15, HEX.amberTint, "question");
  text(s, "Can a world model discover action sequences that recover from actual VLA failure states, and can those imagined recoveries be distilled into the VLA so it recovers without search at inference?",
    { x: 0.9, y: 1.5, w: 11.5, h: 0.9, fontSize: 17, italic: true, color: HEX.ink, fontFace: "Times New Roman" });
  const steps = [
    ["Failure states", "LIBERO-Recover scenes where SmolVLA failed"],
    ["Search", "propose 50-step recovery segments, score them"],
    ["Verify", "run best candidates in the simulator + SmolVLA continuation"],
    ["Distill", "fine-tune SmolVLA on verified recoveries"],
    ["Evaluate", "recovery success on held-out failures, no search"],
  ];
  steps.forEach(([h, d], i) => {
    const x = 0.6 + i * 2.45;
    card(s, C, x, 3.0, 2.2, 2.3, i === 1 ? HEX.ink : HEX.tint, "step" + i);
    badge(s, C, i + 1, x + 0.2, 3.2, i === 1 ? HEX.amber : HEX.ink2);
    text(s, h, { x: x + 0.2, y: 3.8, w: 1.85, h: 0.45, fontSize: 18, bold: true, color: i === 1 ? HEX.white : HEX.ink });
    text(s, d, { x: x + 0.2, y: 4.3, w: 1.85, h: 0.95, fontSize: 13, color: i === 1 ? "DDE3E8" : HEX.ink2 });
    if (i < 4) s.addShape("rightArrow", { x: x + 2.22, y: 4.0, w: 0.2, h: 0.3, fill: { color: HEX.line }, line: { color: HEX.line } });
  });
  text(s, "Search is where the world model enters: it replaces the simulator as the thing that says which candidates are good.",
    { x: 0.6, y: 5.75, w: 12.1, h: 0.5, fontSize: 15, color: HEX.ink2 });
  s.addNotes("Stage 1-6 from the research plan. M1 uses the simulator as the scorer (oracle). M2/M3 replace it with a latent world model + critic.");

  // ---------- 2b. Pipeline overview ----------
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Intro" });
  s.addText("Pipeline overview", { placeholder: "title" });
  img(s, "pipeline_overview.png", 0.6, 1.2, 12.1, 5.75, { alt: "RetryVLA pipeline overview diagram" });
  s.addNotes("1 failure states, 2 search (simulator scorer in M1, world model + critic in M3), 3 verify in simulator, 4 recovery dataset, 5 distill into SmolVLA, 6 evaluate on held-out failures without search. Dashed: WM training (M2), critic self-improvement (M5), iterative loop (M6).");

  // ---------- 3. Setup ----------
  pres.addSection({ title: "Setup" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Setup" });
  s.addText("Setting: benchmark, policy, compute", { placeholder: "title" });
  const cols = [
    ["Benchmark", HEX.blue, ["LIBERO-Recover, Spatial suite", "168 failure scenes, 84 failed episodes, 9 tasks",
      "official protocol: 5 closed-gripper + 5 open-gripper trials, ±2 cm object noise, 520 steps",
      "metric: Recovery Success Rate (RSR)"]],
    ["Policy", HEX.amber, ["SmolVLA, checkpoint HuggingFaceVLA/smolvla_libero (450M)",
      "chunk 50 actions, replan every 10", "7-D OSC delta actions, gripper ±1",
      "pi0 = checkpoint as-is; pi1 = after recovery fine-tuning"]],
    ["Compute", HEX.teal, ["1 local A100 + idle A100s borrowed from other shells (avoid-list)",
      "MuJoCo rendered on GPU (user-local NVIDIA EGL), ~20 ms/step",
      "bit-exact sim snapshots for branching", "all runs tracked: config, git sha, metrics, registry"]],
  ];
  cols.forEach(([h, col, items], i) => {
    const x = 0.6 + i * 4.1;
    card(s, C, x, 1.4, 3.85, 4.9, HEX.tint, "col" + i);
    s.addShape("ellipse", { x: x + 0.25, y: 1.65, w: 0.32, h: 0.32, fill: { color: col }, line: { color: col } });
    text(s, h, { x: x + 0.7, y: 1.62, w: 3.0, h: 0.4, fontSize: 20, bold: true });
    bullets(s, items, { x: x + 0.3, y: 2.25, w: 3.35, h: 3.9, fontSize: 14, color: HEX.ink2 });
  });
  s.addNotes("Python 3.12, lerobot 0.5.1. Every eval uses the same seeds so arms see identical perturbations.");

  // ---------- 4. Failure states ----------
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Setup" });
  s.addText("What a LIBERO-Recover failure state looks like", { placeholder: "title" });
  img(s, "failure_grid.png", 0.6, 1.3, 6.4, 5.4, { align: "left", alt: "20 restored failure states" });
  card(s, C, 7.4, 1.3, 5.3, 5.4, HEX.tint, "explain");
  text(s, "Each scene = a .bddl task + the exact physics state recorded when SmolVLA was failing",
    { x: 7.7, y: 1.55, w: 4.8, h: 0.9, fontSize: 16, bold: true });
  bullets(s, [
    "bowl half-grasped, pushed off the plate, stuck in a drawer, arm in an odd pose",
    "the gripper is forced closed in half of the trials: an initial state SmolVLA never saw in training",
    "restore is bit-exact: state, controller, observables. The same actions always give the same pixels",
    "no L1-L4 difficulty labels released yet",
  ], { x: 7.7, y: 2.6, w: 4.8, h: 3.9, fontSize: 14, color: HEX.ink2 });
  caption(s, "20 restored failure states (agent camera), M0b check: 20/20 deterministic", 0.6, 6.72, 6.4);

  // ---------- 5. Split ----------
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Setup" });
  s.addText("Train / pilot / test split, grouped by failure episode", { placeholder: "title" });
  img(s, "split.png", 0.6, 1.3, 7.4, 4.9, { align: "left", alt: "scenes per task and split" });
  stat(s, "112", "train scenes (search, WM data, SFT)", 8.5, 1.5, 4.2, HEX.amber);
  stat(s, "60", "pilot scenes (subset of train, search targets)", 8.5, 2.95, 4.2, HEX.amber);
  stat(s, "56", "test scenes: never searched, never trained on", 8.5, 4.4, 4.2, HEX.ink);
  text(s, "step_0180 and step_0200 of one failed episode are near-duplicates, so they always stay in the same split. Code asserts test scenes never enter search or training.",
    { x: 0.6, y: 6.3, w: 12.1, h: 0.6, fontSize: 13, color: HEX.ink2 });

  // ---------- 6. M0 ----------
  pres.addSection({ title: "M0 baseline" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "M0 baseline" });
  s.addText("M0: SmolVLA is decent at nominal tasks, bad at recovery", { placeholder: "title" });
  stat(s, "0.66", "nominal LIBERO-Spatial success (500 ep)\nlerobot-eval cross-check: 0.68", 0.8, 1.6, 3.6, HEX.ink);
  stat(s, "0.12", "recovery success (RSR) on 56 held-out failure scenes, 560 ep", 0.8, 3.3, 3.6, HEX.red);
  stat(s, "5 / 160", "drawer-task failures recovered", 0.8, 5.0, 3.6, HEX.red);
  img(s, "strip_demo.png", 4.7, 1.3, 8.0, 3.0, { alt: "nominal LIBERO demo strip" });
  caption(s, "Training data SmolVLA learned from: a nominal LIBERO-Spatial demo (agent + wrist camera, action per step)", 4.7, 4.25, 8.0);
  card(s, C, 4.7, 4.75, 8.0, 1.9, HEX.tint, "m0notes");
  bullets(s, [
    "checkpoint is weaker than the ~0.90 reported for SmolVLA, but our engine matches lerobot-eval, so the interface is correct",
    "fine-tuning on the Spatial demos alone reaches 0.80 nominal. That checkpoint (pi1-nominal) becomes the control and the stronger base",
    "replanning every step (n_action_steps=1) gives +5 points at 10x the cost, so we keep 10",
  ], { x: 4.95, y: 4.9, w: 7.5, h: 1.7, fontSize: 13, color: HEX.ink2 });

  V.m0(pres, C);

  // ---------- 7. M1 how one candidate is tried ----------
  pres.addSection({ title: "M1 oracle search" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "M1 oracle search" });
  s.addText("M1: how one recovery candidate is tried", { placeholder: "title" });
  const phases = [
    ["Restore", "load the exact failure snapshot", HEX.ink2],
    ["Segment (50 steps)", "open-loop actions from search: SmolVLA samples, smooth perturbations, primitives", HEX.amber],
    ["SmolVLA continues", "frozen pi0 drives closed-loop up to 300 steps", HEX.blue],
    ["Judge", "simulator checks the BDDL goal: bowl on plate?", HEX.teal],
  ];
  phases.forEach(([h, d, col], i) => {
    const x = 0.6 + i * 3.1;
    card(s, C, x, 1.4, 2.85, 2.0, HEX.tint, "ph" + i);
    s.addShape("ellipse", { x: x + 0.22, y: 1.6, w: 0.32, h: 0.32, fill: { color: col }, line: { color: col } });
    text(s, h, { x: x + 0.65, y: 1.58, w: 2.1, h: 0.4, fontSize: 16, bold: true });
    text(s, d, { x: x + 0.22, y: 2.1, w: 2.45, h: 1.2, fontSize: 13, color: HEX.ink2 });
  });
  img(s, "strip_primitive.png", 0.6, 3.65, 12.1, 2.85, { alt: "primitive recovery then SmolVLA" });
  caption(s, "Orange = searched segment (here a hand-written 'open + lift' primitive). Blue = SmolVLA takes over and places the bowl. Competence-targeted: the segment does not solve the task, it returns the robot to a state SmolVLA can finish from.", 0.6, 6.55, 12.1);

  V.m1(pres, C);

  // ---------- 8. M1 search loop ----------
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "M1 oracle search" });
  s.addText("M1: search = propose, score in the simulator, refine", { placeholder: "title" });
  const loop = [["1", "Propose 64", "24 SmolVLA samples (different noise), 24 smooth perturbations, 16 primitives"],
    ["2", "Score", "simulate each: success × speed + small goal-progress shaping − jerk"],
    ["3", "Keep elites", "8 best candidates"],
    ["4", "Refine", "CEM: sample around elites, up to 4 rounds (≤256 sims / instance)"]];
  loop.forEach(([n, h, d], i) => {
    const y = 1.4 + i * 1.25;
    badge(s, C, n, 0.6, y + 0.05, HEX.amber);
    text(s, h, { x: 1.2, y, w: 4.6, h: 0.4, fontSize: 17, bold: true });
    text(s, d, { x: 1.2, y: y + 0.42, w: 4.6, h: 0.75, fontSize: 13, color: HEX.ink2 });
  });
  img(s, "m1_search_overview.png", 6.0, 1.3, 6.7, 2.4, { alt: "M1 search overview" });
  card(s, C, 6.0, 3.95, 6.7, 2.75, HEX.redTint, "lesson");
  text(s, "What we learned", { x: 6.3, y: 4.1, w: 6.2, h: 0.4, fontSize: 17, bold: true, color: HEX.red });
  bullets(s, [
    "~70% of pilot failures have a recovery reachable in 50 steps",
    "the first CEM was broken: averaging different strategies into one Gaussian drops success 11% → 1%",
    "fix: refine around ONE elite at a time. On hard instances it then solves 68% vs 55% for random at equal budget",
    "~1/3 of recoveries are trivial (bowl already over plate: just open)",
  ], { x: 6.3, y: 4.55, w: 6.2, h: 2.1, fontSize: 13, color: HEX.ink2 });

  // ---------- 9. M1 data examples ----------
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "M1 oracle search" });
  s.addText("What the recovery training data looks like", { placeholder: "title" });
  img(s, "strip_trivial.png", 0.6, 1.3, 2.4, 2.9, { alt: "trivial recovery" });
  text(s, "Trivial", { x: 0.6, y: 4.25, w: 2.4, h: 0.35, fontSize: 15, bold: true });
  text(s, "2 steps: release the bowl already above the plate. Removing these HURT RSR (0.254 → 0.220)",
    { x: 0.6, y: 4.6, w: 2.4, h: 1.3, fontSize: 12, color: HEX.ink2 });
  img(s, "strip_vla_sample.png", 3.3, 1.3, 9.4, 2.25, { alt: "vla sample recovery" });
  caption(s, "One of 24 SmolVLA samples (different noise) happens to open and back off; continuation succeeds", 3.3, 3.55, 9.4);
  img(s, "strip_cem_bad.png", 3.3, 4.0, 9.4, 2.25, { alt: "noisy CEM recovery" });
  caption(s, "Old CEM on a hard instance: jerky, gripper chatter (3-4 flips / 50 steps vs 0.9 in demos). Searched segments are 4-6x jerkier than demos", 3.3, 6.25, 9.4);

  V.m1bad(pres, C);

  // ---------- 10. M1 results ----------
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "M1 oracle search" });
  s.addText("Distilling simulator-verified recoveries works", { placeholder: "title" });
  img(s, "m1_policy_results.png", 0.6, 1.3, 8.3, 2.8, { align: "left", alt: "M1 policy results" });
  const rows = [
    [{ text: "SFT arm", options: { bold: true, color: HEX.white, fill: { color: HEX.ink } } },
     { text: "recovery data", options: { bold: true, color: HEX.white, fill: { color: HEX.ink } } },
     { text: "test RSR", options: { bold: true, color: HEX.white, fill: { color: HEX.ink } } },
     { text: "nominal", options: { bold: true, color: HEX.white, fill: { color: HEX.ink } } }],
    ["pi0", "-", "0.118", "0.656"], ["pi1-nominal (control)", "demos only", "0.138", "0.796"],
    ["pi1-random", "241 random-search", "0.279", "0.698"], ["pi1-CEM", "241 CEM", "0.286", "0.690"],
    ["ratio 0.25", "241 random", "0.254", "0.726"], ["ratio 0.25, no trivial", "170 random", "0.220", "0.766"],
  ];
  s.addTable(rows, { x: 0.6, y: 4.3, w: 8.3, colW: [2.6, 2.3, 1.7, 1.7], fontSize: 12, color: HEX.ink,
    border: { type: "solid", color: HEX.line, pt: 0.75 }, rowH: 0.3, fill: { color: HEX.white } });
  card(s, C, 9.3, 1.3, 3.4, 5.4, HEX.tint, "take");
  stat(s, "2.4×", "test RSR vs pi0 (0.118 → 0.28), +14 pts over the fine-tune-only control", 9.55, 1.55, 3.0, HEX.teal);
  bullets(s, [
    "CEM ≈ random here: the old CEM's useful data all came from its random first round",
    "cost: nominal 0.80 → 0.70 at a 50% mix. Trade-off set by the ratio",
    "trivial recoveries are useful supervision",
  ], { x: 9.55, y: 3.35, w: 3.0, h: 3.2, fontSize: 13, color: HEX.ink2 });
};
