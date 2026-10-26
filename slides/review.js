// Progress review 2026-10-26: new main-flow slides (intro, statistics, Q1-Q3, nominal, Object, related work, plan)
const { HEX, card, text, bullets, img, badge, stat, caption } = require("./lib");

// header row for native tables
const H = (t) => ({ text: t, options: { bold: true, color: HEX.white, fill: { color: HEX.ink } } });
const B = (t, col) => ({ text: t, options: { bold: true, color: col || HEX.ink } });
function table(s, rows, o) {
  s.addTable(rows, Object.assign({ fontSize: 12, color: HEX.ink, fontFace: "Times New Roman",
    border: { type: "solid", color: HEX.line, pt: 0.75 }, fill: { color: HEX.white }, valign: "middle" }, o));
}
function content(pres, sec, title) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sec });
  s.addText(title, { placeholder: "title" });
  return s;
}
function divider(pres, sec, title, body) {
  pres.addSection({ title: sec });
  const s = pres.addSlide({ masterName: "SECTION", sectionTitle: sec });
  s.addText(title, { placeholder: "title" });
  if (body) s.addText(body, { placeholder: "body" });
  return s;
}

// ---------------------------------------------------------------- intro
function intro(pres, C) {
  pres.addSection({ title: "Intro" });
  let s = pres.addSlide({ masterName: "TITLE", sectionTitle: "Intro" });
  s.addText("RetryVLA: Policy-Competence-Guided Recovery Search and Distillation", { placeholder: "title" });
  s.addText("Progress review · 2026-10-26\nSmolVLA on LIBERO-Recover: Spatial (dev) and LIBERO-Object (held-out)", { placeholder: "body" });
  s.addNotes("Review meeting 2026-10-26. New framing from experiments/PLAN.md: two modules (policy-aware recovery planner, recovery-preserving distillation), evaluated independently. Numbers from experiments/RESULTS.md and runs/_analysis.");

  // executive summary
  s = content(pres, "Intro", "Executive summary");
  const rows = [
    ["Proven", HEX.teal, "Competence is what matters: searching for states from which the frozen policy itself can finish (C^pi) beats latent-goal and progress objectives, in search (32 / 2 failures; Object 66 / 8) and after SFT (0.417 vs 0.187, 3 seeds, +0.230 [+0.161, +0.298])."],
    ["Best numbers", HEX.amber, "RetryVLA pi1: test RSR 0.118 → 0.418 ± 0.021 (3 seeds), nominal 0.785 vs 0.796 for pi1-nominal: inside the 2 pt margin, no extra module at test time."],
    ["Proven", HEX.teal, "Distillation is needed: demo-only 0.127, online best-of-8 +0.027 (n.s.), imagination-only (RISE-inspired) 0.068-0.113, residual RL 0.295 but nominal 0.022. Held-out Object: +0.181 over demo-only."],
    ["Did not work", HEX.red, "The WM is mostly a filter over a proposal library: planning without it solves 0.398 vs 0.458 (M11). On Object, WM data is not better than random-search data at equal count (0.354 vs 0.362)."],
    ["Next", HEX.blue, "C^pi v3 from the Cpi-pool labels, more seeds (baselines, Object eps-test), matched-sim-budget distillation (P4), cost accounting (P5), LIBERO-Goal / LIBERO-10. Real-world adaptation: not decided."],
  ];
  rows.forEach(([lab, col, d], i) => {
    const y = 1.3 + i * 1.1;
    card(s, C, 0.6, y, 12.1, 0.98, i % 2 ? HEX.white : HEX.tint, "exec" + i);
    s.addShape("ellipse", { x: 0.85, y: y + 0.33, w: 0.32, h: 0.32, fill: { color: col }, line: { color: col } });
    text(s, lab, { x: 1.35, y: y + 0.12, w: 1.75, h: 0.74, fontSize: 16, bold: true, color: col, valign: "middle" });
    text(s, d, { x: 3.15, y: y + 0.1, w: 9.35, h: 0.8, fontSize: 13, color: HEX.ink2, valign: "middle" });
  });

  // agenda
  s = content(pres, "Intro", "Agenda");
  const ag = [["Problem and framing", "two modules, pre-registered nominal margin"], ["Timeline M0-M11", "what each milestone answered"],
    ["Statistics done right", "cluster bootstrap, 3 seeds"], ["Q1: why competence?", "search + SFT, critic variants"],
    ["Q2: why a world model?", "cost vs real dynamics, WM as planner"], ["Q3: why distill?", "RISE-inspired, residual RL, best-of-N, demo-only"],
    ["Nominal preservation", "recovery-ratio Pareto, L_preserve"], ["Generalization", "LIBERO-Object, frozen recipe"],
    ["Qualitative", "clickable videos: A / B / C explainers, end-to-end"], ["Related work", "comparison table + method figures"],
    ["Limitations", "honest negatives"], ["Plan and decisions", "P-list, scale plan, open questions"]];
  ag.forEach(([h, d], i) => {
    const col = Math.floor(i / 6), row = i % 6;
    const x = 0.6 + col * 6.15, y = 1.3 + row * 0.86;
    card(s, C, x, y, 5.95, 0.74, HEX.tint, "ag" + i);
    badge(s, C, i + 1, x + 0.18, y + 0.16, i >= 2 && i <= 7 ? HEX.amber : HEX.ink2);
    text(s, h, { x: x + 0.78, y: y + 0.1, w: 2.6, h: 0.54, fontSize: 15, bold: true, valign: "middle" });
    text(s, d, { x: x + 3.35, y: y + 0.1, w: 2.5, h: 0.54, fontSize: 12, color: HEX.ink2, valign: "middle" });
  });
  text(s, "Appendix: the full M0-M5 slides (oracle search, world model, critic, M5 search arms and SFT) are kept after the plan.",
    { x: 0.6, y: 6.55, w: 12.1, h: 0.35, fontSize: 13, italic: true, color: HEX.muted });

  // problem and framing
  s = content(pres, "Intro", "Problem and framing");
  card(s, C, 0.6, 1.25, 12.1, 1.05, HEX.amberTint, "q");
  text(s, "Given a pretrained VLA and a compatible world model, can we search for recovery actions that return the robot to states from which the ORIGINAL policy resumes successfully, and distill them into the policy without sacrificing nominal performance?",
    { x: 0.85, y: 1.35, w: 11.6, h: 0.85, fontSize: 15, italic: true, valign: "middle" });
  img(s, "pipeline_overview.png", 0.6, 2.5, 7.4, 4.05, { align: "left", alt: "RetryVLA pipeline overview" });
  caption(s, "Pipeline as built in M1-M5 (numbers in the figure are the M1-era ones; current pi1: RSR 0.418 ± 0.021)", 0.6, 6.6, 7.4);
  const mods = [["M-A  Policy-aware recovery planner", HEX.amber, HEX.amberTint,
    ["frozen pi + WM + failure state → 50-step bridge", "maximize C^pi(ŝ_H) − λ·cost − β·uncertainty", "C^pi trained only on real continuations of pi; verify every bridge in sim"]],
  ["M-B  Recovery-preserving distillation", HEX.teal, HEX.tealTint,
    ["verified bridges + nominal replay → pi'", "goal: RSR up, NSR(pi') ≥ NSR(pi) − 2 pt", "pre-registered, tested over ≥ 3 seeds"]]];
  mods.forEach(([h, col, tint, items], i) => {
    const y = 2.5 + i * 2.1;
    card(s, C, 8.3, y, 4.4, 1.95, tint, "mod" + i);
    text(s, h, { x: 8.5, y: y + 0.12, w: 4.0, h: 0.4, fontSize: 15, bold: true, color: col });
    bullets(s, items, { x: 8.5, y: y + 0.58, w: 4.05, h: 1.3, fontSize: 12, color: HEX.ink2 });
  });
  s.addNotes("Two contributions evaluated independently: the planner is VLA-architecture independent, distillation is the VLA upgrade. Novelty = recovery-specific objective + policy-grounded calibration + verification, not the WM architecture.");

  // timeline
  s = content(pres, "Intro", "Timeline: what each milestone answered");
  const tl = [[H("milestone"), H("question"), H("answer"), H("key number")],
    ["M0", "can pi0 recover?", "almost never", "RSR 0.118, nominal 0.656"],
    ["M1", "distill oracle-search recoveries?", "yes, ~2× RSR", "0.286 (CEM) / 0.279 (random)"],
    ["M2-M3", "latent WM + critic; better than random?", "yes, on 53 hard failures", "solved 0.36 vs 0.17"],
    ["M5 / M5b-c", "WM data vs random data; search arms", "WM diverse wins; round-robin = pi1", "solved 0.547 vs 0.363; pi1 0.439 (s0)"],
    ["M6", "loop pi1 → pi2", "no gain beyond more training", "0.463 vs control 0.480"],
    ["M7 / M8", "weak tasks (stove, cookie box)", "no transfer / data skew", "M8 0.457 vs 0.504"],
    ["M9", "why competence, WM, distill; 3 seeds", "competence ≫ goal; WM ≈ real prefix", "pi1 0.418 ± 0.021; 0.417 vs 0.187"],
    ["M10", "LIBERO-Object, frozen recipe", "distill + competence replicate", "comp 0.354 vs demo-only 0.173"],
    ["M11", "WM plans without the library?", "not yet", "0.398 vs 0.458 solved"],
    ["B2-B4 (P2)", "policy-grounding or any value?", "policy-grounding (pi0 labels)", "progress 0.213 vs C^pi 0.480"],
    ["B3.2 / B3.3", "imagination-only; residual RL", "below pi0; kills nominal", "0.068-0.113; 0.295 / nom 0.022"],
    ["P3", "recovery-nominal Pareto", "r 0.5 best inside eps", "r 0.7: 0.436 but nominal 0.766"]];
  table(s, tl, { x: 0.6, y: 1.25, w: 12.1, colW: [1.5, 3.7, 3.4, 3.5], fontSize: 12, rowH: 0.41 });
}

// ---------------------------------------------------------------- results
function results(pres, C) {
  // statistics
  pres.addSection({ title: "Statistics" });
  let s = content(pres, "Statistics", "Statistics: cluster bootstrap over scenes and seeds");
  img(s, "rev_forest.png", 0.6, 1.25, 8.0, 3.85, { align: "left", alt: "forest plot of key paired comparisons" });
  caption(s, "Amber = our claims, grey = n.s. checks, red / blue = baselines vs pi1, teal = LIBERO-Object (held-out)", 0.6, 5.15, 8.0);
  card(s, C, 8.9, 1.25, 3.8, 5.55, HEX.tint, "stats");
  text(s, "What changed", { x: 9.15, y: 1.42, w: 3.3, h: 0.4, fontSize: 17, bold: true });
  bullets(s, [
    "10 trials of a scene are not independent: unit = test scene (56) for policies, failure instance for search",
    "CI = scene × training-seed bootstrap; p = paired sign-flip (policies) or exact McNemar (search)",
    "effect: M5 WM vs random p 4e-7 (episode level) → 0.003 (scene level); still significant",
    "3 seeds: pi1 0.439 / 0.396 / 0.420 = 0.418 ± 0.021; competence and goal SFT, M7 also ×3",
    "Spatial test is a dev set from M7 on; clean numbers come from frozen Object",
  ], { x: 9.15, y: 1.9, w: 3.35, h: 4.8, fontSize: 12, color: HEX.ink2 });
  card(s, C, 0.6, 5.55, 8.0, 1.25, HEX.amberTint, "statsrule");
  text(s, "Rule we now apply: differences under ~4 pt RSR on 1 seed are noise; nominal is judged against the pre-registered eps = 2 pt with a 10-task bootstrap.",
    { x: 0.85, y: 5.68, w: 7.5, h: 1.0, fontSize: 13, color: HEX.ink2, valign: "middle" });

  // Q1
  divider(pres, "Q1 competence", "Q1: why policy competence?",
    "score a bridge by 'will THIS policy finish from here', not by goal distance or task progress");
  s = content(pres, "Q1 competence", "Q1: competence beats goal and progress, in search and after SFT");
  img(s, "rev_q1.png", 0.6, 1.2, 12.1, 4.3, { alt: "search solve rate per critic and test RSR per SFT data source" });
  const q1 = [["Search", "C^pi v2 0.480 vs goal latent 0.240, progress critic 0.213, random 0.339 (16 sims / failure). Progress is worse than random."],
    ["SFT", "competence 0.417 ± 0.011 vs goal 0.187 ± 0.002 (3 seeds), +0.230 [+0.161, +0.298]; 47 / 56 scenes better."],
    ["Count-matched", "progress data 0.177 vs competence n=60 0.430 (nominal 0.774 vs 0.798); B4-success data 0.427 / 0.796."]];
  q1.forEach(([h, d], i) => {
    const x = 0.6 + i * 4.1;
    card(s, C, x, 5.6, 3.9, 1.25, i === 1 ? HEX.amberTint : HEX.tint, "q1c" + i);
    text(s, h, { x: x + 0.2, y: 5.68, w: 3.5, h: 0.32, fontSize: 14, bold: true });
    text(s, d, { x: x + 0.2, y: 6.0, w: 3.55, h: 0.8, fontSize: 12, color: HEX.ink2 });
  });

  s = content(pres, "Q1 competence", "Q1: which critic? The gain comes from pi0 continuation labels");
  const ct = [[H("critic (ranks WM-imagined candidates)"), H("supervision"), H("n"), H("solved"), H("hits / sim")],
    ["random (no WM)", "-", "127", "0.339", "0.061"],
    ["goal latent (DINO-WM style)", "goal image latent, no critic", "125", "0.240", "0.046"],
    ["B3 progress", "demos + failure 0 + prefix 1", "127", "0.213", "0.038"],
    [B("C^pi v2 (PoC)", HEX.amber), "pi0 continuations", "127", B("0.480"), "0.156"],
    ["B4 success", "pi0 + pi1 + pi1more + demo (1/4 each)", "127", B("0.559"), "0.182"],
    ["B4-noPi0", "pi1 + pi1more + demo, no pi0 labels", "129", "0.450", "0.093"],
    [B("Cpi-pool", HEX.amber), "B4's pi0 labels only (7,160), B4 trainer", "129", B("0.550"), "0.156"]];
  table(s, ct, { x: 0.6, y: 1.3, w: 7.9, colW: [2.45, 3.2, 0.6, 0.8, 0.85], fontSize: 12, rowH: 0.47 });
  caption(s, "Same pi0 failures, proposal pool, WM v1 and 16 verified sims; only the critic changes", 0.6, 5.2, 7.9);
  card(s, C, 8.8, 1.3, 3.9, 5.5, HEX.tealTint, "b4read");
  text(s, "Reading", { x: 9.05, y: 1.48, w: 3.4, h: 0.4, fontSize: 17, bold: true, color: HEX.teal });
  bullets(s, [
    "B4 − B4-noPi0: +0.119 [+0.048, +0.198], p = 0.004: drop pi0 labels and the gain is gone",
    "Cpi-pool ≈ B4 (0.540 vs 0.556, p = 0.82): pi0 labels alone recover it",
    "labels from other policies and demos add nothing; progress supervision < random",
    "so B4 is a better competence critic, not evidence for policy-agnostic value → candidate C^pi v3",
  ], { x: 9.05, y: 1.95, w: 3.45, h: 4.75, fontSize: 13, color: HEX.ink2 });
  card(s, C, 0.6, 5.65, 7.9, 1.15, HEX.amberTint, "b4q");
  text(s, "Question answered: does B4 win because it is policy-agnostic, or because of more / re-filtered pi0 continuation labels? The latter.",
    { x: 0.85, y: 5.75, w: 7.4, h: 0.95, fontSize: 13, color: HEX.ink2, valign: "middle" });

  // Q2
  divider(pres, "Q2 world model", "Q2: why a world model?",
    "honest answer: acquisition efficiency (more verified recoveries per simulator rollout), not a planner yet");
  s = content(pres, "Q2 world model", "Q2: the WM buys efficiency; it does not plan well on its own yet");
  img(s, "rev_q2.png", 0.6, 1.2, 12.1, 4.3, { alt: "WM vs real-dynamics prefix cost, and M11 WM as planner" });
  card(s, C, 0.6, 5.6, 5.95, 1.25, HEX.amberTint, "q2a");
  text(s, "What the WM buys", { x: 0.85, y: 5.68, w: 5.5, h: 0.32, fontSize: 14, bold: true });
  text(s, "Same solve rate as a real-dynamics prefix at ~1/3 of the transitions (5.0k vs 16.5k / failure); 3.1× (Spatial) and 2.2× (Object) verified hits per sim vs random.",
    { x: 0.85, y: 6.0, w: 5.55, h: 0.8, fontSize: 12, color: HEX.ink2 });
  card(s, C, 6.75, 5.6, 5.95, 1.25, HEX.redTint, "q2b");
  text(s, "What it does not (M11)", { x: 7.0, y: 5.68, w: 5.5, h: 0.32, fontSize: 14, bold: true, color: HEX.red });
  text(s, "No primitives: −0.175 vs library (p = 3e-4). Planning from noise: 0.398 vs 0.458 (n.s., −0.64 hits / failure, p = 0.02); only 2 failures solved outside the library. WM = mostly a filter.",
    { x: 7.0, y: 6.0, w: 5.55, h: 0.8, fontSize: 12, color: HEX.ink2 });
  s.addNotes("M11 arm B: wm_plan_v2 = same planner with WM v2 + critic v3 trained on broad-action data: +0.025 over wm_plan (n.s.). Its successes are the jerkiest and farthest from SmolVLA's manifold; 83% of its success strategy cells are unseen by library arms. Broad-action WM data does not close the gap.");

  // Q3
  divider(pres, "Q3 distillation", "Q3: why distill?",
    "every alternative at the same budget is below pi1, or pays with nominal success or a test-time module");
  s = content(pres, "Q3 distillation", "Q3: all baselines vs RetryVLA (Spatial dev split)");
  img(s, "rev_baselines.png", 0.6, 1.2, 8.9, 5.6, { align: "left", alt: "test RSR and nominal SR for every baseline" });
  card(s, C, 9.75, 1.25, 2.95, 5.55, HEX.tint, "q3read");
  stat(s, "0.418", "RetryVLA pi1 test RSR (3 seeds, ± 0.021), nominal 0.785", 9.95, 1.4, 2.6, HEX.amber);
  bullets(s, [
    "only competence-searched, sim-verified data reaches pi1's RSR AND stays above the nominal floor",
    "goal, progress and random data all fall below the floor",
    "no extra module at test time",
  ], { x: 9.95, y: 3.0, w: 2.6, h: 3.7, fontSize: 12, color: HEX.ink2 });

  s = content(pres, "Q3 distillation", "Q3: four ways to avoid distillation, and why they fail");
  const alt = [["Demo-only SFT", "0.127", "nominal 0.820", "same init + steps, no recovery data: −0.292 [−0.380, −0.207] vs pi1. Recovery data is what helps.", HEX.ink2],
    ["Online best-of-8", "+0.027", "on pi1 (n.s.); pi0 +0.000", "WM + critic pick among 8 VLA samples every replan: costly at test time and only re-ranks the policy's own samples: no gain at all on pi0.", HEX.blue],
    ["Imagination-only (RISE-inspired)", "0.068-0.113", "nominal 0.718-0.776", "learn from imagined advantage, no verification: imagined positives V 0.81 vs real success 0.185 (~4 / 5 wrong).", HEX.blue],
    ["Residual RL (PPO bridge)", "0.295", "nominal 0.022", "full matched budget 3.47M: learns recovery but fires on every episode start (25% budget: 0.111 / 0.400).", HEX.red]];
  alt.forEach(([h, big, sub, d, col], i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 1.3 + Math.floor(i / 2) * 2.75;
    card(s, C, x, y, 5.95, 2.55, HEX.tint, "alt" + i);
    text(s, h, { x: x + 0.25, y: y + 0.18, w: 5.5, h: 0.4, fontSize: 16, bold: true });
    text(s, big, { x: x + 0.25, y: y + 0.65, w: 2.3, h: 0.7, fontSize: 30, bold: true, color: col });
    text(s, "test RSR\n" + sub, { x: x + 0.25, y: y + 1.38, w: 2.3, h: 0.9, fontSize: 12, color: HEX.muted });
    text(s, d, { x: x + 2.7, y: y + 0.7, w: 3.05, h: 1.7, fontSize: 13, color: HEX.ink2 });
  });

  // nominal preservation
  pres.addSection({ title: "Nominal" });
  s = content(pres, "Nominal", "Recovery-nominal Pareto: r 0.5 is the best point inside the margin");
  img(s, "rev_p3_pareto.png", 0.6, 1.2, 7.6, 5.2, { align: "left", alt: "P3 recovery-nominal Pareto" });
  caption(s, "P3 sweep, seed 0 per point; pi1 = r 0.5 (3 seeds). Includes the final r 0.3 / r 0.7 nominal evals", 0.6, 6.45, 7.6);
  card(s, C, 8.5, 1.25, 4.2, 5.55, HEX.tint, "p3read");
  text(s, "Reading", { x: 8.75, y: 1.42, w: 3.7, h: 0.4, fontSize: 17, bold: true });
  bullets(s, [
    "RSR rises with the recovery ratio: 0.323 / 0.375 / 0.418 / 0.436 for r 0.1 / 0.3 / 0.5 / 0.7",
    "r 0.7: +0.017 RSR (n.s.) but nominal 0.766: fails the 2 pt margin",
    "r 0.1 / 0.3 keep nominal (0.788 / 0.814) but lose RSR (−0.095, −0.043)",
    "L_preserve: 0.429 / 0.806, inside pi1's seed spread (0.396-0.439 / 0.766-0.808): not needed",
    "pi1 dNSR −0.011 [−0.046, +0.024]: constraint met",
  ], { x: 8.75, y: 1.9, w: 3.75, h: 4.8, fontSize: 13, color: HEX.ink2 });

  // Object
  pres.addSection({ title: "Generalization" });
  s = content(pres, "Generalization", "LIBERO-Object (held-out): what replicates and what does not");
  img(s, "rev_object.png", 0.6, 1.2, 12.1, 4.3, { alt: "LIBERO-Object test RSR and search solve rate" });
  const ob = [["Replicates", HEX.teal, HEX.tealTint, "competence − demo-only +0.181 [+0.092, +0.277], p = 0.001; nominal 0.942 vs 0.944 (1 seed)."],
    ["Replicates in search", HEX.teal, HEX.tealTint, "competence vs goal 66 / 8 failures, p = 2e-12; SFT +0.112 [−0.004, +0.227], p = 0.08 (directional)."],
    ["Does not replicate", HEX.red, HEX.redTint, "competence vs random data at equal count −0.008 (p = 0.93). Claim → acquisition efficiency: 0.134 vs 0.060 hits / sim."]];
  ob.forEach(([h, col, tint, d], i) => {
    const x = 0.6 + i * 4.1;
    card(s, C, x, 5.6, 3.9, 1.25, tint, "obc" + i);
    text(s, h, { x: x + 0.2, y: 5.68, w: 3.5, h: 0.32, fontSize: 14, bold: true, color: col });
    text(s, d, { x: x + 0.2, y: 6.0, w: 3.55, h: 0.8, fontSize: 12, color: HEX.ink2 });
  });
  s.addNotes("Split object_v1: 77 scenes, train 51 / test 26 (frozen). WM + critic retrained on Object train data (M2 recipe). Search on 247 pi0 failures, 16 verifications / arm, 123 trajectories per arm (size of the smallest bank, wm_goal), round-robin. Goal bank covers only 56 unique failures (comp 114, random 100).");
}

// ---------------------------------------------------------------- related work
function related(pres, C) {
  pres.addSection({ title: "Related work" });
  let s = content(pres, "Related work", "Related work: where RetryVLA sits");
  const y = "yes", n = "no";
  const rt = [[H("method"), H("recovery"), H("world model"), H("objective / value"), H("policy update"), H("verification"), H("extra at test time")],
    [B("RetryVLA (ours)", HEX.amber), y, "latent DINOv2 (filter)", B("competence C^pi of the frozen pi"), "verified SFT + 50% nominal", "sim, pi continues", B("none")],
    ["RePO-VLA", y, n, "progress-aware semantic value", "value-conditioned SFT", "planner / human data", "none (v = 1 token)"],
    ["Dream2Fix", "predicts fix", "video WM makes failures", "learned verifier filter", "SFT of a VLM", "learned verifier", "VLM + correction"],
    ["ReCoVLA", y, n, "VLM-compiled reward", "residual PPO, VLA frozen", "sim training", "VLM + residual policy"],
    ["RACER", y, n, "none (rich language)", "SFT on perturbed demos", "sim data generation", "VLM supervisor"],
    ["Neuro-symbolic", y, "scene-graph model", "distance to sub-goal", "none (re-plan)", n, "detector + search"],
    ["DINO-WM", n, "latent DINOv2", "distance to goal image", "none (MPC / CEM)", n, "planning each step"],
    ["TD-MPC2", n, "implicit latent", "reward + TD value", "online RL", "online interaction", "MPPI planning"],
    ["RISE", "indirect", "video WM + value", "progress value", "RL in imagination", "none", "none"],
    ["WMPO", "emergent", "pixel video WM", "binary success reward", "GRPO in the WM", "few real rollouts", "none"],
    ["MimicGen", n, n, "task success on replay", "BC on generated data", "sim replay", "none"]];
  table(s, rt, { x: 0.6, y: 1.25, w: 12.1, colW: [1.55, 1.05, 1.85, 2.4, 1.95, 1.65, 1.65], fontSize: 11, rowH: 0.39 });
  text(s, "Different signal (competence of the frozen policy, not goal / progress / reward), different update (verified SFT, not RL), cheap test time (no VLM, no planning). Our WM is a filter; verification sets the labels.",
    { x: 0.6, y: 6.15, w: 12.1, h: 0.7, fontSize: 13, italic: true, color: HEX.ink2 });

  const grids = [
    ["Related work: recovery and failure correction", [
      ["rw_repovla.png", "RePO-VLA (arXiv 2026)", "progress value + value-conditioned SFT; recovery data from planner / humans. We search for it."],
      ["rw_dream2fix.png", "Dream2Fix (arXiv 2026)", "video WM dreams failures; a VLM predicts corrections. Not distilled, verifier is learned."],
      ["rw_recovla.png", "ReCoVLA (arXiv 2026)", "VLM-compiled reward + residual PPO on a frozen VLA. Ours: no reward, no residual at test."],
      ["rw_racer.png", "RACER (ICRA 2025)", "perturbed expert demos + language supervisor every step. Ignores whether pi can continue."]]],
    ["Related work: world-model planning and WM-assisted VLAs", [
      ["rw_dinowm.png", "DINO-WM (ICML 2025)", "our WM architecture; plans to a goal image. Goal-latent search: 0.240 vs C^pi 0.480 for us."],
      ["rw_tdmpc2.png", "TD-MPC2 (ICLR 2024)", "short latent plan + value at the horizon: our structure, but RL value and planning at test."],
      ["rw_rise_overview.png", "RISE (RSS 2026)", "RL in imagination with a progress value. Our RISE-inspired arm: 0.068-0.113 < pi0."],
      ["rw_wmpo.png", "WMPO (ICLR 2026)", "GRPO inside a pixel video WM, whole task. Ours: short bridges, verified SFT."]]]];
  grids.forEach(([title, items]) => {
    s = content(pres, "Related work", title);
    items.forEach(([f, h, d], i) => {
      const x = 0.6 + (i % 2) * 6.15, yy = 1.25 + Math.floor(i / 2) * 2.85;
      card(s, C, x, yy, 5.95, 2.7, HEX.tint, "rw" + i);
      img(s, f, x + 0.15, yy + 0.12, 5.65, 1.75, { alt: h });
      text(s, h, { x: x + 0.2, y: yy + 1.92, w: 5.55, h: 0.3, fontSize: 14, bold: true });
      text(s, d, { x: x + 0.2, y: yy + 2.22, w: 5.55, h: 0.45, fontSize: 12, color: HEX.ink2 });
    });
  });
  s.addNotes("Figures cropped from the arXiv PDFs (docs/related_work/figs). Benchmark: LIBERO-Recover (arXiv 2609.05178), RSR metric, 4 levels L1-L4 (not released for our scenes yet).");
}

// ---------------------------------------------------------------- assessment and plan
function plan(pres, C) {
  pres.addSection({ title: "Assessment and plan" });
  let s = content(pres, "Assessment and plan", "Limitations and honest negatives");
  const cols = [["Method", HEX.red, [
    "WM is mostly a filter over the proposal library (VLA samples + primitives): no library −0.175; planning from noise 0.398 vs 0.458",
    "WM data is not better than random-search data at equal count on Object (0.354 vs 0.362): only an efficiency claim holds",
    "self-improvement loop (M6) gave no gain beyond extra training; weak tasks (stove, cookie box) stay low",
    "one VLA backbone (SmolVLA, weak base 0.66 nominal) and one WM: no generality claim yet"]],
  ["Evidence and process", HEX.ink2, [
    "many baselines are 1 seed (random, progress, RISE, RL, best-of-8, P3, all of Object): < ~4 pt is noise",
    "Object nominal 0.942 vs 0.944 is 1 seed: no eps-test yet",
    "Spatial test is a dev set from M7 on (used to choose M7 / M8); clean numbers only from Object",
    "infrastructure cost: ≥ 6 quota (EDQUOT) incidents, lost tmux servers, MuJoCo segfaults, OOM: several SFTs retrained (code now patched)"]]];
  cols.forEach(([h, col, items], i) => {
    const x = 0.6 + i * 6.15;
    card(s, C, x, 1.3, 5.95, 5.5, i === 0 ? HEX.redTint : HEX.tint, "lim" + i);
    s.addShape("ellipse", { x: x + 0.25, y: 1.5, w: 0.32, h: 0.32, fill: { color: col }, line: { color: col } });
    text(s, h, { x: x + 0.7, y: 1.46, w: 5.0, h: 0.4, fontSize: 18, bold: true, color: col });
    bullets(s, items, { x: x + 0.3, y: 2.05, w: 5.4, h: 4.6, fontSize: 14, color: HEX.ink2, paraSpaceAfter: 10 });
  });

  s = content(pres, "Assessment and plan", "Where we stand");
  card(s, C, 0.6, 1.3, 5.95, 5.5, HEX.tealTint, "works");
  text(s, "Supported", { x: 0.85, y: 1.48, w: 5.4, h: 0.4, fontSize: 18, bold: true, color: HEX.teal });
  bullets(s, [
    "competence-guided search + verified distillation: RSR 0.118 → 0.418 ± 0.021, nominal within 2 pt (3 seeds)",
    "competence ≫ goal / progress in search and SFT; search result replicated on held-out Object",
    "distillation ≫ demo-only, best-of-N, imagination-only and residual RL",
    "B4 question answered: the value is in pi0 continuation labels",
    "WM: 2-3× more verified recoveries per sim; matches a real-dynamics prefix at ~1/3 transitions",
  ], { x: 0.85, y: 2.0, w: 5.45, h: 4.7, fontSize: 16, color: HEX.ink2, paraSpaceAfter: 12 });
  card(s, C, 6.75, 1.3, 5.95, 5.5, HEX.redTint, "open");
  text(s, "Open", { x: 7.0, y: 1.48, w: 5.4, h: 0.4, fontSize: 18, bold: true, color: HEX.red });
  bullets(s, [
    "the WM does not yet plan bridges without the proposal library (M11)",
    "acquisition efficiency → better policy at matched sim budget? (P4, not run)",
    "end-to-end cost accounting incl. WM / critic training (P5, partial)",
    "1-seed baselines and Object; eps-test on Object nominal",
    "one VLA, one WM, two suites; LIBERO-Goal / LIBERO-10 not run",
  ], { x: 7.0, y: 2.0, w: 5.45, h: 4.7, fontSize: 16, color: HEX.ink2, paraSpaceAfter: 12 });

  s = content(pres, "Assessment and plan", "Next steps: PoC plan status (PLAN.md)");
  const pt = [[H("#"), H("experiment"), H("question"), H("status")],
    ["P1", "WM as planner (M11): no library, broad-noise CEM, WM v2", "can the WM plan bridges itself?", B("done: no, WM = filter", HEX.red)],
    ["P2", "progress / task-success critic baselines (B2-B4)", "policy grounding or any value?", B("done: policy grounding", HEX.teal)],
    ["P3", "recovery-nominal Pareto + L_preserve", "M-B constraint, forgetting", B("done: r 0.5, no L_preserve", HEX.teal)],
    ["P4", "matched-sim-budget distillation on Object, 3 seeds", "does efficiency give a better policy?", B("todo", HEX.amber)],
    ["P5", "end-to-end cost accounting (WM + critic amortized)", "efficiency claim", B("partial", HEX.amber)]];
  table(s, pt, { x: 0.6, y: 1.3, w: 12.1, colW: [0.6, 5.0, 3.6, 2.9], fontSize: 13, rowH: 0.5 });
  const nx = [["C^pi v3", "B4-style trainer on the Cpi-pool pi0 labels (+0.064 solved over v2), after a held-out check"],
    ["More seeds", "3 seeds for key 1-seed baselines and for Object (nominal eps-test)"],
    ["Suites", "LIBERO-Goal and LIBERO-10 with the frozen recipe; L1-L4 breakdown"],
    ["Real world", "adaptation not decided"]];
  nx.forEach(([h, d], i) => {
    const x = 0.6 + i * 3.05;
    card(s, C, x, 4.6, 2.85, 2.2, i === 3 ? HEX.redTint : HEX.amberTint, "nx" + i);
    badge(s, C, i + 1, x + 0.18, 4.78, i === 3 ? HEX.red : HEX.amber);
    text(s, h, { x: x + 0.75, y: 4.8, w: 2.0, h: 0.38, fontSize: 15, bold: true });
    text(s, d, { x: x + 0.2, y: 5.35, w: 2.5, h: 1.35, fontSize: 13, color: HEX.ink2 });
  });

  s = content(pres, "Assessment and plan", "Long-term plan");
  const phases = [
    ["Now", "PoC", "close out: P4 on Object (3 seeds), P5 cost accounting, more seeds, C^pi v3", HEX.amber],
    ["Next", "M-A", "generality of the planner: a second VLA backbone and / or a second (pixel / video) WM; claim 'general' only after", HEX.blue],
    ["Then", "Suites", "LIBERO-Goal and LIBERO-10 with the frozen recipe (Setting A), LIBERO-Recover L1-L4 breakdown", HEX.teal],
    ["Later", "Loop", "pi_k → C^{pi_k} → search → verified distillation → pi_{k+1}, critic re-grounded on pi_k each round", HEX.ink2],
    ["Open", "Analysis", "action coverage of verified bridges vs pi0 proposals; learned proposal head only if coverage is the bottleneck; real world not decided", HEX.red],
  ];
  phases.forEach(([when, tag, d, col], i) => {
    const yy = 1.35 + i * 1.08;
    card(s, C, 0.6, yy, 12.1, 0.92, HEX.tint, "phase" + i);
    s.addShape("ellipse", { x: 0.75, y: yy + 0.13, w: 0.66, h: 0.66, fill: { color: col }, line: { color: col } });
    text(s, tag, { x: 0.75, y: yy + 0.13, w: 0.66, h: 0.66, fontSize: 10, bold: true, color: HEX.white, align: "center", valign: "middle" });
    text(s, when, { x: 1.6, y: yy + 0.27, w: 1.1, h: 0.4, fontSize: 16, bold: true });
    text(s, d, { x: 2.8, y: yy + 0.12, w: 9.7, h: 0.72, fontSize: 14, color: HEX.ink2, valign: "middle" });
  });

  s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Assessment and plan" });
  s.addText("Next decisions", { placeholder: "title" });
  const dec = ["adopt C^pi v3 (Cpi-pool labels, B4 trainer) for all future search, after a held-out check?",
    "run P4 on Object with 3 seeds before scaling to LIBERO-Goal / LIBERO-10?",
    "WM as planner: invest further, or report the WM honestly as an efficient filter?",
    "real-world adaptation: not decided"];
  s.addText(dec.map((t, i) => ({ text: t, options: { bullet: { type: "number" }, breakLine: i < dec.length - 1, paraSpaceAfter: 6 } })),
    { x: 0.8, y: 4.05, w: 11.7, h: 2.5, fontSize: 18, color: "F3C98B", valign: "top", margin: 0, isTextBox: true });
}

function appendixDivider(pres) {
  divider(pres, "Appendix", "Appendix: M0-M5 details",
    "the earlier progress slides, kept as reference · single seed, Spatial split · current numbers are in the main part");
}

module.exports = { intro, results, related, plan, appendixDivider, H, B, table, content, divider };
