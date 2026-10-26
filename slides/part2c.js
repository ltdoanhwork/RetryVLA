// part 2c: M5 / M5b / M5c results + clickable demo videos
const { HEX, card, text, bullets, img, badge, stat, caption } = require("./lib");
const V = require("./videos");

function appendix(pres, C) {
  pres.addSection({ title: "Appx: M5 results" });

  // section divider
  let s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Appx: M5 results" });
  s.addText("M5: WM data vs random data (Spatial, seed 0)", { placeholder: "title" });
  s.addText("test RSR 0.25 (random data) → 0.44 (WM data, seed 0; 3 seeds: 0.418 ± 0.021), same #trajectories\non held-out Object the equal-count gap did not replicate (0.354 vs 0.362)", { placeholder: "body" });

  // 1. search arms
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appx: M5 results" });
  s.addText("M5b: 4 search arms, 256 fresh failures, same 16 sims each", { placeholder: "title" });
  img(s, "m5b_search.png", 0.6, 1.25, 8.3, 3.6, { alt: "search arms solve rate and hit rate" });
  const arms = [["random", "16 uniform picks from the proposal pool (no WM)", HEX.muted],
    ["WM diverse", "pool + latent CEM, k-means diverse top-16 (M5 default)", HEX.amber],
    ["WM noise-CEM", "NEW: CEM in SmolVLA's flow-matching noise: on-manifold by construction", HEX.blue],
    ["WM quality-div.", "NEW: MAP-Elites over 60 strategy cells (gripper × motion)", HEX.teal]];
  arms.forEach(([h, d, col], i) => {
    const y = 1.3 + i * 0.9;
    s.addShape("rect", { x: 9.2, y: y + 0.05, w: 0.12, h: 0.7, fill: { color: col }, line: { color: col } });
    text(s, h, { x: 9.45, y, w: 3.3, h: 0.35, fontSize: 14, bold: true });
    text(s, d, { x: 9.45, y: y + 0.33, w: 3.3, h: 0.55, fontSize: 11, color: HEX.ink2 });
  });
  card(s, C, 0.6, 5.05, 12.1, 1.7, HEX.tint, "m5bread");
  bullets(s, ["WM diverse: 2.8× more verified recoveries per simulator rollout than random (0.171 vs 0.062), p = 4e-9. Drawer: 0.43 vs 0.07 solved",
    "noise-CEM is the smoothest and closest to SmolVLA, but WORSE than random: it inherits SmolVLA's 'keep grasping' habit (drawer 0.02)",
    "quality-diversity: most distinct strategies per solved failure (4.0 vs 2.2), still beats random (p = 0.002)"],
    { x: 0.85, y: 5.2, w: 11.6, h: 1.5, fontSize: 13, color: HEX.ink2 });

  // 2. strategies
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appx: M5 results" });
  s.addText("Which strategies the WM finds", { placeholder: "title" });
  img(s, "m5_strategies.png", 0.6, 1.25, 12.1, 3.75, { alt: "strategy statistics" });
  card(s, C, 0.6, 5.2, 5.9, 1.55, HEX.amberTint, "s1");
  text(s, "Top recovery = open + lift: let go, get clear, let SmolVLA re-approach as from a normal start. WM found it ~230×, random ~35×.",
    { x: 0.85, y: 5.35, w: 5.4, h: 1.3, fontSize: 13, color: HEX.ink2 });
  card(s, C, 6.8, 5.2, 5.9, 1.55, HEX.tint, "s2");
  text(s, "Drawer: random spends 55% of sims on 'grasp again' (0 / 732 succeed); WM spends 83% on 'stay open, lift' (79 successes).",
    { x: 7.05, y: 5.35, w: 5.4, h: 1.3, fontSize: 13, color: HEX.ink2 });

  // 6. SFT results
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appx: M5 results" });
  s.addText("M5c (Spatial, seed 0): WM data > random data for the VLA", { placeholder: "title" });
  img(s, "m5_sft.png", 0.6, 1.2, 8.4, 3.55, { alt: "test RSR per SFT variant" });
  stat(s, "0.439", "test RSR, +208 WM recoveries (vs 0.252 random, same count, same init)", 9.3, 1.3, 3.4, HEX.amber);
  stat(s, "0.808", "nominal SR: no longer lost (demo-only SFT: 0.796)", 9.3, 2.8, 3.4, HEX.teal);
  card(s, C, 0.6, 4.95, 12.1, 1.8, HEX.tint, "sftread");
  bullets(s, ["M5 (241 vs 241): WM 0.402 vs random 0.286, +0.116 [+0.045, +0.191] scene-bootstrap CI, paired McNemar p = 4e-7",
    "M5c (208 vs 208, round-robin subset): WM diverse 0.439, quality-div. 0.354, random 0.252",
    "more is not better: all 423 WM trajectories 0.389 < 208 balanced. One training seed per variant"],
    { x: 0.85, y: 5.1, w: 11.6, h: 1.6, fontSize: 13, color: HEX.ink2 });

  // 7. subset bug
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appx: M5 results" });
  s.addText("A data-selection bug that hid the gain", { placeholder: "title" });
  img(s, "m5_task_rsr.png", 0.6, 1.2, 8.4, 3.9, { alt: "per task RSR" });
  card(s, C, 9.3, 1.25, 3.4, 5.5, HEX.redTint, "bug");
  text(s, "What went wrong", { x: 9.55, y: 1.45, w: 2.9, h: 0.4, fontSize: 16, bold: true, color: HEX.red });
  bullets(s, ["to match counts we kept the top-208 trajectories by critic score",
    "the critic scores drawer recoveries low, so only 4 drawer trajectories survived (of 44)",
    "drawer test RSR collapsed to 0.11",
    "fix: round-robin per failure (each failure's best, then 2nd best, ...): drawer 0.44, overall 0.30 → 0.44"],
    { x: 9.55, y: 1.95, w: 2.95, h: 4.7, fontSize: 12, color: HEX.ink2 });
  caption(s, "Lesson: the distribution of the distilled data matters as much as the search", 0.6, 5.3, 8.4);
}

function qualitative(pres, C) {
  pres.addSection({ title: "Qualitative" });
  let s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Qualitative" });
  s.addText("Qualitative: what the recoveries look like", { placeholder: "title" });
  s.addText("A = SmolVLA alone, B = random search, C = WM + C^pi search; then the distilled pi1 with no search\nvideos are clickable: click to play in PowerPoint", { placeholder: "body" });
  // 3-5. explainer videos
  const ex = [
    ["m5_explain_drawer", "Video: drawer failure, A alone / B random / C WM",
      "C lifts 1.13 → 1.38 m with the gripper open (orange bar), then SmolVLA re-approaches, grasps and places the bowl at step 198. A and B hover at the drawer for 360 steps."],
    ["m5_explain_cabinet", "Video: cabinet failure, A alone / B random / C WM",
      "B tries to re-grasp in place (open → close, shift +y) and fails. C lets go and lifts; SmolVLA then finishes at step 142."],
    ["m5_explain_ramekin", "Video: ramekin failure, A alone / B random / C WM",
      "Here the WM segment alone finishes the task (step 39): a short grasp-in-place the WM predicted to work. Random 0/16 on this failure."],
  ];
  ex.forEach(([key, title, desc]) => {
    s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Qualitative" });
    s.addText(title, { placeholder: "title" });
    V.clip(s, key, 0.6, 1.2, 12.1, 4.95, title);
    text(s, "All 3 columns start from the SAME failure state. Only the first 50 steps differ; after that SmolVLA drives every column. " + desc,
      { x: 0.6, y: 6.25, w: 12.1, h: 0.65, fontSize: 13, color: HEX.ink2 });
  });

  // 8-9. end-to-end videos
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Qualitative" });
  s.addText("Video: end-to-end, no search, unseen TEST failure", { placeholder: "title" });
  V.clip(s, "m5_e2e_test_drawer", 0.6, 1.2, 12.1, 4.15, "end to end test drawer");
  const e2e = [["π0 SmolVLA", "0/5"], ["π1-nominal", "0/5"], ["π1-random", "0/5"], ["π1-WM (RetryVLA)", "3/5"]];
  e2e.forEach(([n, v], i) => {
    const x = 0.6 + i * 3.05;
    card(s, C, x, 5.45, 2.9, 1.05, i === 3 ? HEX.amberTint : HEX.tint, "e2e" + i);
    text(s, v, { x: x + 0.2, y: 5.5, w: 1.2, h: 0.6, fontSize: 28, bold: true, color: i === 3 ? HEX.amber : HEX.ink2 });
    text(s, n + "\nseeds solved", { x: x + 1.35, y: 5.56, w: 1.5, h: 0.9, fontSize: 12, color: HEX.ink2 });
  });
  caption(s, "π1-WM = RetryVLA pi1, seed-0 checkpoint (test RSR 0.439; 3-seed mean 0.418 ± 0.021, nominal 0.785)", 0.6, 6.6, 12.1);

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Qualitative" });
  s.addText("The VLA learned the WM's move: get clear, then retry", { placeholder: "title" });
  img(s, "m5_e2e_test_keys.png", 0.6, 1.2, 7.6, 5.6, { alt: "end to end keyframes" });
  card(s, C, 8.5, 1.2, 4.2, 5.6, HEX.tint, "e2eread");
  text(s, "Same unseen drawer failure, seed 0 for every row", { x: 8.75, y: 1.4, w: 3.7, h: 0.7, fontSize: 15, bold: true });
  bullets(s, ["rows: π0, π1-nominal, π1-random, π1-WM",
    "π0 / π1-nominal hover at 1.09-1.10 m for all 520 steps",
    "π1-random moves a little higher (1.14-1.24 m), still stuck",
    "π1-WM lifts itself to 1.31 m by step 40, comes back down, grasps at ~140, places at 163",
    "no world model or simulator at test time: the behaviour was distilled",
    "train-split episode (ep031): 0/5, 0/5, 0/5, 2/5",
    "π1-WM = RetryVLA pi1 seed 0 (RSR 0.439); 3 seeds 0.418 ± 0.021"],
    { x: 8.75, y: 2.2, w: 3.75, h: 4.5, fontSize: 12, color: HEX.ink2 });

  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Qualitative" });
  s.addText("Video: end-to-end on the training-split drawer failure", { placeholder: "title" });
  V.clip(s, "m5_e2e_train_drawer", 0.6, 1.2, 12.1, 4.3, "end to end train drawer");
  text(s, "The same drawer failure as the A/B/C explainer (ep031 open6). π0 0/5, π1-nominal 0/5, π1-random 0/5, π1-WM 2/5 seeds. " +
    "Still far from reliable on this failure. π1-WM = RetryVLA pi1, seed 0 (3-seed test RSR 0.418 ± 0.021); the plan now verifies bridges under ≥ 2 of 3 continuation seeds.",
    { x: 0.6, y: 5.7, w: 12.1, h: 0.9, fontSize: 14, color: HEX.ink2 });
}

module.exports = { appendix, qualitative };
