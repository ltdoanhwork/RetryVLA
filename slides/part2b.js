// part 2b: M3 results, M4 distillation, M5 running, honest assessment, long-term plan
const { HEX, card, text, bullets, img, badge, stat, caption } = require("./lib");
const V = require("./videos");

module.exports = function part2b(pres, C) {
  // M3 headline result
  let s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "World model" });
  s.addText("M3: 2× more recoveries on unseen hard failures", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [
    { name: "WM + critic (16 sims)", labels: ["instances solved (of 53)", "verified hits (of 848)"], values: [19, 57] },
    { name: "random from same pool (16 sims)", labels: ["instances solved (of 53)", "verified hits (of 848)"], values: [9, 13] },
  ], { x: 0.6, y: 1.35, w: 6.6, h: 4.4, barDir: "col", barGapWidthPct: 60, chartColors: ["E8962E", "AAB4BE"],
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: "23303B", dataLabelFontSize: 12, dataLabelFontFace: "+mn-lt",
    catAxisLabelColor: "6B7785", valAxisLabelColor: "6B7785", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
    valGridLine: { color: "E3E7EB", size: 0.5 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b",
    legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: "3C4A57", valAxisHidden: true });
  stat(s, "36% vs 17%", "hard failures solved, same simulator budget (sign test p = 0.04)", 7.6, 1.45, 5.1, HEX.amber);
  stat(s, "35 / 45", "recoveries found by latent CEM: plans the WM invented, never simulated before verification", 7.6, 3.0, 5.1, HEX.ink);
  card(s, C, 7.6, 4.6, 5.1, 2.1, HEX.redTint, "caveat");
  text(s, "Caveat", { x: 7.85, y: 4.72, w: 4.6, h: 0.35, fontSize: 15, bold: true, color: HEX.red });
  text(s, "The gain is concentrated in the drawer task (WM 12/20 vs random 1/20). Other tasks: 7/33 vs 8/33. N = 53, one seed.",
    { x: 7.85, y: 5.12, w: 4.6, h: 1.5, fontSize: 13, color: HEX.ink2 });
  caption(s, "Picks are also smoother: jerk 0.017 vs 0.027", 0.6, 5.85, 6.6);

  // M3 side-by-side
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "World model" });
  s.addText("Same failure state: SmolVLA alone vs WM-guided", { placeholder: "title" });
  img(s, "compare_grid.png", 0.6, 1.3, 12.1, 3.95, { alt: "side by side comparison" });
  card(s, C, 0.6, 5.45, 5.9, 1.3, HEX.tint, "alone");
  text(s, "Top: SmolVLA alone hovers at the drawer mouth for 350 steps. 0/3 seeds, 0/4 pre-checks, random 0/16.",
    { x: 0.85, y: 5.6, w: 5.4, h: 1.05, fontSize: 13, color: HEX.ink2 });
  card(s, C, 6.8, 5.45, 5.9, 1.3, HEX.amberTint, "wmguided");
  text(s, "Bottom: the WM picks 'lift straight up' (orange). From there SmolVLA reaches in, grasps and places the bowl. 3/16 verified.",
    { x: 7.05, y: 5.6, w: 5.4, h: 1.05, fontSize: 13, color: HEX.ink2 });

  V.m3(pres, C);

  // WM chosen segment actions
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "World model" });
  s.addText("What the WM chose, step by step", { placeholder: "title" });
  img(s, "strip_wm_drawer.png", 0.6, 1.3, 12.1, 2.85, { alt: "WM-chosen drawer recovery strip" });
  img(s, "actions_wm_drawer.png", 0.6, 4.25, 7.2, 2.6, { align: "left", alt: "WM recovery actions" });
  card(s, C, 8.1, 4.3, 4.6, 2.45, HEX.tint, "readact");
  bullets(s, ["shaded = the 50 steps the WM chose: one clean dz = +0.5 move, gripper open, no chatter",
    "white = SmolVLA's own actions afterwards; finger closes at t≈95 (red), bowl placed at t=183",
    "contrast with the jerky old-CEM segment in M1"], { x: 8.35, y: 4.45, w: 4.1, h: 2.2, fontSize: 13, color: HEX.ink2 });

  // M4
  pres.addSection({ title: "Distillation" });
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Distillation" });
  s.addText("M4: does WM-found data improve the VLA? Not yet measurable", { placeholder: "title" });
  const rows = [
    [{ text: "policy (init)", options: { bold: true, color: HEX.white, fill: { color: HEX.ink } } },
     { text: "recovery data", options: { bold: true, color: HEX.white, fill: { color: HEX.ink } } },
     { text: "test RSR", options: { bold: true, color: HEX.white, fill: { color: HEX.ink } } },
     { text: "drawer", options: { bold: true, color: HEX.white, fill: { color: HEX.ink } } },
     { text: "nominal", options: { bold: true, color: HEX.white, fill: { color: HEX.ink } } }],
    ["pi1-random (pi0)", "241 random", "0.279", "18/160", "0.698"],
    ["A (pi1-nominal)", "241 random", "0.293", "24/160", "0.756"],
    ["B (pi1-nominal)", "241 random + 45 WM", "0.298", "28/160", "0.748"],
  ];
  s.addTable(rows, { x: 0.6, y: 1.45, w: 7.7, colW: [2.1, 2.2, 1.1, 1.1, 1.2], fontSize: 13, color: HEX.ink,
    border: { type: "solid", color: HEX.line, pt: 0.75 }, rowH: 0.42, fill: { color: HEX.white } });
  bullets(s, ["stronger init (pi1-nominal) recovers nominal 0.70 → 0.75 and nudges RSR up",
    "+45 WM trajectories: +0.005 RSR, drawer 24 → 28. Paired episodes 61 vs 58: noise",
    "the dose is tiny: 45 WM trajectories ≈ 8% of training samples"],
    { x: 0.6, y: 3.45, w: 7.7, h: 1.8, fontSize: 14, color: HEX.ink2 });
  card(s, C, 8.7, 1.45, 4.0, 5.25, HEX.amberTint, "fix");
  text(s, "So M5 asks it properly", { x: 8.95, y: 1.65, w: 3.5, h: 0.4, fontSize: 17, bold: true });
  bullets(s, ["generate WM data on all 108 pilot failures (not just 53 hard)", "same trajectory count WM vs random",
    "WM-only vs random-only SFT, same init, same recipe", "measure RSR, drawer, nominal"],
    { x: 8.95, y: 2.2, w: 3.5, h: 4.3, fontSize: 14, color: HEX.ink2 });

  V.distill(pres, C);

};
