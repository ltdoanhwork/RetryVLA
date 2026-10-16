// part 3: assessment, long-term plan, next steps; + build script
const path = require("path");
const { applyTheme } = require("/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.claude/skills/synced/767c841f-370a-4813-a6b8-8ffb7e9c7104_bc975de3-db28-45d8-b6e8-e7e5e2840da9/pptx/scripts/apply_theme.js");
const { THEME, HEX, makeDeck, card, text, bullets, img, badge, stat, caption } = require("./lib");

function part3(pres, C) {
  pres.addSection({ title: "Assessment and plan" });
  let s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Assessment and plan" });
  s.addText("Where we stand", { placeholder: "title" });
  card(s, C, 0.6, 1.4, 5.9, 5.3, HEX.tealTint, "works");
  text(s, "Supported so far", { x: 0.85, y: 1.6, w: 5.4, h: 0.4, fontSize: 18, bold: true, color: HEX.teal });
  bullets(s, [
    "recoveries exist and can be found: ~70% of failures, 50-step segments",
    "WM + competence critic: 2.8× more verified recoveries per simulator rollout than random (N = 256, p = 4e-9)",
    "WM data makes a better VLA: test RSR 0.44 vs 0.25 (random data, same count), nominal kept at 0.81",
    "the distilled VLA performs the WM's 'get clear, then retry' by itself on unseen failures",
  ], { x: 0.85, y: 2.15, w: 5.4, h: 4.4, fontSize: 14, color: HEX.ink2 });
  card(s, C, 6.8, 1.4, 5.9, 5.3, HEX.redTint, "open");
  text(s, "Open problems", { x: 7.05, y: 1.6, w: 5.4, h: 0.4, fontSize: 18, bold: true, color: HEX.red });
  bullets(s, [
    "one training seed per SFT variant; 56 test scenes, one suite",
    "recoveries are stochastic: some verified ones fail on re-simulation",
    "critic barely improved by its own labels (v2 yield 0.186 vs 0.161)",
    "noise-space search inherits SmolVLA's failure habit",
    "data selection is fragile (top-score cut dropped the drawer task)",
    "weak base checkpoint (0.66 vs ~0.90 reported)",
  ], { x: 7.05, y: 2.15, w: 5.4, h: 4.4, fontSize: 14, color: HEX.ink2 });

  // long-term plan
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Assessment and plan" });
  s.addText("Long-term plan", { placeholder: "title" });
  const phases = [
    ["Done", "M5", "WM data > random data for the VLA (0.44 vs 0.25); 4 search arms compared on 256 paired failures", HEX.amber],
    ["Next", "M6", "robust data: verify each recovery under 2-3 seeds; 3 SFT seeds; then loop π1 → π2 on π1's own remaining failures", HEX.blue],
    ["Then", "M7", "hybrid search (WM diverse + QD cells), parameterized primitives, multi-chunk / MCTS for multi-stage recoveries", HEX.teal],
    ["Scale", "M8", "Object, Goal, LIBERO-10 suites; L1-L4 levels once released; 3 seeds; stronger base VLAs", HEX.ink2],
    ["Stretch", "V2", "neural video WM generates RGB for imagined recoveries (no simulator at all), then real-robot recovery", HEX.red],
  ];
  phases.forEach(([when, tag, d, col], i) => {
    const y = 1.35 + i * 1.08;
    card(s, C, 0.6, y, 12.1, 0.92, HEX.tint, "phase" + i);
    s.addShape("ellipse", { x: 0.8, y: y + 0.2, w: 0.52, h: 0.52, fill: { color: col }, line: { color: col } });
    text(s, tag, { x: 0.8, y: y + 0.2, w: 0.52, h: 0.52, fontSize: 12, bold: true, color: HEX.white, align: "center", valign: "middle" });
    text(s, when, { x: 1.55, y: y + 0.27, w: 1.2, h: 0.4, fontSize: 16, bold: true });
    text(s, d, { x: 2.8, y: y + 0.12, w: 9.7, h: 0.72, fontSize: 13, color: HEX.ink2, valign: "middle" });
  });

  // paper story
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Assessment and plan" });
  s.addText("The paper story we are building toward", { placeholder: "title" });
  card(s, C, 0.6, 1.4, 12.1, 1.35, HEX.amberTint, "story");
  text(s, "Competence-targeted world-model search generates recovery actions that restore the policy's ability to finish the task, and distilling them lets the VLA recover from unseen failures without search.",
    { x: 0.9, y: 1.55, w: 11.5, h: 1.1, fontSize: 17, italic: true, fontFace: "Times New Roman" });
  const abl = [["Goal-progress vs competence critic", "does scoring by 'can SmolVLA finish' beat 'is the bowl closer'?"],
    ["Random vs oracle vs WM selection", "same #trajectories: is the gain from the WM, not just more data?"],
    ["Sim-efficiency curve", "verified recoveries per simulator rollout: random, oracle CEM, WM"],
    ["Iterations π0 → π1 → π2", "does the loop keep improving RSR on held-out failures?"]];
  abl.forEach(([h, d], i) => {
    const x = 0.6 + (i % 2) * 6.15, y = 3.05 + Math.floor(i / 2) * 1.85;
    card(s, C, x, y, 5.95, 1.65, HEX.tint, "abl" + i);
    badge(s, C, i + 1, x + 0.2, y + 0.2, HEX.ink2);
    text(s, h, { x: x + 0.8, y: y + 0.22, w: 4.95, h: 0.4, fontSize: 15, bold: true });
    text(s, d, { x: x + 0.8, y: y + 0.7, w: 4.95, h: 0.85, fontSize: 13, color: HEX.ink2 });
  });

  // next steps (dark close)
  s = pres.addSlide({ masterName: "SECTION", sectionTitle: "Assessment and plan" });
  s.addText("Next decisions", { placeholder: "title" });
  s.addText("WM data wins (0.44 vs 0.25). Next: robust multi-seed verification + 3 SFT seeds, then the π1 → π2 loop\nand scaling to more suites",
    { placeholder: "body" });
}

async function build() {
  const { pres, C } = makeDeck();
  require("./part1")(pres, C);
  require("./part2a")(pres, C);
  require("./part2b")(pres, C);
  require("./part2c")(pres, C);
  part3(pres, C);
  const out = path.join(__dirname, "RetryVLA_progress.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
}

build().catch((e) => { console.error(e); process.exit(1); });
