// build script: main review flow (review.js) + clickable videos + appendix with the M0-M5 slides
const path = require("path");
const { applyTheme } = require("/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.claude/skills/synced/767c841f-370a-4813-a6b8-8ffb7e9c7104_bc975de3-db28-45d8-b6e8-e7e5e2840da9/pptx/scripts/apply_theme.js");
const { THEME, makeDeck } = require("./lib");

async function build() {
  const { pres, C } = makeDeck();
  const R = require("./review");
  const P1 = require("./part1");
  const P2c = require("./part2c");
  R.intro(pres, C);          // title, executive summary, agenda, framing, timeline
  P1.setup(pres, C);         // benchmark / policy / compute, failure states, split
  R.results(pres, C);        // statistics, Q1, Q2, Q3, nominal Pareto, LIBERO-Object
  P2c.qualitative(pres, C);  // clickable A/B/C explainers + end-to-end videos
  R.related(pres, C);        // comparison table + method figures
  R.plan(pres, C);           // limitations, where we stand, next steps, long-term plan, decisions
  R.appendixDivider(pres);
  P1.history(pres, C);       // appendix: M0, M1
  require("./part2a")(pres, C);  // appendix: world model, critic
  require("./part2b")(pres, C);  // appendix: M3, M4
  P2c.appendix(pres, C);     // appendix: M5 / M5b / M5c
  const out = path.join(__dirname, "RetryVLA_progress.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
}

build().catch((e) => { console.error(e); process.exit(1); });
