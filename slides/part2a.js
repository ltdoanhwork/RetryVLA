// part 2a: world model, critic, how WM search works
const { HEX, card, text, bullets, img, badge, stat, caption } = require("./lib");

module.exports = function part2a(pres, C) {
  pres.addSection({ title: "World model" });
  let s = pres.addSlide({ masterName: "SECTION", sectionTitle: "World model" });
  s.addText("Replacing the simulator with a world model", { placeholder: "title" });
  s.addText("M2: latent world model + competence critic   ·   M3: search in imagination", { placeholder: "body" });

  // why
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "World model" });
  s.addText("Why a world model if we already have a simulator?", { placeholder: "title" });
  card(s, C, 0.6, 1.4, 5.8, 4.9, HEX.tint, "oracle");
  text(s, "Oracle search (M1)", { x: 0.9, y: 1.6, w: 5.3, h: 0.45, fontSize: 20, bold: true });
  bullets(s, ["scores a candidate by running it for real", "up to 256 simulator rollouts per failure",
    "exact, but only exists where a simulator of THIS scene exists", "upper bound on what search can find"],
    { x: 0.9, y: 2.25, w: 5.3, h: 3.8, fontSize: 15, color: HEX.ink2 });
  card(s, C, 6.9, 1.4, 5.8, 4.9, HEX.amberTint, "wm");
  text(s, "World-model search (M3)", { x: 7.2, y: 1.6, w: 5.3, h: 0.45, fontSize: 20, bold: true, color: HEX.ink });
  bullets(s, ["scores a candidate by imagining it in latent space", "~800 candidates in a few GPU seconds",
    "simulator only to verify the top 16 (later: real robot, or none)", "needs: accurate dynamics on failure states + a critic that knows what SmolVLA can finish"],
    { x: 7.2, y: 2.25, w: 5.3, h: 3.8, fontSize: 15, color: HEX.ink2 });
  text(s, "Claim to test: at a fixed number of real rollouts, WM-guided search finds more recoveries than proposing at random.",
    { x: 0.6, y: 6.45, w: 12.1, h: 0.45, fontSize: 14, italic: true, color: HEX.ink2 });

  // latent state
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "World model" });
  s.addText("State = frozen DINOv2 patch tokens + robot state", { placeholder: "title" });
  img(s, "dino_tokens.png", 0.6, 1.3, 8.2, 2.85, { align: "left", alt: "DINOv2 tokens PCA" });
  caption(s, "Same drawer failure state. Token colors = first 3 PCA components: bowl, drawer, arm and table separate without any training", 0.6, 4.15, 8.2);
  card(s, C, 9.1, 1.3, 3.6, 5.4, HEX.tint, "state");
  text(s, "h_t (per time step)", { x: 9.35, y: 1.5, w: 3.1, h: 0.4, fontSize: 17, bold: true });
  bullets(s, ["agent cam 6×6 tokens", "wrist cam 6×6 tokens", "robot state q (8-D): eef pos, axis-angle, finger",
    "= 72 tokens × 384-D + 1 state token", "spatial tokens, not one global vector: L3/L4 failures depend on where things are"],
    { x: 9.35, y: 2.0, w: 3.1, h: 4.5, fontSize: 13, color: HEX.ink2 });
  card(s, C, 0.6, 4.6, 8.2, 2.1, HEX.amberTint, "c1");
  text(s, "C1 training data for the WM and critic", { x: 0.85, y: 4.75, w: 7.7, h: 0.4, fontSize: 16, bold: true });
  bullets(s, ["144 TRAIN failure instances × 32 candidate segments = 4,608 simulated trajectories",
    "latents every 5 env steps + the actions + outcome label (did SmolVLA finish afterwards?)",
    "the failure episodes containing the 53 hard search targets are excluded: those failures stay UNSEEN"],
    { x: 0.85, y: 5.2, w: 7.7, h: 1.45, fontSize: 13, color: HEX.ink2 });

  // C1 example pair
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "World model" });
  s.addText("One failure state, two candidates, two labels", { placeholder: "title" });
  img(s, "strip_c1_success.png", 0.6, 1.3, 12.1, 2.35, { alt: "C1 success sample" });
  caption(s, "Label 1: opens at t=21 and lifts. The bowl stays on the plate and SmolVLA finishes", 0.6, 3.65, 12.1);
  img(s, "strip_c1_fail.png", 0.6, 4.05, 12.1, 2.35, { alt: "C1 failure sample" });
  caption(s, "Label 0: also opens, but drags the bowl off the plate (t≥35), so SmolVLA cannot finish. The critic must tell these apart from latents alone", 0.6, 6.4, 12.1);

  // dynamics model
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "World model" });
  s.addText("M2: latent dynamics h(t+1) = h(t) + F(h(t), 5 actions)", { placeholder: "title" });
  const arch = [["h_t", "72 patch + 1 state token", HEX.ink2], ["+ action token", "MLP(5 × 7-D actions)", HEX.amber],
    ["Transformer", "6 layers, width 384", HEX.ink], ["4 heads", "ensemble Δh: mean = prediction, spread = uncertainty", HEX.blue]];
  arch.forEach(([h, d, col], i) => {
    const x = 0.6 + i * 1.95;
    card(s, C, x, 1.45, 1.75, 1.7, col, "arch" + i);
    text(s, h, { x: x + 0.15, y: 1.6, w: 1.45, h: 0.45, fontSize: 15, bold: true, color: HEX.white });
    text(s, d, { x: x + 0.15, y: 2.1, w: 1.45, h: 1.0, fontSize: 11, color: "E8ECEF" });
  });
  bullets(s, ["loss = 1-step teacher-forced + 10-step open-loop rollout (search needs 50-step imagination)",
    "11.6M params, 20k steps, ~1.5 h on one A100"], { x: 0.6, y: 3.4, w: 7.6, h: 0.9, fontSize: 13, color: HEX.ink2 });
  img(s, "wm_horizon.png", 8.4, 1.3, 4.3, 2.9, { alt: "WM error vs horizon" });
  img(s, "wm_imagination.png", 0.6, 4.35, 12.1, 2.25, { alt: "imagined vs real gripper height" });
  caption(s, "Held-out failures: gripper height, real (black) vs imagined from the start frame + actions only (amber). Error is 0.36-0.63× the copy-last-frame baseline", 0.6, 6.6, 12.1);

  // critic
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "World model" });
  s.addText("M2: the competence critic V(z, q, task)", { placeholder: "title" });
  card(s, C, 0.6, 1.4, 6.0, 1.3, HEX.amberTint, "q");
  text(s, "“From this (possibly imagined) state, will frozen SmolVLA finish the task?”",
    { x: 0.85, y: 1.55, w: 5.5, h: 1.0, fontSize: 17, italic: true, fontFace: "Cambria" });
  bullets(s, ["small transformer over the same tokens + learned task embedding",
    "trained on real AND WM-imagined end states (50/50), so it is calibrated for what search feeds it",
    "competence-targeted: lifting out of a drawer scores high even though the bowl gets no closer to the plate",
    "held-out AUC 0.71 real / 0.67 imagined; top-4 picks succeed 2.1× the base rate"],
    { x: 0.6, y: 2.95, w: 6.0, h: 3.6, fontSize: 14, color: HEX.ink2 });
  img(s, "critic_scores.png", 6.9, 1.35, 5.8, 3.6, { alt: "critic score histogram" });
  card(s, C, 6.9, 5.15, 5.8, 1.55, HEX.redTint, "calib");
  text(s, "Ranks well, calibrated badly", { x: 7.15, y: 5.28, w: 5.3, h: 0.35, fontSize: 15, bold: true, color: HEX.red });
  text(s, "Almost every pick gets V≈1 while ~12% verify. Fine for ranking. Needs fixing before V can weight training (AWR).",
    { x: 7.15, y: 5.68, w: 5.3, h: 0.95, fontSize: 12, color: HEX.ink2 });

  // search in imagination
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "World model" });
  s.addText("M3: how the world model searches", { placeholder: "title" });
  const flow = [["Encode", "real failure frame → DINOv2 latent z0, robot state q0"],
    ["Propose 256", "same proposal pool as M1: SmolVLA samples, perturbations, primitives"],
    ["Imagine", "WM rolls each 50-step candidate forward: z0 → z50 (10 steps of 5 actions)"],
    ["Score", "S = V(z50) − λ·uncertainty − β·jerkiness"],
    ["Refine in imagination", "2 rounds of elite-local CEM, ~800 candidates, zero simulator calls"],
    ["Verify top-16", "only these touch the simulator; successes become training data"]];
  flow.forEach(([h, d], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 3.3, y = 1.4 + row * 1.75;
    card(s, C, x, y, 3.1, 1.55, i === 5 ? HEX.tealTint : (i >= 2 && i <= 4 ? HEX.amberTint : HEX.tint), "flow" + i);
    badge(s, C, i + 1, x + 0.18, y + 0.18, i === 5 ? HEX.teal : HEX.amber);
    text(s, h, { x: x + 0.72, y: y + 0.2, w: 2.25, h: 0.4, fontSize: 15, bold: true });
    text(s, d, { x: x + 0.2, y: y + 0.7, w: 2.75, h: 0.8, fontSize: 12, color: HEX.ink2 });
  });
  img(s, "drawer_failure_frame.png", 7.4, 1.4, 2.55, 2.55, { alt: "drawer failure agent view" });
  img(s, "drawer_failure_wrist.png", 10.15, 1.4, 2.55, 2.55, { alt: "drawer failure wrist view" });
  caption(s, "Input: one real failure observation (agent + wrist)", 7.4, 4.0, 5.3);
  card(s, C, 7.4, 4.45, 5.3, 2.25, HEX.tint, "fair");
  text(s, "Fair comparison", { x: 7.65, y: 4.6, w: 4.8, h: 0.35, fontSize: 15, bold: true });
  bullets(s, ["random arm: 16 candidates drawn from the same 256-proposal pool",
    "both arms: exactly 16 simulator rollouts per instance",
    "53 hard instances whose failure episodes the WM/critic never saw"],
    { x: 7.65, y: 5.0, w: 4.8, h: 1.6, fontSize: 12, color: HEX.ink2 });
};
