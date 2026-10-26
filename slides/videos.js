// Video slides: one per stage, real outputs embedded (click to play in PowerPoint)
const fs = require("fs");
const path = require("path");
const { HEX, card, text, bullets, caption } = require("./lib");

const VD = path.join(__dirname, "assets", "video");
const MAN = fs.existsSync(path.join(VD, "manifest.json")) ? JSON.parse(fs.readFileSync(path.join(VD, "manifest.json"))) : {};

// embed a video fitted in a box (keeps aspect), poster frame as cover; returns false if missing
function video(s, key, x, y, w, h, alt) {
  const m = MAN[key];
  if (!m) return false;
  const r = m.w / m.h;
  let vw = w, vh = w / r;
  if (vh > h) { vh = h; vw = h * r; }
  const vx = x + (w - vw) / 2, vy = y + (h - vh) / 2;
  s.addMedia({ type: "video", path: path.join(VD, key + ".mp4"), cover: "data:image/png;base64," + fs.readFileSync(path.join(VD, key + "_poster.png")).toString("base64"),
    x: vx, y: vy, w: vw, h: vh, objectName: "video_" + key, altText: alt || key });
  return { x: vx, y: vy, w: vw, h: vh, sec: m.sec };
}

function vtag(s, label, x, y, color) {
  s.addShape("roundRect", { x, y, w: 1.25, h: 0.32, rectRadius: 0.08, fill: { color }, line: { color } });
  text(s, label, { x, y, w: 1.25, h: 0.32, fontSize: 11, bold: true, color: HEX.white, align: "center", valign: "middle" });
}

module.exports = {
  // generic clickable video with a VIDEO tag in its top-left corner
  clip(s, key, x, y, w, h, alt) {
    const v = video(s, key, x, y, w, h, alt);
    if (v) vtag(s, "▶ VIDEO", v.x + v.w - 1.33, v.y + 0.06, HEX.amber);
    else text(s, "(video missing: " + key + ")", { x, y, w, h: 0.4, fontSize: 12, color: HEX.red });
    return v;
  },

  m0(pres, C) {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appx: M0 baseline" });
    s.addText("Video: SmolVLA on held-out failure states", { placeholder: "title" });
    const a = video(s, "m0_pi0_fail_drawer", 0.6, 1.35, 6.0, 4.5, "pi0 fails on a drawer failure");
    const b = video(s, "m0_pi0_fail_ramekin", 6.7, 1.35, 6.0, 4.5, "pi0 fails on a ramekin failure");
    vtag(s, "VIDEO", 0.6, 1.4, HEX.ink2); vtag(s, "VIDEO", 6.7, 1.4, HEX.ink2);
    caption(s, "Drawer failure: pi0 never recovers (" + (a ? a.sec : "?") + " s)", 0.6, 5.95, 6.0);
    caption(s, "Ramekin failure: pi0 keeps trying the same grasp (" + (b ? b.sec : "?") + " s)", 6.7, 5.95, 6.0);
    text(s, "Official protocol, 520 steps. These are the 88% of test episodes we want to fix.", { x: 0.6, y: 6.4, w: 12.1, h: 0.4, fontSize: 14, color: HEX.ink2 });
  },

  m1(pres, C) {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appx: M1 oracle search" });
    s.addText("Video: oracle-search recoveries", { placeholder: "title" });
    video(s, "m1_trio", 0.6, 1.3, 12.1, 4.75, "two verified recoveries and a nominal demo");
    vtag(s, "VIDEO", 0.6, 1.3, HEX.ink2);
    text(s, "Orange bar = searched 50-step segment, blue = SmolVLA continuing, green = nominal demo. Overlay: step, segment, action given to the policy, finger opening.",
      { x: 0.6, y: 6.2, w: 12.1, h: 0.6, fontSize: 14, color: HEX.ink2 });
  },

  m1bad(pres, C) {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appx: M1 oracle search" });
    s.addText("Video: why random-noise search proves little", { placeholder: "title" });
    video(s, "m1_cem_bad", 0.6, 1.35, 6.4, 5.2, "jerky CEM recovery");
    vtag(s, "VIDEO", 0.6, 1.4, HEX.red);
    card(s, C, 7.4, 1.35, 5.3, 5.2, HEX.redTint, "bad");
    text(s, "Old CEM on a hard instance", { x: 7.65, y: 1.55, w: 4.8, h: 0.4, fontSize: 17, bold: true, color: HEX.red });
    bullets(s, ["arm thrashes, gripper flips open/closed with no effect on the fingers",
      "it still 'succeeds' afterwards, so it lands in the training set",
      "4-6× jerkier than demos: a likely reason recovery SFT costs nominal success",
      "motivates: smoothness cost, elite-local refinement, and scoring by a model of what SmolVLA can finish"],
      { x: 7.65, y: 2.1, w: 4.8, h: 4.3, fontSize: 14, color: HEX.ink2 });
  },

  c1(pres, C) {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appx: world model" });
    s.addText("Video: WM data, same state, opposite labels", { placeholder: "title" });
    video(s, "c1_pair", 0.6, 1.3, 7.6, 5.4, "C1 success and failure samples");
    card(s, C, 8.6, 1.3, 4.1, 5.4, HEX.tint, "c1v");
    text(s, "Same failure state, two candidate segments", { x: 8.85, y: 1.5, w: 3.6, h: 0.75, fontSize: 16, bold: true });
    bullets(s, ["left: opens and lifts, bowl stays on the plate, so SmolVLA can finish (label 1)",
      "right: opens but drags the bowl off the plate, so SmolVLA cannot (label 0)",
      "4,608 such segments teach the WM how latents move and the critic which end states are recoverable",
      "replayed from the stored actions: bit-exact"], { x: 8.85, y: 2.4, w: 3.6, h: 4.1, fontSize: 13, color: HEX.ink2 });
  },

  m3(pres, C) {
    let s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appx: world model" });
    s.addText("Video: SmolVLA alone vs WM-guided, same failure", { placeholder: "title" });
    video(s, "m3_compare", 0.6, 1.3, 12.1, 4.9, "side by side");
    vtag(s, "VIDEO", 0.6, 1.35, HEX.amber);
    text(s, "Left: SmolVLA alone hovers at the drawer for 350 steps. Right: WM-chosen 50-step lift (orange) then SmolVLA grasps and places the bowl.",
      { x: 0.6, y: 6.3, w: 12.1, h: 0.5, fontSize: 14, color: HEX.ink2 });

    s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appx: world model" });
    s.addText("Video: WM-found recoveries on unseen failures", { placeholder: "title" });
    video(s, "m3_wm_pair", 0.6, 1.3, 7.6, 5.4, "two WM-guided recoveries");
    card(s, C, 8.6, 1.3, 4.1, 5.4, HEX.amberTint, "m3v");
    text(s, "Picked by WM + critic, verified in sim", { x: 8.85, y: 1.5, w: 3.6, h: 0.75, fontSize: 16, bold: true });
    bullets(s, ["orange: the 50 steps the WM chose in imagination (lift clear of the drawer)",
      "blue: SmolVLA reaches in, grasps, places the bowl",
      "neither failure episode was in the WM's training data",
      "same strategy found on different failures: the WM generalizes the 'get clear first' idea"],
      { x: 8.85, y: 2.4, w: 3.6, h: 4.1, fontSize: 13, color: HEX.ink2 });
  },

  distill(pres, C) {
    if (!MAN["distill_drawer"] && !MAN["distill_stove"]) return;
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appx: distillation" });
    s.addText("Video: after distillation the VLA recovers by itself (no search)", { placeholder: "title" });
    const keys = ["distill_drawer", "distill_stove"].filter((k) => MAN[k]);
    keys.forEach((k, i) => {
      const y = keys.length === 1 ? 1.4 : 1.35 + i * 2.6;
      video(s, k, 0.6, y, 8.4, keys.length === 1 ? 4.8 : 2.45, k);
      vtag(s, "VIDEO", 0.6, y + 0.05, HEX.teal);
    });
    card(s, C, 9.3, 1.35, 3.4, 5.3, HEX.tealTint, "dist");
    text(s, "Held-out test failure", { x: 9.55, y: 1.55, w: 2.9, h: 0.4, fontSize: 16, bold: true, color: HEX.teal });
    bullets(s, ["left: pi0 (original SmolVLA)", "right: pi1, fine-tuned on simulator-verified recoveries",
      "same restored state, same protocol, no world model or simulator help at test time",
      "test RSR 0.12 → 0.29 overall"], { x: 9.55, y: 2.1, w: 2.95, h: 4.4, fontSize: 13, color: HEX.ink2 });
  },
};
