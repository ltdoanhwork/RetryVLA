// Shared theme, layouts and helpers for the RetryVLA deck.
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const A = (f) => path.join(__dirname, "assets", f);
const sizeOf = (f) => {  // PNG size from IHDR
  const b = fs.readFileSync(A(f));
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
};

const THEME = {
  name: "RetryVLA",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "23303B", lt1: "FFFFFF", dk2: "3C4A57", lt2: "F2F4F6",
    accent1: "E8962E", accent2: "1F8FE5", accent3: "2A9D8F", accent4: "C8553D",
    accent5: "6B7785", accent6: "F3C98B", hlink: "1F8FE5", folHlink: "6B7785",
  },
};
const HEX = { ink: "23303B", ink2: "3C4A57", muted: "6B7785", line: "D5DBE1", tint: "F2F4F6",
  amber: "E8962E", amberTint: "FCEFDC", blue: "1F8FE5", blueTint: "E3F1FC", teal: "2A9D8F",
  tealTint: "E0F2EF", red: "C8553D", redTint: "F8E3DE", white: "FFFFFF" };

function makeDeck() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
  pres.title = "RetryVLA: world-model-guided recovery learning";
  pres.author = "ltdoanhwork";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;

  pres.defineSlideMaster({
    title: "TITLE", background: { color: HEX.ink },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.6, fontSize: 44, bold: true,
        color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.95, w: 11.7, h: 1.4, fontSize: 18,
        color: "F3C98B", valign: "top", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "SECTION", background: { color: HEX.ink },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.7, w: 11.7, h: 1.2, fontSize: 40, bold: true,
        color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.0, w: 11.7, h: 1.0, fontSize: 18,
        color: "F3C98B", valign: "top", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: 12.3, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: "AAB4BE" },
  });
  pres.defineSlideMaster({
    title: "CONTENT", background: { color: HEX.white },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.75, fontSize: 30, bold: true,
        color: C.text1, valign: "middle", margin: 0 }, text: "" } },
      { text: { text: "RetryVLA  ·  LIBERO-Recover Spatial", options: { x: 0.6, y: 7.0, w: 6, h: 0.3, fontSize: 10,
        color: C.accent5, margin: 0 } } },
    ],
    slideNumber: { x: 12.3, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5 },
  });
  return { pres, C };
}

// ---------- helpers (each call builds fresh option objects) ----------
const shadow = () => ({ type: "outer", color: "23303B", opacity: 0.12, blur: 6, offset: 2, angle: 90 });

function card(s, C, x, y, w, h, fill, name) {
  s.addShape("roundRect", { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill },
    shadow: shadow(), objectName: name || "card" });
}

function text(s, t, o) {
  s.addText(t, Object.assign({ isTextBox: true, margin: 0, valign: "top", fontSize: 14, color: HEX.ink }, o));
}

function bullets(s, items, o) {
  const runs = items.map((it, i) => {
    const t = typeof it === "string" ? { text: it } : it;
    return { text: t.text, options: Object.assign({ bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6 },
      t.options || {}) };
  });
  s.addText(runs, Object.assign({ isTextBox: true, margin: 0, valign: "top", fontSize: 14, color: HEX.ink }, o));
}

// image fitted inside a box, keeping aspect ratio; align: "center" | "left" | "top"
function img(s, file, x, y, w, h, opts = {}) {
  const { w: pw, h: ph } = sizeOf(file);
  const r = pw / ph;
  let iw = w, ih = w / r;
  if (ih > h) { ih = h; iw = h * r; }
  const ix = opts.align === "left" ? x : x + (w - iw) / 2;
  const iy = opts.valign === "top" ? y : y + (h - ih) / 2;
  s.addImage({ path: A(file), x: ix, y: iy, w: iw, h: ih, altText: opts.alt || file, objectName: opts.name || file });
  return { x: ix, y: iy, w: iw, h: ih };
}

// number badge in a circle (motif)
function badge(s, C, n, x, y, fill) {
  s.addShape("ellipse", { x, y, w: 0.42, h: 0.42, fill: { color: fill || HEX.amber }, line: { color: fill || HEX.amber },
    objectName: "badge" });
  s.addText(String(n), { x, y, w: 0.42, h: 0.42, fontSize: 14, bold: true, color: HEX.white, align: "center",
    valign: "middle", margin: 0, isTextBox: true });
}

function stat(s, big, small, x, y, w, color) {
  text(s, big, { x, y, w, h: 0.75, fontSize: 40, bold: true, color: color || HEX.ink, fontFace: "Cambria" });
  text(s, small, { x, y: y + 0.78, w, h: 0.6, fontSize: 12, color: HEX.muted });
}

function caption(s, t, x, y, w) {
  text(s, t, { x, y, w, h: 0.3, fontSize: 11, color: HEX.muted, italic: true });
}

module.exports = { THEME, HEX, makeDeck, card, text, bullets, img, badge, stat, caption, sizeOf };
