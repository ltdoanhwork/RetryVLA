#!/usr/bin/env python3
"""Render a flow-diagram spec (YAML/JSON) to SVG + PNG + HTML in the diagram-design it-state style.

usage: python render.py SPEC [--out DIR] [--slug NAME] [--scale 2]
Default out dir: $RETRYVLA_ROOT/runs/_analysis/diagrams (or ./diagrams).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
from xml.sax.saxutils import escape

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import style as S  # noqa: E402

from layout import label_w, layout, node_lines, text_w  # noqa: E402

ICONS = json.loads((S.SKILL / "assets" / "icons.json").read_text())


def _f(v):
    return f"{v:.1f}".rstrip("0").rstrip(".")


def path_d(points, r=8):
    """Orthogonal polyline -> path with quadratic rounded corners (reference style)."""
    d = [f"M {_f(points[0][0])},{_f(points[0][1])}"]
    for k in range(1, len(points) - 1):
        (x0, y0), (x1, y1), (x2, y2) = points[k - 1], points[k], points[k + 1]
        l1 = abs(x1 - x0) + abs(y1 - y0)
        l2 = abs(x2 - x1) + abs(y2 - y1)
        rr = min(r, l1 / 2, l2 / 2)
        if rr < 0.5:
            d.append(f"L {_f(x1)},{_f(y1)}")
            continue
        ax = x1 - rr * (x1 - x0) / l1 if l1 else x1
        ay = y1 - rr * (y1 - y0) / l1 if l1 else y1
        bx = x1 + rr * (x2 - x1) / l2 if l2 else x1
        by = y1 + rr * (y2 - y1) / l2 if l2 else y1
        d.append(f"L {_f(ax)},{_f(ay)} Q {_f(x1)},{_f(y1)} {_f(bx)},{_f(by)}")
    d.append(f"L {_f(points[-1][0])},{_f(points[-1][1])}")
    return " ".join(d)


def icon(name, x, y, color):
    if not name:
        return ""
    if name not in ICONS:
        print(f"warning: unknown icon '{name}' (see assets/icons.json); using 'file'", file=sys.stderr)
        name = "file"
    body = ICONS[name]["body"].replace("currentColor", color)
    return (f'<g transform="translate({_f(x)},{_f(y)})" fill="none" stroke="{color}" stroke-width="1.5" '
            f'stroke-linecap="round" stroke-linejoin="round">{body}</g>')


def mono(x, y, txt, fill, size=10, weight=400, anchor="start", spacing=0, opacity=1.0):
    ls = f' letter-spacing="{spacing}"' if spacing else ""
    op = f' fill-opacity="{opacity}"' if opacity < 1 else ""
    return (f'<text x="{_f(x)}" y="{_f(y)}" font-family="{S.MONO}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}"{op} text-anchor="{anchor}"{ls}>{escape(txt)}</text>')


def svg_doc(spec):
    L = layout(spec)
    W = L["width"]
    leg, leg_rows = legend(spec, L, W)
    H = L["height"] + 22 * leg_rows
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(W)} {_f(H)}" width="{_f(W)}" height="{_f(H)}">']
    o.append("<defs>")
    for k, (c, _) in S.EDGE.items():
        o.append(f'<marker id="arrow-{k}" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto" '
                 f'markerUnits="userSpaceOnUse"><polygon points="0 0, 8 3, 0 6" fill="{c}"/></marker>')
    o.append("</defs>")
    o.append(f'<rect width="{_f(W)}" height="{_f(H)}" fill="{S.PAPER}"/>')
    if spec.get("title"):
        if spec.get("eyebrow"):
            o.append(mono(S.PAD, 22, spec["eyebrow"].upper(), S.MUTED, 8, 500, spacing=1.4))
        o.append(f'<text x="{S.PAD}" y="44" font-family="{S.SANS}" font-size="18" font-weight="500" '
                 f'fill="{S.INK}">{escape(spec["title"])}</text>')
        if spec.get("subtitle"):
            o.append(f'<text x="{S.PAD}" y="62" font-family="{S.SANS}" font-size="10.5" fill="{S.MUTED}">'
                     f'{escape(spec["subtitle"])}</text>')
    # panels
    for z in L["zones"]:
        o.append(f'<rect x="{_f(z["x"])}" y="{_f(z["y"])}" width="{_f(z["w"])}" height="{_f(z["h"])}" rx="{S.ZONE_RX}" '
                 f'fill="{S.INK}" fill-opacity="0.02" stroke="{S.INK}" stroke-opacity="0.10" stroke-width="0.8"/>')
        if z["name"]:
            lw = text_w(z["name"].upper(), "mono", 400, 8, 1.12)
            o.append(f'<rect x="{_f(z["x"] + 20)}" y="{_f(z["y"] - 8)}" width="{_f(lw + 8)}" height="16" fill="{S.PAPER}"/>')
            o.append(mono(z["x"] + 24, z["y"] + 3, z["name"].upper(), S.INK, 8, 400, spacing=1.12, opacity=0.45))
    # edges
    for e in L["edges"]:
        kind = e.get("kind", "flow")
        c, w = S.EDGE[kind]
        dash = ' stroke-dasharray="4 3"' if e.get("dashed") else ""
        o.append(f'<path d="{path_d(e["points"])}" fill="none" stroke="{c}" stroke-width="{w}"{dash} '
                 f'marker-end="url(#arrow-{kind})"/>')
    # nodes
    for b in L["boxes"].values():
        fill, fo, st, so, sw, sd, tink, iink = S.NODE[b.kind]
        dash = f' stroke-dasharray="{sd}"' if sd else ""
        geo = f'x="{_f(b.x)}" y="{_f(b.y)}" width="{_f(b.w)}" height="{_f(b.h)}" rx="{S.NODE_RX}"'
        o.append(f'<rect {geo} fill="{S.PAPER}"/>')
        o.append(f'<rect {geo} fill="{fill}" fill-opacity="{fo}" stroke="{st}" stroke-opacity="{so}" '
                 f'stroke-width="{sw}"{dash}/>')
        o.append(icon(b.n.get("icon"), b.x + 12, b.y + (b.h - 24) / 2, iink))
        tx = b.x + (44 if b.n.get("icon") else 16)
        lines = node_lines(b.n)
        ty = b.y + (26 if lines else 26)
        o.append(f'<text x="{_f(tx)}" y="{_f(ty)}" font-family="{S.SANS}" font-size="12" font-weight="600" '
                 f'fill="{tink}">{escape(b.n["title"])}</text>')
        for k, (t, red) in enumerate(lines):
            o.append(mono(tx, ty + 16 + S.LINE_H * k, t, S.ACCENT if red else S.MUTED, 10))
    # edge labels (masked, source-bound)
    for e in L["edges"]:
        if not e.get("label"):
            continue
        txt = e["label"].upper()
        c = S.EDGE[e.get("kind", "flow")][0]
        lw = label_w(txt)
        mode, x, y = e["label_pos"]
        if mode == "above":
            rx, ry = x - lw / 2, y - 22
        elif mode == "on":
            rx, ry = x - lw / 2, y - 8
        else:
            rx, ry = x + 6, y
        o.append(f'<rect x="{_f(rx)}" y="{_f(ry)}" width="{_f(lw)}" height="16" rx="2" fill="{S.PAPER}"/>')
        o.append(mono(rx + lw / 2, ry + 11, txt, c, 8, 600, "middle", 0.64))
    o.append(leg)
    o.append("</svg>")
    return "\n".join(x for x in o if x), L


def legend(spec, L, W):
    used_e = {e.get("kind", "flow") for e in L["edges"]}
    used_n = {b.kind for b in L["boxes"].values()}
    dashed = any(e.get("dashed") for e in L["edges"])
    labels = {**S.LEGEND_LABELS, **(spec.get("legend") or {})}
    items = [k for k in ("flow", "pain", "neutral") if k in used_e]
    items += ["dashed"] if dashed else []
    items += [k for k in ("external", "bottleneck", "storage") if k in used_n]
    items = spec.get("legend_items") or items
    y = L["content_bottom"] + 28
    o = [f'<line x1="{S.PAD}" y1="{_f(y)}" x2="{_f(W - S.PAD)}" y2="{_f(y)}" stroke="{S.INK}" stroke-opacity="0.12" stroke-width="0.8"/>',
         mono(S.PAD, y + 20, "LEGEND", S.MUTED, 8, 400, spacing=1.12)]
    x, row = 80, 0
    for k in items:
        step = 32 + text_w(labels[k], "mono", 400, 8) + 36
        if x + step - 36 > W - S.PAD:
            x, row = 80, row + 1
        yy = y + 22 * row
        if k in S.EDGE:
            c, w = S.EDGE[k]
            o.append(f'<line x1="{x}" y1="{_f(yy + 16)}" x2="{x + 24}" y2="{_f(yy + 16)}" stroke="{c}" stroke-width="{w}"/>')
        elif k == "dashed":
            o.append(f'<line x1="{x}" y1="{_f(yy + 16)}" x2="{x + 24}" y2="{_f(yy + 16)}" stroke="{S.MUTED}" stroke-width="1.2" stroke-dasharray="4 3"/>')
        else:
            fill, fo, st, so, sw, sd, *_ = S.NODE[k]
            dash = f' stroke-dasharray="{sd}"' if sd else ""
            o.append(f'<rect x="{x + 4}" y="{_f(yy + 10)}" width="16" height="12" rx="2" fill="{fill}" fill-opacity="{fo}" '
                     f'stroke="{st}" stroke-opacity="{so}" stroke-width="0.8"{dash}/>')
        o.append(mono(x + 32, yy + 20, labels[k], S.MUTED, 8))
        x += step
    return "\n".join(o), row


def ensure_fonts():
    """Install bundled Geist fonts for fontconfig (user-level, idempotent) so cairo finds them."""
    dst = pathlib.Path(os.environ.get("XDG_DATA_HOME", pathlib.Path.home() / ".local/share")) / "fonts" / "geist"
    missing = [f for f in S.FONT_DIR.glob("*.ttf") if not (dst / f.name).exists()]
    if missing:
        dst.mkdir(parents=True, exist_ok=True)
        for f in missing:
            shutil.copy2(f, dst / f.name)
        subprocess.run(["fc-cache", "-f", str(dst)], check=False, capture_output=True)


def load_spec(p):
    txt = pathlib.Path(p).read_text()
    if p.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(txt)
    return json.loads(txt)


HTML = """<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>{title}</title>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>body{{margin:0;background:{paper};display:flex;justify-content:center;padding:2rem}} svg{{max-width:1200px;width:100%;height:auto}}</style>
</head><body>{svg}</body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("--out", default=None)
    ap.add_argument("--slug", default=None)
    ap.add_argument("--scale", type=float, default=2.0)
    a = ap.parse_args()
    spec = load_spec(a.spec)
    root = os.environ.get("RETRYVLA_ROOT")
    out = pathlib.Path(a.out or (pathlib.Path(root) / "runs/_analysis/diagrams" if root else "diagrams"))
    out.mkdir(parents=True, exist_ok=True)
    slug = a.slug or spec.get("slug") or pathlib.Path(a.spec).stem
    svg, _ = svg_doc(spec)
    (out / f"{slug}.svg").write_text(svg)
    (out / f"{slug}.html").write_text(HTML.format(title=escape(spec.get("title", slug)), paper=S.PAPER, svg=svg))
    (out / f"{slug}.spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False))
    ensure_fonts()
    import cairosvg
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(out / f"{slug}.png"), scale=a.scale)
    print((out / f"{slug}.png").resolve())


if __name__ == "__main__":
    main()
