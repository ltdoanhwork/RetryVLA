"""Deterministic layout + routing for column flow diagrams."""
from __future__ import annotations

from functools import lru_cache

from PIL import ImageFont

import style as S

FONT_FILES = {
    ("sans", 600): "Geist-SemiBold.ttf", ("sans", 500): "Geist-Medium.ttf", ("sans", 400): "Geist-Regular.ttf",
    ("mono", 600): "GeistMono-SemiBold.ttf", ("mono", 500): "GeistMono-Medium.ttf", ("mono", 400): "GeistMono-Regular.ttf",
}


@lru_cache(maxsize=None)
def _font(fam, weight, size):
    return ImageFont.truetype(str(S.FONT_DIR / FONT_FILES[(fam, weight)]), size)


def text_w(txt, fam="sans", weight=400, size=12, spacing=0.0):
    return _font(fam, weight, size).getlength(txt) + spacing * max(len(txt) - 1, 0)


def label_w(txt):
    return text_w(txt, "mono", 600, 8, 0.64) + 12


def node_lines(n):
    """Lines under the title: red note lines first, then mono subtitle lines."""
    notes = n.get("note") or []
    subs = n.get("subtitle") or []
    notes = [notes] if isinstance(notes, str) else list(notes)
    subs = [subs] if isinstance(subs, str) else list(subs)
    return [(t, True) for t in notes] + [(t, False) for t in subs]


def node_h(n):
    return 56 + S.LINE_H * max(len(node_lines(n)) - 1, 0) if node_lines(n) else 44


def node_need_w(n):
    tx = 44 if n.get("icon") else 16
    w = text_w(n["title"], "sans", 600, 12)
    for t, _ in node_lines(n):
        w = max(w, text_w(t, "mono", 400, 10))
    return tx + w + 16


class Box:
    def __init__(self, n, col, row):
        self.n, self.col, self.row = n, col, row
        self.id = n["id"]
        self.kind = n.get("kind", "normal")
        self.h = node_h(n)
        self.x = self.y = self.w = 0.0

    @property
    def r(self):
        return self.x + self.w

    @property
    def b(self):
        return self.y + self.h

    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2


def place(spec):
    cols = spec["columns"]
    boxes, by_col = {}, []
    for ci, c in enumerate(cols):
        lst, row = [], 0
        for n in c["nodes"]:
            row = n.get("row", row)
            b = Box(n, ci, row)
            if b.id in boxes:
                raise SystemExit(f"duplicate node id {b.id}")
            boxes[b.id] = b
            lst.append(b)
            row += 1
        by_col.append(lst)
    n_rows = 1 + max(b.row for b in boxes.values())
    row_h = [max([b.h for b in boxes.values() if b.row == r] or [56]) for r in range(n_rows)]
    gap = spec.get("row_gap", S.ROW_GAP)
    gaps = spec.get("row_gaps") or [gap] * n_rows  # per-gap override (gap after row r)
    return cols, boxes, by_col, n_rows, row_h, gaps


def _spread(center, n, pitch=S.SLOT):
    return [center + pitch * (i - (n - 1) / 2) for i in range(n)]


def _classify(e, boxes, by_col):
    s, d = boxes[e["from"]], boxes[e["to"]]
    route = e.get("route")
    if s.col == d.col:
        rows = sorted(b.row for b in by_col[s.col])
        lo, hi = sorted((s.row, d.row))
        between = [r for r in rows if lo < r < hi]
        return "gutter" if (between or route == "gutter") else "v"
    if route in ("top", "bottom", "chan"):
        return route
    if d.col == s.col + 1:
        return "h"
    return "top" if d.col > s.col else "bottom"


def layout(spec):
    cols, boxes, by_col, n_rows, row_h, gaps = place(spec)
    edges = [dict(e) for e in spec.get("edges", [])]
    for e in edges:
        for k in ("from", "to"):
            if e[k] not in boxes:
                raise SystemExit(f"edge references unknown node '{e[k]}'")
        e["type"] = _classify(e, boxes, by_col)
    n_top = sum(e["type"] == "top" for e in edges)
    n_bot = sum(e["type"] == "bottom" for e in edges)

    header_h = 60 if spec.get("title") else 0
    zone_y = header_h + (40 if header_h else 52)
    top_space = max(S.TOP_SPACE, 18 + 6 * n_top)
    # rows
    row_y, y = [], zone_y + top_space
    for r in range(n_rows):
        row_y.append(y)
        y += row_h[r] + gaps[min(r, len(gaps) - 1)]
    last_bottom = row_y[-1] + row_h[-1]
    zone_h = last_bottom + max(28, 20 + 6 * n_bot) - zone_y
    # columns
    x = S.PAD
    zones = []
    for ci, c in enumerate(cols):
        need = max([node_need_w(b.n) for b in by_col[ci]] + [S.NODE_MIN_W,
                   text_w(c.get("name", ""), "mono", 400, 8, 1.12) + 16])
        cw = max(c.get("width", 0) - 2 * S.ZONE_INSET, need)
        zw = cw + 2 * S.ZONE_INSET
        zones.append(dict(x=x, y=zone_y, w=zw, h=zone_h, name=c.get("name", "")))
        for b in by_col[ci]:
            b.x, b.w, b.y = x + S.ZONE_INSET, cw, row_y[b.row]
        x += zw + S.ZONE_GAP
    width = x - S.ZONE_GAP + S.PAD

    # ---- attach-point allocation -------------------------------------------------
    sides = {}  # (node_id, side) -> list of (edge_idx, sort_key)

    def want(i, nid, side, key):
        sides.setdefault((nid, side), []).append((i, key))

    for i, e in enumerate(edges):
        s, d, t = boxes[e["from"]], boxes[e["to"]], e["type"]
        if t == "v":
            down = d.row > s.row
            want(i, s.id, "bottom" if down else "top", d.cx)
            want(i, d.id, "top" if down else "bottom", s.cx)
        elif t == "gutter":
            want(i, s.id, "right", d.cy)
            want(i, d.id, "right", s.cy)
        else:
            want(i, s.id, "right", -1e9 if t == "top" else (1e9 if t == "bottom" else d.cy))
            top_entry = t == "top" and min(b.row for b in by_col[d.col]) == d.row
            e["top_entry"] = top_entry
            if top_entry:
                want(i, d.id, "top", s.cx)
            else:
                want(i, d.id, "left", -1e9 if t == "top" else (1e9 if t == "bottom" else s.cy))
    pts = {}  # (edge_idx, node_id, side) -> coordinate along side
    for (nid, side), lst in sides.items():
        b = boxes[nid]
        lst.sort(key=lambda t: t[1])
        horiz = side in ("top", "bottom")
        pitch = 24 if horiz else S.SLOT
        center = b.cx if horiz else b.cy
        for (i, _), c in zip(lst, _spread(center, len(lst), pitch)):
            pts[(i, nid, side)] = c

    # ---- channels ------------------------------------------------------------------
    chan_l, chan_gap, chan_r = {}, {}, {}
    lane_top, lane_bot = [], []
    for i, e in enumerate(edges):
        s, d, t = boxes[e["from"]], boxes[e["to"]], e["type"]
        if t in ("chan", "h", "top", "bottom") and not e.get("top_entry"):
            chan_l.setdefault(d.col, []).append(i)
        if t in ("top", "bottom"):
            chan_gap.setdefault(s.col, []).append(i)
            (lane_top if t == "top" else lane_bot).append(i)
        if t == "gutter":
            chan_r.setdefault(s.col, []).append(i)

    # straight horizontals (adjacent columns, overlapping rows) need no channel
    straight = set()
    for i, e in enumerate(edges):
        if e["type"] == "h":
            s, d = boxes[e["from"]], boxes[e["to"]]
            yd = pts[(i, d.id, "left")]
            if s.y + 10 <= yd <= s.b - 10:
                straight.add(i)
    for c in chan_l:
        chan_l[c] = [i for i in chan_l[c] if i not in straight]

    xl = {}
    for c, lst in chan_l.items():
        comp_x = by_col[c][0].x
        lst.sort(key=lambda i: pts[(i, edges[i]["to"], "left")])
        for k, i in enumerate(lst):
            xl[i] = comp_x - 8 - 8 * (len(lst) - 1 - k)
    xg = {}
    for c, lst in chan_gap.items():
        base = zones[c]["x"] + zones[c]["w"] + S.ZONE_GAP / 2 if c < len(zones) - 1 else by_col[c][0].r + 10
        for i, xx in zip(lst, _spread(base, len(lst), 5)):
            xg[i] = xx
    xr = {}
    for c, lst in chan_r.items():
        for k, i in enumerate(lst):
            xr[i] = by_col[c][0].r + 8 + 5 * k
    yl = {i: zone_y + 12 + 6 * k for k, i in enumerate(lane_top)}
    yl.update({i: last_bottom + 14 + 6 * k for k, i in enumerate(lane_bot)})

    # ---- polylines + label anchors ------------------------------------------------
    for i, e in enumerate(edges):
        s, d, t = boxes[e["from"]], boxes[e["to"]], e["type"]
        if t == "v":
            down = d.row > s.row
            xs = pts[(i, s.id, "bottom" if down else "top")]
            xd = pts[(i, d.id, "top" if down else "bottom")]
            y0, y1 = (s.b, d.y) if down else (s.y, d.b)
            if abs(xs - xd) < 1:
                p = [(xs, y0), (xs, y1)]
            else:
                ym = (y0 + y1) / 2
                p = [(xs, y0), (xs, ym), (xd, ym), (xd, y1)]
            e["label_pos"] = ("right", xs, y0 + 8 if down else y0 - 24)
        elif t == "gutter":
            ys, yd = pts[(i, s.id, "right")], pts[(i, d.id, "right")]
            p = [(s.r, ys), (xr[i], ys), (xr[i], yd), (d.r, yd)]
            e["label_pos"] = ("on", xr[i], (ys + yd) / 2)
        else:
            ys = pts[(i, s.id, "right")]
            if i in straight:
                yd = pts[(i, d.id, "left")]
                p = [(s.r, yd), (d.x, yd)]
            elif t in ("h", "chan"):
                yd = pts[(i, d.id, "left")]
                p = [(s.r, ys), (xl[i], ys), (xl[i], yd), (d.x, yd)]
            else:
                p = [(s.r, ys), (xg[i], ys), (xg[i], yl[i])]
                if e.get("top_entry"):
                    xd = pts[(i, d.id, "top")]
                    p += [(xd, yl[i]), (xd, d.y)]
                else:
                    yd = pts[(i, d.id, "left")]
                    p += [(xl[i], yl[i]), (xl[i], yd), (d.x, yd)]
            x0, x1 = p[0][0], p[1][0]
            e["label_pos"] = ("above", (x0 + x1) / 2, p[0][1])
            if t == "bottom":  # loop-back: label sits on the long lane segment
                e["label_pos"] = ("on", (p[2][0] + p[3][0]) / 2, p[2][1])
        e["points"] = p

    content_bottom = zone_y + zone_h
    height = content_bottom + 64
    return dict(width=width, height=height, header_h=header_h, zones=zones,
                boxes=boxes, edges=edges, content_bottom=content_bottom)
