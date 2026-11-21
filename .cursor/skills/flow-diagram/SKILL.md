---
name: flow-diagram
description: >-
  Answers questions about a flow, pipeline, data flow, workflow, process or architecture
  with a rendered diagram (PNG/SVG/HTML) in the diagram-design "it-state" style: grey canvas,
  labelled column panels, white icon cards, blue data-flow arrows, red pain-points, dashed
  external, red bottleneck, green storage, legend. Use whenever the user asks what a flow or
  pipeline looks like or how parts connect, e.g. "flow đang làm là gì", "luồng xử lý",
  "pipeline thế nào", "quy trình", "kiến trúc", "vẽ flow", "explain the pipeline",
  "how does data flow", "architecture of X", "what is the current workflow".
---

# Flow diagram (diagram-design it-state style)

When the user asks about a flow / pipeline / workflow / process / architecture, answer
with a rendered diagram + a short text explanation, not only prose or Mermaid.

## Workflow

1. **Ground it.** Read the relevant code / configs / docs so every card and arrow is real
   (no invented stages). Keep 2-4 columns, ≤ 5 cards per column, ≤ 14 edges.
2. **Write a spec** (YAML) at `runs/_analysis/diagrams/<slug>.yaml` (format below).
3. **Render** (RetryVLA env has `cairosvg` + Pillow + PyYAML):

   ```bash
   source env/activate.sh
   python .cursor/skills/flow-diagram/scripts/render.py runs/_analysis/diagrams/<slug>.yaml
   ```

   Writes `<slug>.png` (2x), `<slug>.svg`, `<slug>.html`, `<slug>.spec.json` to
   `$RETRYVLA_ROOT/runs/_analysis/diagrams/` (override with `--out DIR`, `--slug NAME`).
   Prints the absolute PNG path.
4. **Check it**: Read the PNG. If labels collide, arrows cross cards, or text is clipped,
   adjust the spec (reorder nodes, set `row`, `route`, shorten labels) and re-render.
5. **Answer**: embed `![<title>](/abs/path/runs/_analysis/diagrams/<slug>.png)`, then
   3-6 bullets: stages in order, what each arrow carries, where the pain-points /
   bottlenecks are and why. Mention the SVG/HTML path only if useful.

## Spec format

```yaml
slug: retryvla-pipeline              # output file stem
eyebrow: RetryVLA · recovery         # optional small mono caps line
title: World-model-guided recovery   # optional; omit for a bare diagram like the reference
subtitle: one-line takeaway          # optional
columns:                             # left -> right panels
  - name: COLLECTION                 # rendered as spaced mono caps on the panel border
    width: 300                       # optional min panel width (auto-fits text otherwise)
    nodes:                           # top -> bottom; same row index across columns aligns
      - id: pos                      # unique, referenced by edges
        icon: file                   # see icon list below
        title: Point-of-Sale Exports # bold sans
        subtitle: "nightly · CSV"    # mono; string or list of lines; use " · " separators
        note: no version control     # optional red mono line(s) above subtitle (the pain)
        kind: normal                 # normal | external | bottleneck | storage
        row: 0                       # optional explicit row (leave gaps to stagger)
edges:
  - from: pos
    to: drive
    label: CSV                       # short, uppercased automatically (≤ 8 chars ideal)
    kind: flow                       # flow (blue) | pain (red/orange) | neutral (grey)
    dashed: false                    # dashed stroke = optional / async / external hand-off
    route: auto                      # auto | top | bottom | chan | gutter (rarely needed)
legend: {neutral: "control / eval"}  # optional label overrides
legend_items: [flow, pain, external, bottleneck]   # optional explicit legend
row_gap: 34                          # optional
```

Semantics (match the reference):

| Spec | Visual | Use for |
|---|---|---|
| `kind: normal` | white card, ink border | a regular stage / component |
| `kind: external` | white card, dashed grey border | outside our control (benchmark, third-party data) |
| `kind: bottleneck` | light red fill, red border, red icon | the stage that leaks / is slow / is the risk |
| `kind: storage` | muted green card + title | stores, simulators, datasets used as ground truth |
| `edge kind: flow` | solid blue arrow, blue label | data moving forward |
| `edge kind: pain` | red arrow, red label | hand-off that carries the problem |
| `edge kind: neutral` | grey arrow | control / load / eval |

Routing (automatic): same column → vertical; next column → horizontal (straight when rows
overlap, otherwise via the channel left of the destination); skip-column forward → over the
top lane; backward (feedback loop) → under the bottom lane. Same column but non-adjacent →
`gutter` on the right. Arrows always end on a side that keeps the arrowhead visible.

## Icons

Stroked Tabler icons from diagram-design (`assets/icons.json`). Common ids:
`file database server desktop laptop cloud users user robot cube chart-line search alert
alert-triangle test test-pipe refresh sync rocket pipeline api queue cache bucket backup
log monitoring terminal lock key git-branch git-merge package stack-2 bolt braces csv excel
sql world internet gateway`. Unknown id → warns and falls back to `file`.

## Examples

- `examples/it-state.yaml` — reproduction of diagram-design `docs/screenshots/it-state.png`
  (rendered: `examples/it-state.png`).
- `examples/retryvla-pipeline.yaml` — RetryVLA failure → WM → critic → sim verify → bank → SFT → π1
  (rendered: `examples/retryvla-pipeline.png`). Start new specs by copying one of these.

## Rendering notes

- Pure Python: layout in `scripts/layout.py`, tokens in `scripts/style.py`, SVG + PNG in
  `scripts/render.py` (cairosvg). No browser needed. If `cairosvg` is missing:
  `pip install cairosvg` inside the RetryVLA env.
- Fonts: Geist + Geist Mono (OFL, bundled in `assets/fonts/`), auto-copied to
  `~/.local/share/fonts/geist` + `fc-cache` on first render. Text widths are measured with
  the same TTFs, so cards auto-size. Missing glyphs fall back to DejaVu.
- `runs/` is git-ignored; diagrams there are scratch artifacts. Copy into `docs/` or
  `reports/` if a diagram must be committed.

## Attribution

Visual language, color tokens and icons are from
[cathrynlavery/diagram-design](https://github.com/cathrynlavery/diagram-design)
(MIT, © 2025 Cathryn Lavery; icons: Tabler Icons, MIT). See `LICENSE-diagram-design.txt`
and `NOTICE.md`. For other diagram types (sequence, ER, swimlane, …) use that upstream skill.
