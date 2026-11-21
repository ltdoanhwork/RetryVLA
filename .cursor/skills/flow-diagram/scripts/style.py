"""Design tokens copied from cathrynlavery/diagram-design (MIT), example-it-state.html."""
import pathlib

SKILL = pathlib.Path(__file__).resolve().parents[1]
FONT_DIR = SKILL / "assets" / "fonts"

PAPER = "#f5f5f5"
INK = "#2d3142"
MUTED = "#4f5d75"
SOFT = "#5a6580"
ACCENT = "#bf4520"   # pain-point / bottleneck
LINK = "#2e5aa8"     # data flow
STORAGE = "#7c8f6f"  # muted green ("survivor") card
WHITE = "#ffffff"

SANS = "Geist"
MONO = "Geist Mono"

# node kinds -> (fill, fill_opacity, stroke, stroke_opacity, stroke_w, dash, title_ink, icon_ink)
NODE = {
    "normal":     (WHITE, 1.0, INK, 1.0, 1.0, None, INK, INK),
    "external":   (WHITE, 1.0, MUTED, 1.0, 1.0, "4 3", INK, MUTED),
    "bottleneck": (ACCENT, 0.07, ACCENT, 1.0, 1.4, None, INK, ACCENT),
    "storage":    (STORAGE, 0.06, STORAGE, 0.45, 1.0, None, STORAGE, STORAGE),
}
# edge kinds -> (color, width)
EDGE = {"flow": (LINK, 1.2), "pain": (ACCENT, 1.4), "neutral": (MUTED, 1.0)}

LEGEND_LABELS = {
    "flow": "data flow", "pain": "pain-point", "neutral": "control",
    "dashed": "optional / async", "external": "external",
    "bottleneck": "bottleneck", "storage": "storage",
}

# geometry (px in viewBox units; reference viewBox is 960 wide)
PAD = 16          # canvas side padding
ZONE_GAP = 20     # gap between column panels
ZONE_INSET = 20   # node inset inside a panel
TOP_SPACE = 28    # panel top -> first row (holds the top lane)
ROW_GAP = 34      # vertical gap between rows
NODE_MIN_W = 176
NODE_RX = 6
ZONE_RX = 8
LINE_H = 16       # sub-line pitch inside a card
SLOT = 16         # spacing between attach points on one side
