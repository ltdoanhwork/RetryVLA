#!/usr/bin/env bash
# RetryVLA-local adaptation (not upstream). Rasterize a diagram-design HTML file to PNG.
#   bash .cursor/skills/diagram-design/scripts/render_png.sh <diagram.html> <out.png> [scale=2]
# 1) Playwright/Chromium (upstream export_png.py, exact fonts + CSS) if installed;
# 2) else fallback: upstream export_svg.py --system-fonts -> cairosvg (approximate CSS,
#    no webfonts). For flow / pipeline / architecture / it-state, prefer the flow-diagram
#    skill's YAML renderer instead (pure Python, no browser).
set -uo pipefail
SRC="$1"; OUT="$2"; SCALE="${3:-2}"
D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$D/scripts/pw_env.sh"
if "$DD_PY" -c "import playwright" 2>/dev/null && ls -d "$PLAYWRIGHT_BROWSERS_PATH"/chromium* >/dev/null 2>&1; then
  if "$DD_PY" "$D/scripts/export_png.py" "$SRC" "$OUT" "$SCALE" 2> >(grep -v -E 'dbus/bus|^\s*$' >&2); then
    echo "[playwright] $(readlink -f "$OUT")"; exit 0
  fi
  echo "[diagram-design] Playwright render failed -> cairosvg fallback" >&2
fi
TMP_SVG="$(mktemp --suffix=.svg)"
env -u LD_LIBRARY_PATH "$DD_PY" "$D/scripts/export_svg.py" --system-fonts "$SRC" "$TMP_SVG" >/dev/null || exit 1
env -u LD_LIBRARY_PATH "$DD_PY" -c "import cairosvg,sys; cairosvg.svg2png(url=sys.argv[1], write_to=sys.argv[2], scale=float(sys.argv[3]))" \
  "$TMP_SVG" "$OUT" "$SCALE" || exit 1
rm -f "$TMP_SVG"
echo "[cairosvg-fallback, approximate] $(readlink -f "$OUT")"
