#!/usr/bin/env bash
# RetryVLA-local adaptation (not upstream). Idempotent, no root, no GPU.
# Installs Playwright (python pkg) + chromium-headless-shell + the missing GTK a11y/X libs
# into $WS/tools/playwright (~530 MB on /pfss). Usage:
#   bash .cursor/skills/diagram-design/scripts/setup_playwright.sh
set -euo pipefail
WS=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge
T="${DD_PW_ROOT:-$WS/tools/playwright}"
PY="$WS/miniforge3/envs/retryvla/bin/python"
mkdir -p "$T"
if [ ! -d "$T/py/playwright" ]; then
  env -u LD_LIBRARY_PATH PIP_CONFIG_FILE=/dev/null "$PY" -m pip install -q --target "$T/py" playwright
fi
if ! ls -d "$T"/browsers/chromium_headless_shell-* >/dev/null 2>&1; then
  PYTHONPATH="$T/py" PLAYWRIGHT_BROWSERS_PATH="$T/browsers" env -u LD_LIBRARY_PATH \
    "$PY" -m playwright install chromium-headless-shell
  rm -rf "$T"/browsers/ffmpeg-*
fi
if [ ! -e "$T/libs/lib/libatk-bridge-2.0.so.0" ]; then
  env -u LD_LIBRARY_PATH "$WS/miniforge3/bin/mamba" create -y -q -p "$T/libs" -c conda-forge \
    at-spi2-atk atk-1.0 at-spi2-core xorg-libxcomposite xorg-libxdamage
fi
# smoke test
D="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$D/scripts/pw_env.sh"
"$DD_PY" "$D/scripts/export_png.py" "$D/assets/example-it-state.html" /tmp/dd-smoke.png 1 \
  2> >(grep -v -E 'dbus|^\s*$' >&2)
echo "OK playwright -> /tmp/dd-smoke.png"
