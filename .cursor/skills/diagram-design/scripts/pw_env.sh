# source .cursor/skills/diagram-design/scripts/pw_env.sh
# RetryVLA-local adaptation (not upstream): point Python + Chromium at the user-local
# Playwright install made by setup_playwright.sh. The container lacks libatk / libatspi /
# libXcomposite / libXdamage, so Chromium loads them (and a matching glib) from a small
# conda-forge prefix. Only use this env for rendering: it prepends that prefix to
# LD_LIBRARY_PATH.
_WS=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge
export DD_PW_ROOT="${DD_PW_ROOT:-$_WS/tools/playwright}"
export PLAYWRIGHT_BROWSERS_PATH="$DD_PW_ROOT/browsers"
export PYTHONPATH="$DD_PW_ROOT/py${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$DD_PW_ROOT/libs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# python with cairosvg/PyYAML (RetryVLA env); falls back to python3
if [ -x "$_WS/miniforge3/envs/retryvla/bin/python" ]; then
  export DD_PY="$_WS/miniforge3/envs/retryvla/bin/python"
else
  export DD_PY="$(command -v python3)"
fi
unset _WS
