#!/usr/bin/env bash
P=$(pgrep -f "scripts/v2r/chain.py" | head -1); echo "PID $P"; cat /proc/$P/wchan; echo; ls -la /proc/$P/fd 2>/dev/null | tail -3
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/env/activate.sh >/dev/null 2>&1
which py-spy >/dev/null && py-spy dump --pid $P 2>&1 | head -40
