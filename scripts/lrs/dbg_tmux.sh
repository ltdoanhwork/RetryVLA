#!/usr/bin/env bash
# LRS-G debug 2026-11-27: start a BASH job in tmux on the shell this runs on (local, or via `det shell open <id> --`).
# usage: dbg_tmux.sh <session rv_*> <script.sh> [args...]   (plain args only: det re-joins argv)
set -euo pipefail
REPO=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA
export PATH="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin:$PATH"
SESS=$1; shift
[[ "$SESS" == rv_* ]] || { echo "REFUSED: session must start with rv_"; exit 1; }
command -v tmux >/dev/null || { echo "NO_TMUX"; exit 3; }
if tmux has-session -t "=$SESS" 2>/dev/null; then echo "SESSION_EXISTS $SESS"; exit 2; fi
mkdir -p "$REPO/runs/_logs"
LOG="$REPO/runs/_logs/${SESS}_$(hostname)_$(date +%Y%m%d-%H%M%S).log"
tmux new-session -d -s "$SESS" "export TMPDIR=/tmp; bash $* > $LOG 2>&1; echo RV_DONE rc=\$? >> $LOG"
echo "LAUNCHED $SESS log=$LOG"
