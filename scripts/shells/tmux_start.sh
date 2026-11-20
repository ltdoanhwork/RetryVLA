#!/usr/bin/env bash
# Runs ON a shell: start tmux session <sess> running `bash <script> <args...>`, log to runs/_logs/<sess>_<host>_<ts>.log.
# Plain args only (det re-joins argv). Refuses to start if the session exists. Only rv_* names.
set -euo pipefail
REPO=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA
TM=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin/tmux
S=$1; shift
case "$S" in rv_*) ;; *) echo "REFUSE non-rv session $S"; exit 2;; esac
$TM has-session -t "=$S" 2>/dev/null && { echo "SESSION_EXISTS $S"; exit 2; }
LOG=$REPO/runs/_logs/${S}_$(hostname)_$(date +%Y%m%d-%H%M%S).log
$TM new-session -d -s "$S" "bash $REPO/$* > $LOG 2>&1; echo RV_DONE rc=\$? >> $LOG"
echo "LAUNCHED $S log=$LOG"
