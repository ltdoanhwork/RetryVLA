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
# output goes through a pipe + restartable grep: a /pfss EDQUOT on the log file then only loses log lines, it can no
# longer raise inside the job's print() (2026-11-22; same filter as remote_tmux.sh)
$TM new-session -d -s "$S" "bash $REPO/$* 2>&1 | { while :; do grep --line-buffered -v -E 'EGLError|egl_context|glCheckError' >> $LOG; [ \$? -eq 2 ] || break; sleep 10; done; }; echo RV_DONE rc=\${PIPESTATUS[0]} >> $LOG"
echo "LAUNCHED $S log=$LOG"
