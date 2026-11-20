#!/usr/bin/env bash
# Runs ON a shell: kill OUR tmux sessions with the given exact rv_* names and their whole process trees
# (tmux kill-session alone orphaned torchrun children before). Never touches non-rv sessions / other processes.
TM=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin/tmux
desc() { for c in $(ps -o pid= --ppid "$1" 2>/dev/null); do desc "$c"; echo "$c"; done; }
for S in "$@"; do
  case "$S" in rv_*) ;; *) echo "REFUSE $S"; continue;; esac
  $TM has-session -t "=$S" 2>/dev/null || { echo "NOSESSION $S"; continue; }
  P=$($TM list-panes -t "=$S" -F '#{pane_pid}')
  PIDS="$(desc $P) $P"
  kill -TERM $PIDS 2>/dev/null; sleep 20; kill -KILL $PIDS 2>/dev/null
  $TM kill-session -t "=$S" 2>/dev/null
  echo "KILLED $S pids: $PIDS"
done
