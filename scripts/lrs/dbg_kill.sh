#!/usr/bin/env bash
# kill one rv_* tmux session + its process tree (TERM, 20 s, KILL). usage: dbg_kill.sh <rv_session>
export PATH="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin:$PATH"
S=$1; [[ "$S" == rv_* ]] || exit 1
P=$(tmux list-panes -t "=$S" -F '#{pane_pid}' 2>/dev/null) || { echo NO_SESSION; exit 0; }
desc() { for c in $(pgrep -P $1); do desc $c; echo $c; done; }
K="$(desc $P) $P"; kill -TERM $K 2>/dev/null; sleep 20; kill -KILL $K 2>/dev/null; tmux kill-session -t "=$S" 2>/dev/null; echo KILLED $S
