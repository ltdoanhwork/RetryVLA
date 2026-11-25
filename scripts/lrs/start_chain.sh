#!/usr/bin/env bash
# start an LRS-G chain in tmux on the current shell. usage: start_chain.sh [chain_script (default lrsg_chain.sh)]
export PATH=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin:$PATH
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; C=${1:-lrsg_chain.sh}; N=rv_${C%.sh}
tmux has-session -t "=$N" 2>/dev/null && { echo SESSION_EXISTS; exit 2; }
tmux new-session -d -s "$N" "bash $R/scripts/lrs/$C > $R/runs/_logs/${N}_$(date +%Y%m%d-%H%M%S).log 2>&1"
echo LAUNCHED $N
