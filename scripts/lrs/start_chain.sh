#!/usr/bin/env bash
# start the LRS-G chain in tmux on the current shell (det re-joins argv: no quoted snippets through det)
export PATH=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin:$PATH
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA
tmux has-session -t =rv_lrsg_chain 2>/dev/null && { echo SESSION_EXISTS; exit 2; }
tmux new-session -d -s rv_lrsg_chain "bash $R/scripts/lrs/lrsg_chain.sh > $R/runs/_logs/rv_lrsg_chain_$(date +%Y%m%d-%H%M%S).log 2>&1"
echo LAUNCHED rv_lrsg_chain
