#!/usr/bin/env bash
# runs ON a V2R shell inside tmux rv_v2r_chain (started via scripts/shells/tmux_start.sh)
REPO=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; cd $REPO
source env/activate.sh >/dev/null 2>&1
export TMPDIR=/tmp
exec python scripts/v2r/chain.py
