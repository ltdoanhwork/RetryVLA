#!/usr/bin/env bash
# kill ONLY the exact tmux session rv_v2r_chain (its task children run in their own sessions and survive; the
# restarted chain sees them via their claim files)
/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin/tmux kill-session -t =rv_v2r_chain 2>/dev/null; echo KILLED
