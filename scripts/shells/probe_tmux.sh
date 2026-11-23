#!/usr/bin/env bash
# Read-only: our tmux sessions + GPU memory / util on this shell (one line each).
echo "TMUX $(/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin/tmux ls 2>/dev/null | cut -d: -f1 | tr '\n' ' ')"
echo "GPUS $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' ';')"
