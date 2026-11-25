#!/usr/bin/env bash
echo "HOST $(hostname)"
echo "TMUX $(/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin/tmux ls 2>/dev/null | cut -d: -f1 | tr '\n' ' ')"
echo "GPUS $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | tr '\n' ';')"
echo "LOAD $(cut -d' ' -f1 /proc/loadavg) $(nproc)"
