#!/usr/bin/env bash
# Read-only probe of the current Determined shell. Machine-readable lines:
#   HOST <container> <agent-node>
#   GPU <idx> <mem_used_MiB> <mem_total_MiB> <util_max_of_5_samples> <driver>
#   CPU <nproc> <load1>
#   TMUX <space-separated sessions>
#   PFSS ok|missing
REPO=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA
echo "HOST $(hostname) ${DET_AGENT_ID:-unknown}"
if command -v nvidia-smi >/dev/null; then
  declare -A UMAX
  for _ in 1 2 3 4 5; do
    while IFS=, read -r i u; do
      u=${u// /}; [ "${u:-0}" -gt "${UMAX[$i]:-0}" ] && UMAX[$i]=$u
    done < <(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)
    sleep 1
  done
  while IFS=, read -r i used tot drv; do
    echo "GPU ${i// /} ${used// /} ${tot// /} ${UMAX[${i// /}]:-0} ${drv// /}"
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,driver_version --format=csv,noheader,nounits)
fi
echo "CPU $(nproc) $(cut -d' ' -f1 /proc/loadavg)"
echo "TMUX $(/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin/tmux ls 2>/dev/null | cut -d: -f1 | tr '\n' ' ')"
[ -d "$REPO" ] && echo "PFSS ok" || echo "PFSS missing"
