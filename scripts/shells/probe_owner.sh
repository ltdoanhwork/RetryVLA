#!/usr/bin/env bash
# Read-only owner-activity probe (runs ON a borrowed shell via `det shell open <id> -- bash probe_owner.sh`).
# Lines:
#   GPU <idx> <mem_used_MiB> <mem_total_MiB> <util_max_of_5_samples>
#   APP <gpu_uuid> <pid> <used_MiB> <ours 0|1>
#     NOTE (2026-11-22): nvidia-smi reports HOST pids inside a Determined container, so <ours> is unreliable for GPU apps
#     (the pid is not in this container's /proc). Processes of OTHER containers are not listed at all. Attribute GPU
#     memory per GPU instead: other-container use = mem_used - sum(listed apps) (see goal_f1_watch.py).
#   GUUID <idx> <uuid>
#   CPU <nproc> <load1>
#   PROC <pid> <cpu_pct_now> <ours 0|1> <cmd...>   (processes >= 20 % CPU in a 3 s top sample)
#   TMUX <sessions>
#   PFSS ok|missing
# "ours" = command line mentions RetryVLA / retryvla / torchrun, or the process descends from an rv_* tmux pane.
REPO=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA
TM=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin/tmux
declare -A OURS
for s in $($TM ls 2>/dev/null | cut -d: -f1 | grep '^rv_'); do
  for p in $($TM list-panes -t "=$s" -F '#{pane_pid}' 2>/dev/null); do
    for q in $p $(ps -o pid= --ppid $p 2>/dev/null); do OURS[$q]=1; done
  done
done
# descendants (3 more levels)
for _ in 1 2 3; do for p in "${!OURS[@]}"; do for q in $(ps -o pid= --ppid $p 2>/dev/null); do OURS[$q]=1; done; done; done
ours() { [ -n "${OURS[$1]}" ] && { echo 1; return; }; tr '\0' ' ' < /proc/$1/cmdline 2>/dev/null | grep -qiE 'retryvla|torchrun' && echo 1 || echo 0; }
declare -A UMAX
for _ in 1 2 3 4 5; do
  while IFS=, read -r i u; do u=${u// /}; [ "${u:-0}" -gt "${UMAX[$i]:-0}" ] && UMAX[$i]=$u; done \
    < <(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)
  sleep 1
done
while IFS=, read -r i used tot uuid; do
  i=${i// /}; echo "GPU $i ${used// /} ${tot// /} ${UMAX[$i]:-0}"; echo "GUUID $i ${uuid// /}"
done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,uuid --format=csv,noheader,nounits)
while IFS=, read -r uuid pid mem; do
  pid=${pid// /}; [ -z "$pid" ] && continue
  echo "APP ${uuid// /} $pid ${mem// /} $(ours $pid)"
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
echo "CPU $(nproc) $(cut -d' ' -f1 /proc/loadavg)"
top -b -n 2 -d 3 -w 200 | awk 'BEGIN{n=0} /^top -/{n++} n==2 && $1 ~ /^[0-9]+$/ && $9+0 >= 20 {print $1, $9, $12, $13, $14}' |
  while read -r pid cpu rest; do echo "PROC $pid $cpu $(ours $pid) $rest"; done
echo "TMUX $($TM ls 2>/dev/null | cut -d: -f1 | tr '\n' ' ')"
[ -d "$REPO" ] && echo "PFSS ok" || echo "PFSS missing"
