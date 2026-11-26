#!/usr/bin/env bash
# E1 remote watchdog (tmux rv_e1_watch on the E1 shell): every 10 min, relaunch any E1 lane whose tmux session is
# gone without its end marker (<= 3 relaunches per lane), kill a lane whose job log has not been written for 45 min
# (stall), and write runs/e1/CHAIN_DONE when post lane says ALL_DONE and E2 says E2_DONE.
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; TM=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin/tmux
cd $R
declare -A N
spec() { case $1 in rv_e1_tr*) g=${1:8:1}; l=${1:9:1}; echo "scripts/e1/train_lane.sh $g $l";;
  rv_e1_post) echo "scripts/e1/post_lane.sh 1";; rv_e1_e2) echo "scripts/e1/e2_lane.sh 1";; esac; }
mark() { case $1 in rv_e1_tr*) echo LANE_DONE;; rv_e1_post) echo ALL_DONE;; rv_e1_e2) echo E2_DONE;; esac; }
while true; do
  for s in rv_e1_tr00 rv_e1_tr01 rv_e1_tr02 rv_e1_tr10 rv_e1_tr11 rv_e1_tr12 rv_e1_post rv_e1_e2; do
    lg=$(ls -t runs/_logs/${s}_*.log 2>/dev/null | head -1)
    [ -n "$lg" ] && grep -q "$(mark $s)" $lg && continue
    if ! $TM has-session -t =$s 2>/dev/null; then
      n=${N[$s]:-0}; [ $n -ge 3 ] && continue; N[$s]=$((n + 1))
      echo "$(date -Is) relaunch $s ($((n + 1)))"; bash scripts/shells/tmux_start.sh $s $(spec $s)
      continue
    fi
    # stall: newest of the lane log and the job logs it writes
    newest=$(ls -t $lg runs/e1/wm/*.log data/wm/e1/picks/* data/wm/e1/true/* runs/_logs/rv_e1_post_*.log 2>/dev/null | head -1)
    if [ $s != rv_e1_post ] && [ $s != rv_e1_e2 ]; then
      cur=$(grep START $lg | tail -1 | awk '{print $4}'); [ -n "$cur" ] && newest=runs/e1/wm/$cur.log
    fi
    if [ -n "$newest" ] && [ $(( $(date +%s) - $(stat -c %Y $newest) )) -gt 2700 ]; then
      echo "$(date -Is) STALL $s ($newest), killing"
      for p in $($TM list-panes -t =$s -F '#{pane_pid}' 2>/dev/null); do kill -TERM -- -$(ps -o pgid= -p $p | tr -d ' ') 2>/dev/null; done
      sleep 20; $TM kill-session -t =$s 2>/dev/null
    fi
  done
  pl=$(ls -t runs/_logs/rv_e1_post_*.log 2>/dev/null | head -1); el=$(ls -t runs/_logs/rv_e1_e2_*.log 2>/dev/null | head -1)
  if [ -n "$pl" ] && grep -q ALL_DONE $pl && [ -n "$el" ] && grep -q E2_DONE $el; then
    date -Is > runs/e1/CHAIN_DONE; echo "$(date -Is) CHAIN_DONE"; break
  fi
  sleep 600
done
