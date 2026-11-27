#!/usr/bin/env bash
# LRS-G debug watcher (local tmux, 2026-11-27): wait for own shell $1 RUNNING -> envcheck -> lanes A (GPU0) / B (GPU1)
# -> kill the shell when both lanes print LANE_DONE (unless scripts/shells/keep_<id8>).
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; ID=$1; S8=${ID:0:8}; cd /tmp; unset LD_LIBRARY_PATH
D="det"; log() { echo "[watch $(date -u +%FT%TZ)] $*"; }
until $D shell list 2>/dev/null | tr -d '\r' | grep "$S8" | grep -q RUNNING; do
  $D shell list 2>/dev/null | tr -d '\r' | grep "$S8" | grep -q TERMINATED && { log "shell terminated"; exit 1; }
  sleep 120; done
log RUNNING; sleep 30
timeout 300 $D shell open $ID -- bash $R/scripts/shells/envcheck.sh 0 2>&1 | tr -d '\r' | grep -E "OK|FAIL"
timeout 120 $D shell open $ID -- bash $R/scripts/lrs/dbg_tmux.sh rv_lrsdbg_A $R/scripts/lrs/dbg_laneA.sh 0 \
  "lrs2gt_s8,8,data/recovery/lrs_goal_v2gt_ds" 2>&1 | tr -d '\r' | grep LAUNCHED
timeout 120 $D shell open $ID -- bash $R/scripts/lrs/dbg_tmux.sh rv_lrsdbg_B $R/scripts/lrs/dbg_laneB.sh 1 \
  "lrsdbg_c1_s7,7,data/recovery/lrsdbg_c1_d0copy" "lrsdbg_c2_s7,7,data/recovery/lrsdbg_c2_v2half" 2>&1 | tr -d '\r' | grep LAUNCHED
while :; do
  sleep 600
  $D shell list 2>/dev/null | tr -d '\r' | grep "$S8" | grep -q RUNNING || { log "shell gone"; exit 1; }
  A=$(grep -l "LANE_DONE A" $R/runs/_logs/rv_lrsdbg_A_*.log 2>/dev/null | xargs -r ls -t | head -1)
  B=$(grep -l "LANE_DONE B" $R/runs/_logs/rv_lrsdbg_B_*.log 2>/dev/null | xargs -r ls -t | head -1)
  for x in $R/runs/_logs/rv_lrsdbg_[AB]_*.log; do :; done
  log "A=${A:-running} B=${B:-running}"
  if [ -n "$A" ] && [ -n "$B" ] && [ ! -f $R/scripts/shells/keep_$S8 ]; then log "both lanes done -> kill $S8"; $D shell kill $ID; exit 0; fi
done
