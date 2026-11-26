#!/usr/bin/env bash
# Local / remote watcher for G4: wait for shell 7485771f to be RUNNING (give up at 30 min -> marker file), envcheck,
# launch both lanes in remote tmux, then wait for LANE_DONE on both and kill the shell.
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; SID=7485771f-f716-4545-b602-f75d61198f1e; cd /tmp
L0="g2_wm5_s9:9:wm5 orig_s10:10:orig g2phys_wm_s10:10:phys g3physr_s10:10:physr g4sw5_s7:7:sw5 g4sw36_s7:7:sw36"
L1="g2_wm18_s9:9:wm18 orig_s11:11:orig g2phys_wm_s11:11:phys g3physr_s11:11:physr g4sw18_s7:7:sw18"
echo "$(date -u +%H:%M) waiting for $SID"
for i in $(seq 1 360); do
  st=$(env -u LD_LIBRARY_PATH det shell list 2>/dev/null | grep ${SID:0:8} | awk -F'|' '{print $4}' | tr -d ' ')
  [ "$st" = RUNNING ] && break
  [ $i -eq 6 ] && echo "$(date -u +%H:%M) QUEUED_30MIN" > $R/runs/goal_rsi/g4_queued_30min
  sleep 300
done
echo "$(date -u +%H:%M) shell $st"
[ "$st" = RUNNING ] || exit 1
sleep 60
for g in 0 1; do env -u LD_LIBRARY_PATH det shell open $SID -- env CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g bash $R/scripts/shells/envcheck.sh 2>&1 | grep -E "^OK"; done
env -u LD_LIBRARY_PATH det shell open $SID -- bash $R/scripts/shells/tmux_start.sh rv_g4_g0 scripts/goal_g4_lane.sh 0 $L0 2>&1 | grep LAUNCHED
env -u LD_LIBRARY_PATH det shell open $SID -- bash $R/scripts/shells/tmux_start.sh rv_g4_g1 scripts/goal_g4_lane.sh 1 $L1 2>&1 | grep LAUNCHED
echo "$(date -u +%H:%M) LAUNCHED" > $R/runs/goal_rsi/g4_launched
while true; do
  sleep 600
  n=$(grep -l "LANE_DONE" $(ls -t $R/runs/_logs/rv_g4_g[01]_*.log 2>/dev/null | head -2) 2>/dev/null | wc -l)
  [ "$n" -ge 2 ] && break
done
echo "$(date -u +%H:%M) both lanes done -> kill $SID"; env -u LD_LIBRARY_PATH det shell kill $SID
