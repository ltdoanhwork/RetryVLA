#!/usr/bin/env bash
# E1 post lane (remote tmux): waits for pools, then for every finished WM (runs/e1/wm/<tag>/DONE, + shipped WMv1):
# imagination search (scripts.e1.search) -> bridge replay (scripts.e1.replay). Analysis when all 18 + WMv1 are done.
# usage: scripts/e1/post_lane.sh <gpu>
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1 TMPDIR=/tmp
W=runs/e1/wm; PK=data/wm/e1/picks; TR=data/wm/e1/true
TM=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin/tmux
while $TM has-session -t =rv_e1_pools 2>/dev/null; do sleep 60; done
echo "[post] pools ready ($(ls data/wm/e1/pools/*.pt | wc -l)) $(date -Is)"
while true; do
  todo=0
  for t in WMv1 $(for s in 1 2 3; do for a in A0 A2 A1 A4 A3 A2x3; do echo ${a}_s$s; done; done); do
    [ -f $TR/$t.pt ] && continue
    todo=$((todo + 1))
    if [ $t = WMv1 ]; then wm=shipped; else [ -f $W/$t/DONE ] || continue; wm=$W/$t/wm.pt; fi
    echo "[post] $t search $(date -Is)"
    python -m scripts.e1.search --wm $wm --tag $t || { echo "[post] search FAILED $t"; sleep 60; continue; }
    echo "[post] $t replay $(date -Is)"
    python -m scripts.e1.replay --tag $t --workers 10 || { echo "[post] replay FAILED $t"; sleep 60; continue; }
    echo "[post] $t DONE $(date -Is)"
  done
  [ $todo -eq 0 ] && break
  sleep 120
done
python -m scripts.e1.analyze > runs/e1/analyze.log 2>&1
echo "[post] ALL_DONE $(date -Is)"
