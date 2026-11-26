#!/usr/bin/env bash
# E1 training lane (remote tmux): scripts/e1/train_lane.sh <gpu> <lane>
# Pulls jobs from the shared ordered list below; a job is claimed by an atomic mkdir of runs/e1/wm/<job>.lock
# (holds host:pid; a stale lock whose pid is gone on this host is reclaimed). Skip-if-DONE. Ends with LANE_DONE.
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1 TMPDIR=/tmp
G=$1; L=$2; W=runs/e1/wm; mkdir -p $W
JOBS=""
for s in 1 2 3; do for arm in A0 A2 A1 A4 A3 A2x3; do JOBS="$JOBS ${arm}_s$s"; done; done
for j in $JOBS; do
  [ -f $W/$j/DONE ] && continue
  if ! mkdir $W/$j.lock 2>/dev/null; then
    own=$(cat $W/$j.lock/owner 2>/dev/null); h=${own%%:*}; p=${own##*:}
    if [ "$h" = "$(hostname)" ] && [ -n "$p" ] && ! kill -0 $p 2>/dev/null; then echo "[lane $G.$L] reclaim stale $j"; else continue; fi
  fi
  echo "$(hostname):$$" > $W/$j.lock/owner
  arm=${j%_s*}; s=${j##*_s}
  echo "[lane $G.$L] START $j $(date -Is)"
  python -m retryvla.wm.train_wm_e1 --arm $arm --seed $s --out $W/$j > $W/$j.log 2>&1
  echo "[lane $G.$L] END $j rc=$? $(date -Is)"
  rm -rf $W/$j.lock
done
echo "[lane $G.$L] LANE_DONE $(date -Is)"
