#!/usr/bin/env bash
# V2R R1 search lane for the SECOND shell: takes over shard k from the first shell's lanes by writing s$k/DONE
# (their skip marker) + s$k/CLAIM2 before starting; FIN2 when finished. Only for shards nobody has started
# (no s$k/p* part yet). bash scripts/v2r/search_lane_b.sh <gpu> <shard> [...]
set -uo pipefail
REPO=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; cd $REPO
source env/activate.sh >/dev/null 2>&1
export TMPDIR=/tmp CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1; shift
for k in "$@"; do
  D=data/recovery/v2r_r1/s$k
  if [ -f $D/FIN2 ]; then continue; fi
  if [ ! -f $D/CLAIM2 ]; then
    if ls -d $D/p* >/dev/null 2>&1; then echo "shard $k already started by shell 1"; continue; fi
    mkdir -p $D; date > $D/CLAIM2; date > $D/DONE
  fi
  for try in 1 2 3; do
    python -m retryvla.search.compare_arms configs/exp/v2r_r1_search.yaml m5b.shard=$k m5b.num_shards=8 && { date > $D/FIN2; break; }
    echo "shard $k try $try failed"; sleep 60
  done
done
