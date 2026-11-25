#!/usr/bin/env bash
# V2R R1 search lane: bash scripts/v2r/search_lane.sh <gpu> <shard> [<shard> ...]   (resumable; marker DONE per shard)
set -uo pipefail
REPO=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; cd $REPO
source env/activate.sh >/dev/null 2>&1
export TMPDIR=/tmp CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1; shift
for k in "$@"; do
  M=data/recovery/v2r_r1/s$k/DONE
  [ -f $M ] && { echo "skip shard $k"; continue; }
  for try in 1 2 3; do
    python -m retryvla.search.compare_arms configs/exp/v2r_r1_search.yaml m5b.shard=$k m5b.num_shards=8 && { mkdir -p $(dirname $M); date > $M; break; }
    echo "shard $k try $try failed"; sleep 60
  done
done
