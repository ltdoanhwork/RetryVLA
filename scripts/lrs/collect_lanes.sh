#!/usr/bin/env bash
# LRS-G collection: one tmux lane per (gpu, lane) on the current shell. usage: collect_lanes.sh <num_lanes> <lanes_per_gpu> [attempt]
export PYTORCH_ALLOC_CONF=expandable_segments:True
NL=$1; PG=$2; AT=${3:-0}
for l in $(seq 0 $((NL - 1))); do
  g=$((l / PG))
  bash /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/scripts/shells/remote_tmux.sh rv_lrsg_a${AT}_l$l $g \
    retryvla.baselines.lrs_goal --lane $l --num_lanes $NL --attempt $AT
done
