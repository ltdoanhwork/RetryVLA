#!/usr/bin/env bash
# LRS-G v2 (servo fix, 2026-11-25): collection lanes with --servo demo into data/recovery/lrs_goal_v2.
# usage: collect_v2.sh <num_lanes> <first_lane> <last_lane> <gpu>
export PATH=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin:$PATH PYTORCH_ALLOC_CONF=expandable_segments:True
NL=$1; L0=$2; L1=$3; G=$4
for l in $(seq $L0 $L1); do
  bash /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/scripts/shells/remote_tmux.sh rv_lrsg2_l$l $G \
    retryvla.baselines.lrs_goal --lane $l --num_lanes $NL --out data/recovery/lrs_goal_v2 --servo demo
done
