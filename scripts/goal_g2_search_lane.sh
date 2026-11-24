#!/usr/bin/env bash
# G2 curated onset search, one shard. usage: goal_g2_search_lane.sh <gpu> <shard> <num_shards>
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1
ls data/recovery/goal_search_onset_cur/s$2/p*/CLOSED >/dev/null 2>&1 && exit 0
python -m retryvla.search.compare_arms configs/exp/goal_search_onset_cur.yaml m5b.shard=$2 m5b.num_shards=$3 eval.n_workers=10
