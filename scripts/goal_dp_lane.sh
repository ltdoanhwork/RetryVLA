#!/usr/bin/env bash
# P2-3 D_P generation, one shard. usage: goal_dp_lane.sh <gpu> <seed> <shard> <num_shards>
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1
[ -f data/recovery/goal_dp_s$2/sh$3/CLOSED ] && exit 0
python -m retryvla.search.goal_prevent --ckpt runs/goal_rsi/m0_s7/ckpt_04000 --seed $2 --n_vla 24 --K 3 --workers 8 \
  --shard $3 --num_shards $4
