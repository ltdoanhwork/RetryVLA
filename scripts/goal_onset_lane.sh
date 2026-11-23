#!/usr/bin/env bash
# Goal campaign stage 2: onset scan of M0's failed audit rollouts. usage: goal_onset_lane.sh <gpu> <seed> [seed ...]
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1; G=$1; shift
for S in "$@"; do
  [ -f data/goal_rsi/onset_s$S/summary.json ] && continue
  python -m retryvla.search.goal_onset --ckpt runs/goal_rsi/m0_s7/ckpt_04000 --seed $S --K 8 --need 2 --grid 20 --workers 12 || exit 1
done
