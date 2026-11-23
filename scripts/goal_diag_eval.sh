#!/usr/bin/env bash
# One val eval of an existing checkpoint (diagnostics), on the given GPU. usage: goal_diag_eval.sh <gpu> <ckpt_dir>
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1
python -m retryvla.eval.eval_goal --ckpt $2 --split val --workers 6
