#!/usr/bin/env bash
# usage (in remote tmux via tmux_start.sh): scripts/e1/run_py.sh <gpu> <python args...>
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1 TMPDIR=/tmp; shift
python "$@"
