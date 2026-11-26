#!/usr/bin/env bash
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1 TMPDIR=/tmp
[ -f runs/_analysis/e2/objdyn.pt ] || python -m scripts.e2.objdyn
python -m scripts.e2.selfexploit && python -m scripts.e1.replay --tag WMv1_objdyn --workers 8
echo "[e2x] E2X_DONE $(date -Is)"
