#!/usr/bin/env bash
# Goal campaign stage 1b: own-failure RSR on E8 (168 seed-8 fixed entries) for a list of checkpoints (resumable).
# usage: goal_rsr_lane.sh <gpu> <ckpt> [ckpt ...]
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1; shift
for C in "$@"; do
  [ -f $C/eval_entries_e8/summary.json ] && continue
  python -m retryvla.eval.eval_goal_entries --ckpt $C --entries data/goal_rsi/audit_m0_s8/entries_s8.jsonl \
    --audit_dir data/goal_rsi/audit_m0_s8 --tag e8 --workers 10
done
