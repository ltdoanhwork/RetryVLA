#!/usr/bin/env bash
# LRS-G v2-GT follow-up (PLAN 2026-11-25 09:15 UTC): sim-GT-filtered v2 windows -> SFT seed 7 on GPU1 after lrs2_s8.
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 MUJOCO_EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1
[ -f data/recovery/lrs_goal_v2gt_ds/meta/info.json ] || python -m scripts.lrs.build_lrs_dataset --src data/recovery/lrs_goal_v2 \
  --out data/recovery/lrs_goal_v2gt_ds --gt_only --workers 6 || exit 1
until [ -f runs/goal_rsi/lrs2_s8/ckpt_02000/eval_entries_e8on/summary.json ]; do sleep 60; done
bash scripts/lrs/lrs_run.sh 1 lrs2gt_s7 7 data/recovery/lrs_goal_v2gt_ds
echo "[lrsg2gt] CHAIN_DONE $(date -Is)"
