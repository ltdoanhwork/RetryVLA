#!/usr/bin/env bash
# LRS-G diagnosis (2026-11-25): GPU0 masked-loss equivalence check then C1 (LRS windows, standard unmasked loss);
# GPU1 C0 (masked path, 0 teacher windows; must reproduce ~Original).
export PATH=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin:$PATH
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; L=$R/runs/_logs; T=$(date +%Y%m%d-%H%M%S)
tmux new-session -d -s rv_lrsd_g0 "cd $R; source env/activate.sh; export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1; python -m scripts.lrs.check_masked_loss > $L/rv_lrsd_check_$T.log 2>&1; bash scripts/lrs/lrs_run.sh 0 lrsdiag_c1_unmasked_s7 7 data/recovery/lrs_goal_ds 0 > $L/rv_lrsd_c1_$T.log 2>&1"
tmux new-session -d -s rv_lrsd_g1 "cd $R; bash scripts/lrs/lrs_run.sh 1 lrsdiag_c0_masked_nowin_s7 7 none 1 > $L/rv_lrsd_c0_$T.log 2>&1"
echo LAUNCHED
