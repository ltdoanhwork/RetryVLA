#!/usr/bin/env bash
export PATH=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin:$PATH
R=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA
tmux new-session -d -s rv_lrsd_v2check "cd $R; source env/activate.sh; export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1; python -m scripts.lrs.check_masked_loss --rec data/recovery/lrs_goal_v2_ds --ckpts runs/goal_rsi/m0_s7/ckpt_04000 --n 8 > $R/runs/_logs/rv_lrsd_v2check.log 2>&1"
echo LAUNCHED
