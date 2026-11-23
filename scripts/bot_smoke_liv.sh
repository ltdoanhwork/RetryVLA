#!/usr/bin/env bash
# one-off smoke: liveness BoT arm + C^pi arm on a C^pi-known task (3 episodes, 150 steps)
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA
source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
B=data/bot/smoke_goal/bot.pt
python -m retryvla.eval.eval_goal --ckpt runs/goal_rsi/m0_s7/ckpt_04000 --split val --workers 3 --max_jobs 3 --max_steps 150 \
  --out runs/bot_goal_smoke/liv --bot '{"arm":"bot","value":"liveness","bot_ckpt":"'$B'","t_c":-100}'
