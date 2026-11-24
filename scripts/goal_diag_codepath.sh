#!/usr/bin/env bash
# F1 diagnosis D2 (pre-registered NOTES 2026-11-24): Original recipe (M0 + 2k on D0 only) through the NEW code path
# (single GPU accum 3, --resume_every 250, compact bf16 resume, forced restart at step 500, bf16 final), then val eval.
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1
O=runs/goal_rsi/diag_orig_newpath_s7; M0=runs/goal_rsi/m0_s7/ckpt_04000
A="--out $O --init $M0 --steps 2000 --save_every 2000 --bs 48 --accum 3 --warmup 200 --decay_steps 2000 --seed 7 --resume_every 250 --save_dtype bf16"
[ -f $O/DONE ] || [ -f $O/resume.pt ] || torchrun --nproc_per_node 1 --master_port 29911 -m retryvla.train.pi0_sft $A --stop_at 500
[ -f $O/DONE ] || torchrun --nproc_per_node 1 --master_port 29912 -m retryvla.train.pi0_sft $A || exit 1
[ -f $O/ckpt_02000/eval_val/summary.json ] || python -m retryvla.eval.eval_goal --ckpt $O/ckpt_02000 --split val --workers 8
