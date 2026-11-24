#!/usr/bin/env bash
# Extra evals for existing ckpts (Original, M0): val with saved actions (into eval_val_ev, separate dir; same protocol)
# -> events E-v1; E8on RSR. usage: goal_eval_only.sh <gpu> <ckpt>
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1; C=$2
[ -f $C/eval_val_ev/events_summary.json ] || { python -m retryvla.eval.eval_goal --ckpt $C --split val --workers 8 --save_actions --out $C/eval_val_ev && python -m retryvla.eval.goal_events --eval_dir $C/eval_val_ev --workers 8; }
python -m retryvla.eval.eval_goal_entries --ckpt $C --entries data/goal_rsi/onset_s8/entries_s8.jsonl --audit_dir data/goal_rsi/onset_s8 --tag e8on --workers 8
[ -f $C/eval_entries_e8/summary.json ] || python -m retryvla.eval.eval_goal_entries --ckpt $C --entries data/goal_rsi/audit_m0_s8/entries_s8.jsonl --audit_dir data/goal_rsi/audit_m0_s8 --tag e8 --workers 8
echo "[g2] RUN_DONE eval_only $C"
