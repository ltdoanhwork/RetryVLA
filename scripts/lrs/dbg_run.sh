#!/usr/bin/env bash
# LRS-G debug 2026-11-27: one control run = their recipe (scripts/lrs/lrs_run.sh) + val eval (+ events) only.
# usage: dbg_run.sh <gpu> <name> <seed> <recovery_root> [entries 0|1]
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1 TMPDIR=/tmp
G=$1; N=$2; S=$3; R=$4; EN=${5:-0}
O=runs/goal_rsi/$N; M0=runs/goal_rsi/m0_s7/ckpt_04000; C=$O/ckpt_02000
echo "[dbg] $N start $(date -Is) gpu=$G host=$(hostname)"
if [ ! -f $O/DONE ]; then
  torchrun --nproc_per_node 1 --master_port $((29800 + 10 * G + S + RANDOM % 50)) -m retryvla.train.pi0_sft --out $O --init $M0 \
    --steps 2000 --save_every 2000 --bs 48 --accum 3 --warmup 200 --decay_steps 2000 --seed $S \
    --resume_every 250 --save_dtype bf16 --recovery $R --masked_recovery --l_min 5 || exit 1
fi
[ -f $C/eval_val/summary.json ] || python -m retryvla.eval.eval_goal --ckpt $C --split val --workers 8 --save_actions || exit 1
[ -f $C/eval_val/events_summary.json ] || python -m retryvla.eval.goal_events --eval_dir $C/eval_val --workers 8
if [ "$EN" = 1 ]; then
  [ -f $C/eval_entries_e8/summary.json ] || python -m retryvla.eval.eval_goal_entries --ckpt $C \
    --entries data/goal_rsi/audit_m0_s8/entries_s8.jsonl --audit_dir data/goal_rsi/audit_m0_s8 --tag e8 --workers 8
  [ -f $C/eval_entries_e8on/summary.json ] || python -m retryvla.eval.eval_goal_entries --ckpt $C \
    --entries data/goal_rsi/onset_s8/entries_s8.jsonl --audit_dir data/goal_rsi/onset_s8 --tag e8on --workers 8
fi
echo "[dbg] RUN_DONE $N $(date -Is)"
