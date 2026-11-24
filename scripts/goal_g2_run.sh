#!/usr/bin/env bash
# Goal G2 / P2-3: one distillation run + all evals (resumable). usage: goal_g2_run.sh <gpu> <name> <seed> <recovery...>
# <recovery> items are passed to pi0_sft --recovery ("glob" or "glob@N"); "none" = Original recipe (D0 only).
# Evals: val (their protocol, --save_actions) -> failure events E-v1; own-failure RSR on E8 (168 seed-8 fixed entries)
# and E8on (53 seed-8 onset entries).
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1
G=$1; N=$2; S=$3; shift 3
O=runs/goal_rsi/$N; M0=runs/goal_rsi/m0_s7/ckpt_04000; C=$O/ckpt_02000
REC=""; [ "$1" != none ] && REC="--recovery $*"
if [ ! -f $O/DONE ]; then
  torchrun --nproc_per_node 1 --master_port $((29700 + 10 * G + S + RANDOM % 50)) -m retryvla.train.pi0_sft --out $O --init $M0 \
    --steps 2000 --save_every 2000 --bs 48 --accum 3 --warmup 200 --decay_steps 2000 --seed $S \
    --resume_every 250 --save_dtype bf16 $REC || exit 1
fi
[ -f $C/eval_val/summary.json ] || python -m retryvla.eval.eval_goal --ckpt $C --split val --workers 8 --save_actions || exit 1
[ -f $C/eval_val/events_summary.json ] || python -m retryvla.eval.goal_events --eval_dir $C/eval_val --workers 8
python -m retryvla.eval.eval_goal_entries --ckpt $C --entries data/goal_rsi/audit_m0_s8/entries_s8.jsonl \
  --audit_dir data/goal_rsi/audit_m0_s8 --tag e8 --workers 8
python -m retryvla.eval.eval_goal_entries --ckpt $C --entries data/goal_rsi/onset_s8/entries_s8.jsonl \
  --audit_dir data/goal_rsi/onset_s8 --tag e8on --workers 8
echo "[g2] RUN_DONE $N"
