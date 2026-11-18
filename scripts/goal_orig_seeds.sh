#!/usr/bin/env bash
# 2610.05151 reverse direction: Original continuation (M0 + 2k on D0, their schedule) for extra seeds, + val eval.
# usage: scripts/goal_orig_seeds.sh <gpu> <seed> [seed ...]   (resumable: skips a seed whose eval exists)
set -u
cd "$(dirname "$0")/.."; source env/activate.sh >/dev/null 2>&1; export HF_HUB_OFFLINE=1
G=$1; shift
export CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G
M0=runs/goal_rsi/m0_s7/ckpt_04000
for S in "$@"; do
  O=runs/goal_rsi/orig_s$S
  if [ ! -f $O/DONE ]; then
    for try in 1 2; do
      torchrun --nproc_per_node 1 --master_port $((29600+S)) -m retryvla.train.pi0_sft --out $O --init $M0 \
        --steps 2000 --save_every 2000 --bs 48 --accum 3 --warmup 200 --decay_steps 2000 --seed $S && break
      sleep 60
    done
  fi
  [ -f $O/ckpt_02000/eval_val/summary.json ] || python -m retryvla.eval.eval_goal --ckpt $O/ckpt_02000 --split val --workers 8
done
