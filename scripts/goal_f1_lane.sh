#!/usr/bin/env bash
# 2610.05151 follow-up F1 (pre-registered RESULTS "main comparison", 2026-11-20): share-matched distillation.
# One GPU lane, runs given as <arm>:<seed> (arm wm -> wm_diverse bank, rand -> random bank), sequential, resumable
# (DONE / eval summary markers + trainer --resume_every). Recipe = the main comparison except: bridge segments only
# (--recovery_segments 0) and 36 bridges round-robin (~1,640 anchors = 4.1 % next to 38,240 D0 anchors).
# usage (on the borrowed shell, inside tmux): scripts/goal_f1_lane.sh <gpu> wm:7 wm:8 wm:9
set -u
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA
source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
G=$1; shift
export CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G
M0=runs/goal_rsi/m0_s7/ckpt_04000
echo "[f1] host=$(hostname) gpu=$G start=$(date -Is) runs: $*"
for R in "$@"; do
  A=${R%%:*}; S=${R##*:}; BANK=$([ "$A" = wm ] && echo wm_diverse || echo random)
  O=runs/goal_rsi/f1_${A}_s$S
  [ -f $O/ckpt_02000/eval_val/summary.json ] && { echo "[f1] $O done"; continue; }
  if [ ! -f $O/DONE ]; then
    echo "[f1] train $O $(date -Is)"
    torchrun --nproc_per_node 1 --master_port $((29800 + 10 * G + S)) -m retryvla.train.pi0_sft --out $O --init $M0 \
      --steps 2000 --save_every 2000 --bs 48 --accum 3 --warmup 200 --decay_steps 2000 --seed $S \
      --max_traj 36 --recovery_segments 0 --recovery "data/recovery/goal_search/s*/p*/$BANK" --resume_every 250 \
      || { echo "[f1] train $O FAILED rc=$?"; exit 1; }
  fi
  echo "[f1] eval $O $(date -Is)"
  python -m retryvla.eval.eval_goal --ckpt $O/ckpt_02000 --split val --workers 8 || { echo "[f1] eval FAILED"; exit 1; }
done
echo "[f1] LANE_DONE $(date -Is)"
