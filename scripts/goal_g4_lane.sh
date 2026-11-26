#!/usr/bin/env bash
# Goal G4 (PLAN 2026-11-26): one GPU lane, runs sequentially; each = M0 + 2k (their recipe) + val eval (+ E8 later).
# usage (inside remote tmux): goal_g4_lane.sh <gpu> <name:seed:recipe> ...
#   recipe: orig | wm5 | wm18 | phys | physr | sw5 | sw18 | sw36
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1; G=$1; shift
CUR="data/recovery/goal_search_onset_cur/s*/p*"; KL="data/recovery/goal_search_onset_cur/phys_wm_diverse.jsonl data/recovery/goal_search_onset_cur/phys_random.jsonl"
M0=runs/goal_rsi/m0_s7/ckpt_04000
for job in "$@"; do
  N=${job%%:*}; rest=${job#*:}; S=${rest%%:*}; RC=${rest#*:}
  case $RC in
    orig)  REC="";;
    wm5)   REC="--recovery $CUR/wm_diverse@5";;
    wm18)  REC="--recovery $CUR/wm_diverse@18";;
    phys)  REC="--recovery $CUR/wm_diverse@36 --keep_list data/recovery/goal_search_onset_cur/phys_wm_diverse.jsonl";;
    physr) REC="--recovery $CUR/wm_diverse@18 $CUR/random@28 --keep_list $KL";;
    sw5)   REC="--recovery $CUR/wm_diverse@2 $CUR/random@3 --keep_list $KL";;
    sw18)  REC="--recovery $CUR/wm_diverse@7 $CUR/random@11 --keep_list $KL";;
    sw36)  REC="--recovery $CUR/wm_diverse@14 $CUR/random@22 --keep_list $KL";;
  esac
  O=runs/goal_rsi/$N; C=$O/ckpt_02000
  echo "[g4] $N seed $S recipe $RC $(date -Is)"
  if [ ! -f $O/DONE ]; then
    torchrun --nproc_per_node 1 --master_port $((29600 + 10 * G + S)) -m retryvla.train.pi0_sft --out $O --init $M0 \
      --steps 2000 --save_every 2000 --bs 48 --accum 3 --warmup 200 --decay_steps 2000 --seed $S \
      --resume_every 250 --save_dtype bf16 $REC || { echo "[g4] FAILED $N"; continue; }
  fi
  [ -f $C/eval_val/summary.json ] || python -m retryvla.eval.eval_goal --ckpt $C --split val --workers 10 --save_actions
  echo "[g4] DONE $N $(python3 -c "import json;print(json.load(open('$C/eval_val/summary.json'))['successes'])" 2>/dev/null) $(date -Is)"
done
echo "[g4] LANE_DONE $(date -Is)"
