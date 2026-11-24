#!/usr/bin/env bash
# F1 diagnosis runs (NOTES 2026-11-24). usage: goal_diag_runs.sh <gpu> <name>
#  D3 diag_f1wm_oldpath_s7 : F1 wm recipe (36 bridges, bridge-only) on the OLD code path (no resume, fp32 final)
#  D4 diag_d0dup_s7        : pooling control: D0 + 36 trajectories of D0 ITSELF as the "recovery" root (no segment filter)
#  D5 diag_f1wm_full_s7    : 36 wm trajectories, bridge + continuation anchors (segment filter off), old path
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1
N=$2; O=runs/goal_rsi/$N; M0=runs/goal_rsi/m0_s7/ckpt_04000
C="--out $O --init $M0 --steps 2000 --save_every 2000 --bs 48 --accum 3 --warmup 200 --decay_steps 2000 --seed 7"
case $N in
  diag_f1wm_oldpath_s7) X="--max_traj 36 --recovery_segments 0 --recovery data/recovery/goal_search/s*/p*/wm_diverse";;
  diag_d0dup_s7)        X="--max_traj 36 --recovery data/goal_rsi/d0_train";;
  diag_f1wm_full_s7)    X="--max_traj 36 --recovery data/recovery/goal_search/s*/p*/wm_diverse";;
esac
[ -f $O/DONE ] || torchrun --nproc_per_node 1 --master_port $((29920 + $1)) -m retryvla.train.pi0_sft $C $X || exit 1
[ -f $O/ckpt_02000/eval_val/summary.json ] || python -m retryvla.eval.eval_goal --ckpt $O/ckpt_02000 --split val --workers 8
