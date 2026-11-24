#!/usr/bin/env bash
# G2 / P2-3 smoke tests (1 entry each + detector on 4 episodes). usage: goal_g2_smoke.sh <gpu>
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1
set -x
python -m retryvla.search.compare_arms configs/exp/goal_search_onset_cur.yaml goal.max_entries=1 m5b.num_shards=1 \
  m5b.budget=2 m5b.out_root=data/recovery/_smoke_g2 m5b.verif_out=data/wm/_smoke_g2_verif eval.n_workers=6 tag=smoke_g2
python -m retryvla.search.goal_prevent --ckpt runs/goal_rsi/m0_s7/ckpt_04000 --seed 7 --n_vla 4 --workers 6 --shard 0 --num_shards 52
python -m retryvla.eval.eval_goal --ckpt runs/goal_rsi/orig_s7/ckpt_02000 --split val --max_jobs 4 --workers 4 --save_actions \
  --out runs/goal_rsi/_smoke_events
python -m retryvla.eval.goal_events --eval_dir runs/goal_rsi/_smoke_events --workers 4
echo SMOKE_DONE
