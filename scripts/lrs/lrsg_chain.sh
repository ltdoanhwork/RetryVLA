#!/usr/bin/env bash
# LRS-G chain on the 2-GPU shell (tmux rv_lrsg_chain): collection lanes -> datasets -> demos -> SFT x 3 seeds + evals.
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
D=data/recovery/lrs_goal
until [ "$(ls $D/a0/l*/DONE 2>/dev/null | wc -l)" -ge 4 ]; do sleep 60; done
echo "[lrsg] collection done $(date -Is)"
export MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0
[ -f data/recovery/lrs_goal_ds/meta/info.json ] || python -m scripts.lrs.build_lrs_dataset --out data/recovery/lrs_goal_ds --target_anchors 1639 --workers 8 || exit 1
[ -f data/recovery/lrs_goal_ds_all/meta/info.json ] || python -m scripts.lrs.build_lrs_dataset --out data/recovery/lrs_goal_ds_all --workers 8 || exit 1
echo "[lrsg] datasets done $(date -Is)"
# demos: false-positive continuations (GPU0, ~5 min) then render on CPU-light EGL in the background
[ -f runs/_analysis/demos/goal/fp/rerun_report.json ] || python -m scripts.demos.goal_fp_rollout
( python -m scripts.demos.render_goal_demos > runs/_logs/rv_lrsg_render_$(date +%H%M%S).log 2>&1 ) &
# trainer smoke (masked path, 3 steps) before committing 2 x 2.4 h of GPU
if [ ! -f runs/goal_rsi/_lrs_smoke/SMOKE_OK ]; then
  torchrun --nproc_per_node 1 --master_port 29911 -m retryvla.train.pi0_sft --out runs/goal_rsi/_lrs_smoke \
    --init runs/goal_rsi/m0_s7/ckpt_04000 --steps 3 --save_every 1000000 --bs 48 --accum 3 --warmup 200 --decay_steps 2000 \
    --seed 7 --save_dtype bf16 --recovery data/recovery/lrs_goal_ds --masked_recovery --l_min 5 2>&1 | grep -v Loading | tail -20
  [ -f runs/goal_rsi/_lrs_smoke/DONE ] || { echo "[lrsg] SMOKE_FAILED"; exit 1; }
  rm -rf runs/goal_rsi/_lrs_smoke/ckpt_*; touch runs/goal_rsi/_lrs_smoke/SMOKE_OK
fi
R=scripts/lrs/lrs_run.sh
( bash $R 0 lrs_s7 7 data/recovery/lrs_goal_ds; bash $R 0 lrs_s9 9 data/recovery/lrs_goal_ds ) > runs/_logs/rv_lrsg_g0_$(date +%H%M%S).log 2>&1 &
( bash $R 1 lrs_s8 8 data/recovery/lrs_goal_ds; bash $R 1 lrs_all_s7 7 data/recovery/lrs_goal_ds_all ) > runs/_logs/rv_lrsg_g1_$(date +%H%M%S).log 2>&1 &
wait
echo "[lrsg] CHAIN_DONE $(date -Is)"
