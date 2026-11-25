#!/usr/bin/env bash
# V2R smoke: save_actions in both evals (tiny), then E-v1s replay check.
set -uo pipefail
REPO=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; cd $REPO; source env/activate.sh >/dev/null 2>&1
export TMPDIR=/tmp CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1
P=runs/20261016-042447_M5_m5c_sft_wm_diverse_rr/checkpoints/006000/pretrained_model
python -m retryvla.eval.eval_nominal configs/exp/m0_baseline_spatial.yaml milestone=V2R nominal_tag=v2r_smoke_nominal policy.path=$P \
  "nominal.task_ids=[0,5]" nominal.episodes_per_task=3 eval.n_workers=4 eval.save_actions=true eval.save_videos=none
D=$(ls -d runs/*_V2R_v2r_smoke_nominal | tail -1)
python -m retryvla.eval.rec_events --run_dir $D --workers 4
python -m retryvla.eval.eval_recover configs/exp/m0_baseline_spatial.yaml milestone=V2R recover_tag=v2r_smoke_recover policy.path=$P \
  "recover_eval.splits=[test]" recover_eval.max_scenes=2 eval.n_workers=4 eval.save_actions=true eval.save_videos=none
D=$(ls -d runs/*_V2R_v2r_smoke_recover | tail -1)
python -m retryvla.eval.rec_events --run_dir $D --kind recover --workers 4
