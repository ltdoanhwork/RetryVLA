#!/usr/bin/env bash
# LRS-G debug lane A (2026-11-27): M0 loss / prediction diagnostic -> C3 dataset -> C3 SFT + val.
# usage: dbg_laneA.sh <gpu>
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1 TMPDIR=/tmp
G=$1
[ -f runs/_analysis/lrs_debug/m0_pred_lrs_goal_v2_ds.npz ] || python -m scripts.lrs.diag_m0_windows --rec data/recovery/lrs_goal_v2_ds --pred_only || exit 1
[ -f data/recovery/lrsdbg_c3_v2m0act/retryvla/trajectories.jsonl ] || python -m scripts.lrs.derive_ctrl_dataset \
  --from data/recovery/lrs_goal_v2_ds --out data/recovery/lrsdbg_c3_v2m0act \
  --actions runs/_analysis/lrs_debug/m0_pred_lrs_goal_v2_ds.npz || exit 1
bash scripts/lrs/dbg_run.sh $G lrsdbg_c3_s7 7 data/recovery/lrsdbg_c3_v2m0act
for x in "${@:2}"; do bash scripts/lrs/dbg_run.sh $G ${x//,/ }; done
echo "[dbg] LANE_DONE A $(date -Is)"
