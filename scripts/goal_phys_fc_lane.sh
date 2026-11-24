#!/usr/bin/env bash
# Goal (b) G2-phys scan of the curated bank + (a) FC labelling of audit seeds 7, 8. CPU-heavy sim, light GPU (none
# needed besides EGL). usage: goal_phys_fc_lane.sh <gpu>
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1
[ -f data/recovery/goal_search_onset_cur/phys_summary.json ] || python -m retryvla.search.goal_phys \
  --bank data/recovery/goal_search_onset_cur --audit_dir data/goal_rsi/onset_s7 --seed 7 --workers 6 || exit 1
for S in 7 8; do [ -f data/goal_rsi/fc_s$S/summary.json ] || python -m retryvla.search.goal_fc --seed $S --workers 6 || exit 1; done
echo PHYS_FC_DONE
