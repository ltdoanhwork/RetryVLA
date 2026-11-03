#!/usr/bin/env bash
# Second CPU shell, after its collect_v1 replay: verif_m5b frame ends in REVERSE file order (meets chain A in the middle).
set -uo pipefail
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA
source env/activate.sh
export CUDA_VISIBLE_DEVICES= LP_NUM_THREADS=1
while [ "$(ls data/foundation/frames/collect_v1 2>/dev/null | grep -c '\.pt$')" -lt 144 ]; do sleep 120; done
sleep 120
LOG=runs/_logs/rv_replayC_$(hostname)_$(date +%Y%m%d-%H%M%S).log
PYTHONUNBUFFERED=1 python -u -m scripts.replay_bridges --src data/wm/verif_m5b --cfg configs/exp/m5b_compare_arms.yaml \
  --kind m5b --no_dino --frames_out data/foundation/frames/verif_m5b --frame_cams agent --frame_ends --max_cands 32 \
  --reverse --workers ${W:-30} >>$LOG 2>&1
echo "[chain] RV_DONE rc=$? $(date -Is)" >>$LOG
