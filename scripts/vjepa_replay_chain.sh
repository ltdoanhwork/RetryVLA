#!/usr/bin/env bash
# Replay chain on the CPU-only shell (bd8bdcf6, Mesa EGL): recorded bridges -> true latents / raw frames.
#   1. RQ3-sweep picks (slim logs): TRUE DINO trajectory (Step 1 high-budget breakdown) + agent/wrist frames
#   2. collect_v1 (WM v1 training data): agent frames every 5 steps (V-JEPA 2-AC fine-tune)
#   3. verif_r1 (critic v2 extra data, pilot instances): agent frames every 5 steps (V-JEPA critic)
#   4. verif_m5b (paired eval set; first 32 = random + wm_diverse picks): first + last agent frame (V-JEPA offline re-scoring, true-end control)
# usage (on a CPU shell): tmux new-session -d -s rv_replay bash scripts/vjepa_replay_chain.sh [A|B]
#   A = 1, 3, 4 (bd8bdcf6); B = 2 (second CPU shell 89995d8d)
set -uo pipefail
REPO=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA
cd $REPO
source env/activate.sh
export CUDA_VISIBLE_DEVICES=
# Mesa llvmpipe spawns nproc render threads per sim worker; 1 thread/worker is ~1.5x faster under 28 workers
export LP_NUM_THREADS=1
TS=$(date +%Y%m%d-%H%M%S)
LOG=runs/_logs/rv_replay${1:-A}_$(hostname)_$TS.log
W=${W:-28}
PART=${1:-A}
run() { echo "[chain] $(date -Is) START $*" >>$LOG; PYTHONUNBUFFERED=1 python -u -m scripts.replay_bridges "$@" >>$LOG 2>&1;
        echo "[chain] $(date -Is) END rc=$? $*" >>$LOG; }
if [ "$PART" = A ]; then
run --src data/wm/verif_rq3_sweep --cfg configs/exp/rq3_verifier_sweep.yaml --kind m5b --out data/wm/true_end_rq3 \
    --frames_out data/foundation/frames/rq3 --frame_cams agent --sources wm_bud_l0 wm_unc_l0 --workers $W
run --src data/wm/verif_r1 --cfg configs/exp/m5_wm_datagen.yaml --kind collect --no_exclude --no_dino \
    --frames_out data/foundation/frames/verif_r1 --frame_cams agent --frame_ends --workers $W
run --src data/wm/verif_m5b --cfg configs/exp/m5b_compare_arms.yaml --kind m5b --no_dino \
    --frames_out data/foundation/frames/verif_m5b --frame_cams agent --frame_ends --max_cands 32 --workers $W
else
run --src data/wm/collect_v1 --cfg configs/exp/m2_collect.yaml --kind collect --no_dino \
    --frames_out data/foundation/frames/collect_v1 --frame_cams agent --workers $W
fi
echo "[chain] RV_DONE $(date -Is)" >>$LOG
