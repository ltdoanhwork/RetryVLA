#!/usr/bin/env bash
set -uo pipefail
REPO=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; cd $REPO; source env/activate.sh >/dev/null 2>&1
export TMPDIR=/tmp CUDA_VISIBLE_DEVICES=$1 MUJOCO_EGL_DEVICE_ID=$1
SB=runs/v2r/smoke_bank; rm -rf $SB; mkdir -p $SB/s0/p0
cp -r data/recovery/m5b/s0/p0/wm_diverse $SB/s0/p0/   # small copy (520 MB)
python - <<'P'
import json
p="runs/v2r/smoke_bank/s0/p0/wm_diverse/retryvla/trajectories.jsonl"
L=open(p).read().splitlines()[:6]; open(p+".all","w").write("\n".join(L)+"\n")
P
mv $SB/s0/p0/wm_diverse/retryvla/trajectories.jsonl.all $SB/s0/p0/wm_diverse/retryvla/trajectories.jsonl
echo "== phys"; python -m retryvla.search.rec_phys --bank $SB --arm wm_diverse --workers 6 | tail -30
echo "== visual"; python scripts/v2r/visual_bank.py smoke "runs/v2r/smoke_bank/s*/p*/wm_diverse" - | tail -12
echo "== sft groups+aux 20 steps"
python -m retryvla.train.sft configs/exp/m5b_sft_wm_diverse.yaml milestone=V2R tag=v2r_smoke_sft train.steps=20 train.save_every=1000000 \
  train.num_workers=2 train.log_every=5 \
  "train.recovery_groups=[{datasets: ['runs/v2r/smoke_bank/s*/p*/wm_diverse'], n: 4, keep_list: ['runs/v2r/smoke_bank/phys_wm_diverse.jsonl'], rank: d3}]" \
  train.aux_beta=0.1 "train.aux_phys_dirs=['runs/v2r/smoke_bank/phys/wm_diverse']" 2>&1 | grep -E "group|aux|step|recovery|Error|error" | tail -15
echo "== dp 1 group"
python -m retryvla.search.rec_prevent --shard 0 --num_shards 1 --max_groups 1 --workers 8 --out runs/v2r/smoke_dp 2>&1 | grep -vE "Warning|warn|Loading|Assets" | tail -8
