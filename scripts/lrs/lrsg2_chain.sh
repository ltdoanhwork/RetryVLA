#!/usr/bin/env bash
# LRS-G v2 chain (servo fix, 2026-11-25; tmux rv_lrsg2_chain on 1123d1ea): lane 3 after C0 training -> v2 dataset ->
# pre-registered gate (saturation < 2 %, rotation std > 0.01) -> SFT seeds 7 / 8 / 9 + evals.
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
D=data/recovery/lrs_goal_v2
until [ -f runs/goal_rsi/lrsdiag_c0_masked_nowin_s7/DONE ]; do sleep 60; done
bash scripts/lrs/collect_v2.sh 4 3 3 1
until [ "$(ls $D/a0/l*/DONE 2>/dev/null | wc -l)" -ge 4 ]; do sleep 60; done
echo "[lrsg2] collection done $(date -Is)"
export MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0
[ -f data/recovery/lrs_goal_v2_ds/meta/info.json ] || python -m scripts.lrs.build_lrs_dataset --src $D --out data/recovery/lrs_goal_v2_ds --target_anchors 1639 --workers 8 || exit 1
python - <<'PY' || { echo "[lrsg2] GATE_FAILED"; exit 1; }
import glob, numpy as np, pandas as pd, sys
df = pd.concat([pd.read_parquet(f, columns=["action"]) for f in glob.glob("data/recovery/lrs_goal_v2_ds/data/*/*.parquet")])
A = np.stack(df.action.to_numpy()); sat = float((np.abs(A[:, :3]) >= 0.99).mean()); rs = float(A[:, 3:6].std(0).min())
print(f"[lrsg2] gate: n {len(A)} sat {sat:.4f} rot_std_min {rs:.4f}")
sys.exit(0 if sat < 0.02 and rs > 0.01 else 1)
PY
R=scripts/lrs/lrs_run.sh
( bash $R 0 lrs2_s7 7 data/recovery/lrs_goal_v2_ds; bash $R 0 lrs2_s9 9 data/recovery/lrs_goal_v2_ds ) > runs/_logs/rv_lrsg2_g0_$(date +%H%M%S).log 2>&1 &
( until [ -f runs/goal_rsi/lrsdiag_c0_masked_nowin_s7/ckpt_02000/eval_entries_e8on/summary.json ]; do sleep 60; done
  bash $R 1 lrs2_s8 8 data/recovery/lrs_goal_v2_ds ) > runs/_logs/rv_lrsg2_g1_$(date +%H%M%S).log 2>&1 &
wait
echo "[lrsg2] CHAIN_DONE $(date -Is)"
