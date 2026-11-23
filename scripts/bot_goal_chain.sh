#!/usr/bin/env bash
# Back on Track (2610.19579) + ForkCast-style (2610.18650) test-time arms on the 2610.05151 LIBERO-Goal protocol.
# Pre-registered: experiments/PLAN.md "Back on Track + ForkCast-style test-time arms" (2026-11-23).
# Runs ON shell 92927808 (vla_RetryVLA_DoanhLT_bot1g, 1 GPU) inside tmux rv_bot_goal; resumable (done markers).
# usage: scripts/bot_goal_chain.sh [smoke]
set -u
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA
source env/activate.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
MODE=${1:-full}
M0=runs/goal_rsi/m0_s7/ckpt_04000
CPI=$(cat data/wm/goal_latest_critic.txt)
if [ "$MODE" = smoke ]; then
  D=data/bot/smoke_goal; R=runs/bot_goal_smoke; CX="--max_jobs 6 --max_steps 80"; EX="--max_jobs 6 --max_steps 120"
  TX="--steps1 200 --steps2 300 --members 2"
else
  D=data/bot/goal_m0; R=runs/bot_goal; CX=""; EX=""; TX=""
fi
mkdir -p $R
echo "[bot] host=$(hostname) mode=$MODE start=$(date -Is)"
# 1) on-policy M0 rollouts on the TRAIN inits (300 inits x policy seeds 8 9 10), latents every 5 steps
if [ ! -f $D/COLLECT_DONE ]; then
  python -m retryvla.baselines.bot.collect --suite goal --policy $M0 --out $D --policy_seeds 8 9 10 --workers 10 \
    --full_tokens $CX && touch $D/COLLECT_DONE || { echo "[bot] collect FAILED"; exit 1; }
fi
# 2) liveness ensemble + progress head + PCA basis (H 10, D 6; play data = our Goal WM's own training actions)
if [ ! -f $D/bot.pt ]; then
  python -m retryvla.baselines.bot.train --data $D --wm_link data/wm/goal_latest_wm.txt --out $D/bot.pt --horizon 10 \
    --max_steps_ep 300 --play_roots data/wm/goal_collect $TX || { echo "[bot] train FAILED"; exit 1; }
fi
BOT=$D/bot.pt
run() {  # name ckpt json
  [ -f $R/$1/summary.json ] && return 0
  echo "[bot] eval $1 $(date -Is)"
  python -m retryvla.eval.eval_goal --ckpt $2 --split val --workers 8 --out $R/$1 --bot "$3" $EX \
    || echo "[bot] eval $1 FAILED"
}
LIV='{"arm":"bot","value":"liveness","bot_ckpt":"'$BOT'","horizon":10,"execute":5}'
BCP='{"arm":"bot","value":"cpi","bot_ckpt":"'$BOT'","critic_ckpt":"'$CPI'","horizon":10,"execute":5}'
FPR='{"arm":"forkcast","value":"progress","bot_ckpt":"'$BOT'"}'
FCP='{"arm":"forkcast","value":"cpi","bot_ckpt":"'$BOT'","critic_ckpt":"'$CPI'"}'
run m0_bot_cpi $M0 "$BCP"
run m0_bot_liv $M0 "$LIV"
run m0_fc_cpi $M0 "$FCP"
run m0_fc_prog $M0 "$FPR"
# best RetryVLA / reference policies on this protocol + BoT (value conditioned on M0, as their "filter on top of
# Filtered-BC" reuses the nominal policy's value): Original s7 and RetryVLA rec_wm s7 (main comparison, F0)
run orig7_bot_cpi runs/goal_rsi/orig_s7/ckpt_02000 "$BCP"
run recwm7_bot_cpi runs/goal_rsi/rec_wm_s7/ckpt_02000 "$BCP"
run orig7_bot_liv runs/goal_rsi/orig_s7/ckpt_02000 "$LIV"
run recwm7_bot_liv runs/goal_rsi/rec_wm_s7/ckpt_02000 "$LIV"
echo "[bot] CHAIN_DONE $(date -Is)"
