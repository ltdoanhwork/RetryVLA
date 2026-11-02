#!/usr/bin/env bash
# Part 4 chain (runs on the AGENT machine in tmux, CPU only: it only launches + polls jobs on the RetryVLA shell).
# futures datagen (already running as rv_af_dg) -> SFT arm 3 -> [SFT arm 5 | arm 3 RSR + nominal] -> arm 5 RSR + nominal
# -> C_pi embedding probe. Each job: scripts/shells/launch_remote.sh (claims keyed by session), retry_oom gate.
# usage: bash scripts/af_chain.sh <shell8> [seed]
set -uo pipefail
cd "$(dirname "$0")/.."
SH=$1; SEED=${2:-0}
L=runs/_logs/af_chain_$(date +%Y%m%d-%H%M%S).log
say() { echo "$(date -Is) $*" | tee -a "$L"; }

launch() {  # name module args... -> echoes log path
  local n=$1; shift
  for k in 1 2 3 4 5 6; do
    out=$(scripts/shells/launch_remote.sh "$SH" 0 "$n" "$@" 2>&1 | tr -d '\r' | grep -E "LAUNCHED|REFUSED|failed|not RUNNING" | tail -1)
    if [[ "$out" == *LAUNCHED* ]]; then echo "${out##*log=}"; return 0; fi
    say "launch $n failed ($out), retry $k in 300 s" >&2; sleep 300
  done
  return 1
}
wait_log() {  # log -> 0 when RV_DONE rc=0, 1 otherwise
  while :; do
    if grep -q "RV_DONE rc=0" "$1" 2>/dev/null; then return 0; fi
    if grep -q "RV_DONE rc=" "$1" 2>/dev/null; then return 1; fi
    sleep 120
  done
}
release() { python3 - "$1" <<'PY'
import json, subprocess, sys
d = json.load(open("scripts/shells/shells.json"))
for k, v in d.get("claims", {}).items():
    if v["tmux"] == "rv_" + sys.argv[1]:
        subprocess.run(["scripts/shells/remote_ctl.sh", "stop", k], capture_output=True)
PY
}
ckpt() { ls -d runs/*_P4_$1/checkpoints/006000/pretrained_model 2>/dev/null | tail -1; }
evals() {  # tag -> launches RSR + nominal, waits for both
  local t=$1 p; p=$(ckpt "$t")
  [ -n "$p" ] || { say "no checkpoint for $t"; return 1; }
  lr=$(launch "${t}_rec" retryvla.retry_oom --min_free_gb 6 retryvla.eval.eval_recover configs/exp/m0_baseline_spatial.yaml \
       milestone=P4 recover_tag=${t}_recover_test policy.path=$p "recover_eval.splits=[test]" eval.n_workers=5 \
       eval.save_videos=none eval.worker_retries=2)
  ln=$(launch "${t}_nom" retryvla.retry_oom --min_free_gb 6 retryvla.eval.eval_nominal configs/exp/m0_baseline_spatial.yaml \
       milestone=P4 nominal_tag=${t}_nominal policy.path=$p eval.n_workers=3 eval.save_videos=none eval.worker_retries=2)
  say "$t evals: $lr $ln"
  wait_log "$lr"; r1=$?; release "${t}_rec"
  wait_log "$ln"; r2=$?; release "${t}_nom"
  say "$t evals done rc $r1 $r2"
}
sft() {  # config tag -> launches SFT, echoes log
  launch "$2" retryvla.retry_oom --min_free_gb 32 retryvla.train.sft "configs/train/$1.yaml" "tag=$2" \
         "train.seed=$SEED" train.save_every=6000
}

say "chain start shell=$SH seed=$SEED"
DG=$(ls -t runs/_logs/rv_af_dg_*.log | head -1)
say "waiting for futures datagen $DG"
wait_log "$DG" || { say "datagen FAILED"; exit 1; }
release af_dg
[ -f data/futures/af_m5b/futures.pt ] || { say "no futures.pt"; exit 1; }
T3=af_arm3_free_filtered_s$SEED; T5=af_arm5_hybrid_s$SEED
if [ -n "$(ckpt "$T3")" ]; then say "sft $T3: checkpoint exists, skip"; else
  l3=$(sft af_arm3_free_filtered "$T3"); say "sft $T3: $l3"
  wait_log "$l3" || { say "sft $T3 FAILED"; exit 1; }
fi
release "$T3"
l5=$(sft af_arm5_hybrid "$T5"); say "sft $T5: $l5"
evals "$T3" &
E3=$!
wait_log "$l5" || say "sft $T5 FAILED"
release "$T5"
wait $E3  # <= 8 sim workers on the shared shell: arm 5 evals start after arm 3 evals
evals "$T5" &
E5=$!
P1=runs/20261016-042447_M5_m5c_sft_wm_diverse_rr/checkpoints/006000/pretrained_model
PN=runs/20261013-080655_M1_sft_nominal_only/checkpoints/006000/pretrained_model
lp=$(launch af_probe retryvla.retry_oom --min_free_gb 6 scripts.af_probe --futures data/futures/af_m5b/futures.pt \
     --out runs/_analysis/af_probe_s$SEED --policy pi1_nominal=$PN --policy pi1=$P1 \
     --policy arm3=$(ckpt "$T3") --policy arm5=$(ckpt "$T5"))
say "probe: $lp"
wait_log "$lp"; say "probe rc $?"; release af_probe
wait $E5
say "chain DONE"
