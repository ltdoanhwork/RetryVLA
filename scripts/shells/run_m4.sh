#!/usr/bin/env bash
# M4 (C5/C6): two arms, one GPU each, fixed slots. Per arm: SFT -> test RSR eval -> nominal eval.
# usage: scripts/shells/run_m4.sh <slotA shell8:gpu> <slotB shell8:gpu>
set -o pipefail
unset LD_LIBRARY_PATH
cd "$(dirname "$0")/../.."
SA=${1:-aa1cf25d:0}; SB=${2:-aa1cf25d:1}
LOG=runs/_logs/run_m4.log
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
done_ok() { grep -a -q "RV_DONE rc=0" runs/_logs/rv_$1_*.log 2>/dev/null; }
failed()  { grep -a -q -E "RV_DONE rc=[1-9]" runs/_logs/rv_$1_*.log 2>/dev/null && ! done_ok "$1"; }
claimed() { python3 -c "import json,sys; c=json.load(open('scripts/shells/shells.json'))['claims']; sys.exit(0 if any(v['tmux']=='rv_$1' for v in c.values()) else 1)"; }
release() { scripts/shells/remote_ctl.sh stop "$1" >/dev/null 2>&1; }

# arm := name slot sft_config
step() {  # step <session> <slot> <module> <args...>  -> 0 done, 1 in progress, 2 failed
  local s=$1 slot=$2; shift 2
  if done_ok "$s"; then return 0; fi
  if failed "$s"; then say "FAILED $s"; echo "AGENT_LOOP_WAKE_m4 {\"prompt\":\"M4 step $s failed\"}"; return 2; fi
  if ! claimed "$s"; then
    local out; out=$(scripts/shells/launch_remote.sh "${slot%%:*}" "${slot##*:}" "$s" "$@" 2>&1)
    say "launch $s on $slot: $(echo "$out" | grep -E 'LAUNCHED|REFUSED|QUOTA|fail' | head -1)"
  fi
  return 1
}

arm() {  # arm <A|B> <slot> <sft_cfg> <tag>
  local a=$1 slot=$2 cfg=$3 tag=$4
  step m4_sft_$a "$slot" retryvla.train.sft "$cfg"; r=$?; [ $r -ne 0 ] && return $r
  release "$slot"
  local ck; ck=$(ls -d runs/*_M4_${tag}/checkpoints/006000/pretrained_model 2>/dev/null | tail -1)
  [ -n "$ck" ] || { say "no checkpoint for $tag"; return 2; }
  step m4_rec_$a "$slot" retryvla.eval.eval_recover configs/exp/m0_baseline_spatial.yaml milestone=M4 \
       recover_tag=${tag}_recover_test policy.path=$ck "recover_eval.splits=[test]" eval.n_workers=14; r=$?
  [ $r -ne 0 ] && return $r
  release "$slot"
  step m4_nom_$a "$slot" retryvla.eval.eval_nominal configs/exp/m0_baseline_spatial.yaml milestone=M4 \
       nominal_tag=${tag}_nominal policy.path=$ck eval.n_workers=14 eval.save_videos=none; r=$?
  [ $r -ne 0 ] && return $r
  release "$slot"
  return 0
}

say "M4 start: A on $SA, B on $SB"
while true; do
  [ -f runs/_logs/M4_STOP ] && { say "stopped by M4_STOP"; exit 0; }
  arm A "$SA" configs/exp/m4_sft_A_base_nom_random.yaml sft_A_base_nom_random; ra=$?
  arm B "$SB" configs/exp/m4_sft_B_base_nom_random_wm.yaml sft_B_base_nom_random_wm; rb=$?
  { [ $ra -eq 2 ] || [ $rb -eq 2 ]; } && exit 1
  if [ $ra -eq 0 ] && [ $rb -eq 0 ]; then say "ALL DONE"; echo "AGENT_LOOP_WAKE_m4 {\"prompt\":\"M4 finished\"}"; exit 0; fi
  sleep 120
done
