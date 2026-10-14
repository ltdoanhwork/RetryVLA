#!/usr/bin/env bash
# M2 -> M3 pipeline that waits for free GPUs (avoid-list respected) and runs stages in order:
#   collect (N shards) -> train_wm -> train_critic -> wm_search on hard instances (M shards)
# Resume-safe: collect skips finished instances; finished stages are detected via RV_DONE rc=0 in logs.
# usage: [ALLOWED_SLOTS="shell8:gpu ..."] scripts/shells/pipeline_m2m3.sh [collect_shards=4] [search_shards=2]
set -o pipefail
unset LD_LIBRARY_PATH
cd "$(dirname "$0")/../.."
NC=${1:-4}; NS=${2:-2}
LOG=runs/_logs/pipeline_m2m3.log; mkdir -p runs/_logs
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }

done_ok() { grep -a -q "RV_DONE rc=0" runs/_logs/rv_$1_*.log 2>/dev/null; }
failed()  { grep -a -q -E "RV_DONE rc=[1-9]" runs/_logs/rv_$1_*.log 2>/dev/null && ! done_ok "$1"; }
running() { python3 -c "import json,sys; c=json.load(open('scripts/shells/shells.json'))['claims']; sys.exit(0 if any(v['tmux']=='rv_$1' for v in c.values()) else 1)"; }
free_slot() { python3 scripts/shells/find_free.py --json 2>/dev/null | python3 -c "
import json,sys,os
allow = set(os.environ.get('ALLOWED_SLOTS','').split())
s = [x for x in json.load(sys.stdin) if not allow or f\"{x['short']}:{x['gpu']}\" in allow]
print(f\"{s[0]['short']} {s[0]['gpu']}\" if s else '')"; }
release() {  # release claim of a finished session
  key=$(python3 -c "import json; c=json.load(open('scripts/shells/shells.json'))['claims']; print(next((k for k,v in c.items() if v['tmux']=='rv_$1'), ''))")
  [ -n "$key" ] && scripts/shells/remote_ctl.sh stop "$key" >/dev/null 2>&1
}
launch() {  # launch <name> <module> <config> [args...]
  local name=$1; shift
  local slot; slot=$(free_slot); [ -n "$slot" ] || return 1
  set -- $slot "$name" "$@"
  local out; out=$(scripts/shells/launch_remote.sh "$@" 2>&1)
  say "launch $name on ${1}:${2}: $(echo "$out" | grep -E 'LAUNCHED|REFUSED|QUOTA|fail' | head -1)"
  [[ "$out" == *LAUNCHED* ]]
}
stage() {  # stage <name> -> 0 done, 1 not done (launched or waiting), 2 failed
  local name=$1; shift
  if done_ok "$name"; then release "$name"; return 0; fi
  if failed "$name" && ! running "$name"; then say "FAILED $name (see runs/_logs/rv_${name}_*.log)"; echo "AGENT_LOOP_WAKE_pipe {\"prompt\":\"pipeline stage $name failed\"}"; return 2; fi
  running "$name" || launch "$name" "$@"
  return 1
}

say "pipeline start: $NC collect shards, $NS search shards"
while true; do
  [ -f runs/_logs/PIPELINE_STOP ] && { say "stopped by PIPELINE_STOP"; exit 0; }
  all=0
  for i in $(seq 0 $((NC - 1))); do
    stage m2_collect_s$i retryvla.wm.collect configs/exp/m2_collect.yaml wm_collect.shard=$i wm_collect.num_shards=$NC eval.n_workers=14
    r=$?; [ $r -eq 2 ] && exit 1; [ $r -eq 0 ] && all=$((all + 1))
  done
  if [ $all -eq $NC ]; then
    stage m2_wm retryvla.wm.train_wm configs/exp/m2_train_wm.yaml; r=$?
    [ $r -eq 2 ] && exit 1
    if [ $r -eq 0 ]; then
      stage m2_critic retryvla.critic.train_critic configs/exp/m2_train_critic.yaml; r=$?
      [ $r -eq 2 ] && exit 1
      if [ $r -eq 0 ]; then
        sdone=0
        for i in $(seq 0 $((NS - 1))); do
          stage m3_search_s$i retryvla.search.wm_search configs/exp/m3_wm_search_hard.yaml m3.shard=$i m3.num_shards=$NS m3.out_root=data/recovery/m3_wm_search_hard_s$i eval.n_workers=14
          r=$?; [ $r -eq 2 ] && exit 1; [ $r -eq 0 ] && sdone=$((sdone + 1))
        done
        if [ $sdone -eq $NS ]; then say "ALL DONE"; echo "AGENT_LOOP_WAKE_pipe {\"prompt\":\"M2->M3 pipeline finished\"}"; exit 0; fi
      fi
    fi
  fi
  sleep 120
done
