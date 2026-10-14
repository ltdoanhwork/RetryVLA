#!/usr/bin/env bash
# Wait for M1-v2 SFT checkpoints, then dispatch test-RSR + nominal evals on free GPUs (avoid-list respected).
set -o pipefail
unset LD_LIBRARY_PATH
cd "$(dirname "$0")/../.."
declare -A DONE=([_]=0); unset "DONE[_]"
ARMS="sft_random_r25 sft_random_r25_nontrivial"
TOTAL=4
while [ ${#DONE[@]} -lt $TOTAL ]; do
  for arm in $ARMS; do
    CK=$(ls -d runs/*_M1_${arm}/checkpoints/006000/pretrained_model 2>/dev/null | tail -1)
    [ -n "$CK" ] && [ -f "$CK/model.safetensors" ] || continue
    T=${arm#sft_}
    for kind in recover nominal; do
      key=${arm}_$kind; [ -n "${DONE[$key]:-}" ] && continue
      SLOT=$(python3 scripts/shells/find_free.py --json 2>/dev/null | python3 -c "import json,sys; s=json.load(sys.stdin); print(f\"{s[0]['short']} {s[0]['gpu']}\" if s else '')")
      [ -n "$SLOT" ] || continue
      set -- $SLOT
      if [ $kind = recover ]; then
        OUT=$(scripts/shells/launch_remote.sh $1 $2 ev_${T}_rec retryvla.eval.eval_recover configs/exp/m0_baseline_spatial.yaml milestone=M1 recover_tag=${T}_recover_test policy.path=$CK "recover_eval.splits=[test]" eval.n_workers=12 2>&1)
      else
        OUT=$(scripts/shells/launch_remote.sh $1 $2 ev_${T}_nom retryvla.eval.eval_nominal configs/exp/m0_baseline_spatial.yaml milestone=M1 nominal_tag=${T}_nominal policy.path=$CK eval.n_workers=12 eval.save_videos=none 2>&1)
      fi
      echo "$(date -Is) $key -> $SLOT: $(echo "$OUT" | grep -E 'LAUNCHED|REFUSED|fail' | head -1)"
      [[ "$OUT" == *LAUNCHED* ]] && DONE[$key]=1
    done
  done
  [ ${#DONE[@]} -lt $TOTAL ] && sleep 120
done
echo "AGENT_LOOP_WAKE_evals2 {\"prompt\":\"M1-v2 SFT evals dispatched\"}"
