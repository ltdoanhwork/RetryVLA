#!/usr/bin/env bash
# Wait for the pi1-random / pi1-CEM SFT checkpoints, then dispatch their test evals on free GPUs.
set -o pipefail
unset LD_LIBRARY_PATH
cd "$(dirname "$0")/../.."
declare -A TAG=([sft_random]=pi1rand [sft_oracle_cem]=pi1cem)
declare -A DONE=([_]=0); unset "DONE[_]"
while [ ${#DONE[@]} -lt 4 ]; do
  for arm in sft_random sft_oracle_cem; do
    CK=$(ls -d runs/*_M1_${arm}/checkpoints/006000/pretrained_model 2>/dev/null | tail -1)
    [ -n "$CK" ] && [ -f "$CK/model.safetensors" ] || continue
    for kind in recover nominal; do
      key=${arm}_$kind; [ -n "${DONE[$key]:-}" ] && continue
      SLOT=$(python3 scripts/shells/find_free.py --json 2>/dev/null | python3 -c "import json,sys; s=json.load(sys.stdin); print(f\"{s[0]['short']} {s[0]['gpu']}\" if s else '')")
      [ -n "$SLOT" ] || continue
      set -- $SLOT
      if [ $kind = recover ]; then
        OUT=$(scripts/shells/launch_remote.sh $1 $2 m1d_${TAG[$arm]}_rec retryvla.eval.eval_recover configs/exp/m0_baseline_spatial.yaml milestone=M1 recover_tag=${TAG[$arm]}_recover_test policy.path=$CK "recover_eval.splits=[test]" eval.n_workers=12 2>&1)
      else
        OUT=$(scripts/shells/launch_remote.sh $1 $2 m1d_${TAG[$arm]}_nom retryvla.eval.eval_nominal configs/exp/m0_baseline_spatial.yaml milestone=M1 nominal_tag=${TAG[$arm]}_nominal policy.path=$CK eval.n_workers=12 eval.save_videos=none 2>&1)
      fi
      echo "$(date -Is) $key -> $SLOT: $(echo "$OUT" | grep -E 'LAUNCHED|REFUSED|fail' | head -1)"
      [[ "$OUT" == *LAUNCHED* ]] && DONE[$key]=1
    done
  done
  [ ${#DONE[@]} -lt 4 ] && sleep 120
done
echo "AGENT_LOOP_WAKE_evals {\"prompt\":\"all 4 pi1 evals dispatched\"}"
