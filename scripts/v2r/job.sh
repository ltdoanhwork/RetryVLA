#!/usr/bin/env bash
# V2R job wrapper (called by scripts/v2r/chain.py with CUDA_VISIBLE_DEVICES / MUJOCO_EGL_DEVICE_ID / V2R_NW set).
#   job.sh eval_nominal <tag> <policy>     -> runs/*_V2R_<tag>_nominal (+ actions, + E-v1s events)
#   job.sh eval_recover <tag> <policy>     -> runs/*_V2R_<tag>_recover_test (+ actions, + jerk)
#   job.sh sft <tag> <seed> [train overrides...]
#   job.sh dp <shard> | phys <bank> <arm> <glob|-> | visual <name> <root glob> <keep|->
set -uo pipefail
REPO=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA; cd $REPO
source env/activate.sh >/dev/null 2>&1
export TMPDIR=/tmp
NW=${V2R_NW:-11}
CFG=configs/exp/m0_baseline_spatial.yaml
newest() { ls -d runs/*_V2R_$1 2>/dev/null | tail -1; }
done_run() { python - "$1" <<'EOF'
import json,sys
try: print(json.load(open(sys.argv[1]+"/summary.json")).get("status"))
except Exception: print("none")
EOF
}
cmd=$1; shift
case $cmd in
  eval_nominal|eval_recover)
    TAG=$1; POL=$2
    if [ $cmd = eval_nominal ]; then SUF=nominal; else SUF=recover_test; fi
    D=$(newest ${TAG}_$SUF)
    if [ -z "$D" ] || [ "$(done_run $D)" != done ]; then
      if [ $cmd = eval_nominal ]; then
        python -m retryvla.eval.eval_nominal $CFG milestone=V2R nominal_tag=${TAG}_nominal policy.path=$POL \
          eval.n_workers=$NW eval.save_actions=true eval.worker_retries=2 || exit 1
      else
        python -m retryvla.eval.eval_recover $CFG milestone=V2R recover_tag=${TAG}_recover_test policy.path=$POL \
          "recover_eval.splits=[test]" eval.n_workers=$NW eval.save_actions=true eval.worker_retries=2 || exit 1
      fi
      D=$(newest ${TAG}_$SUF)
    fi
    if [ $cmd = eval_nominal ]; then
      python -m retryvla.eval.rec_events --run_dir $D --kind nominal --workers $NW || exit 1
    else
      python -m retryvla.eval.rec_events --run_dir $D --kind recover --no_events || exit 1
    fi ;;
  sft)
    TAG=$1; SEED=$2; shift 2
    D=$(newest $TAG)
    if [ -n "$D" ] && [ "$(done_run $D)" = done ] && [ -d $D/checkpoints/006000/pretrained_model ]; then echo "sft done $D"; exit 0; fi
    python -m retryvla.train.sft configs/exp/m5b_sft_wm_diverse.yaml milestone=V2R tag=$TAG train.seed=$SEED \
      train.save_every=6000 train.num_workers=6 "$@" ;;
  dp)
    python -m retryvla.search.rec_prevent --shard $1 --num_shards 2 --workers $NW ;;
  phys)
    if [ "$3" = "-" ]; then python -m retryvla.search.rec_phys --bank $1 --arm $2 --workers $NW
    else python -m retryvla.search.rec_phys --bank $1 --arm $2 --glob "$3" --workers $NW; fi ;;
  visual)
    python scripts/v2r/visual_bank.py "$1" "$2" "$3" ;;
  *) echo "unknown $cmd"; exit 2 ;;
esac
