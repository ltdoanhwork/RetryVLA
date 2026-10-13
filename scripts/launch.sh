#!/usr/bin/env bash
# Launch any RetryVLA entrypoint in a detached tmux session with a log file.
# usage: scripts/launch.sh <session_name> <python module> <config.yaml> [key=value ...]
# e.g.   scripts/launch.sh m0a retryvla.eval.eval_nominal configs/exp/m0_baseline_spatial.yaml
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
NAME=$1; MOD=$2; shift 2
mkdir -p "$REPO/runs/_logs"
LOG="$REPO/runs/_logs/${NAME}_$(date +%Y%m%d-%H%M%S).log"
CMD="cd $REPO && source env/activate.sh && PYTHONUNBUFFERED=1 python -u -m $MOD $* 2>&1 | grep --line-buffered -v -E 'EGLError|egl_context|binding_utils|OpenGL/error|glCheckError|Exception ignored|macro file|setup_macros|^Traceback \(most recent call last\):$|^  File .*(egl|binding_utils|OpenGL)' | tee $LOG"
tmux new-session -d -s "$NAME" "bash -lc \"$CMD\"; sleep 600"
echo "tmux session '$NAME' -> $LOG"
