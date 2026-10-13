#!/usr/bin/env bash
# Executed ON the remote shell (via `det shell open <id> -- bash remote_tmux.sh ...`).
# usage: remote_tmux.sh <session> <gpu_idx> <python.module> <config.yaml> [key=value ...]
# Plain args only: det re-joins argv, so never pass quoted shell snippets through it.
set -euo pipefail
REPO=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA
SESS=$1; GPU=$2; shift 2
export PATH="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin:$PATH"   # tmux lives on /pfss
command -v tmux >/dev/null || { echo "NO_TMUX"; exit 3; }
if tmux has-session -t "=$SESS" 2>/dev/null; then echo "SESSION_EXISTS $SESS"; exit 2; fi
mkdir -p "$REPO/runs/_logs" "$REPO/runs/_jobs"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$REPO/runs/_logs/${SESS}_$(hostname)_${TS}.log"
JOB="$REPO/runs/_jobs/${SESS}_${TS}.sh"
cat > "$JOB" <<EOF
#!/usr/bin/env bash
cd $REPO
export CUDA_VISIBLE_DEVICES=$GPU MUJOCO_EGL_DEVICE_ID=$GPU
source env/activate.sh
echo "[rv] host=\$(hostname) gpu=$GPU start=\$(date -Is) args: $*"
PYTHONUNBUFFERED=1 python -u -m $* 2>&1 | grep --line-buffered -v -E 'EGLError|egl_context|binding_utils|OpenGL/error|glCheckError|Exception ignored|macro file|setup_macros|^Traceback \(most recent call last\):\$|^  File .*(egl|binding_utils|OpenGL)'
echo "[rv] RV_DONE rc=\${PIPESTATUS[0]} end=\$(date -Is)"
EOF
chmod +x "$JOB"
tmux new-session -d -s "$SESS" "bash $JOB > $LOG 2>&1"
echo "LAUNCHED $SESS gpu=$GPU log=$LOG"
