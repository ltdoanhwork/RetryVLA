# source env/activate.sh   -- activate the RetryVLA environment (bash)
WS=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge
RETRYVLA_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export RETRYVLA_ROOT
source "$WS/miniforge3/etc/profile.d/conda.sh"
conda activate "$WS/miniforge3/envs/retryvla"

# repo-local LIBERO config (the shared ~/.libero points at someone else's checkout)
export LIBERO_CONFIG_PATH="$RETRYVLA_ROOT/.libero"
# headless MuJoCo rendering
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0}
# The container only mounts NVIDIA *compute* libs, so EGL falls back to Mesa (software,
# ~250ms/step). Use user-local NVIDIA EGL libs matching the host driver (scripts/02_setup_nvidia_egl.sh).
_DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
_EGL_DIR="$WS/tools/nvidia-egl-${_DRV}"
if [ -n "$_DRV" ] && [ -f "$_EGL_DIR/vendor/10_nvidia.json" ]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="$_EGL_DIR/vendor/10_nvidia.json"
  export LD_LIBRARY_PATH="$_EGL_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
  echo "[retryvla] WARNING: NVIDIA EGL libs for driver '$_DRV' not found -> software rendering. Run scripts/02_setup_nvidia_egl.sh" >&2
fi
# cluster pip.conf has an unreachable NGC extra index
export PIP_CONFIG_FILE=/dev/null
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$RETRYVLA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
