#!/usr/bin/env bash
# Create the dedicated RetryVLA conda env (python 3.12, lerobot 0.5.1 + smolvla + libero).
# usage: bash scripts/00_setup_env.sh
set -euo pipefail
WS=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge
ENV_PREFIX=${ENV_PREFIX:-$WS/miniforge3/envs/retryvla}
REPO=$(cd "$(dirname "$0")/.." && pwd)

source "$WS/miniforge3/etc/profile.d/conda.sh"
[ -d "$ENV_PREFIX" ] || conda create -y -p "$ENV_PREFIX" python=3.12
conda activate "$ENV_PREFIX"

# The cluster-wide pip.conf adds an unreachable NGC extra index; bypass it.
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple

pip install -e "$REPO[dev]"
# lerobot pulls opencv-python (needs libxcb); other cluster images lack X libs -> force headless
pip uninstall -y opencv-python 2>/dev/null || true
pip install "opencv-python-headless>=4.9,<4.14"
# lerobot[libero] installs `hf-libero`; make sure its config points at bundled assets (no prompt).
python - <<'PY'
import os, pathlib, yaml, importlib.util
spec = importlib.util.find_spec("libero")
root = pathlib.Path(list(spec.submodule_search_locations)[0]) / "libero"
cfg_dir = pathlib.Path(os.environ.get("LIBERO_CONFIG_PATH", pathlib.Path.home() / ".libero"))
cfg_dir.mkdir(parents=True, exist_ok=True)
cfg = {
    "benchmark_root": str(root),
    "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"),
    "datasets": str(root.parent / "datasets"),
    "assets": str(root / "assets"),
}
print("libero root:", root)
PY
python -c "import lerobot, libero, robosuite, mujoco, torch; print('lerobot', lerobot.__version__, 'mujoco', mujoco.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
