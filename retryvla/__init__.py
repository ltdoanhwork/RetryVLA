"""RetryVLA: world-model-guided synthetic recovery learning for VLAs."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"
CONFIG_DIR = REPO_ROOT / "configs"

__version__ = "0.1.0"
