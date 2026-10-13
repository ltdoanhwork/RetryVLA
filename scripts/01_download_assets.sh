#!/usr/bin/env bash
# Download LIBERO-Recover scene assets (ModelScope ataier/LIBERO_Recovery_Assets) + HF checkpoint/dataset.
# usage: bash scripts/01_download_assets.sh [--full]   (--full also extracts the dense per-20-step scene library)
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
A="$REPO/data/assets"; mkdir -p "$A"; cd "$A"
MS="https://www.modelscope.cn/api/v1/datasets/ataier/LIBERO_Recovery_Assets/repo?Revision=master&FilePath="
[ -f scenes.tar.gz ] || curl -fsSL -o scenes.tar.gz "${MS}scenes.tar.gz"
[ -d scenes/scenes ] || { mkdir -p scenes && tar xzf scenes.tar.gz -C scenes; }
if [ "${1:-}" = "--full" ]; then
  [ -f scenes_full.tar.gz ] || curl -fsSL -o scenes_full.tar.gz "${MS}scenes_full.tar.gz"
  [ -d scenes_full/assets_times ] || { mkdir -p scenes_full && tar xzf scenes_full.tar.gz -C scenes_full; }
fi
for d in scenes/scenes/*; do echo "$(basename "$d"): $(find "$d" -name '*.bddl' | wc -l) scenes"; done

# LIBERO-Recovery code (reference eval client) -- pinned commit
TP="$REPO/third_party/LIBERO-Recovery"
if [ ! -d "$TP/.git" ]; then
  git clone https://github.com/liulin815/LIBERO-Recovery "$TP"
  git -C "$TP" checkout 7584d1b
fi

# HF: policy + nominal dataset (cached in HF_HOME)
cd "$REPO" && source env/activate.sh
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("HuggingFaceVLA/smolvla_libero")
snapshot_download("lerobot/libero", repo_type="dataset")
print("HF downloads ok")
PY
