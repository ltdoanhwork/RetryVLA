#!/usr/bin/env bash
# Extract NVIDIA EGL userspace libs matching the host driver into a user-local dir.
# Needed because the container exposes only compute libs (NVIDIA_DRIVER_CAPABILITIES=compute,utility),
# so MuJoCo EGL silently falls back to Mesa software rendering (~12x slower).
set -euo pipefail
WS=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
D="$WS/tools/nvidia-egl-$DRV"
if [ -f "$D/vendor/10_nvidia.json" ]; then echo "already set up: $D"; exit 0; fi
TMP=$(mktemp -d -p "$WS/tmp")
cd "$TMP"
curl -fsSL -o drv.run "https://us.download.nvidia.com/XFree86/Linux-x86_64/$DRV/NVIDIA-Linux-x86_64-$DRV.run"
sh drv.run --extract-only --target ex >/dev/null
mkdir -p "$D/lib" "$D/vendor"
for f in libEGL_nvidia libnvidia-eglcore libnvidia-glcore libnvidia-glsi libnvidia-gpucomp; do
  cp "ex/$f.so.$DRV" "$D/lib/"
done
ln -sf "libEGL_nvidia.so.$DRV" "$D/lib/libEGL_nvidia.so.0"
# libglvnd loader: some cluster images (e.g. other users' shells) ship no libEGL.so.1 at all
cp ex/libEGL.so.1.1.0 ex/libGLdispatch.so.0 ex/libOpenGL.so.0 "$D/lib/"
ln -sf libEGL.so.1.1.0 "$D/lib/libEGL.so.1"; ln -sf libEGL.so.1 "$D/lib/libEGL.so"
cat > "$D/vendor/10_nvidia.json" <<EOF
{"file_format_version":"1.0.0","ICD":{"library_path":"$D/lib/libEGL_nvidia.so.0"}}
EOF
rm -rf "$TMP"
echo "installed NVIDIA EGL libs -> $D"
