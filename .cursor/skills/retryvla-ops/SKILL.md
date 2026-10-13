---
name: retryvla-ops
description: Operate the RetryVLA repo (LIBERO-Recover, SmolVLA, world-model-guided recovery search). Use for any RetryVLA task involving the env, LIBERO/MuJoCo rendering, launching runs in tmux, GPU/CPU usage, assets, or checkpoints.
---

# RetryVLA Ops

Repo: `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA`. Read `README.md` and `experiments/REGISTRY.md` first.

## Environment

Always `source env/activate.sh`. It:
- activates conda env `miniforge3/envs/retryvla` (python 3.12, lerobot 0.5.1 + smolvla + hf-libero, torch cu128).
  Never use or modify the shared `miniforge3/envs/lerobot` env (editable install from another project).
- sets `LIBERO_CONFIG_PATH=$REPO/.libero`. The shared `~/.libero` points at someone else's checkout.
- sets `MUJOCO_GL=egl` and points EGL at user-local NVIDIA libs (`tools/nvidia-egl-<driver>`).
  The container only exposes compute libs, and without these MuJoCo silently uses Mesa (about 250 ms/step, 12x slower).
  If it warns "software rendering", run `scripts/02_setup_nvidia_egl.sh`.
- sets `PIP_CONFIG_FILE=/dev/null`. The cluster pip.conf adds an unreachable NGC index, which makes pip hang.

Fresh setup: `bash scripts/00_setup_env.sh && bash scripts/01_download_assets.sh && bash scripts/02_setup_nvidia_egl.sh`.

## Assets

- LIBERO-Recover scenes: `data/assets/scenes/scenes/selected_bddl_scenes_<suite>/<task>/<episode>/step_XXXX{.bddl,_sim_state.npy}`
  (spatial: 168 scenes, 84 failure episodes, 9 tasks; no L1-L4 labels released).
- `data/assets/scenes_full/` holds denser per-20-step states (not the benchmark; possible extra train states).
- Policy: `HuggingFaceVLA/smolvla_libero` (chunk 50, n_obs 1, MEAN_STD state/action). Nominal data: `lerobot/libero`.

## Launching

```bash
scripts/launch.sh <session> <python.module> <config.yaml> [key=value ...]
# logs -> runs/_logs/<session>_<ts>.log ; run outputs -> runs/<ts>_<milestone>_<tag>/
```
- This node is shared. Check `top`/`nvidia-smi` first; use `eval.n_workers` ~ free cores (16-24).
- Never `pkill -f <pattern>`: it matches the agent's own shell. Kill by `tmux kill-session -t <name>`
  or by explicit PID.
- One model on the GPU per process. Search and eval each load SmolVLA once (about 2 GB). SFT needs about 20-30 GB at bs32.

## Sim conventions (`retryvla/envs/libero_env.py`)

- Images = raw render flipped H and W; state = eef_pos(3) + axis-angle(3) + gripper_qpos(2); actions in [-1,1], gripper -1 open / +1 close.
- `load_scene` = official protocol: ±2 cm xy noise on free-joint objects, OSC goal re-anchor, gripper settle 20 steps.
- Branching: use `snapshot()` / `restore_snapshot()`. They are bit-exact (MuJoCo integration state, controller,
  gripper, observable timers). Restoring the flat `sim_state` alone is not deterministic.
- `SimPool` (spawn processes) needs a real `__main__` file. Do not run pool code from `python - <<EOF`.
