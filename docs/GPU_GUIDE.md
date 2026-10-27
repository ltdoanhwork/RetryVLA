# GPU usage guide for agents: Determined shell `0097c39c` (RetryVLA GPU)

Written 2026-10-23 for shell `49b70230` (now gone); updated 2026-10-27 for the new RetryVLA shell `0097c39c`.
Read all of it before launching anything.

## 0. Important: where the agent runs is NOT where RetryVLA runs

The Cursor / agent session may be attached to a **different** Determined shell (2026-10-27: `ac1da4a3`,
`benchmark_RoboWorM_Thanhnv_1`, node cn43, whose GPU belongs to another project). Never run GPU / simulation work
on the machine the agent happens to run on. RetryVLA jobs must be launched **on `0097c39c`** via
`scripts/shells/launch_remote.sh 0097c39c 0 <name> <python.module> [args]` or `dag_run.py --slots 0097c39c:0`
(a remote slot), **not** as `local:` lanes. `/pfss` is shared, so outputs / logs land in the repo either way.
Check `hostname` / `nvidia-smi` before assuming anything is "local".

## 1. The machine

| item | value |
|---|---|
| Determined shell | `0097c39c-553a-4bdd-b81a-09a55ac8c034`, description `vla_RetryVLA_DoanhLT_1`, user `hn49pymo` |
| node / hostname | agent node `cn69`; container hostname `dfe3b0366526` (changes when the container restarts -- do not hard-code it) |
| GPU | 1x NVIDIA A100 80 GB (index 0), driver 610.43.02 |
| CPU / RAM | 32 cores shared with other containers on the node (load 69 at the 2026-10-27 probe) |
| storage | `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge` (shared workspace quota -- it HAS filled up and killed jobs) |
| probe 2026-10-27 04:50 UTC | GPU 7.7 / 80 GB used, util ~85 % (user's own jobs, tmux `cab_eu_queue4`, `cab_lidtr_r2`); `envcheck.sh`: OK video decode (cuda True), OK sim 50 steps 1.1 s (hardware EGL) |

Connect from another Determined shell or a login node:

```bash
env -u LD_LIBRARY_PATH det shell open 0097c39c-553a-4bdd-b81a-09a55ac8c034            # interactive
env -u LD_LIBRARY_PATH det shell open 0097c39c-553a-4bdd-b81a-09a55ac8c034 -- bash /path/to/script.sh   # one command
env -u LD_LIBRARY_PATH det shell open 0097c39c-553a-4bdd-b81a-09a55ac8c034 -- bash RetryVLA/scripts/shells/probe.sh     # read-only state
env -u LD_LIBRARY_PATH det shell open 0097c39c-553a-4bdd-b81a-09a55ac8c034 -- bash RetryVLA/scripts/shells/envcheck.sh  # EGL + env check
```
(use absolute paths for the scripts.)

`env -u LD_LIBRARY_PATH` matters: a conda `LD_LIBRARY_PATH` breaks `ssh` (OpenSSL mismatch).
`det` re-joins argv, so pass a script path + plain args, never a quoted shell snippet.

## 2. What may already be running here (do not disturb)

The user's own jobs can share this GPU (2026-10-27: tmux `cab_eu_queue4`, `cab_lidtr_r2`), so it is in
`shells.json:share_ok`. RetryVLA jobs run in tmux sessions named `rv_*` (job sessions) and `rv_dag_*` (DAG runners).

- **Never kill, attach-and-type-into, or rename sessions you did not start.**
- Claims (`0097c39c:0` etc.) in `RetryVLA/scripts/shells/shells.json:claims` are written by `launch_remote.sh`;
  clear your own claim when the job is done.
- Typical per-job GPU memory: SmolVLA SFT ~12 GB (L_preserve variant ~18 GB), eval / search / PPO 2-5 GB each.

Check the live state before you start (ON the shell, or via `probe.sh` from elsewhere):

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin/tmux ls
uptime                                    # CPU load (node shared)
```

## 3. Rules for your jobs

1. **Memory budget**: keep the GPU total below ~74 GB. If `memory.used` + your job's peak would exceed that, wait or
   shrink the batch. Out-of-memory kills the other jobs too.
2. **Compute is shared** with the user's own jobs: your job will be slower than on an idle GPU.
   Prefer short / light jobs. Ask the user before starting anything that needs the whole GPU for hours.
3. **CPU**: the node is shared and loaded. Keep data-loader / simulator workers <= 8.
4. **Always run in tmux** (processes started with `nohup` / `&` inside `det shell open` die when the connection closes):
   ```bash
   T=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin/tmux      # tmux is not on PATH in this image
   $T new-session -d -s <yourprefix>_<job> "bash /path/to/job.sh > /path/to/log 2>&1"
   ```
   Use your own session prefix (NOT `rv_`). Kill only your own sessions, by exact name: `$T kill-session -t =<name>`.
5. **Never** `pkill -f <pattern>` or `kill` processes you did not start (it has killed agent shells before).
6. **Disk**: the workspace quota is shared and has run out several times. Write outputs incrementally, save checkpoints
   atomically (write `*.tmp` then rename) with retries, and delete your own temporary files. Before a big write:
   `dd if=/dev/zero of=<your dir>/.probe bs=1M count=64 && rm <your dir>/.probe`.
7. Set the device explicitly: `export CUDA_VISIBLE_DEVICES=0`. For MuJoCo / robosuite rendering also
   `export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0`.
8. Do not modify anything under `RetryVLA/` (code, configs, data, runs) unless the user asks you to work on RetryVLA.

## 4. Environment

- Conda: `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/miniforge3` (`source .../miniforge3/etc/profile.d/conda.sh`).
  Use your own env; the RetryVLA env is `retryvla` (do not `pip install` into it).
- Hardware EGL for MuJoCo (the image only mounts compute libs): RetryVLA ships user-space NVIDIA EGL + glvnd at
  `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/tools/nvidia-egl-610.43.02`; see `RetryVLA/env/activate.sh` for the
  exact variables (`__EGL_VENDOR_LIBRARY_FILENAMES`, `LD_LIBRARY_PATH`). Without it rendering falls back to software
  EGL (~250 ms / step).
- The image has no X libs: use `opencv-python-headless`; ffmpeg comes from conda-forge.

## 5. If you work ON RetryVLA (same repo)

Use the repo's launchers (remote slots on `0097c39c`, never `local:` lanes from a different shell):

```bash
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA
# one job (tmux rv_<name> on 0097c39c, log runs/_logs/rv_<name>_<host>_<ts>.log, ends with RV_DONE rc=N)
scripts/shells/launch_remote.sh 0097c39c 0 <name> <python.module> [config.yaml] [k=v ...]   # e.g. scripts.<module> --arg
# a DAG json: {"name": "...", "tasks": [{"name": "...", "module": "retryvla....", "args": [...], "deps": []}]}
T=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin/tmux
$T new-session -d -s rv_dag_<name> "cd $PWD && python3 scripts/shells/dag_run.py configs/dag/<name>.json --slots 0097c39c:0 --poll 60 >> runs/_logs/dag_<name>.stdout 2>&1"
```

- The DAG runner itself is light (CPU only) and may run on the agent's shell; its tasks run on `0097c39c`.
- Status / stop: `scripts/shells/remote_ctl.sh`. A task is done when its log ends with `RV_DONE rc=0`.
- Read `RetryVLA/.cursor/skills/retryvla-ops/SKILL.md`, `retryvla-multi-gpu-shells/SKILL.md` and
  `experiments/PLAN.md` first. Commit with `scripts/git_commit.sh` (author ltdoanhwork, no Cursor trailer).
- Only GPUs in `shells.json:allow_only` may be used (`0097c39c`; `8a2808f9` = user's shared GPU, light eval jobs only).
  Do not borrow other shells.

## 6. Quick checklist

- [ ] I am launching on `0097c39c` (not on the shell the agent runs on)
- [ ] `probe.sh`: enough free memory for my peak (+ margin, total < 74 GB)?
- [ ] tmux session (`rv_` via launch_remote, or my own prefix), logging to a file
- [ ] `CUDA_VISIBLE_DEVICES=0`, workers <= 8
- [ ] outputs written incrementally / atomically, disk probe OK
- [ ] I will not touch sessions I did not start; I clear my claim in `shells.json` when done
