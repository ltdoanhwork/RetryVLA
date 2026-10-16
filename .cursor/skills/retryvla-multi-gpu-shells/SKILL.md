---
name: retryvla-multi-gpu-shells
description: >-
  Run RetryVLA jobs on more GPUs by borrowing idle GPUs on other Determined shells (each shell ID is one
  container on one node with 1-4 A100 80GB). Covers the avoid-list of shells that must never be touched,
  read-only probing (`scripts/shells/find_free.py`), dispatching into tmux on a borrowed GPU
  (`scripts/shells/launch_remote.sh`), status/stop (`remote_ctl.sh`), claims in `shells.json`,
  cross-image portability (headless OpenCV, user-local NVIDIA EGL + glvnd), and known traps.
  Use whenever a RetryVLA eval/search/SFT needs a GPU beyond the local one, when the user says
  to use other accounts' GPUs ("dùng ké"), or when a remote job dies, logs stay empty, or a shell is reclaimed.
---

# RetryVLA multi-GPU: borrowing idle GPUs on Determined shells

All shells share `/pfss/mlde/workspaces/...` (code, env, data, runs are the same tree), and run under the
same uid. A shell ID is one container on one node (`DET_AGENT_ID`, e.g. `cn62`). It may expose 1-4 GPUs,
and its **32 CPUs are shared across them**. That matters: MuJoCo sim workers are CPU-bound.

## 0. Never touch these

**Current mode (user, 2026-10-16): NO borrowing.** Run only on the GPU the user gave (`shells.json:allow_only`,
now `49b70230` = this machine, via `local:0:a` / `local:0:b` lanes in `dag_run.py`). `launch_remote.sh` and
`dag_run.py` refuse anything else. Only lift this when the user explicitly says borrowing is OK again.

`scripts/shells/shells.json:avoid` is the contract. Re-read it each session, and extend it when the user names more:

| id | why |
|---|---|
| 8c7cc072, f1fcfcfe, d4c06aca | roboworm (DreamDojo running / reserved) |
| eae733b2 | VLA_doanh_newmethod (14B) |
| 661efdd6 | VLA_doanh_bench8 (Ctrl-World 4-view) |
| 15f251a1 | VLA_doanh_timing (DreamGen) |
| 8fcf8097, 7a52c6b5 | retryvla shells the user listed (nodes shared with other jobs) |
| f6fb5d66 | VLA_doanh_user: the user's own shell. Never probe, launch on, or kill it |
| 4aec7049, 433d27b6 | VLA_doanh_cpu*: no GPU |

`launch_remote.sh` refuses avoided IDs. The local box (`831c8e6df2ea`, 1x A100) is fine for RetryVLA too
(`scripts/launch.sh`), but other users' jobs share its CPUs.

## 1. Find borrowable GPUs (read-only)

```bash
python3 scripts/shells/find_free.py          # table: node, load/nproc, per-GPU used MiB / peak util, tmux
python3 scripts/shells/find_free.py --json   # free (shell, gpu) slots
```
A slot is free when it has at least 40 GB free, peak util of at most 30% over 5 samples, load per core of at most 0.6, `/pfss` visible, and no existing RetryVLA claim
(rules in `shells.json:borrow_rules`).
- IDs are ephemeral. Re-run this before every dispatch and never hardcode an ID from a doc or log.
- Memory used with 0% util (e.g. `9dfdca1e`) means the owner is holding the GPU. Borrowing the leftover memory is OK,
  but cap the job (`eval.n_workers` at 8 or below, SFT `train.batch_size` at 16 or below) and expect the owner to resume.
- Empty output everywhere means the Determined token has expired. Run `det auth login`. Do not trust a "0 MiB".

## 2. Dispatch (always tmux, via the helper)

```bash
scripts/shells/launch_remote.sh <shell8> <gpu_idx> <name> <python.module> <config.yaml> [k=v ...]
# e.g.
scripts/shells/launch_remote.sh 22d385b6 2 m0c retryvla.eval.eval_recover configs/exp/m0_baseline_spatial.yaml \
    eval.n_workers=8
```
The helper:
1. refuses avoided shells, and refuses if the GPU is busy right now (re-probed);
2. runs `remote_tmux.sh` on the shell, which writes `runs/_jobs/rv_<name>_<ts>.sh` and starts tmux session
   `rv_<name>` with `CUDA_VISIBLE_DEVICES=<gpu> MUJOCO_EGL_DEVICE_ID=<gpu>` (log goes to `runs/_logs/rv_<name>_<host>_<ts>.log`,
   ending in `RV_DONE rc=N`);
3. records the claim `shell8:gpu` in `shells.json:claims` (only if `LAUNCHED`).

Any module works, e.g. `lerobot.scripts.lerobot_eval --policy.path=...`. Session names always start with `rv_`,
so they can never collide with the owners' sessions.

Workers per GPU: on a 4-GPU shell, keep the sum of `eval.n_workers` across our jobs at about 28 or below on that shell.

## 3. Monitor / stop

```bash
scripts/shells/remote_ctl.sh status            # every claim: ALIVE/ENDED/SHELL_GONE + last progress line
tail -f runs/_logs/rv_<name>_*.log             # logs are on /pfss: no need to connect
scripts/shells/remote_ctl.sh stop <shell8:gpu> # kills only rv_* sessions, releases the claim
```
Release claims for finished jobs (`stop` on an ENDED session is safe). Results are in `runs/<ts>_<M>_<tag>/` like local runs.
Register them with `scripts/register_run.py` as usual.

## 4. Splitting work across GPUs

- Different arms on different GPUs (easiest): e.g. oracle CEM search on one, random search on another, pi0 RSR eval on a third.
- One search, sharded: `run_search ... search_on.shard=i search_on.num_shards=N dataset.out_root=data/recovery/<name>_s<i>`
  (one dataset per shard; list them all in `train.recovery_datasets`).
- SFT needs a whole GPU (about 25 GB at bs32 plus dataloader CPU). Do not co-locate it with a borrowed, half-used GPU.

## Traps (all hit for real)

| symptom | cause / fix |
|---|---|
| `NO_TMUX` | tmux is not on PATH in other images. Helpers use `/pfss/.../.local/bin/tmux` |
| `ImportError: libxcb.so.1` | other images have no X libs. The env uses `opencv-python-headless` (keep it; lerobot reinstalls may bring `opencv-python` back. Uninstall it) |
| `'NoneType' object has no attribute 'eglQueryString'` | image has no libglvnd. `tools/nvidia-egl-<drv>/lib` ships `libEGL.so.1` too (`scripts/02_setup_nvidia_egl.sh`) |
| torchcodec `libavutil.so.6x: cannot open` (SFT / dataset video decode) | other images ship no ffmpeg. The env has conda-forge `ffmpeg 7` and `activate.sh` puts `$CONDA_PREFIX/lib` on `LD_LIBRARY_PATH` |
| `CXXABI_1.3.15 not found` (libopenvino) | system libstdc++ is too old for conda's ffmpeg deps. Same fix: `$CONDA_PREFIX/lib` first on `LD_LIBRARY_PATH` |
| `OpenSSL version mismatch` from `det shell open` | `env/activate.sh` was sourced in the calling shell (conda libs break ssh). The shell tools unset `LD_LIBRARY_PATH` themselves. For raw `det` calls use `env -u LD_LIBRARY_PATH det ...` |
| unsure whether a shell's image works | `det shell open <id> -- bash scripts/shells/envcheck.sh` should print `OK video decode` and `OK sim 50 steps ~1s` |
| sim about 250 ms/step | Mesa software EGL. `source env/activate.sh` must find `tools/nvidia-egl-<driver>`. The driver must match (all nodes 610.43.02 now) |
| job gone, log empty | started with nohup/& inside `det shell open`. Only tmux survives |
| quoted command arrives mangled | det re-joins argv. Pass a script path + plain args (that is what `remote_tmux.sh` is for) |
| `pkill -f <pattern>` killed the agent's own shell (happened twice) | never pkill by pattern. Kill tmux sessions by exact name `-t =rv_x` or by PID |
| `tmux has-session -t foo` matches `foo2` | use `-t =name` (exact) |
| shell disappears mid-run | reclaimed. Outputs written incrementally to /pfss survive. Re-probe and relaunch the remaining work |
| chain script relaunches the same eval every loop (100+ partial runs) | releasing a claim **by slot** killed the next step running on that GPU. Release claims by session name only (`run_m4.sh::release`) |
