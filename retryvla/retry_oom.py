"""Run `python -m <module> <args>` with GPU-memory gating and retry on CUDA OOM (shared GPU, other users' jobs).

python -m retryvla.retry_oom [--min_free_gb G] [--tries N] [--wait_s S] <module> [args ...]

Before every attempt waits until the GPU has >= G GB free (nvidia-smi), then runs the module as a subprocess
(output streamed). If it fails with a CUDA out-of-memory error it waits and retries (up to N attempts); any other
failure is returned as is. Used by configs/dag/b2p3.json so one transient OOM does not stop the DAG.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time


def free_gb() -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60).stdout.split()
        return float(out[0]) / 1024
    except Exception:
        return 1e9


GATE = f"/tmp/rv_gpu_gate_{os.getuid()}"


def pass_gate(min_free: float, hold_s: float) -> bool:
    """Free-memory gate shared by all retry_oom processes on this machine (2026-10-24): several waiters polling
    the same GPU used to pass together the moment memory freed (the new job allocates only ~1-2 min after start),
    overshooting the GPU. Under an flock, a waiter passes only if >= min_free GB are free AND no other waiter
    passed in the last hold_s seconds; it then stamps the gate. hold_s = 0 -> old behaviour (plain check)."""
    if hold_s <= 0:
        return free_gb() >= min_free
    with open(GATE, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            try:
                last = float(f.read().strip() or 0)
            except ValueError:
                last = 0.0
            if time.time() - last < hold_s or free_gb() < min_free:
                return False
            f.seek(0)
            f.truncate()
            f.write(str(time.time()))
            f.flush()
            return True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def main() -> None:
    argv = sys.argv[1:]
    opts = {"--min_free_gb": 0.0, "--tries": 6, "--wait_s": 600, "--gate_hold_s": 240.0}
    while argv and argv[0] in opts:
        opts[argv[0]] = type(opts[argv[0]])(argv[1])
        argv = argv[2:]
    hold = opts["--gate_hold_s"] if opts["--min_free_gb"] > 0 else 0.0
    rc = 1
    for k in range(int(opts["--tries"])):
        t0 = time.time()
        while not pass_gate(opts["--min_free_gb"], hold):
            if time.time() - t0 > 6 * 3600:
                print(f"[retry_oom] GPU never had {opts['--min_free_gb']} GB free in 6 h, trying anyway", flush=True)
                break
            print(f"[retry_oom] waiting for {opts['--min_free_gb']} GB free GPU memory (now {free_gb():.1f}; "
                  f"gate hold {hold:.0f} s)", flush=True)
            time.sleep(60 if hold > 0 else 120)
        p = subprocess.Popen([sys.executable, "-u", "-m", *argv], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        oom = False
        for line in p.stdout:
            sys.stdout.write(line)
            if "CUDA out of memory" in line or "OutOfMemoryError" in line:
                oom = True
        rc = p.wait()
        if rc == 0 or not oom:
            sys.exit(rc)
        print(f"[retry_oom] attempt {k + 1} died of CUDA OOM, retry in {opts['--wait_s']} s", flush=True)
        time.sleep(int(opts["--wait_s"]))
    sys.exit(rc)


if __name__ == "__main__":
    main()
