"""Part 4 smoke checks on a GPU (run via scripts/shells/launch_remote.sh <shell> 0 af_smoke scripts.af_smoke):

1. default path unchanged: pi1's exact SFT recipe (M5c) for N steps with the pre-Part-4 code (git worktree at BASE_REV)
   and with the current code -> per-step train/loss must be identical (same data stream, same FM noise).
2. future path runs: arm 5 (hybrid) and arm 3 configs for N steps on the smoke futures file; L_future logged, FM loss
   finite, head saved.
Checkpoints written by these short runs are deleted afterwards (/pfss quota). Summary -> runs/_analysis/af_smoke.json
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE_REV = "c6d3f8c"
N = int(os.environ.get("AF_SMOKE_STEPS", "20"))
PI1 = ["configs/exp/m5b_sft_wm_diverse.yaml", "train.max_recovery_trajectories=208", "train.subset_mode=roundrobin"]
COMMON = [f"train.steps={N}", "train.log_every=1", f"train.save_every={10 * N}", "train.num_workers=4",
          "train.warmup_steps=300"]


def worktree() -> Path:
    wt = REPO.parent / "RetryVLA_af_base_wt"
    if not wt.exists():
        subprocess.run(["git", "worktree", "add", "--detach", str(wt), BASE_REV], cwd=REPO, check=True)
    for d in ("data", "runs", "env", "tools", ".libero"):
        _link(REPO / d, wt / d)
    return wt


def _link(src: Path, dst: Path) -> None:
    """Symlink src -> dst; if dst is a tracked dir already, link its missing children (recursively one level)."""
    if not src.exists():
        return
    if not dst.exists() and not dst.is_symlink():
        dst.symlink_to(src)
    elif dst.is_dir() and not dst.is_symlink():
        for c in src.iterdir():
            _link(c, dst / c.name)


def run(cwd: Path, args: list[str], tag: str) -> Path:
    before = set((REPO / "runs").glob(f"*_{tag}"))
    env = dict(os.environ, PYTHONPATH=str(cwd))
    cmd = [sys.executable, "-u", "-m", "retryvla.train.sft", *args, *COMMON, f"tag={tag}", "milestone=P4"]
    print("[af_smoke]", cwd, " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)
    new = sorted(set((REPO / "runs").glob(f"*_{tag}")) - before)
    assert new, tag
    return new[-1]


def losses(run_dir: Path, key: str = "train/loss") -> list[float]:
    out = []
    for l in open(run_dir / "metrics.jsonl"):
        r = json.loads(l)
        if key in r:
            out.append(r[key])
    return out


def clean(run_dir: Path) -> None:
    shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)


def main() -> None:
    out = REPO / "runs" / "_analysis" / "af_smoke.json"
    res, t0 = (json.loads(out.read_text()) if out.exists() else {}), time.time()
    done = []
    if "--skip_default" not in sys.argv:
        wt = worktree()
        a = run(wt, PI1, "af_smoke_default_base")
        b = run(REPO, PI1, "af_smoke_default_new")
        la, lb = losses(a), losses(b)
        res["default"] = {"base_run": a.name, "new_run": b.name, "base": la, "new": lb, "identical": la == lb,
                          "max_abs_diff": max(abs(x - y) for x, y in zip(la, lb)) if la else None}
        print("[af_smoke] default identical:", la == lb, res["default"]["max_abs_diff"], flush=True)
        done += [a, b]
    fut = os.environ.get("AF_SMOKE_FUTURES", "data/futures/af_smoke/futures.pt")
    for arm in next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--arms=")), "af_arm5_hybrid,af_arm3_free_filtered").split(","):
        r = run(REPO, [f"configs/train/{arm}.yaml",
                       f"train.future_datasets=[{{path: {fut}, variant: filtered, n: 208}}]"], f"af_smoke_{arm}")
        res[arm] = {"run": r.name, "loss": losses(r), "future_loss": losses(r, "train/future_loss"),
                    "future_frac": losses(r, "train/future_frac"), "head_saved": (r / "future_head.pt").exists()}
        print("[af_smoke]", arm, res[arm], flush=True)
        done.append(r)
    for r in done:
        clean(r)
    res["wall_s"] = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1))
    tmp.replace(out)
    print("[af_smoke] DONE", json.dumps({k: (v.get("identical") if isinstance(v, dict) else v) for k, v in res.items()}))


if __name__ == "__main__":
    main()
