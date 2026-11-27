"""LRS-G debug 2026-11-27: derive control datasets in the exact LRS window format (q, run_end, segment columns; one
episode per window; same LeRobotDataset writer as scripts/lrs/build_lrs_dataset.py), without any sim replay.

C1 (D0 relabelled as teacher):  --from data/goal_rsi/d0_train --frames 1725 --q 1.0
C2 (0.5 % share of v2 windows): --from data/recovery/lrs_goal_v2_ds --frames 200      (whole windows, seed 0)
C3 (v2 frames, M0 actions):     --from data/recovery/lrs_goal_v2_ds --actions runs/_analysis/lrs_debug/m0_pred_lrs_goal_v2_ds.npz

python -m scripts.lrs.derive_ctrl_dataset --from <root> --out <root> [--frames N] [--actions npz] [--q 1.0] [--seed 0]
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from retryvla import REPO_ROOT


def _img(v, root: Path):
    from PIL import Image
    b = v["bytes"] if isinstance(v, dict) else v
    if b is None:
        b = (root / v["path"]).read_bytes()
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=0, help="> 0: keep whole episodes (shuffled, seed) until >= N frames")
    ap.add_argument("--actions", default=None, help="npz with `first` [n_frames, 7]: replace every action (C3)")
    ap.add_argument("--q", type=float, default=None, help="constant q for sources without a q column (C1)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src = REPO_ROOT / a.src
    fs = sorted((src / "data").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in fs]).sort_values("index").reset_index(drop=True)
    tasks = pd.read_parquet(src / "meta/tasks.parquet")
    t_of = {int(r.task_index): name for name, r in tasks.iterrows()}
    if a.actions:
        z = np.load(REPO_ROOT / a.actions)
        assert len(z["first"]) == len(df), (len(z["first"]), len(df))
        assert np.allclose(z["label"], np.stack(df.action.values), atol=1e-6), "npz rows != dataset rows"
        df["action"] = list(z["first"].astype(np.float32))
    eps = sorted(df.episode_index.unique())
    if a.frames:
        rng = np.random.default_rng(a.seed)
        rng.shuffle(eps)
        sel, n = [], 0
        for e in eps:
            if n >= a.frames:
                break
            sel.append(e)
            n += int((df.episode_index == e).sum())
        eps = sorted(sel)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    out = REPO_ROOT / a.out
    if out.exists():
        shutil.rmtree(out)
    img = {"dtype": "image", "shape": (128, 128, 3), "names": ["height", "width", "channel"]}
    feats = {"observation.images.image": dict(img), "observation.images.image2": dict(img),
             "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
             "action": {"dtype": "float32", "shape": (7,), "names": None},
             "q": {"dtype": "float32", "shape": (1,), "names": None},
             "run_end": {"dtype": "int64", "shape": (1,), "names": None},
             "segment": {"dtype": "int64", "shape": (1,), "names": None}}
    ds = LeRobotDataset.create(repo_id="retryvla/lrs_ctrl", fps=20, features=feats, root=out, robot_type="panda",
                               use_videos=False, image_writer_threads=8)
    src_meta = {}
    if (src / "retryvla/trajectories.jsonl").exists():
        src_meta = {m["episode_index"]: m for m in map(json.loads, open(src / "retryvla/trajectories.jsonl"))}
    meta = []
    for k, e in enumerate(eps):
        g = df[df.episode_index == e].sort_values("frame_index")
        T = len(g)
        for _, r in g.iterrows():
            q = float(np.asarray(r["q"]).reshape(-1)[0]) if "q" in g.columns else float(a.q)
            ds.add_frame({"observation.images.image": _img(r["observation.images.image"], src),
                          "observation.images.image2": _img(r["observation.images.image2"], src),
                          "observation.state": np.asarray(r["observation.state"], np.float32),
                          "action": np.asarray(r["action"], np.float32),
                          "q": np.asarray([q], np.float32), "run_end": np.asarray([T - 1], np.int64),
                          "segment": np.asarray([0], np.int64), "task": t_of[int(r["task_index"])]})
        ds.save_episode()
        m = dict(src_meta.get(int(e), {"instance_id": f"{Path(a.src).name}:{int(e)}"}))
        meta.append(m | {"episode_index": k, "src_episode": int(e), "n": T})
    ds.finalize() if hasattr(ds, "finalize") else None
    (out / "retryvla").mkdir(exist_ok=True)
    with open(out / "retryvla/trajectories.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    (out / "retryvla/derive_args.json").write_text(json.dumps(vars(a), indent=1))
    print("episodes", len(eps), "frames", sum(m["n"] for m in meta), flush=True)


if __name__ == "__main__":
    main()
