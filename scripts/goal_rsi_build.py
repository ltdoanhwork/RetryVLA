"""2610.05151 protocol (reverse direction): custom LIBERO-Goal split + training data rebuilt by replay in OUR sim.

python scripts/goal_rsi_build.py [--workers 12] [--res 128]

1. Split (deviation D1, their exact seed-7 assignment is unknown): per task, np.random.default_rng(7).permutation(50)
   over the 50 hdf5 demos -> first 30 train, next 10 val, last 10 test. A "scene" = task + demo init_state.
   -> data/splits/goal_rsi_v1.json
2. D0 = the 300 TRAIN demos replayed from their init_state in LiberoSim at `res` px (their 128x128), recording the
   PRE-action observation of every action (our obs convention: images flipped H+W, state = eef pos + axis-angle +
   2 gripper qpos). Equivalent to their "post-action obs i paired with action i+1" alignment.
   -> data/goal_rsi/d0_train (LeRobotDataset, keys as lerobot/libero) + replay_report.json (replay success per demo).
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HDF = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--yifengzhu-hf--LIBERO-datasets/snapshots/*/libero_goal/*_demo.hdf5")))


def bddl_path(task: str) -> str:
    from libero.libero import get_libero_path
    return os.path.join(get_libero_path("bddl_files"), "libero_goal", f"{task}.bddl")


def make_split() -> dict:
    import h5py
    out = {"name": "goal_rsi_v1", "protocol": "2610.05151 reverse direction (D1: our seed-7 permutation)",
           "seed": 7, "tasks": {}}
    for f in HDF:
        task = Path(f).name.replace("_demo.hdf5", "")
        with h5py.File(f) as h:
            n = len(h["data"].keys())
            lang = json.loads(h["data"].attrs["problem_info"])["language_instruction"]
        assert n == 50, (f, n)
        p = np.random.default_rng(7).permutation(50).tolist()
        out["tasks"][task] = {"hdf5": f, "language": lang, "train": sorted(p[:30]), "val": sorted(p[30:40]),
                              "test": sorted(p[40:])}
    return out


def _records(args):
    """Their alignment (paper 4.1): hdf5 post-action obs index i paired with action i + 1, final frame dropped.
    Images flipped H + W to our sim convention (checked: flipped hdf5 frame vs our render MAE ~5-7 / 255, unflipped
    ~73); state = ee pos + axis-angle (ee_states) + 2 gripper qpos (matches our sim state to < 2 mm at t 0)."""
    task, f, demo, res = args
    import h5py
    with h5py.File(f) as h:
        d = h[f"data/demo_{demo}"]
        im = np.asarray(d["obs/agentview_rgb"])[:, ::-1, ::-1]
        im2 = np.asarray(d["obs/eye_in_hand_rgb"])[:, ::-1, ::-1]
        st = np.concatenate([np.asarray(d["obs/ee_states"]), np.asarray(d["obs/gripper_states"])], 1)
        acts = np.asarray(d["actions"], np.float32)
    T = len(acts) - 1
    return {"task": task, "demo": demo, "image": np.ascontiguousarray(im[:T]), "image2": np.ascontiguousarray(im2[:T]),
            "state": st[:T].astype(np.float32), "action": acts[1:], "success": True, "success_t": None, "T": T}


def _replay(args):
    task, f, demo, res = args
    import h5py
    from retryvla.envs.libero_env import LiberoSim
    sim = _replay.sim if hasattr(_replay, "sim") else None
    if sim is None or sim.resolution != res:
        sim = _replay.sim = LiberoSim(res)
    with h5py.File(f) as h:
        d = h[f"data/demo_{demo}"]
        init = np.asarray(d.attrs["init_state"]) if "init_state" in d.attrs else np.asarray(d["states"][0])
        acts = np.asarray(d["actions"], np.float32)
    o = sim.load_init_state(bddl_path(task), init, num_steps_wait=0, return_snapshot=False)["obs"]
    img, img2, st = [], [], []
    succ_t = None
    for t, a in enumerate(acts):
        img.append(o["image"]); img2.append(o["image2"]); st.append(o["state"])
        r = sim.step(a)
        o = r.obs
        if r.success and succ_t is None:
            succ_t = t + 1
    return {"task": task, "demo": demo, "image": np.stack(img), "image2": np.stack(img2),
            "state": np.stack(st).astype(np.float32), "action": acts, "success": succ_t is not None,
            "success_t": succ_t, "T": len(acts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--out", default="data/goal_rsi/d0_train")
    ap.add_argument("--max_demos", type=int, default=0)
    ap.add_argument("--source", default="hdf5", choices=["hdf5", "replay"])
    a = ap.parse_args()
    split = make_split()
    sp = REPO / "data/splits/goal_rsi_v1.json"
    sp.write_text(json.dumps(split, indent=1))
    print("split ->", sp)
    jobs = [(t, v["hdf5"], d, a.res) for t, v in split["tasks"].items() for d in v["train"]]
    if a.max_demos:
        jobs = jobs[: a.max_demos]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    out = REPO / a.out
    if out.exists():
        import shutil
        shutil.rmtree(out)
    img = {"dtype": "image", "shape": (a.res, a.res, 3), "names": ["height", "width", "channel"]}
    feats = {"observation.images.image": dict(img), "observation.images.image2": dict(img),
             "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
             "action": {"dtype": "float32", "shape": (7,), "names": None}}
    ds = LeRobotDataset.create(repo_id="retryvla/goal_rsi_d0", fps=20, features=feats, root=out,
                               robot_type="panda", use_videos=False, image_writer_threads=8)
    rep = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap(_records if a.source == "hdf5" else _replay, jobs)):
            lang = split["tasks"][r["task"]]["language"]
            for t in range(r["T"]):
                ds.add_frame({"observation.images.image": r["image"][t], "observation.images.image2": r["image2"][t],
                              "observation.state": r["state"][t], "action": r["action"][t], "task": lang})
            ds.save_episode()
            rep.append({k_: r[k_] for k_ in ("task", "demo", "success", "success_t", "T")})
            if k % 20 == 0:
                print(f"{k + 1}/{len(jobs)} replay success so far {np.mean([x['success'] for x in rep]):.3f}", flush=True)
    ds.finalize() if hasattr(ds, "finalize") else None
    (out / "replay_report.json").write_text(json.dumps(rep, indent=0, default=int))
    print("frames", sum(x["T"] for x in rep), "replay success", np.mean([x["success"] for x in rep]))


if __name__ == "__main__":
    main()
