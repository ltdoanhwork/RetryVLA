#!/usr/bin/env python
"""Visualize (a) C1 WM training data (replayed from the stored actions) and (b) C4 WM-guided recoveries.

C1 stores only DINOv2 latents + actions + outcomes (no RGB). We re-create RGB by restoring the same
failure instance (official protocol, same seed -> bit-exact) and replaying the stored 50-step segment.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.viz_trajectories import frames_of, render  # noqa: E402

OUT = REPO / "runs" / "_analysis" / "viz"


def c1_summary():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for f in sorted(glob.glob(str(REPO / "data/wm/collect_v1/*.pt"))):
        d = torch.load(f, weights_only=False)
        for k in range(len(d["success"])):
            rows.append({"inst": d["instance_id"], "src": d["source"][k].split("_")[0], "succ": bool(d["success"][k]),
                         "prefix": bool(d["prefix_success"][k]), "frames": int(d["valid"][k].sum())})
    import pandas as pd

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    df.groupby("src").succ.mean().plot.bar(ax=ax[0], rot=0, color="tab:blue")
    ax[0].set(title="C1: outcome by candidate source", ylabel="success rate (segment + SmolVLA)")
    per = df.groupby("inst").succ.mean()
    ax[1].hist(per, bins=20, color="tab:orange")
    ax[1].set(title=f"C1: success rate per failure instance (n={len(per)})", xlabel="fraction of 32 candidates that succeed")
    df.frames.value_counts().sort_index().plot.bar(ax=ax[2], rot=0, color="gray")
    ax[2].set(title="C1: latent frames per trajectory (11 = full 50 steps)", xlabel="frames")
    plt.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT / "c1_summary.png", dpi=100)
    print("C1:", len(df), "trajectories,", df.inst.nunique(), "instances, success", round(df.succ.mean(), 3),
          "| instances with 0 successes:", int((per == 0).sum()))
    return df


def c1_replay(instance_id: str, ks: list[int]):
    """Replay stored segments of one C1 instance in the simulator (prefix only, 50 steps)."""
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from retryvla.envs.libero_env import LiberoSim, discover_scenes
    from retryvla.eval.eval_recover import noise_seed

    d = torch.load(REPO / "data/wm/collect_v1" / f"{instance_id}.pt", weights_only=False)
    tag, gi = instance_id.rsplit("__", 1)
    grip, trial = ("close", int(gi[5:])) if gi.startswith("close") else ("open", int(gi[4:]))
    sc = next(s for s in discover_scenes(REPO / "data/assets/scenes/scenes/selected_bddl_scenes_spatial") if s.tag == tag)
    sim = LiberoSim()
    for k in ks:
        out = sim.load_scene(sc.bddl, sc.npy, gripper_close=grip == "close", noise_xy=0.02,
                             noise_seed=noise_seed(7, sc.tag, trial), settle_steps=20, build_seed=7)
        frames, obs = [], out["obs"]
        for a in d["actions"][k].numpy():
            frames.append({"img": obs["image"], "wrist": obs["image2"], "state": obs["state"], "action": a, "seg": 0})
            obs = sim.step(a).obs
        lab = "SUCCEEDS after SmolVLA continuation" if d["success"][k] else "FAILS even after SmolVLA continuation"
        render(f"c1_{instance_id.split('bowl_')[1][:30]}_k{k}", frames,
               f"C1 training sample (replayed segment only, label: {lab})  src={d['source'][k]}")
    sim.close()


def c4_wm_found(n: int = 3):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from retryvla.data.mix import patch_lerobot_index_mapping

    patch_lerobot_index_mapping()
    rows = []
    for r in sorted(glob.glob(str(REPO / "data/recovery/m3_wm_search_hard_s*/retryvla/trajectories.jsonl"))):
        for l in open(r):
            j = json.loads(l)
            j["root"] = str(Path(r).parents[1])
            rows.append(j)
    rows = [r for r in rows if r["continuation_len"] > 20]
    rows.sort(key=lambda r: ("drawer" not in r["instance_id"], -r["score"]))
    seen = set()
    for r in rows:
        key = r["instance_id"]
        if key in seen:
            continue
        seen.add(key)
        ds = LeRobotDataset("retryvla/recovery", root=r["root"], episodes=[int(r["episode_index"])])
        name = "c4_wm_" + r["instance_id"].split("bowl_")[1][:22] + "_" + r["instance_id"].split("__", 1)[1].replace("__", "_")
        render(name, frames_of(ds), f"C4 WM-guided recovery (picked by WM+critic, verified in sim)  "
               f"pred V={r.get('pred_value', 0):.2f}  src={r['source']}  [{r['instance_id'].split('__', 1)[1]}]")
        if len(seen) >= n:
            break


if __name__ == "__main__":
    df = c1_summary()
    per = df.groupby("inst").succ.mean()
    mid = per[(per > 0.1) & (per < 0.5)].index[0]
    sub = df[df.inst == mid].reset_index(drop=True)
    ks = [int(sub.index[sub.succ][0]), int(sub.index[~sub.succ][0])]
    c1_replay(mid, ks)
    c4_wm_found(3)
