#!/usr/bin/env python
"""Summarize M1 search outputs: per-arm stats, per-source success, plots, sample recovery videos.

python scripts/analyze_m1.py  ->  experiments/figs/m1_*.png, runs/_analysis/videos/*.mp4, printed table
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ARMS = {
    "CEM": ["m1_oracle_cem_spatial_s0", "m1_oracle_cem_spatial_s0b", "m1_oracle_cem_spatial_s1"],
    "random": ["m1_random_spatial_s0", "m1_random_spatial_s0b", "m1_random_spatial_s1"],
}
FIG = REPO / "experiments" / "figs"
VID = REPO / "runs" / "_analysis" / "videos"


def load_arm(names):
    br, tr, inst = [], [], []
    for n in names:
        side = REPO / "data" / "recovery" / n / "retryvla"
        if not side.exists():
            continue
        if (side / "branches.parquet").exists():
            b = pd.read_parquet(side / "branches.parquet").drop(columns=["actions"])
            b["part"] = n
            br.append(b)
        t = [json.loads(l) for l in open(side / "trajectories.jsonl")]
        for r in t:
            r["part"] = n
        tr += t
        inst += [json.loads(l) | {"part": n} for l in open(side / "instances.jsonl")]
    return (pd.concat(br) if br else pd.DataFrame()), pd.DataFrame(tr), pd.DataFrame(inst)


def source_group(s: str) -> str:
    if s.startswith("vla"):
        return "SmolVLA samples"
    if s == "perturb":
        return "perturbed"
    if s.startswith("prim"):
        return "primitives"
    return "CEM refit" if s == "cem" else s


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    stats, data = {}, {}
    for arm, names in ARMS.items():
        br, tr, inst = load_arm(names)
        data[arm] = (br, tr, inst)
        # branches.parquet is rewritten every 5 instances: instances with branches are the searched ones
        searched = br["instance_id"].nunique() if len(br) else 0
        solved = tr["instance_id"].nunique() if len(tr) else 0
        stats[arm] = {
            "instances searched": searched,
            "instances solved": solved,
            "solve rate": solved / max(1, searched),
            "trajectories": len(tr),
            "sim episodes (branches)": len(br),
            "branch success rate": float(br["success"].mean()) if len(br) else float("nan"),
            "mean recovery len": float(tr["recovery_len"].mean()) if len(tr) else float("nan"),
            "mean total len": float(tr["length"].mean()) if len(tr) else float("nan"),
        }
    df = pd.DataFrame(stats)
    print(df.to_string(float_format=lambda x: f"{x:.3f}"))

    # 1) success rate per iteration (CEM should rise, random flat)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for arm, (br, tr, inst) in data.items():
        if len(br):
            g = br.groupby("iteration")["success"].mean()
            ax[0].plot(g.index, g.values, marker="o", label=arm)
    ax[0].set(title="Branch success rate per search iteration", xlabel="iteration", ylabel="success rate")
    ax[0].legend()
    # 2) success by proposal source (iteration 0 = shared proposal pool)
    rows = []
    for arm, (br, _, _) in data.items():
        if len(br):
            b = br.assign(src=br["source"].map(source_group))
            for s, v in b.groupby("src")["success"]:
                rows.append((arm, s, v.mean(), len(v)))
    sdf = pd.DataFrame(rows, columns=["arm", "source", "rate", "n"])
    piv = sdf.pivot(index="source", columns="arm", values="rate")
    piv.plot.bar(ax=ax[1], rot=20)
    ax[1].set(title="Branch success rate by proposal source", ylabel="success rate")
    # 3) per-task solve rate
    rows = []
    for arm, (br, tr, inst) in data.items():
        if not len(br):
            continue
        task = br["instance_id"].str.extract(r"pick_up_the_black_bowl_(.*)_and_place")[0]
        per = pd.DataFrame({"inst": br["instance_id"], "task": task}).drop_duplicates()
        solved = set(tr["instance_id"]) if len(tr) else set()
        per["solved"] = per["inst"].isin(solved)
        for t, v in per.groupby("task")["solved"]:
            rows.append((arm, t.replace("_", " ")[:28], v.mean(), len(v)))
    tdf = pd.DataFrame(rows, columns=["arm", "task", "rate", "n"])
    tdf.pivot(index="task", columns="arm", values="rate").plot.barh(ax=ax[2])
    ax[2].set(title="Instances solved, by task (pilot)", xlabel="fraction solved")
    plt.tight_layout()
    plt.savefig(FIG / "m1_search_overview.png", dpi=110)
    print("saved", FIG / "m1_search_overview.png")
    sdf.to_csv(FIG / "m1_source_success.csv", index=False)
    df.to_csv(FIG / "m1_arm_stats.csv")

    # 4) sample recovery videos (agent | wrist), recovery segment tinted
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import imageio

    VID.mkdir(parents=True, exist_ok=True)
    br, tr, _ = data["CEM"]
    finished = [n for n in ARMS["CEM"] if (REPO / "data/recovery" / n / "meta/episodes").exists()
                and not n.endswith("s0b")]
    tr = tr[tr["part"].isin(finished)]
    picks = tr.sort_values("score", ascending=False).drop_duplicates("instance_id")
    picks = picks.groupby(picks["instance_id"].str.extract(r"bowl_(.*?)_and")[0]).head(1).head(6)
    for _, r in picks.iterrows():
        ds = LeRobotDataset("retryvla/recovery", root=REPO / "data" / "recovery" / r["part"], episodes=[int(r["episode_index"])])
        frames = []
        for i in range(ds.num_frames):
            it = ds[i]
            a = (it["observation.images.image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            w = (it["observation.images.image2"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            f = np.concatenate([a, w], 1).copy()
            seg = int(it["segment"])
            f[:6] = (255, 140, 0) if seg == 0 else (0, 170, 255)  # orange bar = searched recovery, blue = SmolVLA
            frames.append(f)
        name = r["instance_id"].replace("pick_up_the_black_bowl_", "")[:70]
        out = VID / f"{name}.mp4"
        imageio.mimwrite(out, frames, fps=20)
        imageio.imwrite(VID / f"{name}_first.png", frames[0])
        print("video", out, f"len={r['length']} recovery={r['recovery_len']} src={r['source']}")


if __name__ == "__main__":
    main()
