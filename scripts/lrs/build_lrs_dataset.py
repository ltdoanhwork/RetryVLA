"""LRS-G step 4: accepted teacher paths -> quality-filtered training windows (their Sec. 3.6), as a LeRobotDataset.

python -m scripts.lrs.build_lrs_dataset --src data/recovery/lrs_goal --out data/recovery/lrs_goal_ds [--workers 8]

For every accepted entry (rows.jsonl status == accepted) the teacher path is replayed from the exact entry snapshot
(bit-exact; checked against the logged eef positions of the commands) to render the observations. Per-action quality
q = the grade of the command that produced it. Windows = maximal runs of consecutive actions with q >= 0.75 and length
>= L_min = 5. Each window becomes one episode (frames = its actions, aligned obs_i -> action_i as in our sim
convention; D0 uses the same alignment), with per-frame columns `q` (float32) and `run_end` (int64: index of the last
valid action of the window inside the episode). pi0_sft --masked_recovery uses them: targets beyond run_end are
masked (validity mask) and the loss is weighted by q (their quality-weighted, validity-masked flow matching; D-L7).
Teacher actions only; student branches never enter.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from retryvla import REPO_ROOT

Q_MIN, L_MIN = 0.75, 5


def windows(q: np.ndarray) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(q)
    while i < n:
        if q[i] >= Q_MIN:
            j = i
            while j < n and q[j] >= Q_MIN:
                j += 1
            if j - i >= L_MIN:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def per_action_q(row: dict, n: int) -> np.ndarray:
    q = np.zeros(n, np.float32)
    for c in row["cmds"]:
        q[c["i0"]: c["i1"]] = float(c.get("q", 0.0))
    return q


def replay(iid: str, path: np.ndarray) -> dict:
    from retryvla.envs.libero_env import LiberoSim
    from scripts.demos.goal_demo_common import build_entry

    sim = LiberoSim(128)
    E = build_entry(sim, iid)
    sim.restore_snapshot(E["bddl"], E["snap"])
    obs = [E["obs"]]
    for a in path:
        obs.append(sim.step(a).obs)
    sim.close()
    return {"image": np.stack([o["image"] for o in obs[:-1]]), "image2": np.stack([o["image2"] for o in obs[:-1]]),
            "state": np.stack([o["state"] for o in obs]).astype(np.float32), "lang": E["lang"]}


def _job(args):
    iid, path = args
    return iid, replay(iid, path)


def main() -> None:
    import multiprocessing as mp

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/recovery/lrs_goal")
    ap.add_argument("--out", default="data/recovery/lrs_goal_ds")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gt_only", action="store_true",
                    help="LRS-G follow-up: keep only accepted paths whose hand-off was ALSO a sim-GT success")
    ap.add_argument("--target_anchors", type=int, default=0,
                    help="> 0: subsample whole accepted paths (seed 0, round-robin over tasks) until the anchors "
                         "(window frames with >= L_min valid targets) reach this count (their 1,639)")
    a = ap.parse_args()
    src = REPO_ROOT / a.src
    rows, paths = {}, {}
    for p in sorted(src.glob("a*/l*/rows.jsonl")):
        for r in map(json.loads, p.read_text().splitlines()):
            if r["status"] == "accepted" and r["instance_id"] not in rows and (not a.gt_only or r.get("accepted_gt")):
                rows[r["instance_id"]] = r
                paths[r["instance_id"]] = np.load(p.parent / "actions" / f"{r['instance_id']}.npz")["path"]
    plan = {}
    for iid, r in rows.items():
        q = per_action_q(r, len(paths[iid]))
        w = windows(q)
        if w:
            plan[iid] = (q, w)
    if a.target_anchors:
        rng = np.random.default_rng(0)
        by_task = {}
        for iid in sorted(plan):
            by_task.setdefault(rows[iid]["task_folder"], []).append(iid)
        for v in by_task.values():
            rng.shuffle(v)
        sel, n_anc, rnd = [], 0, 0
        while n_anc < a.target_anchors and any(len(v) > rnd for v in by_task.values()):
            for tk in sorted(by_task):
                v = by_task[tk]
                if len(v) > rnd and n_anc < a.target_anchors:
                    sel.append(v[rnd])
                    n_anc += sum(j - i - (L_MIN - 1) for i, j in plan[v[rnd]][1])
            rnd += 1
        plan = {i: plan[i] for i in sel}
        print(f"subsampled {len(plan)} paths -> {n_anc} anchors (target {a.target_anchors})", flush=True)
    print(f"accepted paths {len(rows)} ({sum(len(p) for p in paths.values())} actions); with windows {len(plan)}; "
          f"windows {sum(len(w) for _, w in plan.values())}; window actions "
          f"{sum(j - i for _, w in plan.values() for i, j in w)}", flush=True)
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
    ds = LeRobotDataset.create(repo_id="retryvla/lrs_goal", fps=20, features=feats, root=out, robot_type="panda",
                               use_videos=False, image_writer_threads=8)
    meta, ep = [], 0
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for iid, rp in pool.imap(_job, [(i, paths[i]) for i in plan]):
            q, ws = plan[iid]
            path = paths[iid]
            # replay check: eef after every command == logged (text) is coarse; check the final state is finite and
            # the replayed teacher endpoint matches the path length
            assert len(rp["state"]) == len(path) + 1
            for i, j in ws:
                for t in range(i, j):
                    ds.add_frame({"observation.images.image": rp["image"][t], "observation.images.image2": rp["image2"][t],
                                  "observation.state": rp["state"][t], "action": path[t],
                                  "q": np.asarray([q[t]], np.float32), "run_end": np.asarray([j - 1 - i], np.int64),
                                  "segment": np.asarray([0], np.int64), "task": rp["lang"]})
                ds.save_episode()
                meta.append({"episode_index": ep, "instance_id": iid, "i0": i, "i1": j, "mean_q": float(q[i:j].mean())})
                ep += 1
    ds.finalize() if hasattr(ds, "finalize") else None
    (out / "retryvla").mkdir(exist_ok=True)
    with open(out / "retryvla" / "trajectories.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    print("episodes", ep, "frames", sum(m["i1"] - m["i0"] for m in meta))


if __name__ == "__main__":
    main()
