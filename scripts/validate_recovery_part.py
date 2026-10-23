"""Validate a (possibly crashed, never-finalized) compare_arms part before marking it CLOSED.

python scripts/validate_recovery_part.py data/recovery/m11_a/s3/p2 [--decode]

Checks per arm dataset: instances.jsonl rows == done.jsonl instances, every trajectory row belongs to a done
instance, LeRobotDataset loads, n_frames / n_episodes match trajectories.jsonl, (--decode) every frame's images
decode. Lists leftover image dirs of episodes that never reached the parquet (half-written, to delete).
Exit 0 = OK to mark CLOSED.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part")
    ap.add_argument("--decode", action="store_true")
    a = ap.parse_args()
    part = Path(a.part)
    done = {json.loads(l)["instance_id"] for l in open(part / "done.jsonl")}
    ok = True
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    for arm in sorted(p for p in part.iterdir() if p.is_dir()):
        side = arm / "retryvla"
        inst = [json.loads(l) for l in open(side / "instances.jsonl")]
        traj = [json.loads(l) for l in open(side / "trajectories.jsonl")]
        bad_i = [r["instance_id"] for r in inst if r["instance_id"] not in done]
        bad_t = [r["episode_index"] for r in traj if r["instance_id"] not in done]
        msg = [f"{arm.name}: {len(inst)} instances, {len(traj)} trajectories"]
        if bad_i or bad_t:
            ok = False
            msg.append(f"UNDONE instance rows {bad_i} / episodes {bad_t}")
        try:
            ds = LeRobotDataset("retryvla/recovery", root=arm)
            n_ep, n_fr = ds.meta.total_episodes, ds.meta.total_frames
            exp_fr = sum(int(r["length"]) for r in traj)
            msg.append(f"lerobot: {n_ep} eps / {n_fr} frames (side: {len(traj)} / {exp_fr})")
            if n_ep != len(traj) or n_fr != exp_fr or len(ds) != n_fr:
                ok = False
                msg.append("MISMATCH")
            if a.decode:
                for i in range(len(ds)):
                    x = ds[i]
                    assert x["observation.images.image"].shape[-1] == 256
                msg.append("all frames decode")
            img_root = arm / "images" / "observation.images.image"
            if img_root.exists():
                eps = sorted(int(p.name.split("-")[1]) for p in img_root.iterdir())
                extra = [e for e in eps if e >= n_ep]
                if extra:
                    ok = False
                    msg.append(f"leftover image dirs for eps {extra} (half-written, delete)")
        except Exception as e:  # noqa: BLE001
            ok = False
            msg.append(f"LOAD FAILED {type(e).__name__}: {e}")
        print(" | ".join(msg), flush=True)
    print("OK" if ok else "NOT OK")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
