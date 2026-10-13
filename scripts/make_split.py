#!/usr/bin/env python
"""Build the scene-level LIBERO-Recover split (group = task folder + failure episode).

python scripts/make_split.py --scene_root data/assets/scenes/scenes/selected_bddl_scenes_spatial \
    --out data/splits/spatial_v1.json
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retryvla.envs.libero_env import discover_scenes  # noqa: E402
from retryvla.envs.splits import build_split, save_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_root", default="data/assets/scenes/scenes/selected_bddl_scenes_spatial")
    ap.add_argument("--out", default="data/splits/spatial_v1.json")
    ap.add_argument("--test_frac", type=float, default=0.35)
    ap.add_argument("--pilot_scenes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    scenes = discover_scenes(a.scene_root)
    split = build_split(scenes, a.test_frac, a.pilot_scenes, a.seed)
    split["scene_root"] = a.scene_root
    save_split(split, a.out)
    tf = {s.tag: s.task_folder for s in scenes}
    for part in ("train", "pilot", "test"):
        c = Counter(tf[t] for t in split[part])
        print(f"{part}: {len(split[part])} scenes")
        for k, v in sorted(c.items()):
            print(f"   {v:3d}  {k}")
    assert not set(split["test"]) & set(split["train"]), "leak"
    assert set(split["pilot"]) <= set(split["train"])
    tg = {t.rsplit("__", 1)[0] for t in split["test"]}
    trg = {t.rsplit("__", 1)[0] for t in split["train"]}
    assert not tg & trg, "group leak"
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
