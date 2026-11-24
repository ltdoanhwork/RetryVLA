"""LRS helper: which image direction is +x / +y / +z of the eef frame used by the teacher tools (sim_geom eef)?
Saves before / after agentview + wrist images for 20-step moves along each axis. Train scene only."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from retryvla import REPO_ROOT
from retryvla.envs.libero_env import LiberoSim
from retryvla.envs.splits import load_split, scenes_for
from retryvla.geom.features import IDX, sim_geom

out = REPO_ROOT / "runs/_analysis/lrs/axes"
out.mkdir(parents=True, exist_ok=True)
split = load_split(REPO_ROOT / "data/splits/spatial_v1.json")
scenes = scenes_for(split, "train", REPO_ROOT / "data/assets/scenes/scenes/selected_bddl_scenes_spatial")
s = scenes[0]
sim = LiberoSim(256)
for ax, name in enumerate("xyz"):
    o = sim.load_scene(s.bddl, s.npy, gripper_close=False, noise_xy=0.0, noise_seed=0)["obs"]
    g0 = sim_geom(sim)
    imgs = [np.concatenate([o["image"], o["image2"]], 1)]
    a = np.zeros(7, np.float32); a[ax] = 1.0; a[6] = -1
    for _ in range(15):
        r = sim.step(a)
    g1 = sim_geom(sim)
    imgs.append(np.concatenate([r.obs["image"], r.obs["image2"]], 1))
    Image.fromarray(np.concatenate(imgs, 0)).save(out / f"move_{name}.png")
    print(name, "eef", g0[[IDX['eef_x'], IDX['eef_y'], IDX['eef_z']]], "->", g1[[IDX['eef_x'], IDX['eef_y'], IDX['eef_z']]],
          "state", o["state"][:3], "->", r.obs["state"][:3], "obj", g0[:3], "goal", g0[3:6], "grip_open", g0[IDX['grip_open']])
print(s.tag, s.instruction("bddl"))
