"""Probe action scale / gripper qpos for the Backtrack baseline (run on a GPU shell; ~1 min)."""
import numpy as np

from retryvla.envs.libero_env import LiberoSim, discover_scenes

if __name__ == "__main__":
    scs = discover_scenes("data/assets/scenes/scenes/selected_bddl_scenes_spatial")
    sim = LiberoSim()
    for sc in scs[:3]:
        for close in (True, False):
            out = sim.load_scene(sc.bddl, sc.npy, close)
            s0 = out["obs"]["state"].copy()
            # close gripper fully with no motion (empty grasp unless something is in between)
            for _ in range(15):
                r = sim.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
            sc_ = r.obs["state"].copy()
            for _ in range(10):
                r = sim.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
            so = r.obs["state"].copy()
            for _ in range(20):
                r = sim.step(np.array([0, 0, 0.6, 0, 0, 0, -1.0]))
            su = r.obs["state"].copy()
            for _ in range(20):
                r = sim.step(np.array([1.0, 0, 0, 0, 0, 0, -1.0]))
            sx = r.obs["state"].copy()
            print(sc.tag[-40:], "close" if close else "open", "pos0", np.round(s0[:3], 3), "grip0", np.round(s0[6:], 4),
                  "| closed", np.round(sc_[6:], 4), "| open", np.round(so[6:], 4),
                  "| dz(20x0.6)", round(float(su[2] - so[2]), 4), "| dx(20x1.0)", round(float(sx[0] - su[0]), 4), flush=True)
