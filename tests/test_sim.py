"""Simulator tests (need MuJoCo + assets): run with  pytest -m sim tests/test_sim.py"""

import numpy as np
import pytest

from retryvla import REPO_ROOT
from retryvla.envs.libero_env import LiberoSim, discover_scenes

SCENE_ROOT = REPO_ROOT / "data/assets/scenes/scenes/selected_bddl_scenes_spatial"
pytestmark = [pytest.mark.sim, pytest.mark.skipif(not SCENE_ROOT.exists(), reason="assets not downloaded")]


@pytest.fixture(scope="module")
def sim_and_scene():
    sc = discover_scenes(SCENE_ROOT)[0]
    sim = LiberoSim()
    yield sim, sc
    sim.close()


def test_snapshot_branching_is_exact(sim_and_scene):
    sim, sc = sim_and_scene
    out = sim.load_scene(sc.bddl, sc.npy, gripper_close=True, noise_seed=3)
    acts = np.random.RandomState(0).uniform(-1, 1, (40, 7)).astype(np.float32)
    for a in acts:
        r1 = sim.step(a)
    sim.restore_snapshot(sc.bddl, out["snapshot"])
    for a in acts:
        r2 = sim.step(a)
    assert np.array_equal(r1.obs["state"], r2.obs["state"])
    assert np.array_equal(r1.obs["image"], r2.obs["image"])


def test_failure_state_not_success(sim_and_scene):
    sim, sc = sim_and_scene
    sim.load_scene(sc.bddl, sc.npy, gripper_close=False, noise_seed=0)
    assert not sim.check_success()
