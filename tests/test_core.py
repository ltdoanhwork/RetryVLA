import json

import numpy as np
import pytest

from retryvla.envs.libero_env import canonical_language, discover_scenes
from retryvla.envs.splits import TestSplitLeak, assert_allowed, build_split, load_split
from retryvla.search.cem import CEM, CEMConfig, roughness, smooth_noise
from retryvla.search.primitives import library
from retryvla.train.awr import awr_weights
from retryvla import REPO_ROOT

SCENE_ROOT = REPO_ROOT / "data/assets/scenes/scenes/selected_bddl_scenes_spatial"
SPLIT = REPO_ROOT / "data/splits/spatial_v1.json"


def test_canonical_language():
    assert canonical_language("KITCHEN_SCENE3_turn_on_the_stove") == "turn on the stove"
    assert canonical_language("pick_up_the_black_bowl_on_the_plate") == "pick up the black bowl on the plate"


def test_cem_shapes_and_bounds():
    cem = CEM(50, 7, CEMConfig(n_samples=16), seed=0)
    x = cem.sample()
    assert x.shape == (16, 50, 7) and x.min() >= -1 and x.max() <= 1
    assert set(np.unique(x[..., 6])) <= {-1.0, 1.0}
    elites, _ = CEM.select_elites(x, np.arange(16.0), 4)
    cem.refit(elites)
    assert cem.mean.shape == (50, 7) and (cem.std >= cem.cfg.min_std).all()
    assert roughness(x).shape == (16,)
    z = smooth_noise(np.random.RandomState(0), 4, 50, 7, 5)
    assert z.shape == (4, 50, 7)


def test_primitives():
    p, names = library(50, np.random.RandomState(0))
    assert p.shape[1:] == (50, 7) and len(names) == len(p)
    assert np.abs(p).max() <= 1


def test_awr_weights_normalized():
    w = awr_weights(np.array([-5.0, 0.0, 5.0]), beta=1.0, clip=2.0)
    assert abs(float(w.mean()) - 1.0) < 1e-5 and float(w[2] / w[1]) == pytest.approx(np.exp(2.0), rel=1e-4)


@pytest.mark.skipif(not SCENE_ROOT.exists(), reason="assets not downloaded")
def test_split_no_leakage():
    scenes = discover_scenes(SCENE_ROOT)
    s = build_split(scenes, seed=0)
    assert not set(s["train"]) & set(s["test"])
    assert set(s["pilot"]) <= set(s["train"])
    groups = {sc.tag: sc.group for sc in scenes}
    assert not {groups[t] for t in s["train"]} & {groups[t] for t in s["test"]}
    with pytest.raises(TestSplitLeak):
        assert_allowed(s["test"][:1], s)
    assert_allowed(s["pilot"], s)


@pytest.mark.skipif(not SPLIT.exists(), reason="split not built")
def test_tracked_split_matches_builder():
    saved = load_split(SPLIT)
    rebuilt = build_split(discover_scenes(SCENE_ROOT), saved["test_frac"], len(saved["pilot"]), saved["seed"])
    assert rebuilt["test"] == saved["test"] and rebuilt["train"] == saved["train"]
