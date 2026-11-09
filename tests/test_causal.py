"""ACR hidden physics + probes (needs the sim; ~1-2 min on one CPU core + EGL).

python -m pytest -q tests/test_causal.py
"""

from __future__ import annotations

import numpy as np
import pytest

from retryvla import REPO_ROOT


@pytest.fixture(scope="module")
def rsim():
    from retryvla.envs.splits import load_split, scenes_for
    from retryvla.envs.vec import _RecordingSim

    split = load_split(REPO_ROOT / "data/splits/spatial_v1.json")
    scenes = scenes_for(split, "train", REPO_ROOT / "data/assets/scenes/scenes/selected_bddl_scenes_spatial")
    s = [x for x in scenes if "between_the_plate" in x.tag][0]
    r = _RecordingSim(128)
    out = r.sim.load_scene(s.bddl, s.npy, gripper_close=False)
    r._scene = (s.bddl, out["snapshot"], out["obs"])
    yield r
    r.sim.close()


H = [{"type": "slippery", "mag": 0.2}, {"type": "anchored", "mag": 2.0}, {"type": "heavy", "mag": 1.0},
     {"type": "weak_grip", "mag": 0.3}]


def test_render_identical(rsim):
    bddl, snap, obs = rsim._scene
    base = rsim.restore_hidden(bddl, snap, None)["obs"]
    for h in H:
        o = rsim.restore_hidden(bddl, snap, h)["obs"]
        assert np.array_equal(o["image"], base["image"]) and np.array_equal(o["image2"], base["image2"]), h
        assert np.array_equal(o["state"], base["state"]), h


def test_nominal_restored_no_leak(rsim):
    from retryvla.causal.probes import library

    bddl, snap, _ = rsim._scene
    acts = library(-1.0)["touch_grasp"]
    rsim.restore_hidden(bddl, snap, None)
    a = rsim.probe(acts, snapshot=False)["feats"]
    rsim.restore_hidden(bddl, snap, H[0])
    rsim.probe(acts, snapshot=False)
    rsim.restore_hidden(bddl, snap, None)
    b = rsim.probe(acts, snapshot=False)["feats"]
    assert a == b  # bit-exact after a hypothesis was applied in between


def test_dynamics_change(rsim):
    """Each hypothesis changes the response of at least one probe (oracle feature set) vs nominal."""
    from retryvla.causal.probes import FEATS_ORACLE, library

    bddl, snap, _ = rsim._scene
    lib = library(-1.0)
    names = ["touch_grasp", "nudge_px", "lift2"]

    def resp(h):
        out = []
        for n in names:
            rsim.restore_hidden(bddl, snap, h)
            f = rsim.probe(lib[n], snapshot=False)["feats"]
            out.append([f[k] for k in FEATS_ORACLE])
        return np.array(out)

    r0 = resp(None)
    for h in H:
        assert not np.allclose(resp(h), r0), h


def test_static_equivalence(rsim):
    """Hidden causes must not reveal themselves at rest: holding still for 15 steps moves the object (privileged
    pose) by < 3 mm more than under nominal physics, and the proprio response of `hold` matches nominal."""
    from retryvla.causal.probes import FEATS_PROPRIO, library

    bddl, snap, _ = rsim._scene
    hold = library(-1.0)["hold"]
    rsim.restore_hidden(bddl, snap, None)
    f0 = rsim.probe(hold, snapshot=False)["feats"]
    for h in H:
        rsim.restore_hidden(bddl, snap, h)
        f = rsim.probe(hold, snapshot=False)["feats"]
        assert f["obj_disp_max"] < f0["obj_disp_max"] + 0.003, (h, f["obj_disp_max"])
        assert all(abs(f[k] - f0[k]) < 2e-3 for k in FEATS_PROPRIO), (h, f, f0)


def test_probe_lengths():
    from retryvla.causal.probes import PROBE_MAX_STEPS, library

    for g in (-1.0, 1.0):
        for k, a in library(g).items():
            assert a.shape[1] == 7 and len(a) <= PROBE_MAX_STEPS and np.all(np.abs(a) <= 1), k
