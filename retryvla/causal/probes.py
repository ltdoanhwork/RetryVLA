"""ACR diagnostic probes: short, safe, mostly reversible open-loop actions executed BEFORE choosing a recovery.

Action = LIBERO OSC delta [dx, dy, dz, droll, dpitch, dyaw, gripper] in [-1, 1] (|dx| = 1 ~ 1 cm / step, world-aligned); gripper
-1 open / +1 close. A probe is a function of the current gripper command g (kept unless the probe changes it).

Response features (computed in the worker from the per-step trace, `response`):
  proprio   (observation-only; default) finger opening min / end / std while closed, achieved eef displacement max,
            achieved eef lift max, eef return error, eef lag (commanded minus achieved path length)
  tactile   + fraction of steps with finger-object contact (a tactile sensor would give this)
  oracle    + privileged object displacement max / end and object lift max ("oracle perception", upper bound only)
"""

from __future__ import annotations

import numpy as np

STEP_M = 0.01           # ~ m of eef motion per step at |a| = 1 (measured: 10 steps of dx = 1 -> 8-12 cm)
PROBE_MAX_STEPS = 45
FEATS_PROPRIO = ("grip_min", "grip_end", "grip_std_closed", "eef_disp_max", "eef_lift_max", "eef_return_err",
                 "eef_lag")
FEATS_TACTILE = FEATS_PROPRIO + ("contact_frac",)
FEATS_ORACLE = FEATS_TACTILE + ("obj_disp_max", "obj_disp_end", "obj_lift_max")
FEATURE_SETS = {"proprio": FEATS_PROPRIO, "tactile": FEATS_TACTILE, "oracle": FEATS_ORACLE}


def _seg(n, d=(0, 0, 0), rot=(0, 0, 0), g=-1.0):
    a = np.zeros((int(n), 7), np.float32)
    a[:, :3] = d
    a[:, 3:6] = rot
    a[:, 6] = g
    return a


def library(g: float) -> dict[str, np.ndarray]:
    """name -> actions [n, 7]; g = current gripper command (+1 closed / -1 open)."""
    v = 0.3                               # ~3 mm / step: lift2 / touch_grasp ~2.4 cm, nudges ~1.2 cm
    P = {
        "hold": _seg(15, g=g),
        "squeeze": np.concatenate([_seg(8, g=1.0), _seg(8, g=-1.0 if g < 0 else 1.0)]),
        "lift2": np.concatenate([_seg(8, g=1.0), _seg(8, (0, 0, v), g=1.0), _seg(4, g=1.0),
                                 _seg(8, (0, 0, -v), g=1.0), _seg(6, g=g)]),
        "touch_grasp": np.concatenate([_seg(4, g=-1.0), _seg(8, (0, 0, -v), g=-1.0), _seg(8, g=1.0),
                                       _seg(8, (0, 0, v), g=1.0), _seg(8, (0, 0, -v), g=1.0), _seg(6, g=-1.0)]),
        "yaw_wiggle": np.concatenate([_seg(6, rot=(0, 0, v), g=g), _seg(12, rot=(0, 0, -v), g=g),
                                      _seg(6, rot=(0, 0, v), g=g)]),
    }
    for name, d in (("nudge_px", (v, 0, 0)), ("nudge_mx", (-v, 0, 0)), ("nudge_py", (0, v, 0)),
                    ("nudge_my", (0, -v, 0))):
        P[name] = np.concatenate([_seg(6, g=1.0), _seg(4, d, g=1.0), _seg(4, tuple(-x for x in d), g=1.0),
                                  _seg(4, g=g)])
    for k, a in P.items():
        assert len(a) <= PROBE_MAX_STEPS, k
    return P


def commanded_path(actions: np.ndarray) -> float:
    return float(np.abs(actions[:, :3]).sum(1).sum() * STEP_M)


def run_probe(rsim, actions: np.ndarray, abort_disp: float = 0.03) -> dict:
    """Worker side (rsim = vec._RecordingSim). Steps the probe, aborting (zero motion, gripper held) once the object
    moved more than abort_disp from its start. Returns features (all sets), geometry start / end, n, aborted."""
    from retryvla.geom.features import IDX, sim_geom

    g0 = sim_geom(rsim.sim)
    eef0, obj0 = g0[[IDX["eef_x"], IDX["eef_y"], IDX["eef_z"]]], g0[[IDX["obj_x"], IDX["obj_y"], IDX["obj_z"]]]
    tr, aborted, success = [g0], False, False
    for a in actions:
        if aborted:
            a = np.concatenate([np.zeros(6, np.float32), a[6:7]])
        out = rsim.step(a)
        g = sim_geom(rsim.sim)
        tr.append(g)
        success = success or bool(out["success"])
        if not aborted and np.linalg.norm(g[[IDX["obj_x"], IDX["obj_y"], IDX["obj_z"]]] - obj0) > abort_disp:
            aborted = True
    T = np.stack(tr)
    feats = response(T, actions, eef0, obj0)
    return {"feats": feats, "geom0": g0, "geom1": T[-1], "n": int(len(actions)), "aborted": aborted,
            "success": success}


def response(T: np.ndarray, actions: np.ndarray, eef0: np.ndarray, obj0: np.ndarray) -> dict[str, float]:
    from retryvla.geom.features import IDX

    eef = T[:, [IDX["eef_x"], IDX["eef_y"], IDX["eef_z"]]]
    obj = T[:, [IDX["obj_x"], IDX["obj_y"], IDX["obj_z"]]]
    grip = T[:, IDX["grip_open"]]
    closed = np.concatenate([[False], actions[:, 6] > 0])
    de = np.linalg.norm(eef - eef0, axis=1)
    do = np.linalg.norm(obj - obj0, axis=1)
    path = float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum())
    return {
        "grip_min": float(grip.min()), "grip_end": float(grip[-1]),
        "grip_std_closed": float(grip[closed].std()) if closed.sum() > 1 else 0.0,
        "eef_disp_max": float(de.max()), "eef_lift_max": float((eef[:, 2] - eef0[2]).max()),
        "eef_return_err": float(de[-1]), "eef_lag": float(commanded_path(actions) - path),
        "contact_frac": float(T[1:, IDX["finger_obj_contact"]].mean()),
        "obj_disp_max": float(do.max()), "obj_disp_end": float(do[-1]),
        "obj_lift_max": float((obj[:, 2] - obj0[2]).max()),
    }


def feat_vec(feats: dict, fset: str) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_SETS[fset]], np.float64)


def risk(geom0: np.ndarray, geom1: np.ndarray) -> float:
    """1 if the probe made the state worse (off_table / tilt / disturb >= 0.5 vs the probe start), else 0."""
    from retryvla.geom.features import geom_terms

    t = geom_terms(geom0[None], geom1[None])
    return float(max(t["off_table"][0], t["tilt"][0] >= 0.5, t["disturb"][0] >= 0.5))


def gripper_cmd(geom0: np.ndarray) -> float:
    """Current gripper command inferred from the finger opening (Panda: ~0.08 open, ~0 closed)."""
    from retryvla.geom.features import IDX

    return 1.0 if float(geom0[IDX["grip_open"]]) < 0.04 else -1.0
