"""ACR Tier B: hidden physical parameters injected into a LIBERO-Recover failure state WITHOUT changing the render.

A hypothesis h = {"type": <str>, "mag": <float>} modifies the MuJoCo model of the manipulated object (first
(:obj_of_interest)) or of the gripper:
  nominal     no change
  slippery    slippery to GRASP: finger geom sliding friction x mag (mag < 1) with the finger geoms' priority raised,
              so the fingers' friction governs finger-object contacts (MuJoCo otherwise takes the max of the two
              geoms). Object-table contacts are unchanged, so the resting state does not move (an earlier version
              lowered the OBJECT's friction: the bowl then slid onto the plate on its own -> not a hidden cause)
  anchored    planar dry friction on the object's free joint: dof_frictionloss[x, y] = mag (N); z / rotations free
              (the object resists sliding / pushing but can be lifted)
  heavy       object total mass = mag (kg; nominal bowl ~0.006 kg), inertia scaled by the same ratio
  weak_grip   gripper actuator force range x mag (mag < 1)
Model parameters do not enter the renderer: the first observation is pixel-identical (tests/test_causal.py).
Every call first restores the NOMINAL parameters (cached per model), so a worker never leaks a previous hypothesis.
"""

from __future__ import annotations

import numpy as np

TYPES = ("nominal", "slippery", "anchored", "heavy", "weak_grip")


def _model_ids(sim):
    """(object geom ids, object body id, object free-joint dof adr, gripper actuator ids) of a LiberoSim."""
    from retryvla.geom.features import _is_child

    env = sim._env.env
    m = env.sim.model
    obj = sim._ooi[0]
    b = int(env.obj_body_id[obj])
    geoms = [i for i in range(m.ngeom) if _is_child(m, int(m.geom_bodyid[i]), b)]
    jnt = int(m.body_jntadr[b])
    dof = int(m.jnt_dofadr[jnt]) if jnt >= 0 and int(m.jnt_type[jnt]) == 0 else -1
    acts = [i for i in range(m.nu) if "gripper" in (m.actuator_id2name(i) or "")]
    bodies = [i for i in range(m.nbody) if _is_child(m, i, b)]
    g = env.robots[0].gripper
    g = g if not isinstance(g, dict) else next(iter(g.values()))
    fingers = sorted({m.geom_name2id(n) for k in ("left_finger", "right_finger") for n in g.important_geoms[k]})
    return geoms, b, bodies, dof, acts, fingers


def _nominal(sim) -> dict:
    """Cache of the nominal parameters, keyed by the underlying mjModel object (rebuilt env = new cache)."""
    m = sim._env.env.sim.model
    key = id(m._model)
    c = getattr(sim, "_acr_nominal", None)
    if c is not None and c["key"] == key:
        return c
    geoms, b, bodies, dof, acts, fingers = _model_ids(sim)
    c = {"key": key, "geoms": geoms, "body": b, "bodies": bodies, "dof": dof, "acts": acts, "fingers": fingers,
         "f_friction": m.geom_friction[fingers].copy(), "f_priority": m.geom_priority[fingers].copy(),
         "friction": m.geom_friction[geoms].copy(), "priority": m.geom_priority[geoms].copy(),
         "mass": m.body_mass[bodies].copy(), "inertia": m.body_inertia[bodies].copy(),
         "frictionloss": m.dof_frictionloss[dof: dof + 6].copy() if dof >= 0 else None,
         "forcerange": m.actuator_forcerange[acts].copy()}
    sim._acr_nominal = c
    return c


def reset_nominal(sim) -> None:
    c = _nominal(sim)
    m = sim._env.env.sim.model
    m.geom_friction[c["geoms"]] = c["friction"]
    m.geom_priority[c["geoms"]] = c["priority"]
    m.geom_friction[c["fingers"]] = c["f_friction"]
    m.geom_priority[c["fingers"]] = c["f_priority"]
    m.body_mass[c["bodies"]] = c["mass"]
    m.body_inertia[c["bodies"]] = c["inertia"]
    if c["dof"] >= 0:
        m.dof_frictionloss[c["dof"]: c["dof"] + 6] = c["frictionloss"]
    m.actuator_forcerange[c["acts"]] = c["forcerange"]


def apply_hidden(sim, h: dict | None) -> dict:
    """Reset to nominal, then apply hypothesis h (None / nominal = nominal). Returns a summary of what was set."""
    reset_nominal(sim)
    if not h or h.get("type", "nominal") == "nominal":
        return {"type": "nominal"}
    c = _nominal(sim)
    m = sim._env.env.sim.model
    t, mag = h["type"], float(h["mag"])
    if t == "slippery":
        f = c["f_friction"].copy()
        f[:, 0] *= mag
        m.geom_friction[c["fingers"]] = f
        m.geom_priority[c["fingers"]] = 10
    elif t == "anchored":
        if c["dof"] < 0:
            raise ValueError("object has no free joint")
        fl = c["frictionloss"].copy()
        fl[0:2] = mag
        m.dof_frictionloss[c["dof"]: c["dof"] + 6] = fl
    elif t == "heavy":
        ratio = mag / max(float(c["mass"].sum()), 1e-9)
        m.body_mass[c["bodies"]] = c["mass"] * ratio
        m.body_inertia[c["bodies"]] = c["inertia"] * ratio
    elif t == "weak_grip":
        m.actuator_forcerange[c["acts"]] = c["forcerange"] * mag
    else:
        raise ValueError(f"unknown hidden type {t}")
    return {"type": t, "mag": mag}


def particles(cfg_types: dict) -> list[dict]:
    """Planning particles: nominal + the two ENDS of every type's magnitude range (never the true magnitude)."""
    out = [{"type": "nominal", "mag": 1.0}]
    for t, rng in cfg_types.items():
        lo, hi = float(rng[0]), float(rng[1])
        out += [{"type": t, "mag": lo}, {"type": t, "mag": hi}]
    return out


def draw_true(cfg_types: dict, t: str, seed: int) -> dict:
    """True h* of a case: magnitude uniform strictly inside the range (25-75 % quantiles), seeded per case."""
    if t == "nominal":
        return {"type": "nominal", "mag": 1.0}
    lo, hi = (float(x) for x in cfg_types[t])
    u = np.random.default_rng(seed).uniform(0.25, 0.75)
    return {"type": t, "mag": float(lo + u * (hi - lo))}
