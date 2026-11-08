"""Geometric state features + geometric energy E_geom for Recursive Policy-Conditioned Geometric Recovery (RGR).

Two sources of the SAME feature vector g (GEOM_KEYS, G = 16 dims):
  * ORACLE: `sim_geom(sim)` reads privileged MuJoCo state (object poses, contacts, gripper) of a LiberoSim
    -> E_geom on the TRUE state (upper bound; never available on a real robot).
  * OBSERVATION-ONLY: `retryvla.geom.head.GeomHead` predicts g from the WM v1 latent (DINOv2 tokens + proprio),
    so E_geom can be computed on WM-IMAGINED states.

Objects: obj = first (:obj_of_interest) of the BDDL (the object to manipulate, e.g. the black bowl),
goal = second (e.g. the plate). Positions are world coordinates (m); z relative to the table top.

E_geom(g_start, g_end) (`geom_energy`) sums terms in [0, 1]; terms that describe the object state are counted
only as a WORSENING relative to the start state (a failure state may already have the bowl tipped / off the plate):
  off_table   any free object fell below the table top / outside the table (oracle) or obj/goal did (head)
  penetrate   max contact penetration depth (oracle; head predicts it but it is noisy) / 1 cm
  tilt        obj up-axis tilt increase (bowl tipped over)
  disturb     goal-object displacement (rigid-body consistency: the plate should not be pushed) / 5 cm
  align       gripper-obj distance / 30 cm (gripper-object alignment; soft, small weight)
  infeasible  obj lifted > 3 cm above its START height without finger contact (floating object = physically
              inconsistent; on true states only transient during a fall)
GeomValid = off_table == 0 and tilt, disturb, penetrate, infeasible < 0.5.
"""

from __future__ import annotations

import numpy as np

GEOM_KEYS = ("obj_x", "obj_y", "obj_z", "goal_x", "goal_y", "goal_z", "obj_upz", "eef_x", "eef_y", "eef_z",
             "grip_open", "d_grip_obj", "finger_obj_contact", "max_pen", "off_table", "obj_lift")
G = len(GEOM_KEYS)
IDX = {k: i for i, k in enumerate(GEOM_KEYS)}
BINARY = ("finger_obj_contact", "off_table")            # head: BCE
WEIGHTS = {"off_table": 1.0, "penetrate": 0.5, "tilt": 0.5, "disturb": 0.5, "align": 0.25, "infeasible": 0.5}


def _upz(quat_wxyz: np.ndarray) -> float:
    """z-component of the body's local z axis in world frame (1 = upright, -1 = upside down)."""
    w, x, y, z = quat_wxyz
    return float(1.0 - 2.0 * (x * x + y * y))


def sim_geom(sim) -> np.ndarray:
    """Privileged geometry vector [G] of a LiberoSim's current state (ORACLE)."""
    env = sim._env.env
    m, d = env.sim.model, env.sim.data
    ooi = list(sim._ooi) + [None, None]
    bid = env.obj_body_id
    tz = float(env.table_offset[2])
    half = np.asarray(env.table_full_size[:2], float) / 2.0
    txy = np.asarray(env.table_offset[:2], float)

    def pos(name):
        return np.asarray(d.body_xpos[bid[name]], float) if name in bid else np.zeros(3)

    obj, goal = ooi[0], ooi[1]
    po, pg = pos(obj), pos(goal)
    upz = _upz(np.asarray(d.body_xquat[bid[obj]], float)) if obj in bid else 1.0
    eef = np.asarray(env.robots[0].controller.ee_pos, float)
    gq = np.asarray(d.qpos[env.robots[0]._ref_gripper_joint_pos_indexes], float)
    grip_open = float(np.abs(gq).sum())
    # contacts: max penetration over all contacts; finger <-> obj contact
    cache = getattr(sim, "_geom_cache", None)
    if cache is None or cache[0] != sim._bddl:
        g = env.robots[0].gripper
        g = g if not isinstance(g, dict) else next(iter(g.values()))
        fingers = {m.geom_name2id(n) for k in ("left_finger", "right_finger") for n in g.important_geoms[k]}
        obj_geoms = set()
        if obj in bid:
            b = bid[obj]
            obj_geoms = {i for i in range(m.ngeom) if _is_child(m, int(m.geom_bodyid[i]), b)}
        sim._geom_cache = cache = (sim._bddl, fingers, obj_geoms)
    _, fingers, obj_geoms = cache
    max_pen, fo = 0.0, 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        max_pen = max(max_pen, -float(c.dist))
        if (c.geom1 in fingers and c.geom2 in obj_geoms) or (c.geom2 in fingers and c.geom1 in obj_geoms):
            fo = 1.0
    off = 0.0
    for name, b in bid.items():
        jnt = int(m.body_jntadr[b])
        if jnt < 0 or int(m.jnt_type[jnt]) != 0:     # free-joint objects only
            continue
        p = np.asarray(d.body_xpos[b], float)
        if p[2] < tz - 0.03 or np.any(np.abs(p[:2] - txy) > half + 0.02):
            off = 1.0
    out = np.array([po[0], po[1], po[2] - tz, pg[0], pg[1], pg[2] - tz, upz, eef[0], eef[1], eef[2] - tz,
                    grip_open, float(np.linalg.norm(eef - po)), fo, max_pen, off, max(0.0, po[2] - tz)],
                   dtype=np.float32)
    return out


def _is_child(m, body: int, root: int) -> bool:
    while body > 0:
        if body == root:
            return True
        body = int(m.body_parentid[body])
    return False


def geom_terms(g0: np.ndarray, g1: np.ndarray, base_lift: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-term energies for transitions g0 -> g1 (arrays [..., G]); values in [0, 1].
    Binary entries may be probabilities (head)."""
    g0, g1 = np.asarray(g0, np.float32), np.asarray(g1, np.float32)
    I = IDX
    off = np.clip(g1[..., I["off_table"]] - g0[..., I["off_table"]], 0, 1)
    pen = np.clip((g1[..., I["max_pen"]] - np.maximum(g0[..., I["max_pen"]], 0.002)) / 0.01, 0, 1)
    tilt = np.clip((g0[..., I["obj_upz"]] - g1[..., I["obj_upz"]]) / 0.6, 0, 1)
    dg = np.linalg.norm(g1[..., I["goal_x"]: I["goal_z"] + 1] - g0[..., I["goal_x"]: I["goal_z"] + 1], axis=-1)
    disturb = np.clip(dg / 0.05, 0, 1)
    align = np.clip(g1[..., I["d_grip_obj"]] / 0.30, 0, 1)
    lifted = np.clip((g1[..., I["obj_lift"]] - g0[..., I["obj_lift"]] - 0.03) / 0.03, 0, 1)
    infeas = lifted * (1.0 - np.clip(g1[..., I["finger_obj_contact"]], 0, 1))
    return {"off_table": off, "penetrate": pen, "tilt": tilt, "disturb": disturb, "align": align,
            "infeasible": infeas}


def geom_energy(g0, g1, weights: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """E_geom [...] (weighted sum of geom_terms), GeomValid [...] bool, terms."""
    w = {**WEIGHTS, **(weights or {})}
    t = geom_terms(g0, g1)
    E = sum(w[k] * v for k, v in t.items())
    valid = (t["off_table"] < 0.5) & (t["tilt"] < 0.5) & (t["disturb"] < 0.5) & (t["penetrate"] < 0.5) & \
            (t["infeasible"] < 0.5)
    return E, valid, t
