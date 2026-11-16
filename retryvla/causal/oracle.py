"""ACR-N oracle failure-cause labels for NATURAL failure states (privileged MuJoCo; design: PLAN "ACR-N").

Worker side (`oracle_tests(rsim, bddl, snap)`, run through the SimPool command `oracle_tests`):
  census   contacts at the failure snapshot, grouped by body ROOT (robot / table / obj / goal / other free object /
           fixture): finger-obj (per finger), gripper-other (fingers / hand touching anything but obj and table),
           obj-table, obj-goal, obj-other (free objects), obj-fixture; gripper opening; obj up-axis; obj height
  lift     from the exact snapshot: close 8 steps, raise the eef ~3 cm (10 steps), measure obj lift / eef lift
  move     from the exact snapshot, robot holding still (zero motion, current gripper command): an external horizontal
           force F = 3 m g on the object's body for 5 control steps (0.25 s) in each of 4 directions; max object displacement
           (movability; does the object respond to a push at all?)
None of this is visible to non-oracle arms; it is never charged to a budget.

`cause_of(t)` maps the measurements to one cause (first match, thresholds pre-registered in PLAN "ACR-N").
"""

from __future__ import annotations

import numpy as np

CAUSES = ("tipped", "held_ok", "held_slip", "stuck", "obstructed", "contact_free", "free")
MERGE = {"tipped": "contact_free", "held_ok": "held", "held_slip": "held", "stuck": "blocked",
         "obstructed": "blocked", "contact_free": "contact_free", "free": "free"}


def _roots(sim):
    env = sim._env.env
    m = env.sim.model
    ooi = list(sim._ooi) + [None, None]
    bid = env.obj_body_id
    obj_b = int(bid[ooi[0]]) if ooi[0] in bid else -1
    goal_b = int(bid[ooi[1]]) if ooi[1] in bid else -1
    free_roots = set()
    for name, b in bid.items():
        jnt = int(m.body_jntadr[b])
        if jnt >= 0 and int(m.jnt_type[jnt]) == 0:
            free_roots.add(int(m.body_rootid[b]))
    robot_root = int(m.body_rootid[m.body_name2id("robot0_base")])
    table = m.body_name2id("table")
    fingers = {m.body_name2id(n): k for k, n in (("l", "gripper0_leftfinger"), ("l", "gripper0_finger_joint1_tip"),
                                                 ("r", "gripper0_rightfinger"), ("r", "gripper0_finger_joint2_tip"))}
    return m, obj_b, goal_b, free_roots, robot_root, table, fingers


def census(sim) -> dict:
    env = sim._env.env
    m, d = env.sim.model, env.sim.data
    m, obj_b, goal_b, free_roots, robot_root, table, fingers = _roots(sim)
    root = m.body_rootid
    obj_r = int(root[obj_b]) if obj_b >= 0 else -2
    goal_r = int(root[goal_b]) if goal_b >= 0 else -2
    out = {"finger_l": 0, "finger_r": 0, "hand_obj": 0, "grip_other": 0, "obj_table": 0, "obj_goal": 0,
           "obj_other": 0, "obj_fixture": 0, "max_pen": 0.0, "grip_other_names": []}

    def kind(b):
        r = int(root[b])
        if r == obj_r:
            return "obj"
        if r == robot_root:
            return "robot"
        if b == table or r == table:
            return "table"
        if r == goal_r:
            return "goal"
        if r in free_roots:
            return "other"
        return "fixture"

    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        k1, k2 = kind(b1), kind(b2)
        out["max_pen"] = max(out["max_pen"], -float(c.dist))
        for (ba, ka), (bb, kb) in (((b1, k1), (b2, k2)), ((b2, k2), (b1, k1))):
            if ka == "obj":
                if kb == "robot":
                    if bb in fingers:
                        out["finger_" + fingers[bb]] = 1
                    else:
                        out["hand_obj"] = 1
                elif kb == "table":
                    out["obj_table"] = 1
                elif kb == "goal":
                    out["obj_goal"] = 1
                elif kb == "other":
                    out["obj_other"] = 1
                elif kb == "fixture":
                    out["obj_fixture"] = 1
            if ka == "robot" and kb not in ("obj", "robot"):
                out["grip_other"] = 1
                out["grip_other_names"].append(m.body_id2name(bb))
    out["grip_other_names"] = sorted(set(out["grip_other_names"]))
    return out


def _obj_pos(sim):
    env = sim._env.env
    return np.asarray(env.sim.data.body_xpos[env.obj_body_id[sim._ooi[0]]], float).copy()


def _eef(sim):
    return np.asarray(sim._env.env.robots[0].controller.ee_pos, float).copy()


def oracle_tests(rsim, bddl: str, snap: dict) -> dict:
    """Worker side. rsim = vec._RecordingSim. Returns census + lift / movability tests + geometry."""
    from retryvla.causal.probes import gripper_cmd
    from retryvla.geom.features import IDX, sim_geom

    rsim.restore_hidden(bddl, snap, None)
    sim = rsim.sim
    g0 = sim_geom(sim)
    t = census(sim)
    t["grip_open"] = float(g0[IDX["grip_open"]])
    t["obj_upz"] = float(g0[IDX["obj_upz"]])
    t["obj_z"] = float(g0[IDX["obj_z"]])
    t["d_obj_goal_xy"] = float(np.linalg.norm(g0[[IDX["obj_x"], IDX["obj_y"]]] - g0[[IDX["goal_x"], IDX["goal_y"]]]))
    t["d_grip_obj"] = float(g0[IDX["d_grip_obj"]])
    g = gripper_cmd(g0)
    # lift test (privileged measurement of the object's response; the robot actions are the lift2 probe's)
    o0, e0 = _obj_pos(sim), _eef(sim)
    a = np.zeros(7, np.float32)
    a[6] = 1.0
    for _ in range(8):
        sim.step(a)
    o1, e1 = _obj_pos(sim), _eef(sim)
    a[2] = 0.3
    for _ in range(10):
        sim.step(a)
    o2, e2 = _obj_pos(sim), _eef(sim)
    t["lift_eef"] = float(e2[2] - e1[2])
    t["lift_obj"] = float(o2[2] - o1[2])
    t["lift_ratio"] = float(t["lift_obj"] / max(t["lift_eef"], 1e-3))
    t["lift_grip_end"] = float(sim_geom(sim)[IDX["grip_open"]])
    # movability test: external force on the object, robot frozen
    env = sim._env.env
    m = env.sim.model
    b = int(env.obj_body_id[sim._ooi[0]])
    mass = float(m.body_subtreemass[b])
    F = 3.0 * mass * 9.81
    disp = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rsim.restore_hidden(bddl, snap, None)
        sim = rsim.sim
        d = sim._env.env.sim.data
        p0 = _obj_pos(sim)
        a = np.zeros(7, np.float32)
        a[6] = g
        for _ in range(5):
            d.xfrc_applied[b, :3] = (F * dx, F * dy, 0.0)
            sim.step(a)
        d.xfrc_applied[b, :] = 0.0
        disp.append(float(np.linalg.norm(_obj_pos(sim)[:2] - p0[:2])))
    t["move_disp"] = disp
    t["move_max"] = float(max(disp))
    t["mass"] = mass
    rsim.restore_hidden(bddl, snap, None)  # leave the worker at the failure state
    t["cause"] = cause_of(t)
    return t


def cause_of(t: dict) -> str:
    """Pre-registered rules (PLAN "ACR-N"), first match."""
    finger = t["finger_l"] or t["finger_r"]
    closed = t["grip_open"] < 0.04
    touches = t["obj_other"] or t["obj_fixture"] or t["obj_goal"]
    if t["obj_upz"] < 0.8:
        return "tipped"
    if finger and closed and t["finger_l"] and t["finger_r"] and t["lift_ratio"] >= 0.6:
        return "held_ok"
    if finger:
        return "held_slip"
    if (touches or t["obj_z"] > 0.03) and t["move_max"] < 0.01:
        return "stuck"
    if t["grip_other"]:
        return "obstructed"
    if touches:
        return "contact_free"
    return "free"
