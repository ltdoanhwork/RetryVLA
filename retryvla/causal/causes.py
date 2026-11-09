"""ACR Tier A: privileged cause label of a NATURAL failure state (no injected physics). Used only as (i) the
oracle_cause arm's knowledge and (ii) training labels of the cross-fitted response -> cause classifier.

Labels (first match, from sim_geom of the failure snapshot):
  held       finger-object contact and gripper closed (width < 0.04): the object is in / against the closed gripper
  blocked    object elevated > 3 cm above the table (in a drawer, on the ramekin / cabinet / stove / cookie box),
             or another free object's centre within 9 cm in xy (cluttered grasp)
  near_goal  object-goal xy distance < 8 cm (resting next to / on the rim of the plate; mostly needs a re-place)
  free       otherwise (object free on the table, far from the goal: approach / localization failure)
"""

from __future__ import annotations

import numpy as np

CAUSES = ("held", "blocked", "near_goal", "free")


def other_objects_min_dist(sim) -> float:
    env = sim._env.env
    m, d = env.sim.model, env.sim.data
    ooi = list(sim._ooi)
    if not ooi:
        return 1e9
    po = np.asarray(d.body_xpos[env.obj_body_id[ooi[0]]], float)
    best = 1e9
    for name, b in env.obj_body_id.items():
        if name in ooi[:2]:
            continue
        jnt = int(m.body_jntadr[b])
        if jnt < 0 or int(m.jnt_type[jnt]) != 0:
            continue
        best = min(best, float(np.linalg.norm(np.asarray(d.body_xpos[b], float)[:2] - po[:2])))
    return best


def cause_label(g: np.ndarray, clutter_dist: float) -> str:
    from retryvla.geom.features import IDX

    if g[IDX["finger_obj_contact"]] > 0.5 and g[IDX["grip_open"]] < 0.04:
        return "held"
    if g[IDX["obj_z"]] > 0.03 or clutter_dist < 0.09:
        return "blocked"
    if np.linalg.norm(g[[IDX["obj_x"], IDX["obj_y"]]] - g[[IDX["goal_x"], IDX["goal_y"]]]) < 0.08:
        return "near_goal"
    return "free"
