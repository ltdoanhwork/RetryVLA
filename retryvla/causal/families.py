"""ACR recovery families: parametric recovery prefixes (scripted, closed-loop waypoint servo in the worker) followed by
the frozen pi continuation (same protocol as every search arm: recovery prefix -> pi).

Coordinates = retryvla.geom.features.sim_geom (x, y world; z above the table top). Object-relative families get the
object / goal positions with SIMULATED PERCEPTION noise (sigma = cfg perception_sigma, default 2.5 cm = median error of
the RGR geometry head on real latents), drawn per variant; never the exact privileged pose.

Families (name -> what the prefix does, then pi takes over):
  vla          nothing (pi from the failure state, new continuation seed per variant)
  hover        open, rise, move above the object (z + 12 cm): pi grasps from a canonical pre-grasp pose
  grasp_slow   open, above the rim, descend, close 15 steps, SLOW lift 8 cm (gentle: slippery / weak grip)
  grasp_firm   open, above the rim, descend 1.5 cm deeper, close 25 steps, fast lift 8 cm
  push_goal    gripper closed, go behind the object (opposite the goal), lower, push along object -> goal, back off
  retreat      open, rise 15 cm, move 10 cm back towards the robot base (re-observe from afar)
"""

from __future__ import annotations

import numpy as np

from retryvla.causal.probes import STEP_M

FAMILIES = ("vla", "hover", "grasp_slow", "grasp_firm", "push_goal", "retreat", "place_goal", "pull_free")
PREFIX_MAX = 140


def wp(pos=None, grip=-1.0, n=20, speed=1.0, tol=0.006):
    """pos None = hold (no arm motion) for n steps; else servo to pos for at most n steps (stops within tol)."""
    return {"pos": None if pos is None else np.asarray(pos, np.float64), "grip": float(grip), "n": int(n),
            "speed": float(speed), "tol": float(tol)}


def build(name: str, g0: np.ndarray, rng: np.random.Generator, sigma: float) -> list[dict]:
    """Waypoints of one variant of family `name` at a state with privileged geometry g0 (only obj / goal / eef
    positions are read; obj / goal get perception noise)."""
    from retryvla.geom.features import IDX

    obj = g0[[IDX["obj_x"], IDX["obj_y"], IDX["obj_z"]]].astype(np.float64)
    goal = g0[[IDX["goal_x"], IDX["goal_y"], IDX["goal_z"]]].astype(np.float64)
    eef = g0[[IDX["eef_x"], IDX["eef_y"], IDX["eef_z"]]].astype(np.float64)
    obj = obj + rng.normal(0, sigma, 3) * np.array([1, 1, 0.5])
    goal = goal + rng.normal(0, sigma, 3) * np.array([1, 1, 0.5])
    up = max(eef[2], obj[2] + 0.12)
    if name == "vla":
        return []
    if name == "retreat":
        back = eef + np.array([-0.10, 0.0, 0.0])
        return [wp(None, -1, 6), wp([eef[0], eef[1], eef[2] + 0.15], -1, 25), wp([back[0], back[1], eef[2] + 0.15], -1, 20)]
    if name == "hover":
        j = rng.normal(0, 0.01, 2)
        return [wp(None, -1, 6), wp([eef[0], eef[1], up], -1, 25),
                wp([obj[0] + j[0], obj[1] + j[1], obj[2] + 0.12 + rng.uniform(-0.02, 0.02)], -1, 40)]
    if name in ("grasp_slow", "grasp_firm"):
        th = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.025, 0.045)                    # bowl rim radius region
        tgt = obj[:2] + r * np.array([np.cos(th), np.sin(th)])
        dz = rng.uniform(0.02, 0.04) - (0.015 if name == "grasp_firm" else 0.0)
        lift_speed = 0.25 if name == "grasp_slow" else 1.0
        close_n = 15 if name == "grasp_slow" else 25
        return [wp(None, -1, 6), wp([eef[0], eef[1], up], -1, 25), wp([tgt[0], tgt[1], up], -1, 30),
                wp([tgt[0], tgt[1], obj[2] + dz], -1, 25, speed=0.5), wp(None, 1, close_n),
                wp([tgt[0], tgt[1], obj[2] + dz + 0.08], 1, 40, speed=lift_speed)]
    if name == "push_goal":
        d = goal[:2] - obj[:2]
        dist = float(np.linalg.norm(d))
        u = d / max(dist, 1e-6)
        behind = obj[:2] - u * rng.uniform(0.07, 0.09)
        push_to = obj[:2] + u * max(0.0, dist - rng.uniform(0.0, 0.02))
        zp = obj[2] + rng.uniform(0.015, 0.03)
        return [wp(None, 1, 6), wp([eef[0], eef[1], up], 1, 25), wp([behind[0], behind[1], up], 1, 30),
                wp([behind[0], behind[1], zp], 1, 25, speed=0.5), wp([push_to[0], push_to[1], zp], 1, 40, speed=0.4),
                wp([push_to[0] - u[0] * 0.05, push_to[1] - u[1] * 0.05, zp + 0.10], -1, 15)]
    if name == "place_goal":
        # ACR-N: keep / close the grasp, rise, carry above the goal, lower, release, rise (secure-grasp option)
        zc = max(eef[2], goal[2] + 0.14)
        gz = goal[2] + rng.uniform(0.06, 0.09)
        return [wp(None, 1, 6), wp([eef[0], eef[1], zc], 1, 25, speed=0.6), wp([goal[0], goal[1], zc], 1, 40, speed=0.6),
                wp([goal[0], goal[1], gz], 1, 20, speed=0.5), wp(None, -1, 10), wp([goal[0], goal[1], zc], -1, 15)]
    if name == "pull_free":
        # ACR-N: gripper closed, from the far side push the object ~5 cm toward the robot (-x: out of clutter / off
        # a fixture edge), then open and back off; pi re-grasps from a freed state
        d = rng.uniform(0.04, 0.06)
        far = obj[:2] + np.array([0.075, rng.normal(0, 0.01)])
        zp = obj[2] + rng.uniform(0.015, 0.03)
        return [wp(None, 1, 6), wp([eef[0], eef[1], up], 1, 25), wp([far[0], far[1], up], 1, 30),
                wp([far[0], far[1], zp], 1, 25, speed=0.5), wp([far[0] - 0.075 - d, far[1], zp], 1, 30, speed=0.4),
                wp([far[0] - 0.075 - d - 0.03, far[1], zp + 0.10], -1, 15)]
    raise ValueError(name)


def run_waypoints(rsim, wps: list[dict], stop_on_success: bool = True, max_steps: int = PREFIX_MAX) -> dict:
    """Worker side: execute waypoints with a proportional servo on the TRUE eef position (proprioception)."""
    from retryvla.geom.features import IDX, sim_geom

    n, success = 0, False
    for w in wps:
        for _ in range(w["n"]):
            if n >= max_steps or (success and stop_on_success):
                break
            a = np.zeros(7, np.float32)
            a[6] = w["grip"]
            if w["pos"] is not None:
                g = sim_geom(rsim.sim)
                err = w["pos"] - g[[IDX["eef_x"], IDX["eef_y"], IDX["eef_z"]]]
                if np.linalg.norm(err) < w["tol"]:
                    break
                a[:3] = np.clip(err / STEP_M, -w["speed"], w["speed"])
            out = rsim.step(a)
            n += 1
            success = success or bool(out["success"])
    return {"n": n, "success": success}
