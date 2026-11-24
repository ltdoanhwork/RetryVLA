"""CRIT-V rich state records (sim side): privileged symbolic state, BDDL goal predicates, contacts and a labelled,
subsampled point cloud from MuJoCo depth + segmentation of the LIBERO cameras.

Used (a) inside a sim worker during recording (`retryvla.envs.vec._RecordingSim`, opt-in `record_cfg["rich"]`), so
verification runs (compare_arms / goal search, `m5b.rich_record`) and the bridge replay (`scripts/crit_rich_replay.py`)
produce the SAME format, and (b) offline by the critic variants (`retryvla.critic.variants`).

Symbolic vector (SYM_KEYS, S dims): RGR geometry (retryvla.geom.features.GEOM_KEYS, 16) + extra contacts / grasp +
up to MAX_PRED BDDL goal-predicate truths (padded with -1) + fraction of goal predicates true.
Point cloud: agent-view + wrist depth at `res` px, back-projected to world, table frame (z relative to table top),
workspace crop, per-class random subsample to fixed quotas -> xyz [N, 3] float16, label [N] uint8 with
PC_CLASSES (255 = padding). Classes: 0 robot (arm + gripper), 1 obj_of_interest[0], 2 obj_of_interest[1],
3 other objects / fixtures (not table / floor / walls).
"""

from __future__ import annotations

import numpy as np

from retryvla.geom.features import GEOM_KEYS, sim_geom

MAX_PRED = 3
EXTRA_KEYS = ("both_finger_contact", "grasped", "obj_goal_contact", "robot_other_contact", "n_robot_contacts")
SYM_KEYS = tuple(GEOM_KEYS) + EXTRA_KEYS + tuple(f"pred{i}" for i in range(MAX_PRED)) + ("pred_frac",)
S = len(SYM_KEYS)
SYM_IDX = {k: i for i, k in enumerate(SYM_KEYS)}
PC_CLASSES = ("robot", "obj", "goal", "other")
PC_QUOTA = (128, 160, 128, 96)   # points per class (sum = PC_N)
PC_N = int(sum(PC_QUOTA))
CAMS = ("agentview", "robot0_eye_in_hand")


def _is_child(m, body: int, root: int) -> bool:
    while body > 0:
        if body == root:
            return True
        body = int(m.body_parentid[body])
    return False


def _resolve(m, bid: dict, name: str) -> tuple[str | None, int | None]:
    """BDDL obj_of_interest name -> (owning object body name, site id or None). Regions / sides such as
    `wooden_cabinet_1_middle_region` or `flat_stove_1_cook_region` are sites of an object (longest-prefix match);
    their position is the site position. Table regions resolve to (None, site)."""
    sid = None
    try:
        sid = int(m.site_name2id(name))
    except Exception:
        sid = None
    if name in bid:
        return name, sid
    best = max((k for k in bid if name.startswith(k + "_")), key=len, default=None)
    if best is not None and any(s_ in best.lower() for s_ in ("table", "floor", "wall")):
        best = None
    return best, sid


def _pos(env, c: dict, k: int) -> np.ndarray:
    name, sid = c["ooi"][k]
    d = env.sim.data
    if sid is not None:
        return np.asarray(d.site_xpos[sid], float)
    if name is not None:
        return np.asarray(d.body_xpos[env.obj_body_id[name]], float)
    return np.zeros(3)


def _setup(sim) -> dict:
    """Per-bddl cache: geom -> class, finger geoms, obj / goal geoms, goal predicates."""
    c = getattr(sim, "_rich_cache", None)
    if c is not None and c["bddl"] == sim._bddl:
        return c
    env = sim._env.env
    m = env.sim.model
    bid = env.obj_body_id
    ooi = [_resolve(m, bid, n) for n in list(sim._ooi)[:2]] + [(None, None), (None, None)]
    roots = {}
    for name, b in bid.items():
        low = name.lower()
        if any(s in low for s in ("table", "floor", "wall")):
            continue
        roots[name] = int(b)
    geom_cls = np.full(m.ngeom, 255, np.uint8)
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        bname = m.body_id2name(b) or ""
        if bname.startswith(("robot0", "gripper0")):
            geom_cls[g] = 0
            continue
        for name, r in roots.items():
            if _is_child(m, b, r):
                geom_cls[g] = 1 if name == ooi[0][0] else 2 if name == ooi[1][0] else 3
                break
    gr = env.robots[0].gripper
    gr = gr if not isinstance(gr, dict) else next(iter(gr.values()))
    left = {m.geom_name2id(n) for n in gr.important_geoms["left_finger"]}
    right = {m.geom_name2id(n) for n in gr.important_geoms["right_finger"]}
    try:
        preds = list(env.parsed_problem["goal_state"])
    except Exception:
        preds = []
    c = {"bddl": sim._bddl, "geom_cls": geom_cls, "left": left, "right": right,
         "obj": set(np.nonzero(geom_cls == 1)[0].tolist()), "goal": set(np.nonzero(geom_cls == 2)[0].tolist()),
         "other": set(np.nonzero(geom_cls == 3)[0].tolist()), "robot": set(np.nonzero(geom_cls == 0)[0].tolist()),
         "preds": preds, "ooi": ooi[:2]}
    sim._rich_cache = c
    return c


def symbolic(sim) -> np.ndarray:
    """[S] float32 privileged symbolic state of the current sim state."""
    c = _setup(sim)
    env = sim._env.env
    d = env.sim.data
    g = sim_geom(sim)
    # obj / goal positions of region-type objects of interest (LIBERO-Goal drawers, stove, rack): site positions
    tz = float(env.table_offset[2])
    po, pg = _pos(env, c, 0), _pos(env, c, 1)
    g[0:3] = [po[0], po[1], po[2] - tz]
    g[3:6] = [pg[0], pg[1], pg[2] - tz]
    g[11] = float(np.linalg.norm(np.asarray(env.robots[0].controller.ee_pos, float) - po))
    lf = rf = og = ro = 0.0
    sup = False
    nrc = 0
    for i in range(d.ncon):
        ct = d.contact[i]
        a, b = int(ct.geom1), int(ct.geom2)
        if (a in c["left"] and b in c["obj"]) or (b in c["left"] and a in c["obj"]):
            lf = 1.0
        if (a in c["right"] and b in c["obj"]) or (b in c["right"] and a in c["obj"]):
            rf = 1.0
        if (a in c["obj"] and b in c["goal"]) or (b in c["obj"] and a in c["goal"]):
            og = 1.0
        if (a in c["obj"]) != (b in c["obj"]):
            oth = b if a in c["obj"] else a
            if oth not in c["robot"]:
                sup = True
        ra, rb = a in c["robot"], b in c["robot"]
        if ra != rb:
            nrc += 1
            other = b if ra else a
            if other in c["other"] or other in c["goal"]:
                ro = 1.0
    both = float(lf and rf)
    # grasped: both fingers touch the object AND the object touches nothing but the robot (held in the air)
    grasped = float(both and not sup)
    pv = np.full(MAX_PRED, -1.0, np.float32)
    ok = []
    for k, st in enumerate(c["preds"][:MAX_PRED]):
        try:
            v = float(bool(env._eval_predicate(st)))
        except Exception:
            v = -1.0
        pv[k] = v
        if v >= 0:
            ok.append(v)
    frac = float(np.mean(ok)) if ok else -1.0
    return np.concatenate([g, np.array([both, grasped, og, ro, float(nrc)], np.float32), pv,
                           np.array([frac], np.float32)]).astype(np.float32)


def _render_seg(s, cam: str, res: int) -> np.ndarray:
    """robosuite MjRenderContext.read_pixels(segmentation=True) with an int32 cast (the original overflows uint8
    under numpy 2). -> [res, res, 2] (objtype, objid), -1 = background."""
    import mujoco

    from robosuite.utils.binding_utils import _MjSim_render_lock

    ctx = s._render_context_offscreen
    with _MjSim_render_lock:
        ctx.render(width=res, height=res, camera_id=s.model.camera_name2id(cam), segmentation=True)
        rgb = np.empty((res, res, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb=rgb, depth=None, viewport=mujoco.MjrRect(0, 0, res, res), con=ctx.con)
    seg = rgb[..., 0].astype(np.int32) + rgb[..., 1].astype(np.int32) * 256 + rgb[..., 2].astype(np.int32) * 65536
    seg[seg >= ctx.scn.ngeom + 1] = 0
    ids = np.full((ctx.scn.ngeom + 1, 2), -1, np.int32)
    for i in range(ctx.scn.ngeom):
        g = ctx.scn.geoms[i]
        if g.segid != -1:
            ids[g.segid + 1, 0] = g.objtype
            ids[g.segid + 1, 1] = g.objid
    return ids[seg]


def _cam_points(sim, cam: str, res: int):
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix

    s = sim._env.env.sim
    _, depth = s.render(camera_name=cam, height=res, width=res, depth=True)
    seg = _render_seg(s, cam, res)
    depth = depth[::-1]
    seg = seg[::-1]
    ext = s.model.stat.extent
    far, near = s.model.vis.map.zfar * ext, s.model.vis.map.znear * ext
    z = near / (1.0 - depth * (1.0 - near / far))
    K = get_camera_intrinsic_matrix(s, cam, res, res)
    R = get_camera_extrinsic_matrix(s, cam)
    v, u = np.mgrid[0:res, 0:res]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pc = np.stack([x, y, z, np.ones_like(z)], -1).reshape(-1, 4) @ R.T
    gid = seg[..., 1].reshape(-1).astype(np.int64)
    gty = seg[..., 0].reshape(-1)
    return pc[:, :3], gid, gty


def point_cloud(sim, res: int = 96, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """-> xyz [PC_N, 3] float16 (table frame), label [PC_N] uint8 (255 = pad)."""
    import mujoco

    c = _setup(sim)
    env = sim._env.env
    rng = rng or np.random.default_rng(0)
    tz = float(env.table_offset[2])
    txy = np.asarray(env.table_offset[:2], float)
    P, L = [], []
    for cam in CAMS:
        p, gid, gty = _cam_points(sim, cam, res)
        okg = (gty == int(mujoco.mjtObj.mjOBJ_GEOM)) & (gid >= 0) & (gid < len(c["geom_cls"]))
        lab = np.full(len(p), 255, np.uint8)
        lab[okg] = c["geom_cls"][gid[okg]]
        P.append(p); L.append(lab)
    p, lab = np.concatenate(P), np.concatenate(L)
    keep = (lab != 255) & (p[:, 2] > tz - 0.05) & (p[:, 2] < tz + 0.8) & (np.abs(p[:, :2] - txy).max(1) < 0.8)
    p, lab = p[keep], lab[keep]
    p = p - np.array([txy[0], txy[1], tz])
    out_p = np.zeros((PC_N, 3), np.float16)
    out_l = np.full(PC_N, 255, np.uint8)
    o = 0
    for k, q in enumerate(PC_QUOTA):
        idx = np.nonzero(lab == k)[0]
        if len(idx):
            # 5 mm voxel dedup before sampling (both cameras see the same surfaces at different densities)
            vox = np.floor(p[idx] / 0.005).astype(np.int64)
            _, first = np.unique(vox, axis=0, return_index=True)
            idx = idx[first]
            sel = rng.choice(idx, size=min(q, len(idx)), replace=False)
            out_p[o: o + len(sel)] = p[sel]
            out_l[o: o + len(sel)] = k
        o += q
    return out_p, out_l


def capture(sim, with_pc: bool, res: int = 96, seed: int = 0) -> dict:
    out = {"sym": symbolic(sim)}
    if with_pc:
        out["pc"], out["pcl"] = point_cloud(sim, res, np.random.default_rng(seed))
    return out
