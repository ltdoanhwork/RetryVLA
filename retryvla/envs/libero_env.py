"""LIBERO simulator wrapper used everywhere in RetryVLA.

Conventions (match lerobot's LIBERO pipeline + the LIBERO-Recovery client):
- Policy/dataset images are the raw MuJoCo renders flipped on H and W (``img[::-1, ::-1]``).
- ``state`` = [eef_pos(3), eef_axis_angle(3), gripper_qpos(2)] (float32, 8-d).
- Actions are 7-d OSC deltas in [-1, 1]; gripper -1 = open, +1 = close.

A ``LiberoSim`` owns one OffScreenRenderEnv and can be re-pointed at any ``.bddl``
(nominal LIBERO task or a LIBERO-Recover failure scene). The env for a given bddl is
cached so branching (restore -> act -> restore) is cheap.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

DUMMY_ACTION = np.array([0.0] * 6 + [-1.0], dtype=np.float32)


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = math.sqrt(max(1.0 - q[3] * q[3], 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


@contextlib.contextmanager
def _no_placement_check():
    """Build samplers with ensure_valid_placement=False (collision check off). Used only as a
    fallback for scenes whose tight custom regions make the throwaway initial layout infeasible."""
    import inspect

    from libero.libero.envs.regions import base_region_sampler as brs
    from robosuite.utils import placement_samplers as ps

    patched = []
    for mod in (brs, ps):
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            init = cls.__dict__.get("__init__")
            if init is None or "ensure_valid_placement" not in inspect.signature(init).parameters:
                continue

            def make(orig):
                sig = inspect.signature(orig)

                def __init__(self, *a, **k):
                    ba = sig.bind_partial(self, *a, **k)
                    ba.arguments["ensure_valid_placement"] = False
                    orig(*ba.args, **ba.kwargs)
                return __init__

            cls.__init__ = make(init)
            patched.append((cls, init))
    try:
        yield
    finally:
        for cls, init in patched:
            cls.__init__ = init


def _grippers(robot) -> dict[str, Any]:
    g = getattr(robot, "gripper", None)
    if g is None:
        return {}
    return dict(g) if isinstance(g, dict) else {"main": g}


def parse_bddl_language(bddl_path: str | Path) -> str:
    with open(bddl_path) as f:
        for line in f:
            m = re.search(r"\(:language\s+(.+?)\s*\)", line)
            if m:
                return m.group(1).strip()
    return Path(bddl_path).stem


def parse_bddl_objects_of_interest(bddl_path: str | Path) -> list[str]:
    txt = Path(bddl_path).read_text()
    m = re.search(r"\(:obj_of_interest(.*?)\)", txt, flags=re.S)
    return m.group(1).split() if m else []


def format_obs(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    """Raw robosuite obs -> {image, image2, state} in policy/dataset convention."""
    img = np.ascontiguousarray(raw["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1])
    state = np.concatenate(
        [raw["robot0_eef_pos"], quat2axisangle(raw["robot0_eef_quat"]), raw["robot0_gripper_qpos"]]
    ).astype(np.float32)
    return {"image": img, "image2": wrist, "state": state}


# ----------------------------------------------------------------------------------------
# LIBERO-Recover scenes
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Scene:
    tag: str            # <task_folder>__<episode>__<step>  (official tag format)
    bddl: str
    npy: str
    task_folder: str    # original LIBERO task name (underscored)
    episode: str        # source failure episode id, e.g. episode_003
    step: int           # step at which the failure state was captured
    language_bddl: str  # (:language ...) of the scene (paraphrased)
    language_canonical: str  # original LIBERO wording from the folder name
    level: str = "unk"  # L1-L4 when available (not released with the assets)

    @property
    def group(self) -> str:
        """Leakage group: scenes from the same failure episode are near-duplicates."""
        return f"{self.task_folder}__{self.episode}"

    def instruction(self, source: str = "bddl") -> str:
        return self.language_bddl if source == "bddl" else self.language_canonical

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_language(task_folder: str) -> str:
    name = re.sub(r"^(KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+_", "", task_folder)
    return name.replace("_", " ")


def discover_scenes(scene_root: str | Path, levels: dict[str, str] | None = None) -> list[Scene]:
    root = Path(scene_root)
    out = []
    for bddl in sorted(root.rglob("*.bddl")):
        npy = bddl.with_name(bddl.stem + "_sim_state.npy")
        if not npy.is_file():
            log.warning("missing sim state for %s", bddl)
            continue
        rel = bddl.relative_to(root).with_suffix("")
        parts = rel.parts
        task_folder = parts[0]
        episode = parts[1] if len(parts) > 2 else "episode_000"
        m = re.search(r"(\d+)$", bddl.stem)
        step = int(m.group(1)) if m else -1
        tag = "__".join(parts)
        out.append(
            Scene(
                tag=tag,
                bddl=str(bddl),
                npy=str(npy),
                task_folder=task_folder,
                episode=episode,
                step=step,
                language_bddl=parse_bddl_language(bddl),
                language_canonical=canonical_language(task_folder),
                level=(levels or {}).get(tag, "unk"),
            )
        )
    return out


# ----------------------------------------------------------------------------------------
# simulator
# ----------------------------------------------------------------------------------------
@dataclass
class StepResult:
    obs: dict[str, np.ndarray]
    success: bool
    info: dict[str, Any] = field(default_factory=dict)


class LiberoSim:
    """Single LIBERO env, re-targetable to any bddl. Not thread/process safe."""

    def __init__(self, resolution: int = 256, placement_max_attempts: int = 24):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        self.resolution = resolution
        self.placement_max_attempts = placement_max_attempts
        self._env = None
        self._bddl: str | None = None
        self._obj_xy_adrs: list[tuple[str, int, int]] = []
        self._ooi: list[str] = []
        self.t = 0

    # ---------------- env construction ----------------
    def _build(self, bddl: str, seed: int = 0) -> None:
        if self._bddl == bddl and self._env is not None:
            return
        from libero.libero.envs import OffScreenRenderEnv
        from robosuite.utils.errors import RandomizationError

        self.close()
        args = {"bddl_file_name": bddl, "camera_heights": self.resolution, "camera_widths": self.resolution}
        for attempt in range(self.placement_max_attempts):
            np.random.seed(seed + attempt)
            try:
                if attempt < 6:
                    self._env = OffScreenRenderEnv(**args)
                else:  # infeasible custom regions: the layout is overwritten by the recorded state anyway
                    with _no_placement_check():
                        self._env = OffScreenRenderEnv(**args)
                break
            except RandomizationError as e:  # throwaway layout, overwritten by set_init_state
                log.debug("placement failed building %s (%d): %s", bddl, attempt, e)
        if self._env is None:
            raise RuntimeError(f"could not build env for {bddl}")
        self._env.seed(seed)
        self._bddl = bddl
        self._obj_xy_adrs = self._free_joint_xy_adrs()
        self._ooi = parse_bddl_objects_of_interest(bddl)

    def _reset(self, seed: int) -> None:
        from robosuite.utils.errors import RandomizationError

        for attempt in range(self.placement_max_attempts):
            np.random.seed(seed + attempt)
            try:
                # NOTE: call robosuite's reset directly: LIBERO's ControlEnv.reset() retries
                # RandomizationError forever (rebuilding the sim each time) on infeasible regions.
                # hard resets rebuild the samplers, so the no-check fallback must wrap the reset.
                ctx = _no_placement_check() if attempt >= 6 else contextlib.nullcontext()
                with ctx:
                    self._env.env.reset()
                return
            except RandomizationError as e:
                log.debug("placement failed on reset (%d): %s", attempt, e)
        raise RuntimeError("could not reset env")

    def _disable_placement_check(self) -> None:
        env = self._env.env
        stack = [getattr(env, k) for k in dir(env) if k.endswith("placement_initializer")]
        while stack:
            s = stack.pop()
            if s is None:
                continue
            if hasattr(s, "ensure_valid_placement"):
                s.ensure_valid_placement = False
            stack += list(getattr(s, "samplers", {}).values())

    def _free_joint_xy_adrs(self) -> list[tuple[str, int, int]]:
        sim = self._env.env.sim
        out = []
        for name, body_id in getattr(self._env.env, "obj_body_id", {}).items():
            jnt_adr = int(sim.model.body_jntadr[body_id])
            if jnt_adr < 0 or int(sim.model.jnt_type[jnt_adr]) != 0:
                continue
            qadr = int(sim.model.jnt_qposadr[jnt_adr])
            out.append((name, qadr, qadr + 1))
        return out

    def _reset_controller_goal(self) -> None:
        for robot in self._env.robots:
            robot.controller.update()
            robot.controller.update_initial_joints(robot.controller.joint_pos)
            robot.controller.reset_goal()

    def _raw_obs(self) -> dict[str, Any]:
        return self._env.env._get_observations(force_update=True)

    # ---------------- public API ----------------
    def load_nominal(self, suite: str, task_id: int, init_idx: int, num_steps_wait: int = 10,
                     seed: int = 0) -> dict[str, Any]:
        """Reset to a LIBERO benchmark init state (same as lerobot LiberoEnv)."""
        import torch
        from libero.libero import benchmark, get_libero_path

        s = benchmark.get_benchmark_dict()[suite]()
        task = s.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        self._build(bddl, seed)
        self._reset(seed)
        init_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        init_states = torch.load(init_path, weights_only=False)  # nosec
        self._env.set_init_state(init_states[init_idx % len(init_states)])
        for _ in range(num_steps_wait):
            self._env.step(DUMMY_ACTION.tolist())
        self.t = 0
        return {"obs": format_obs(self._raw_obs()), "language": task.language, "task_name": task.name,
                "bddl": bddl}

    def load_scene(self, bddl: str, npy: str, gripper_close: bool | None, noise_xy: float = 0.02,
                   noise_seed: int = 0, settle_steps: int = 20, num_steps_wait: int = 0,
                   build_seed: int = 0) -> dict[str, Any]:
        """Official LIBERO-Recovery restore protocol: perturb free-joint objects xy, restore,
        re-anchor OSC goal, drive gripper to closed/open (None = leave as recorded)."""
        self._build(bddl, build_seed)
        self._reset(build_seed)
        base = np.load(npy)
        rng = np.random.RandomState(noise_seed)
        s = np.array(base, dtype=np.float64, copy=True)
        deltas = {}
        for name, ax, ay in self._obj_xy_adrs:
            dx, dy = float(rng.uniform(-noise_xy, noise_xy)), float(rng.uniform(-noise_xy, noise_xy))
            s[ax] += dx
            s[ay] += dy
            deltas[name] = (dx, dy)
        self._env.set_init_state(s)
        self._reset_controller_goal()
        if gripper_close is not None:
            g = 1.0 if gripper_close else -1.0
            for _ in range(settle_steps):
                self._env.step([0.0] * 6 + [g])
        for _ in range(num_steps_wait):
            self._env.step(DUMMY_ACTION.tolist())
        self.t = 0
        return {"obs": format_obs(self._raw_obs()), "deltas": deltas, "state": self.get_state(),
                "snapshot": self.snapshot(), "language": parse_bddl_language(bddl)}

    def get_state(self) -> np.ndarray:
        return np.asarray(self._env.get_sim_state(), dtype=np.float64).copy()

    # ---- exact snapshots (flat sim state is NOT enough for bit-exact branching: MuJoCo warmstart,
    # the Panda gripper's internal `current_action` and OSC controller goals live outside it) ----
    _CTRL_SKIP = {"sim", "robot_name", "eef_name", "joint_index", "qpos_index", "qvel_index",
                  "actuator_range", "interpolator_pos", "interpolator_ori", "ref_name"}

    @staticmethod
    def _np_attrs(obj, skip=()) -> dict[str, Any]:
        out = {}
        for k, v in vars(obj).items():
            if k in skip:
                continue
            if isinstance(v, np.ndarray):
                out[k] = v.copy()
            elif isinstance(v, (float, int, bool, np.floating)) and not isinstance(v, type):
                out[k] = v
        return out

    def snapshot(self) -> dict[str, Any]:
        import mujoco

        m, d = self._env.env.sim.model._model, self._env.env.sim.data._data
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        buf = np.empty(mujoco.mj_stateSize(m, spec), dtype=np.float64)
        mujoco.mj_getState(m, d, buf, spec)
        robots = []
        for r in self._env.robots:
            robots.append({
                "controller": self._np_attrs(r.controller, self._CTRL_SKIP),
                "grippers": {arm: self._np_attrs(g) for arm, g in _grippers(r).items()},
            })
        return {"mj": buf, "flat": self.get_state(), "robots": robots,
                "observables": {k: {"t": float(o._time_since_last_sample), "sampled": bool(o._sampled),
                                    "delay": float(o._current_delay),
                                    "value": np.array(o._current_observed_value, copy=True)}
                                for k, o in self._env.env._observables.items()},
                "obs_cache": {k: np.array(v, copy=True) for k, v in self._env.env._obs_cache.items()}}

    def restore_snapshot(self, bddl: str, snap: dict[str, Any], build_seed: int = 0) -> dict[str, Any]:
        import mujoco

        if self._bddl != bddl:
            self._build(bddl, build_seed)
            self._reset(build_seed)
        m, d = self._env.env.sim.model._model, self._env.env.sim.data._data
        mujoco.mj_setState(m, d, np.asarray(snap["mj"], dtype=np.float64), mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(m, d)
        for r, rs in zip(self._env.robots, snap["robots"]):
            for k, v in rs["controller"].items():
                setattr(r.controller, k, v.copy() if isinstance(v, np.ndarray) else v)
            for arm, g in _grippers(r).items():
                for k, v in rs["grippers"].get(arm, {}).items():
                    setattr(g, k, v.copy() if isinstance(v, np.ndarray) else v)
        self._env.env._post_process()
        for k, v in snap.get("observables", {}).items():
            o = self._env.env._observables.get(k)
            if o is None:
                continue
            o._time_since_last_sample, o._sampled, o._current_delay = v["t"], v["sampled"], v["delay"]
            o._current_observed_value = np.array(v["value"], copy=True)
        if "obs_cache" in snap:
            self._env.env._obs_cache = {k: np.array(v, copy=True) for k, v in snap["obs_cache"].items()}
        self._env.env.timestep = 0
        self._env.env.done = False
        self.t = 0
        # read the restored observed values; do NOT force-update (that would shift sampling phase)
        return {"obs": format_obs(self._env.env._get_observations(force_update=False))}

    def restore(self, bddl: str, state: np.ndarray, build_seed: int = 0) -> dict[str, Any]:
        """Branch from an exact flat sim state (e.g. a failure instance after gripper settle)."""
        if self._bddl != bddl:
            self._build(bddl, build_seed)
            self._reset(build_seed)
        self._env.set_init_state(np.asarray(state, dtype=np.float64))
        self._reset_controller_goal()
        # robosuite horizon bookkeeping: restart the episode clock for this branch
        self._env.env.timestep = 0
        self._env.env.done = False
        self.t = 0
        return {"obs": format_obs(self._raw_obs())}

    def step(self, action: np.ndarray) -> StepResult:
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        raw, _r, _done, _info = self._env.step(a.tolist())
        self.t += 1
        success = bool(self._env.check_success())
        return StepResult(format_obs(raw), success)

    def progress(self) -> float:
        """Dense goal-progress proxy: -xy distance between the first two objects of interest
        (spatial tasks: bowl -> plate). Used for shaping / goal-progress ablation only."""
        if len(self._ooi) < 2:
            return 0.0
        env = self._env.env
        try:
            p0 = env.sim.data.body_xpos[env.obj_body_id[self._ooi[0]]]
            p1 = env.sim.data.body_xpos[env.obj_body_id[self._ooi[1]]]
        except KeyError:
            return 0.0
        return -float(np.linalg.norm(p0[:2] - p1[:2]))

    def check_success(self) -> bool:
        return bool(self._env.check_success())

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        self._env = None
        self._bddl = None
