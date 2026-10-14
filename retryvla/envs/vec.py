"""Process pool of LiberoSim workers. The policy stays in the main process (GPU) and batches
observations across workers; workers only simulate + render.

Every worker supports all ``LiberoSim`` public methods by name, plus:
  - ``step_seq(actions, stop_on_success)``: execute an open-loop sequence in one round trip
  - ``record(on)`` / ``pop_record(keep)``: per-step (obs, action) buffer kept in the worker so that
    successful rollouts can be fetched without a deterministic replay.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from multiprocessing.connection import Connection, wait
from typing import Any

import numpy as np


class _RecordingSim:
    def __init__(self, resolution: int):
        from retryvla.envs.libero_env import LiberoSim

        self.sim = LiberoSim(resolution=resolution)
        self._rec_on = False
        self._buf: list[tuple[dict, np.ndarray, int]] = []
        self._last_obs: dict | None = None
        self.segment = 0

    def _track(self, out: dict) -> dict:
        if isinstance(out, dict) and "obs" in out:
            self._last_obs = out["obs"]
        return out

    # recording
    def record(self, on: bool = True, segment: int = 0, stride: int = 1, img_segments=None,
               img_until: int | None = None) -> None:
        """stride: keep images every `stride` env steps; img_segments: segments whose images are kept
        (None = all); img_until: no images after this many recorded steps. Actions/states/segments are
        always kept for every step."""
        self._rec_on, self.segment = on, segment
        self._stride, self._img_segments = max(1, int(stride)), (None if img_segments is None else set(img_segments))
        self._img_until = img_until
        self._t_rec = 0
        if on:
            self._buf = []

    def set_segment(self, segment: int) -> None:
        self.segment = segment

    def pop_record(self, keep: bool = True) -> dict[str, np.ndarray] | None:
        buf, self._buf = self._buf, []
        self._rec_on = False
        if not keep or not buf:
            return None
        img_t = [i for i, b in enumerate(buf) if b[0].get("image") is not None]
        out = {
            "state": np.stack([b[0]["state"] for b in buf]),
            "action": np.stack([b[1] for b in buf]).astype(np.float32),
            "segment": np.asarray([b[2] for b in buf], dtype=np.int64),
            "img_t": np.asarray(img_t, dtype=np.int64),
        }
        if img_t:
            out["image"] = np.stack([buf[i][0]["image"] for i in img_t])
            out["image2"] = np.stack([buf[i][0]["image2"] for i in img_t])
        if self._last_obs is not None:  # observation AFTER the last action (needed for end-state latents)
            out["final_image"], out["final_image2"] = self._last_obs["image"], self._last_obs["image2"]
            out["final_state"] = self._last_obs["state"]
        return out

    def step(self, action) -> dict:
        a = np.asarray(action, dtype=np.float32)
        if self._rec_on and self._last_obs is not None:
            keep_img = (self._t_rec % getattr(self, "_stride", 1) == 0) and (
                getattr(self, "_img_segments", None) is None or self.segment in self._img_segments) and (
                getattr(self, "_img_until", None) is None or self._t_rec <= self._img_until)
            o = self._last_obs if keep_img else {"state": self._last_obs["state"]}
            self._buf.append((o, a.copy(), self.segment))
            self._t_rec += 1
        r = self.sim.step(a)
        self._last_obs = r.obs
        return {"obs": r.obs, "success": r.success, "t": self.sim.t}

    def step_seq(self, actions, stop_on_success: bool = True, return_obs: bool = True, segment: int = 0) -> dict:
        """segment 0 = open-loop recovery prefix, 1 = policy continuation chunk."""
        out = {"success": False, "n": 0}
        self.segment = segment
        for a in np.asarray(actions, dtype=np.float32):
            out = self.step(a) | {"n": out["n"] + 1}
            if out["success"] and stop_on_success:
                break
        self.segment = 1  # whatever follows is policy continuation
        out["progress"] = self.sim.progress()
        if not return_obs:
            out.pop("obs", None)
        return out

    def begin(self, load_cmd: str, load_kwargs: dict, record: bool = False,
              return_snapshot: bool = False, record_cfg: dict | None = None) -> dict:
        out = dict(getattr(self.sim, load_cmd)(**load_kwargs))
        self._last_obs = out["obs"]
        self.record(bool(record), segment=1, **(record_cfg or {}))
        if not return_snapshot:
            out.pop("snapshot", None)
        return out

    def finish(self, keep_record: bool = False, snapshot: bool = False) -> dict:
        out = {"record": self.pop_record(keep_record), "progress": self.sim.progress()}
        if snapshot:
            out["snapshot"] = self.sim.snapshot()
        return out

    def __getattr__(self, name):
        fn = getattr(self.sim, name)
        if not callable(fn):
            return fn

        def wrapped(*a, **k):
            return self._track(fn(*a, **k))

        return wrapped


def _worker(conn: Connection, resolution: int, env: dict[str, str]) -> None:
    os.environ.update(env)
    import logging

    logging.basicConfig(level=logging.WARNING)
    sim = _RecordingSim(resolution)
    while True:
        try:
            cmd, args, kwargs = conn.recv()
        except EOFError:
            break
        if cmd == "__close__":
            sim.sim.close()
            conn.send(("ok", None))
            break
        try:
            conn.send(("ok", getattr(sim, cmd)(*args, **kwargs)))
        except Exception as e:  # report, keep worker alive
            conn.send(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


class SimPool:
    def __init__(self, n_workers: int, resolution: int = 256):
        ctx = mp.get_context("spawn")
        env = {k: v for k, v in os.environ.items()
               if k.startswith(("MUJOCO", "PYOPENGL", "__EGL", "LD_LIBRARY", "LIBERO", "PYTHONPATH"))}
        self.conns: list[Connection] = []
        self.procs = []
        for _ in range(n_workers):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, resolution, env), daemon=True)
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self._busy = [False] * n_workers

    @property
    def n(self) -> int:
        return len(self.conns)

    def send(self, i: int, cmd: str, *args, **kwargs) -> None:
        assert not self._busy[i], f"worker {i} busy"
        self.conns[i].send((cmd, args, kwargs))
        self._busy[i] = True

    def recv(self, i: int) -> Any:
        status, out = self.conns[i].recv()
        self._busy[i] = False
        if status == "err":
            raise RuntimeError(f"worker {i}: {out}")
        return out

    def call(self, i: int, cmd: str, *args, **kwargs) -> Any:
        self.send(i, cmd, *args, **kwargs)
        return self.recv(i)

    def ready(self, idxs: list[int], timeout: float | None = None) -> list[int]:
        """Indices among ``idxs`` (busy workers) whose reply is available."""
        conns = {self.conns[i]: i for i in idxs}
        return [conns[c] for c in wait(list(conns), timeout=timeout)]

    def broadcast(self, cmd: str, *args, **kwargs) -> list[Any]:
        for i in range(self.n):
            self.send(i, cmd, *args, **kwargs)
        return [self.recv(i) for i in range(self.n)]

    def close(self) -> None:
        for i, c in enumerate(self.conns):
            try:
                if self._busy[i]:
                    c.recv()
                c.send(("__close__", (), {}))
                c.recv()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def __enter__(self) -> "SimPool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
