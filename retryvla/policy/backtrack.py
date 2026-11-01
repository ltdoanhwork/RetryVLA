"""Backtrack baseline (no world model, no training), after Landeg & Tambe, "If at First You Don't Succeed, Backtrack"
(arXiv 2610.32941): stop the frozen VLA, move the end-effector back (gripper open), hand control back to the VLA.

Our re-implementation for LIBERO-Recover (no learned failure head; the start state is a failure by construction):
- ``single``: at t = 0, back off to a safe pose ABOVE the current one (open gripper, +``lift`` m in z; LIBERO-Recover
  scenes are snapshots, so there is no executed motion to retrace yet), then pi0 runs for the rest of the budget.
- ``retry``: ``single`` + up to ``max_retries`` further backtracks inside the same 520-step budget, each triggered by a
  proprioceptive failure detector (no privileged sim info): (a) STALL: the EE moved < ``stall_dist`` m over the last
  ``stall_steps`` steps, or (b) ATTEMPT TIMEOUT: ``attempt_steps`` steps since the last backtrack while the gripper is
  commanded OPEN (so a carry in progress is never interrupted). An "empty grasp" trigger (finger gap while closed) is
  implemented but OFF (``empty_gap = 0``): on LIBERO-Spatial the gap while holding the bowl rim (demos: median 0.0049,
  1st pct 0.0035) overlaps the empty-closed gap (0.001-0.009), so it cannot be told apart. A triggered
  backtrack RETRACES the executed motion: target = EE position ``k`` steps earlier (+``retry_lift`` m), gripper open.
Backtrack motion = OSC position deltas, P-controlled at chunk granularity (the VLA also only observes every
``n_action_steps``), at most ``max_bt_steps`` env steps per backtrack; rotation untouched. All backtrack steps count
against the episode budget (same ``recover.max_steps`` as every other arm).

Episode bookkeeping: the rollout engine (``eval/rollout.py``) seeds chunk calls with ``policy_seed * 100003 + n_calls``,
so ``seed // 100003`` identifies the episode and ``seed % 100003`` the chunk index (call 0 = episode start). Requires
seeds (always passed by ``run_jobs``). ``self.events[policy_seed]`` lists the chunk indices that were backtrack chunks
(used by ``baselines/backtrack_datagen.py`` to label segment 0 = backtrack, 1 = VLA in distilled data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

SEED_MULT = 100_003  # must match rollout.run_jobs


@dataclass
class BacktrackCfg:
    mode: str = "single"          # single | retry
    lift: float = 0.08            # m, t = 0 back-off height
    k: int = 40                   # retry: retrace to the EE position k env steps earlier
    retry_lift: float = 0.03      # m added to the retrace target
    max_retries: int = 3          # retry: extra backtracks after the t = 0 one
    min_gap: int = 60             # env steps of VLA control between two backtracks
    empty_gap: float = 0.0        # finger gap (qpos[0]-qpos[1]) below this while closed = empty grasp (0 = off)
    stall_dist: float = 0.015     # m
    stall_steps: int = 60
    attempt_steps: int = 160
    max_bt_steps: int = 40        # per backtrack
    tol: float = 0.012            # m, target reached
    gain: float = 1.0 / 0.15      # action per metre of error (one chunk covers ~2/3 of a 5 cm error)
    max_action: float = 0.8
    z_max: float = 1.25           # never lift above this (m, world z; table top ~0.9)

    @classmethod
    def from_dict(cls, d: dict | None) -> "BacktrackCfg":
        d = dict(d or {})
        return cls(**{k: type(getattr(cls, k))(v) for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class _Ep:
    hist: list = field(default_factory=list)   # EE position at each chunk boundary (every n_act steps)
    target: np.ndarray | None = None
    bt_steps: int = 0
    n_bt: int = 0
    last_bt_end: int = 0                       # env step at which the last backtrack ended
    last_grip_cmd: float = -1.0
    bt_calls: list = field(default_factory=list)
    triggers: list = field(default_factory=list)


class BacktrackPolicy:
    def __init__(self, base, cfg: BacktrackCfg | dict | None = None):
        self.base = base
        self.cfg = cfg if isinstance(cfg, BacktrackCfg) else BacktrackCfg.from_dict(cfg)
        self.eps: dict[int, _Ep] = {}
        self._done: list[tuple[int, list]] = []  # (n_bt, triggers) of dropped (finished) episodes

    @property
    def n_action_steps(self) -> int:
        return self.base.n_action_steps

    def __getattr__(self, k):  # chunk_size, action_dim, ...
        return getattr(self.base, k)

    # ---------------------------------------------------------------------------------------------------------
    def events(self, policy_seed: int) -> dict:
        e = self.eps.get(int(policy_seed))
        return {} if e is None else {"bt_calls": list(e.bt_calls), "n_bt": e.n_bt, "triggers": list(e.triggers)}

    def stats(self) -> dict:
        allx = self._done + [(e.n_bt, e.triggers) for e in self.eps.values()]
        n = [a for a, _ in allx]
        trig = [t for _, ts in allx for t in ts]
        return {"episodes": len(n), "mean_backtracks": float(np.mean(n)) if n else 0.0,
                "triggers": {k: trig.count(k) for k in sorted(set(trig))}}

    def drop(self, policy_seed: int) -> None:
        e = self.eps.pop(int(policy_seed), None)
        if e is not None:
            self._done.append((e.n_bt, list(e.triggers)))

    def _trigger(self, e: _Ep, t: int, state: np.ndarray) -> str | None:
        c, n = self.cfg, self.n_action_steps
        if c.mode != "retry" or e.n_bt >= 1 + c.max_retries or t - e.last_bt_end < c.min_gap:
            return None
        gap = float(state[6] - state[7])
        if e.last_grip_cmd > 0 and gap < c.empty_gap:
            return "empty_grasp"
        w = max(1, c.stall_steps // n)
        if len(e.hist) > w and np.linalg.norm(e.hist[-1] - e.hist[-1 - w]) < c.stall_dist:
            return "stall"
        if t - e.last_bt_end >= c.attempt_steps and e.last_grip_cmd < 0:
            return "timeout"
        return None

    def _bt_chunk(self, e: _Ep, pos: np.ndarray) -> np.ndarray | None:
        c = self.cfg
        d = e.target - pos
        if np.linalg.norm(d) < c.tol or e.bt_steps >= c.max_bt_steps:
            return None
        a = np.zeros((int(getattr(self.base, "chunk_size", 50) or 50), 7), np.float32)
        a[:, :3] = np.clip(d * c.gain, -c.max_action, c.max_action)
        a[:, 6] = -1.0  # open
        if e.bt_steps == 0:  # first chunk of a backtrack: open the gripper before moving (avoid dragging the object)
            a[:4, :3] = 0.0
        return a

    def _start_bt(self, e: _Ep, t: int, pos: np.ndarray, why: str) -> None:
        c, n = self.cfg, self.n_action_steps
        if why == "start":
            tgt = pos + np.array([0.0, 0.0, c.lift])
        else:
            back = max(1, c.k // n) + (max(1, c.stall_steps // n) if why == "stall" else 0)  # stall: before the stall
            j = max(0, len(e.hist) - 1 - back)
            tgt = e.hist[j] + np.array([0.0, 0.0, c.retry_lift])
        tgt[2] = min(tgt[2], c.z_max)
        e.target, e.bt_steps = tgt.astype(np.float64), 0
        e.n_bt += 1
        e.triggers.append(why)

    def predict_chunk(self, obs: Sequence[dict], tasks: Sequence[str], seeds: Sequence[int] | None = None,
                      noise_scale: float = 1.0) -> np.ndarray:
        if len(obs) == 0 or seeds is None:
            return self.base.predict_chunk(obs, tasks, seeds, noise_scale)
        n = self.n_action_steps
        out: list[np.ndarray | None] = [None] * len(obs)
        for b, (o, s) in enumerate(zip(obs, seeds)):
            ep, call = int(s) // SEED_MULT, int(s) % SEED_MULT
            if call == 0 or ep not in self.eps:
                self.eps[ep] = _Ep()
            e = self.eps[ep]
            st = np.asarray(o["state"], np.float64)
            pos, t = st[:3], call * n
            e.hist.append(pos.copy())
            if call == 0:
                self._start_bt(e, t, pos, "start")
            elif e.target is None:
                why = self._trigger(e, t, st)
                if why:
                    self._start_bt(e, t, pos, why)
            if e.target is not None:
                ch = self._bt_chunk(e, pos)
                if ch is None:
                    e.target, e.last_bt_end = None, t
                else:
                    e.bt_steps += n
                    e.bt_calls.append(call)
                    e.last_grip_cmd = -1.0
                    out[b] = ch
        need = [b for b, x in enumerate(out) if x is None]
        if need:
            ch = self.base.predict_chunk([obs[b] for b in need], [tasks[b] for b in need],
                                         [seeds[b] for b in need], noise_scale)
            for j, b in enumerate(need):
                out[b] = np.asarray(ch[j], np.float32)
                e = self.eps[int(seeds[b]) // SEED_MULT]
                e.last_grip_cmd = float(np.asarray(ch[j])[min(n, len(ch[j])) - 1, 6])
        T = max(len(x) for x in out)
        return np.stack([x if len(x) == T else np.concatenate([x, np.repeat(x[-1:], T - len(x), 0)]) for x in out])
