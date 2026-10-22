"""Batched episode runner for the residual bridge baseline (same async pattern as retryvla/eval/rollout.py).

Episode = restore the exact failure snapshot (identical to compare_arms verification jobs: `restore_snapshot`
with the precheck snapshot) -> decisions every n_act steps: pi0 chunk (batched over workers) [+ residual inside
the bridge window, batched] -> step_seq -> ... until success or max_steps. Every env step is counted.
"""

from __future__ import annotations

import collections
import logging
import time

import numpy as np
import torch

from retryvla.baselines.recovery_rl.bridge import SEED_MOD, ResidualBridge
from retryvla.envs.vec import SimPool

log = logging.getLogger(__name__)


class _W:
    __slots__ = ("eid", "ep", "phase", "steps", "obs", "n_calls", "success", "prog", "decs", "bridge_success")

    def __init__(self):
        self.phase = "idle"


def run_episodes(pool: SimPool, base, bridge: ResidualBridge, episodes: list[dict], sample: bool,
                 gen: torch.Generator | None = None, noise_scale: float = 1.0, batch_wait_s: float = 0.03,
                 desc: str = "rl") -> list[dict]:
    """episodes: dicts with bddl, snap, task, policy_seed, max_steps (+ any meta). Returns one dict per episode:
    {success, steps, success_step, bridge_success, decisions: [ {img, low, a_cont, a_grip, logp, value,
    prog0, prog1, delta_sq, done_in_bridge} ], **meta}."""
    n_act, H = base.n_action_steps, bridge.horizon
    pending = collections.deque(range(len(episodes)))
    ws = [_W() for _ in range(pool.n)]
    out: dict[int, dict] = {}
    t0, last = time.time(), 0.0
    while pending or any(w.phase != "idle" for w in ws):
        for i, w in enumerate(ws):
            if w.phase == "idle" and pending:
                w.eid = pending.popleft()
                w.ep = episodes[w.eid]
                w.steps, w.n_calls, w.success, w.decs, w.bridge_success = 0, 0, False, [], False
                pool.send(i, "restore_snapshot", w.ep["bddl"], w.ep["snap"])
                w.phase = "loading"
        need = [i for i, w in enumerate(ws) if w.phase == "act"]
        if need:
            obs = [ws[i].obs for i in need]
            tasks = [ws[i].ep["task"] for i in need]
            seeds = [int(ws[i].ep["policy_seed"]) * SEED_MOD + ws[i].n_calls for i in need]
            chunks = base.predict_chunk(obs, tasks, seeds=seeds, noise_scale=noise_scale)
            br = [k for k, i in enumerate(need) if ws[i].steps < H]
            if br:
                mod, info = bridge.decide([obs[k] for k in br], [tasks[k] for k in br], chunks[br],
                                          [ws[need[k]].n_calls for k in br], sample=sample, gen=gen)
                chunks = np.array(chunks, copy=True)
                chunks[br] = mod
                for j, k in enumerate(br):
                    ws[need[k]].decs.append({"img": info["img"][j], "low": info["low"][j],
                                             "a_cont": info["a_cont"][j], "a_grip": info["a_grip"][j],
                                             "logp": info["logp"][j], "value": info["value"][j],
                                             "delta_sq": float(info["delta_sq"][j].mean()),
                                             "prog0": ws[need[k]].prog})
            for k, i in enumerate(need):
                w = ws[i]
                w.n_calls += 1
                n = min(n_act, int(w.ep["max_steps"]) - w.steps)
                if w.steps < H:  # never let one bridge chunk run past the window
                    n = min(n, H - w.steps)
                pool.send(i, "step_seq", np.clip(chunks[k][:n], -1, 1), stop_on_success=True,
                          segment=0 if w.steps < H else 1)
                w.phase = "stepping"
        busy = [i for i, w in enumerate(ws) if w.phase in ("loading", "prog0", "stepping")]
        if not busy:
            continue
        ready = pool.ready(busy)
        more = pool.ready([i for i in busy if i not in ready], timeout=batch_wait_s) if len(ready) < len(busy) else []
        for i in ready + more:
            w = ws[i]
            r = pool.recv(i)
            if w.phase == "loading":
                w.obs = r["obs"]
                pool.send(i, "progress")
                w.phase = "prog0"
                continue
            if w.phase == "prog0":
                w.prog = float(r)
                w.phase = "act"
                continue
            # stepping
            was_bridge = w.steps < H
            w.steps += int(r["n"])
            w.obs, w.success, w.prog = r["obs"], bool(r["success"]), float(r["progress"])
            if was_bridge and w.decs:
                w.decs[-1]["prog1"] = w.prog
                w.decs[-1]["done_in_bridge"] = w.success
                w.bridge_success = w.bridge_success or w.success
            if w.success or w.steps >= int(w.ep["max_steps"]):
                out[w.eid] = {**{k: v for k, v in w.ep.items() if k != "snap"}, "success": w.success,
                              "steps": w.steps, "success_step": w.steps if w.success else None,
                              "bridge_success": w.bridge_success, "decisions": w.decs}
                w.phase, w.ep, w.obs, w.decs = "idle", None, None, []
            else:
                w.phase = "act"
        now = time.time()
        if now - last > 60:
            last = now
            sr = np.mean([o["success"] for o in out.values()]) if out else float("nan")
            log.info("[%s] %d/%d episodes | SR %.3f | %.0fs", desc, len(out), len(episodes), sr, now - t0)
    return [out[k] for k in range(len(episodes))]
