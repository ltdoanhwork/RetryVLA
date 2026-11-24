"""LRS: faithful-as-possible re-implementation of arXiv 2610.05151 (Recursive Self-Improvement through Local Recovery
Supervision) in the RetryVLA stack, for a paired comparison with the WM + C^pi generator. TRAIN split only.

python -m retryvla.baselines.lrs_teacher configs/exp/lrs_pilot.yaml [m5b.shard=0] [lrs.lane=0 lrs.num_lanes=3]

Per failure instance (the SAME M5b / RGR instances, exact snapshots, as `single` / `random`):
  K independent attempts. Attempt = stages r = 1..max_stages of a VLM tool teacher (one fresh session per stage:
  SPEC -> bounded tool commands -> completion claim), each followed by the frozen-student hand-off test of the paper
  (Eq. 7: progress + confirmation, (H_S, Delta, C) = (60, 10, 10)); a failed test restores the TEACHER endpoint and
  appends a further stage (<= 5 stages, <= 300 teacher actions). Accepted paths are additionally re-executed from the
  failure snapshot + frozen pi0 for <= 300 steps (RetryVLA's full-task acceptance, reported separately).
Deviations from the paper (docs/related_work/COMPARE_2610.05151.md): local Qwen3-VL-8B teacher instead of gpt-6.1-sol;
no auditor (entry = the benchmark failure state, same for all arms); spec written by the teacher itself; programmatic
privileged-state observer instead of an LLM observer; <= lrs.max_turns LLM turns per stage; whole-path imitation.
Attempt processing order: (a3) generation order, (d) C^pi(real stage-1 endpoint) order; both stop at the first
accepted attempt; arms / budgets are computed offline from the per-attempt log (scripts/analyze_lrs.py).
"""

from __future__ import annotations

import json
import logging
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from retryvla import REPO_ROOT, tracking
from retryvla.baselines import lrs_prompts as P
from retryvla.baselines.lrs_llm import VLMTeacher, parse_commands
from retryvla.config import cli_config, to_dict
from retryvla.data.recovery_writer import RecoveryWriter
from retryvla.envs.libero_env import parse_bddl_objects_of_interest
from retryvla.envs.splits import assert_allowed, load_split, scenes_for
from retryvla.envs.vec import SimPool
from retryvla.eval.rollout import Job, run_jobs
from retryvla.geom.features import IDX
from retryvla.policy import SmolVLAWrapper
from retryvla.search.compare_arms import load_known_precheck, precheck, select_jobs
from retryvla.utils import setup_logging

log = logging.getLogger("lrs")
STEP_M = 0.01      # ~ m of eef motion per step at |a| = 1 (retryvla.causal.probes)
EEF = [IDX["eef_x"], IDX["eef_y"], IDX["eef_z"]]


# ----------------------------------------------------------------------------------------------- sim session
class Sess:
    """Interactive teacher session on pool worker 0 (exact snapshots; every executed action is recorded)."""

    def __init__(self, pool: SimPool, bddl: str, names: list[str], geo: bool):
        self.pool, self.bddl, self.names, self.geo = pool, bddl, names, geo
        self.obs, self.success, self.g = None, False, None

    def restore(self, snap: dict) -> None:
        out = self.pool.call(0, "restore_snapshot", self.bddl, snap)
        self.obs, self.success = out["obs"], False
        self.g = self.pool.call(0, "state_info")["geom"]

    def snapshot(self) -> dict:
        return self.pool.call(0, "snapshot")

    def step(self, a: np.ndarray) -> None:
        out = self.pool.call(0, "step", np.asarray(a, np.float32))
        self.obs, self.success = out["obs"], bool(out["success"])
        self.g = self.pool.call(0, "state_info")["geom"]

    def eef(self) -> np.ndarray:
        return np.asarray(self.g[EEF], float)

    def state_txt(self) -> str:
        e = self.eef()
        s = f"gripper at ({e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f}), finger opening {self.g[IDX['grip_open']]:.3f}"
        if self.geo:
            o, q = self.g[0:3], self.g[3:6]
            s += (f"; {self.names[0]} at ({o[0]:.3f}, {o[1]:.3f}, {o[2]:.3f}); "
                  f"{self.names[1]} at ({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f})")
        return s


def exec_cmd(s: Sess, c: dict, grip: float, left: int) -> tuple[list[np.ndarray], float, str]:
    """Execute one tool command (bounded); returns executed actions, gripper command, feedback text."""
    acts: list[np.ndarray] = []
    tool = str(c.get("tool", ""))
    if tool in ("move_to", "move_by"):
        try:
            v = np.asarray(c.get("xyz") if tool == "move_to" else c.get("dxyz"), float).reshape(3)
        except Exception:
            return acts, grip, f"{tool}: invalid arguments {c}"
        tgt = v if tool == "move_to" else s.eef() + v
        tgt[2] = max(tgt[2], 0.0)
        n = int(np.clip(int(c.get("max_steps", 40) or 40), 1, 40))
        n = min(n, left)
        for _ in range(n):
            err = tgt - s.eef()
            if np.linalg.norm(err) < 0.008:
                break
            a = np.zeros(7, np.float32)
            a[:3] = np.clip(err / STEP_M, -1.0, 1.0)
            a[6] = grip
            s.step(a)
            acts.append(a)
        d = float(np.linalg.norm(tgt - s.eef()))
        return acts, grip, (f"{tool} -> target ({tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f}): {len(acts)} steps, "
                            f"remaining error {d:.3f} m{' (reached)' if d < 0.01 else ' (NOT reached)'}")
    if tool == "gripper":
        g = 1.0 if str(c.get("cmd", "")).lower().startswith("c") else -1.0
        for _ in range(min(10, left)):
            a = np.zeros(7, np.float32)
            a[6] = g
            s.step(a)
            acts.append(a)
        return acts, g, f"gripper {'close' if g > 0 else 'open'}: {len(acts)} steps"
    return acts, grip, f"unknown tool {tool!r} ignored"


# ------------------------------------------------------------------------------------------ hand-off observer
def held(g: np.ndarray, z_rest: float) -> bool:
    return bool(g[IDX["finger_obj_contact"]] > 0.5 and g[IDX["obj_z"]] > z_rest + 0.02)


def obj_goal_xy(g: np.ndarray) -> float:
    return float(np.linalg.norm(g[0:2] - g[3:5]))


Z_REST = 0.012     # bowl centre height above the table when resting (measured on train scenes: 0.010)


def handoff_test(s: Sess, policy, task: str, seed: int, H_S: int = 60, delta: int = 10, C: int = 10) -> dict:
    """Paper Eq. 7 with a programmatic observer on privileged state (spec fixed for the Spatial pick-place family):
    B  = teacher-achieved possession is not lost unless the bowl is placed (task success);
    P  = {possession newly established, bowl-plate xy distance reduced >= 4 cm while held};  f = task success;
    check every `delta` student actions up to H_S; the first supported advance must survive C more actions.
    Student actions are never imitation targets. Returns accept, steps, reason, a student-visible summary."""
    gT = s.g.copy()
    hT, dT, succT = held(gT, Z_REST), obj_goal_xy(gT), s.success
    e0, o0 = s.eef(), float(gT[IDX["grip_open"]])
    steps, k, anchor = 0, 0, None
    obs = s.obs
    prev_success = succT

    def run_chunk(n):
        nonlocal obs, steps, k
        ch = policy.predict_chunk([obs], [task], seeds=[seed * 100_003 + k])[0][:n]
        k += 1
        for a in ch:
            s.step(a)
            steps += 1
        obs = s.obs

    def summary(why):
        e1 = s.eef()
        return (f"{why}; in {steps} actions the student moved the gripper from ({e0[0]:.3f}, {e0[1]:.3f}, {e0[2]:.3f}) "
                f"to ({e1[0]:.3f}, {e1[1]:.3f}, {e1[2]:.3f}), finger opening {o0:.3f} -> {s.g[IDX['grip_open']]:.3f}")

    while steps < H_S:
        run_chunk(delta)
        g, succ = s.g, s.success
        hn, dn = held(g, Z_REST), obj_goal_xy(g)
        if hT and not hn and not succ:
            return {"accept": False, "steps": steps, "reason": "lost_possession",
                    "summary": summary("the student lost the object you were holding")}
        if succ and prev_success:
            anchor = "complete"
        elif not hT and hn:
            anchor = "possession"
        elif hn and dn <= dT - 0.04:
            anchor = "transport"
        elif succ:
            anchor = "complete"
        prev_success = succ
        if anchor:
            break
    if anchor is None:
        return {"accept": False, "steps": steps, "reason": "no_progress",
                "summary": summary("no supported task progress")}
    run_chunk(C)                       # confirmation
    g, succ = s.g, s.success
    hn, dn = held(g, Z_REST), obj_goal_xy(g)
    ok = succ or (anchor == "possession" and hn) or (anchor == "transport" and hn and dn <= dT - 0.04)
    return {"accept": bool(ok), "steps": steps, "reason": f"{anchor}_{'confirmed' if ok else 'not_confirmed'}",
            "summary": summary(f"progress ({anchor}) was not sustained") if not ok else "accepted"}


# --------------------------------------------------------------------------------------------- teacher stage
def run_stage(teacher: VLMTeacher, s: Sess, task: str, stage: int, prev_cmds: list[str], student_txt: str,
              grip: float, act_left: int, rc, seed: int, transcript: list) -> dict:
    """One fresh teacher session from the CURRENT sim state. Returns executed actions + how the stage ended."""
    budget = lambda t, a: f"Budget: {rc.max_turns - t} replies, {a} robot steps left in this attempt."
    ext = P.extension_text(stage, prev_cmds, student_txt) if stage > 0 else ""
    msgs = [{"role": "system", "content": [{"type": "text", "text": P.SYSTEM}]},
            {"role": "user", "content": [{"type": "image", "image": s.obs["image"]},
                                         {"type": "image", "image": s.obs["image2"]},
                                         {"type": "text", "text": P.first_user(task, s.state_txt(),
                                                                               budget(0, act_left), ext)}]}]
    acts, cmds, end = [], [], "turn_cap"
    for turn in range(int(rc.max_turns)):
        reply = teacher.chat(msgs, seed=seed * 131 + turn)
        transcript.append({"stage": stage, "turn": turn, "reply": reply[-1500:]})
        cl = parse_commands(reply)
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
        if not cl:
            fb = "no valid JSON {\"commands\": [...]} found; nothing executed"
        else:
            fbs = []
            for c in cl[:4]:
                t = str(c.get("tool", ""))
                if t == "done":
                    end = "claimed" if acts else "claimed_no_actions"
                    break
                if t == "abort":
                    end = "abort"
                    break
                if len(acts) >= act_left:
                    end = "action_cap"
                    break
                a, grip, f = exec_cmd(s, c, grip, act_left - len(acts))
                acts += a
                cmds.append(json.dumps(c)[:160])
                fbs.append(f)
            fb = "; ".join(fbs) or "nothing executed"
            if end != "turn_cap":
                break
            if len(acts) >= act_left:
                end = "action_cap"
                break
        # keep the context small: only the latest images
        for m in msgs:
            if m["role"] == "user":
                m["content"] = [c for c in m["content"] if c["type"] == "text"]
        msgs.append({"role": "user", "content": [{"type": "image", "image": s.obs["image"]},
                                                 {"type": "image", "image": s.obs["image2"]},
                                                 {"type": "text", "text": P.feedback_user(
                                                     fb, s.state_txt(), budget(turn + 1, act_left - len(acts)))}]})
    return {"actions": acts, "cmds": cmds, "end": end, "grip": grip, "turns": turn + 1}


# ----------------------------------------------------------------------------------------------- attempts
class Attempt:
    def __init__(self, k: int):
        self.k = k
        self.path: list[np.ndarray] = []      # accumulated teacher actions (continuous path from the failure state)
        self.stages: list[dict] = []
        self.cmds: list[str] = []
        self.end_snap = None
        self.grip = 0.0
        self.teacher_steps = 0
        self.student_steps = 0
        self.status = "new"                   # new | stage1 | accepted | failed
        self.cpi1 = float("nan")
        self.transcript: list = []
        self.student_txt = ""

    def log(self) -> dict:
        return {"k": self.k, "status": self.status, "n_stages": len(self.stages), "path_len": len(self.path),
                "teacher_steps": self.teacher_steps, "student_steps": self.student_steps, "cpi1": self.cpi1,
                "stages": self.stages}


def stage_once(A: Attempt, teacher, s: Sess, inst, rc, seed: int) -> bool:
    """Run the next teacher stage of attempt A from its saved endpoint (or the failure state). True if the stage
    ended with a completion claim (or at the action cap) and the hand-off test should run."""
    s.restore(A.end_snap if A.end_snap is not None else inst.snapshot)
    r = len(A.stages)
    st = run_stage(teacher, s, inst.task, r, A.cmds, A.student_txt, A.grip, int(rc.max_actions) - len(A.path),
                   rc, seed + 1009 * r, A.transcript)
    A.path += st["actions"]
    A.cmds += st["cmds"]
    A.grip = st["grip"]
    A.teacher_steps += len(st["actions"])
    A.end_snap = s.snapshot()
    A.end_g = s.g.copy()
    A.end_obs = s.obs
    A.stages.append({"stage": r, "end": st["end"], "n_actions": len(st["actions"]), "turns": st["turns"]})
    return st["end"] in ("claimed", "action_cap")


def test_and_extend(A: Attempt, teacher, s: Sess, policy, inst, rc, seed: int) -> None:
    """Hand-off test from the saved teacher endpoint; on failure restore the endpoint and extend (paper Sec. 3.5)."""
    while True:
        s.restore(A.end_snap)
        t = handoff_test(s, policy, inst.task, seed + 7 * len(A.stages), int(rc.H_S), int(rc.delta), int(rc.C))
        A.student_steps += t["steps"]
        A.stages[-1]["handoff"] = {k: t[k] for k in ("accept", "steps", "reason")}
        if t["accept"]:
            A.status = "accepted"
            return
        A.student_txt = t["summary"]
        if len(A.stages) >= int(rc.max_stages) or len(A.path) >= int(rc.max_actions) \
                or A.stages[-1]["end"] == "action_cap":
            A.status = "failed"
            return
        if not stage_once(A, teacher, s, inst, rc, seed):
            A.status = "failed"
            return


def full_verify(pool, policy, inst, path: list[np.ndarray], cont: int, seed: int, record: bool):
    pre = np.asarray(path, np.float32).reshape(-1, 7)
    job = Job("restore_snapshot", dict(bddl=inst.bddl, snap=inst.snapshot), task=inst.task,
              max_steps=len(pre) + cont, prefix=pre if len(pre) else None, record="all" if record else "none",
              policy_seed=seed % (2**31 - 1))
    return run_jobs(pool, policy, [job], desc="full")[0]


# ------------------------------------------------------------------------------------------------------ main
def main() -> None:
    setup_logging()
    cfg = cli_config()
    mb, rc = cfg.m5b, cfg.lrs
    split = load_split(REPO_ROOT / cfg.split.file)
    scenes = scenes_for(split, "train", REPO_ROOT / cfg.recover.scene_root)
    assert_allowed([s.tag for s in scenes], split)
    lane, nl = int(rc.lane), int(rc.num_lanes)
    cfg.tag = f"{cfg.tag}_s{mb.shard}of{mb.num_shards}_l{lane}of{nl}"
    cont, K = int(cfg.search.continuation_max_steps), int(rc.n_attempts)
    root = REPO_ROOT / rc.out_root / f"s{mb.shard}" / f"l{lane}"
    root.mkdir(parents=True, exist_ok=True)
    done = set()
    for p_ in root.glob("p*/rows.jsonl"):
        done |= {json.loads(x)["instance_id"] for x in p_.read_text().splitlines() if x.strip()}
    part = root / f"p{len(list(root.glob('p*')))}"
    part.mkdir()
    jobs = select_jobs(scenes, cfg, mb)
    dev = torch.device(cfg.policy.device)
    with tracking.start(cfg) as run:
        run.summary.update({"part": str(part)})
        policy = SmolVLAWrapper(cfg.policy.path, cfg.policy.device, cfg.policy.n_action_steps)
        ranker = None
        if bool(rc.get("cpi", True)):
            from retryvla.search.wm_search import WMCriticRanker

            wm_ck = (REPO_ROOT / mb.wm_link).read_text().strip()
            cr_ck = (REPO_ROOT / mb.critic_link).read_text().strip()
            ranker = WMCriticRanker(wm_ck, cr_ck, cfg.wm_collect.encoder, int(cfg.wm_collect.pool_grid), dev,
                                    0.0, float(mb.beta_rough))
        teacher = VLMTeacher(str(rc.model), cfg.policy.device, int(rc.max_new_tokens), float(rc.temperature))
        wr = {"their_accept": RecoveryWriter(part / "their_accept", overwrite=True),
              "full_verified": RecoveryWriter(part / "full_verified", overwrite=True)}
        t0 = time.time()
        with SimPool(1, int(cfg.env.resolution)) as pool:
            insts, pre = precheck(pool, policy, jobs, set(), known=load_known_precheck(mb.get("precheck_from") or []))
            log.info("shard %s lane %d: %d pi0 failures, %d done", mb.shard, lane, len(insts), len(done))
            n_run = 0
            for n, inst in enumerate(insts):
                if n % nl != lane or inst.instance_id in done:
                    continue
                if int(rc.get("max_instances", 0)) and n_run >= int(rc.max_instances):
                    break
                n_run += 1
                ti, llm0 = time.time(), teacher.stats()
                seed = zlib.crc32(inst.instance_id.encode()) % (2**31)
                names = [x.split("_1")[0].replace("_", " ") for x in parse_bddl_objects_of_interest(inst.bddl)][:2]
                names += ["object", "target"][len(names):]
                s = Sess(pool, inst.bddl, names, bool(rc.geo))
                atts = [Attempt(k) for k in range(K)]
                for A in atts:   # stage 1 of every attempt (independent fresh sessions, different LLM seeds)
                    ok = stage_once(A, teacher, s, inst, rc, seed + 104_729 * A.k)
                    A.status = "stage1" if ok else "failed"
                    if ranker is not None:
                        z, q = ranker.encode(A.end_obs)
                        A.cpi1 = float(ranker.score_real(z, q, inst.task)[0])
                orders = {"a3": list(range(K)), "d": [int(i) for i in np.argsort(-np.array([a.cpi1 for a in atts]),
                                                                                 kind="stable")]}
                budget = int(rc.budget_steps)
                for name, order in orders.items():
                    used = sum(a.teacher_steps for a in atts) if name == "d" else 0
                    for k in order:
                        A = atts[k]
                        if name == "a3":
                            used += A.stages[0]["n_actions"]
                        if used > budget:
                            break
                        if A.status == "stage1":
                            test_and_extend(A, teacher, s, policy, inst, rc, seed + 104_729 * A.k)
                        used += A.teacher_steps - A.stages[0]["n_actions"] + A.student_steps
                        if A.status == "accepted":
                            break
                # RetryVLA's full-task acceptance on the final path of every processed attempt (measurement)
                for A in atts:
                    A.full = None
                    if A.status not in ("accepted", "failed") or not A.path:
                        continue
                    rec = A.status == "accepted"
                    r = full_verify(pool, policy, inst, A.path, cont, seed, record=rec)
                    A.full = {"success": bool(r.success), "steps": int(r.steps), "prefix_success": bool(r.prefix_success)}
                    if rec:
                        meta = {"arm": "lrs", "attempt": A.k, "scene": inst.scene_tag, "full_success": bool(r.success),
                                "n_stages": len(A.stages), "cpi1": A.cpi1}
                        for wn, keep in (("their_accept", True), ("full_verified", bool(r.success))):
                            if keep:
                                wr[wn].add_instance(inst.instance_id, {"scene": inst.scene_tag, "task": inst.task})
                                wr[wn].add_trajectory(r.record, inst.task, inst.instance_id, meta,
                                                      keep_continuation=(wn == "full_verified"))
                    r.record = None
                llm1 = teacher.stats()
                row = {"instance_id": inst.instance_id, "scene": inst.scene_tag, "task_folder": inst.meta["task_folder"],
                       "gripper_init": inst.meta["gripper_init"], "level": inst.meta.get("level"),
                       "geo": bool(rc.geo), "orders": orders,
                       "attempts": [{**A.log(), "full": A.full} for A in atts],
                       "llm_calls": llm1["llm_calls"] - llm0["llm_calls"], "llm_s": llm1["llm_s"] - llm0["llm_s"],
                       "tok_in": llm1["tok_in"] - llm0["tok_in"], "tok_out": llm1["tok_out"] - llm0["tok_out"],
                       "t_s": time.time() - ti}
                with open(part / "rows.jsonl", "a") as f:
                    f.write(json.dumps(row, default=float) + "\n")
                with open(part / "transcripts.jsonl", "a") as f:
                    f.write(json.dumps({"instance_id": inst.instance_id,
                                        "attempts": [A.transcript for A in atts]}) + "\n")
                log.info("[%d/%d] %s | %s | llm %d calls %.0fs | %.0fs (total %.0fs)", n + 1, len(insts),
                         inst.instance_id[-38:],
                         " ".join(f"A{A.k}:{A.status[:4]}/{len(A.stages)}st/{len(A.path)}a/"
                                  f"full={None if A.full is None else int(A.full['success'])}/cpi={A.cpi1:.2f}"
                                  for A in atts), row["llm_calls"], row["llm_s"], row["t_s"], time.time() - t0)
        for w in wr.values():
            w.close()
        (part / "CLOSED").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        run.summary["config"] = to_dict(cfg)


if __name__ == "__main__":
    main()
