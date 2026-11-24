"""LRS-G: the 2610.05151 Recovery pipeline on the LIBERO-Goal protocol (PLAN "LRS-G", pre-registered 2026-11-24).

python -m retryvla.baselines.lrs_goal --lane 0 --num_lanes 4 [--max_entries N] [--out data/recovery/lrs_goal]

Per audit entry (the 168 seed-7 fixed entries, same snapshots as goal_search), ONE attempt (their final protocol):
  stage r = 1..5: fresh Qwen3-VL-8B tool-teacher session from the current endpoint (frozen JSON spec in reply 1,
  bounded move_to / move_by / gripper tools, <= 8 replies, <= 300 teacher actions in total) -> hand-off test (Eq. 7):
  frozen pi0 M0 (execute 5 / replan) runs <= 60 actions, LLM observer every 10 actions on the spec's progress
  conditions, the first advance must survive 10 more actions -> accept, else restore the teacher endpoint and extend.
  EXTRA (not in the paper): every hand-off run continues to <= 300 student steps / BDDL success = sim ground truth.
Outputs per lane: rows.jsonl (one row per entry: stages, specs, commands with action index ranges, observer
verdicts, GT), actions/<entry>.npz (teacher path + every student branch), transcripts.jsonl. Resumable.
Deviations D-L1..D-L8: experiments/PLAN.md "LRS-G".
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np

from retryvla import REPO_ROOT
from retryvla.baselines import lrs_goal_prompts as GP
from retryvla.baselines.lrs_llm import parse_commands

log = logging.getLogger("lrs_goal")
TABLE_Z = 0.9          # LIBERO-Goal table top (env.table_offset[2])
STEP_M = 0.01          # ~ m of eef motion per step at |a| = 1 (retryvla.baselines.lrs_teacher)


class GSess:
    """Teacher / student session on an in-process LiberoSim (exact snapshots)."""

    def __init__(self, sim, bddl: str):
        self.sim, self.bddl = sim, bddl
        self.obs, self.success = None, False

    def restore(self, snap: dict) -> None:
        self.obs, self.success = self.sim.restore_snapshot(self.bddl, snap)["obs"], False

    def snapshot(self) -> dict:
        return self.sim.snapshot()

    def step(self, a) -> None:
        r = self.sim.step(np.asarray(a, np.float32))
        self.obs, self.success = r.obs, bool(r.success)

    def eef(self) -> np.ndarray:
        e = np.asarray(self.obs["state"][:3], float).copy()
        e[2] -= TABLE_Z
        return e

    def opening(self) -> float:
        return float(np.abs(self.obs["state"][6:8]).sum())

    def state_txt(self) -> str:
        e = self.eef()
        return f"gripper at ({e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f}), finger opening {self.opening():.3f}"

    def privileged(self) -> str:
        """Offline-evaluator summary: object positions (z above table), goal predicates."""
        env = self.sim._env.env
        d = env.sim.data
        objs = ", ".join(f"{n} ({d.body_xpos[b][0]:.3f}, {d.body_xpos[b][1]:.3f}, {d.body_xpos[b][2] - TABLE_Z:.3f})"
                         for n, b in env.obj_body_id.items())
        preds = []
        for st in env.parsed_problem["goal_state"]:
            try:
                preds.append(f"{' '.join(st)} = {bool(env._eval_predicate(st))}")
            except Exception:
                pass
        return f"{self.state_txt()}; objects: {objs}; goal predicates: {'; '.join(preds)}"


def exec_cmd(s: GSess, c: dict, grip: float, left: int) -> tuple[list[np.ndarray], float, str]:
    """One bounded tool command (same servo as retryvla.baselines.lrs_teacher.exec_cmd)."""
    acts: list[np.ndarray] = []
    tool = str(c.get("tool", ""))
    if tool in ("move_to", "move_by"):
        try:
            v = np.asarray(c.get("xyz") if tool == "move_to" else c.get("dxyz"), float).reshape(3)
        except Exception:
            return acts, grip, f"{tool}: invalid arguments {c}"
        tgt = v if tool == "move_to" else s.eef() + v
        tgt[2] = max(tgt[2], 0.0)
        try:
            n = int(np.clip(int(c.get("max_steps", 40) or 40), 1, 40))
        except Exception:
            n = 40
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


def parse_json_obj(text: str, key: str) -> dict | None:
    """Last JSON object containing `key` (tolerant to fences / prose)."""
    for m in reversed(list(re.finditer(r"\{", text))):
        depth, j = 0, m.start()
        for k in range(j, len(text)):
            depth += text[k] == "{"
            depth -= text[k] == "}"
            if depth == 0:
                try:
                    o = json.loads(text[j: k + 1])
                    if isinstance(o, dict) and key in o:
                        return o
                except json.JSONDecodeError:
                    pass
                break
    return None


def img_msg(obs) -> list[dict]:
    return [{"type": "image", "image": obs["image"]}, {"type": "image", "image": obs["image2"]}]


# ------------------------------------------------------------------------------------------------ teacher stage
def run_stage(teacher, s: GSess, task: str, stage: int, prev_cmds: list[str], student_txt: str, grip: float,
              act_left: int, max_turns: int, seed: int, transcript: list) -> dict:
    budget = lambda t, a: f"Budget: {max_turns - t} replies, {a} robot steps left in this attempt."
    ext = GP.extension_text(stage, prev_cmds, student_txt) if stage > 0 else ""
    msgs = [{"role": "system", "content": [{"type": "text", "text": GP.SYSTEM}]},
            {"role": "user", "content": img_msg(s.obs) + [{"type": "text", "text": GP.first_user(
                task, s.state_txt(), budget(0, act_left), ext)}]}]
    acts, cmds, end, spec = [], [], "turn_cap", None
    turn = 0
    for turn in range(max_turns):
        reply = teacher.chat(msgs, seed=seed * 131 + turn)
        transcript.append({"stage": stage, "turn": turn, "reply": reply[-2000:]})
        if spec is None:
            o = parse_json_obj(reply, "spec")
            spec = o["spec"] if o and isinstance(o.get("spec"), dict) else None
        cl = parse_commands(reply)
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
        fbs = []
        for c in (cl or [])[:4]:
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
            pb = s.privileged()
            a, grip, f = exec_cmd(s, c, grip, act_left - len(acts))
            cmds.append({"cmd": json.dumps(c)[:200], "i0": len(acts), "i1": len(acts) + len(a), "before": pb,
                         "after": s.privileged(), "fb": f})
            acts += a
            fbs.append(f)
        fb = "; ".join(fbs) or ("no valid JSON {\"commands\": [...]} found; nothing executed" if not cl
                                else "nothing executed")
        if end != "turn_cap":
            break
        if len(acts) >= act_left:
            end = "action_cap"
            break
        for m in msgs:   # keep the context small: only the latest images
            if m["role"] == "user":
                m["content"] = [c for c in m["content"] if c["type"] == "text"]
        msgs.append({"role": "user", "content": img_msg(s.obs) + [{"type": "text", "text": GP.feedback_user(
            fb, s.state_txt(), budget(turn + 1, act_left - len(acts)))}]})
    if spec is None:   # D-L3 fallback: the teacher never wrote a parseable spec -> task-level spec
        spec = {"local_goal": task, "completion_checks": [], "preserve": [],
                "student_progress": [f"the robot makes visible progress toward: {task}"], "completed_task": task}
    if not spec.get("student_progress"):   # D-L3: empty progress conditions -> completion checks / task progress
        spec["student_progress"] = list(spec.get("completion_checks") or []) or \
            [f"the robot makes visible progress toward: {task}"]
    if end == "turn_cap" and acts:   # D-L4b: our reply cap << their 100 requests; a capped stage is handed off
        end = "turn_cap_handoff"
    return {"actions": acts, "cmds": cmds, "end": end, "grip": grip, "turns": turn + 1, "spec": spec}


# --------------------------------------------------------------------------------------- hand-off (Eq. 7)
def observe(teacher, task: str, spec: dict, obs0, obs1, st0: str, st1: str, n: int, confirm: bool, log_: list) -> dict:
    msgs = [{"role": "system", "content": [{"type": "text", "text": GP.OBSERVER_SYSTEM}]},
            {"role": "user", "content": img_msg(obs0) + img_msg(obs1) + [{"type": "text", "text": GP.observer_user(
                task, json.dumps(spec)[:1500], st0, st1, n, confirm)}]}]
    reply = teacher.chat(msgs, seed=0, greedy=True, max_new_tokens=120)
    o = parse_json_obj(reply, "progress") or {}
    v = {"n": n, "confirm": confirm, "progress": bool(o.get("progress", False)),
         "completed_task": bool(o.get("completed_task", False)), "preserved": bool(o.get("preserved", True)),
         "reason": str(o.get("reason", reply[-200:]))[:200], "parsed": bool(o)}
    log_.append(v)
    return v


def handoff(teacher, policy, s: GSess, task: str, spec: dict, seed: int, H_S=60, delta=10, C=10, gt_max=300) -> dict:
    """Paper Eq. 7 with an LLM observer; then (EXTRA) the same student run continues to <= gt_max steps for the sim
    ground truth. Student actions are never imitation targets."""
    obs0, st0 = s.obs, s.state_txt()
    acts, k, verdicts = [], 0, []
    gt_succ, gt_step = False, None

    def chunk(n):
        nonlocal k, gt_succ, gt_step
        ch = policy.predict_chunk([s.obs], [task], seeds=[seed * 100_003 + k])[0]
        k += 1
        done = 0
        while done < n:
            for a in ch[: min(5, n - done)]:
                s.step(a)
                acts.append(np.asarray(a, np.float32))
                done += 1
                if s.success and not gt_succ:
                    gt_succ, gt_step = True, len(acts)
            if done < n:
                ch = policy.predict_chunk([s.obs], [task], seeds=[seed * 100_003 + k])[0]
                k += 1

    accept, anchor = False, None
    while len(acts) < H_S:
        chunk(delta)
        v = observe(teacher, task, spec, obs0, s.obs, st0, s.state_txt(), len(acts), False, verdicts)
        if (v["progress"] or v["completed_task"]) and v["preserved"]:
            anchor = len(acts)
            break
    if anchor is not None:
        chunk(C)
        v = observe(teacher, task, spec, obs0, s.obs, st0, s.state_txt(), len(acts), True, verdicts)
        accept = bool((v["progress"] or v["completed_task"]) and v["preserved"])
    n_test = len(acts)
    e1 = s.eef()
    summary = (f"{'progress was not sustained' if anchor is not None else 'no supported progress'}; in {n_test} "
               f"actions the student moved the gripper to ({e1[0]:.3f}, {e1[1]:.3f}, {e1[2]:.3f}), finger opening "
               f"{s.opening():.3f}; observer: {verdicts[-1]['reason'] if verdicts else ''}")
    while len(acts) < gt_max and not gt_succ:      # EXTRA: sim ground truth of the same student run
        chunk(min(5, gt_max - len(acts)))
    return {"accept": accept, "anchor": anchor, "n_test": n_test, "verdicts": verdicts, "summary": summary,
            "gt_success": gt_succ, "gt_step": gt_step, "student_actions": np.asarray(acts, np.float32)}


# ----------------------------------------------------------------------------------------- quality (offline)
def grade(teacher, task: str, spec: dict, cmds: list[dict], final_txt: str) -> list[float]:
    """Offline action-quality step (D-L6: one grade per teacher command, applied to all its actions)."""
    rows = [f"{i + 1}. {c['cmd']} ({c['i1'] - c['i0']} actions)\n   before: {c['before']}\n   after: {c['after']}"
            for i, c in enumerate(cmds)]
    msgs = [{"role": "system", "content": [{"type": "text", "text": GP.EVAL_SYSTEM}]},
            {"role": "user", "content": [{"type": "text", "text": GP.eval_user(task, json.dumps(spec)[:1200], rows,
                                                                               final_txt)}]}]
    reply = teacher.chat(msgs, seed=0, greedy=True, max_new_tokens=40 + 8 * len(cmds))
    o = parse_json_obj(reply, "q") or {}
    q = o.get("q") if isinstance(o.get("q"), list) else []
    out = []
    for i in range(len(cmds)):
        try:
            v = float(q[i])
        except (IndexError, TypeError, ValueError):
            v = 0.5   # unparseable -> neutral (excluded by the 0.75 threshold)
        out.append(float(min([0, .25, .5, .75, 1], key=lambda x: abs(x - v))))
    return out


# ------------------------------------------------------------------------------------------------- attempt
def run_entry(E: dict, sim, teacher, policy, a, seed: int) -> tuple[dict, dict, list]:
    s = GSess(sim, E["bddl"])
    s.restore(E["snap"])
    task, iid = E["lang"], E["entry"]["entry_id"]
    t_e = int(E["entry"]["t"])
    grip = float(np.sign(E["audit"][t_e - 1][6])) if t_e > 0 else -1.0
    path, cmds_all, stages, branches, transcript = [], [], [], {}, []
    student_txt, end_snap, status = "", None, "failed"
    for r in range(int(a.max_stages)):
        if end_snap is not None:
            s.restore(end_snap)
        st = run_stage(teacher, s, task, r, [c["cmd"] for c in cmds_all], student_txt, grip,
                       int(a.max_actions) - len(path), int(a.max_turns), seed + 1009 * r, transcript)
        off = len(path)
        for c in st["cmds"]:
            cmds_all.append({**c, "stage": r, "i0": c["i0"] + off, "i1": c["i1"] + off})
        path += st["actions"]
        grip = st["grip"]
        rec = {"stage": r, "end": st["end"], "n_actions": len(st["actions"]), "turns": st["turns"], "spec": st["spec"],
               "teacher_success": bool(s.success)}
        stages.append(rec)
        if st["end"] not in ("claimed", "action_cap", "turn_cap_handoff"):
            status = st["end"]
            break
        end_snap = s.snapshot()
        h = handoff(teacher, policy, s, task, st["spec"], seed + 7 * r, int(a.H_S), int(a.delta), int(a.C),
                    int(a.gt_max))
        branches[f"student_{r}"] = h.pop("student_actions")
        rec["handoff"] = {k: h[k] for k in ("accept", "anchor", "n_test", "verdicts", "gt_success", "gt_step")}
        rec["path_len"] = len(path)
        if h["accept"]:
            status = "accepted"
            break
        student_txt = h["summary"]
        if len(path) >= int(a.max_actions) or st["end"] == "action_cap":
            status = "budget"
            break
    q = []
    if status == "accepted":
        last = stages[-1]["handoff"]
        final_txt = (f"after the teacher, the student continued and the observer accepted the hand-off "
                     f"(anchor {last['anchor']})")
        q = grade(teacher, task, stages[-1]["spec"], cmds_all, final_txt) if cmds_all else []
        for c, g in zip(cmds_all, q):
            c["q"] = g
    acts = {"path": np.asarray(path, np.float32).reshape(-1, 7), **branches}
    row = {"instance_id": iid, "scene": E["entry"]["scene"], "task_folder": E["entry"]["task"], "t": t_e,
           "status": status, "n_stages": len(stages), "path_len": len(path), "stages": stages,
           "cmds": [{k: v for k, v in c.items() if k not in ("before", "after")} for c in cmds_all],
           "accepted_gt": bool(stages[-1].get("handoff", {}).get("gt_success")) if status == "accepted" else None,
           "seed": seed}
    return row, acts, transcript


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/recovery/lrs_goal")
    ap.add_argument("--lane", type=int, default=0)
    ap.add_argument("--num_lanes", type=int, default=1)
    ap.add_argument("--max_entries", type=int, default=0)
    ap.add_argument("--entries", nargs="*", default=None, help="explicit entry ids (demos / smoke)")
    ap.add_argument("--attempt", type=int, default=0, help="D-L8: 1 = second attempt on entries without acceptance")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_turns", type=int, default=12)
    ap.add_argument("--max_stages", type=int, default=5)
    ap.add_argument("--max_actions", type=int, default=300)
    ap.add_argument("--H_S", type=int, default=60)
    ap.add_argument("--delta", type=int, default=10)
    ap.add_argument("--C", type=int, default=10)
    ap.add_argument("--gt_max", type=int, default=300)
    a = ap.parse_args()

    import torch

    from retryvla.baselines.lrs_llm import VLMTeacher
    from retryvla.envs.libero_env import LiberoSim
    from retryvla.policy.pi0_wrapper import Pi0Wrapper
    from scripts.demos.goal_demo_common import AUDIT_DIR, AUDIT_SEED, M0_CKPT, build_entry, entry_seed

    out = REPO_ROOT / a.out / f"a{a.attempt}" / f"l{a.lane}"
    (out / "actions").mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=1))
    ids = a.entries or [json.loads(l)["entry_id"] for l in open(REPO_ROOT / AUDIT_DIR / f"entries_s{AUDIT_SEED}.jsonl")]
    if a.attempt > 0:   # D-L8: only entries without an accepted path in earlier attempts
        acc = set()
        for p in (REPO_ROOT / a.out).glob("a*/l*/rows.jsonl"):
            acc |= {r["instance_id"] for r in map(json.loads, p.read_text().splitlines()) if r["status"] == "accepted"}
        ids = [i for i in ids if i not in acc]
    ids = ids[a.lane:: a.num_lanes][: a.max_entries or None]
    done = set()
    if (out / "rows.jsonl").exists():
        done = {json.loads(l)["instance_id"] for l in (out / "rows.jsonl").read_text().splitlines() if l.strip()}
    todo = [i for i in ids if i not in done]
    log.info("lane %d/%d attempt %d: %d entries, %d done, %d to run", a.lane, a.num_lanes, a.attempt, len(ids),
             len(done), len(todo))
    if not todo:
        (out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        return
    policy = Pi0Wrapper(M0_CKPT, "cuda", 5)
    teacher = VLMTeacher(a.model, "cuda", a.max_new_tokens, a.temperature)
    sim = LiberoSim(128)
    t0 = time.time()
    for n, iid in enumerate(todo):
        ti, s0 = time.time(), teacher.stats()
        E = build_entry(sim, iid)
        seed = (entry_seed(iid) + 104_729 * a.attempt) % (2**31 - 1)
        row, acts, tr = run_entry(E, sim, teacher, policy, a, seed)
        s1 = teacher.stats()
        row.update({"attempt": a.attempt, "llm_calls": s1["llm_calls"] - s0["llm_calls"],
                    "llm_s": round(s1["llm_s"] - s0["llm_s"], 1), "tok_out": s1["tok_out"] - s0["tok_out"],
                    "t_s": round(time.time() - ti, 1)})
        np.savez_compressed(out / "actions" / f"{iid}.npz", **acts)
        with open(out / "transcripts.jsonl", "a") as f:
            f.write(json.dumps({"instance_id": iid, "transcript": tr}) + "\n")
        with open(out / "rows.jsonl", "a") as f:
            f.write(json.dumps(row, default=float) + "\n")
        hs = [st.get("handoff", {}) for st in row["stages"]]
        log.info("[%d/%d] %s | %s %dst %da | judge %s | gt %s | llm %d calls %.0fs | %.0fs (total %.0f min)",
                 n + 1, len(todo), iid[-45:], row["status"], row["n_stages"], row["path_len"],
                 "".join("A" if h.get("accept") else ("r" if h else "-") for h in hs),
                 "".join("S" if h.get("gt_success") else ("f" if h else "-") for h in hs),
                 row["llm_calls"], row["llm_s"], row["t_s"], (time.time() - t0) / 60)
        torch.cuda.empty_cache()
    (out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))


if __name__ == "__main__":
    main()
