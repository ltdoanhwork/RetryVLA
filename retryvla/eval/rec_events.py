"""V2R failure-event detector E-v1s (Spatial port of retryvla.eval.goal_events E-v1; PLAN "V2R", frozen with it).

python -m retryvla.eval.rec_events --run_dir runs/<id>_nominal [--workers 12]
python -m retryvla.eval.rec_events --run_dir runs/<id>_recover_test --kind recover   (jerk / steps; events optional)

Needs an eval run made with `eval.save_actions=true` (episodes.jsonl rows carry `job`, actions.npz keyed by job).
Every episode is replayed bit-exactly from its own load command (load_nominal / load_scene, rebuilt from the run's
config exactly like eval_nominal / eval_recover) with the saved actions; privileged state every 5 steps
(`vec.events_scan_load`, same sampler as Goal) -> `goal_events.detect` (same thresholds). Replay check: the replay's
first success step must equal the logged one (success episodes) / no success (failures); mismatches are excluded
from the event metrics and counted. Also per episode: closed-loop action jerk (mean ||Δ³a||, dims 0-5) and steps.
Output: <run_dir>/events.jsonl + events_summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from retryvla import REPO_ROOT
from retryvla.eval.goal_events import EVERY, detect

VERSION = "E-v1s"


def jerk(a: np.ndarray) -> float:
    a = np.asarray(a, np.float32)[:, :6]
    return float(np.linalg.norm(np.diff(a, n=3, axis=0), axis=1).mean()) if len(a) > 3 else float("nan")


def rebuild_jobs(cfg: dict, kind: str):
    from omegaconf import OmegaConf

    c = OmegaConf.create(cfg)
    if kind == "nominal":
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()[c.env.suite]()
        tids = list(c.nominal.task_ids) if c.nominal.get("task_ids") else list(range(len(suite.tasks)))
        out = []
        for tid in tids:
            for ep in range(int(c.nominal.episodes_per_task)):
                out.append(("load_nominal", dict(suite=c.env.suite, task_id=tid, init_idx=ep,
                                                 num_steps_wait=c.env.nominal_num_steps_wait, seed=c.eval.seed + ep)))
        return out
    from retryvla.envs.splits import load_split, scenes_for
    from retryvla.eval.eval_recover import instance_jobs

    split = load_split(REPO_ROOT / c.split.file)
    root = REPO_ROOT / c.recover.scene_root
    jobs = []
    for part in c.recover_eval.splits:
        scenes = scenes_for(split, part, root)
        tf = list(c.recover_eval.get("task_filter") or [])
        if tf:
            scenes = [s for s in scenes if any(x in s.task_folder for x in tf)]
        if c.recover_eval.get("max_scenes"):
            scenes = scenes[: int(c.recover_eval.max_scenes)]
        jobs += instance_jobs(scenes, c.recover, int(c.eval.seed), c.recover.instruction, part=part)
    return [(j.load_cmd, j.load_kwargs) for j in jobs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--kind", default="nominal", choices=["nominal", "recover"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--no_events", action="store_true", help="jerk / steps only (no replay)")
    a = ap.parse_args()
    d = Path(a.run_dir)
    if (d / "events_summary.json").exists():
        print("exists")
        return
    eps = [json.loads(l) for l in open(d / "episodes.jsonl")]
    acts = np.load(d / "actions.npz")
    cfg = yaml.safe_load(open(d / "config.yaml"))
    loads = rebuild_jobs(cfg, a.kind)
    rows = []
    todo = [e for e in eps]
    if a.no_events:
        for e in todo:
            ac = acts[str(e["job"])]
            rows.append({**{k: e[k] for k in e if k in ("task_id", "episode", "scene", "gripper_init", "trial")},
                         "success": e["success"], "steps": e["steps"], "jerk": jerk(ac)})
    else:
        from retryvla.envs.vec import SimPool

        with SimPool(a.workers, int(cfg["env"]["resolution"])) as pool:
            while todo:
                batch, todo = todo[: pool.n], todo[pool.n:]
                for w, e in enumerate(batch):
                    cmd, kw = loads[int(e["job"])]
                    pool.send(w, "events_scan_load", cmd, kw, acts[str(e["job"])], EVERY)
                for w, e in enumerate(batch):
                    r = pool.recv(w)
                    ss = e["steps"] if e["success"] else None
                    ok = (r["success_step"] == ss) if e["success"] else (r["success_step"] is None)
                    ev = detect(r["samples"], ss)
                    rows.append({**{k: e[k] for k in e if k in ("task_id", "episode", "scene", "gripper_init", "trial")},
                                 "success": e["success"], "steps": e["steps"], "replay_ok": bool(ok),
                                 "replay_success_step": r["success_step"], "events": ev, "n_events": len(ev),
                                 "first_attempt_success": bool(e["success"] and not ev),
                                 "jerk": jerk(acts[str(e["job"])])})
                print(json.dumps({"done": len(rows), "of": len(eps)}), flush=True)
    with open(d / "events.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    good = [r for r in rows if r.get("replay_ok", True)]
    succ = [r for r in rows if r["success"]]
    types: dict[str, int] = {}
    for r in good:
        for e in r.get("events", []):
            types[e["type"]] = types.get(e["type"], 0) + 1
    summ = {"version": VERSION, "kind": a.kind, "n": len(rows), "n_replay_ok": len(good),
            "success": float(np.mean([r["success"] for r in rows])),
            "jerk_mean": float(np.nanmean([r["jerk"] for r in rows])),
            "completion_steps_mean": float(np.mean([r["steps"] for r in succ])) if succ else None,
            "completion_steps_median": float(np.median([r["steps"] for r in succ])) if succ else None}
    if not a.no_events and good:
        summ.update({"failure_incidence": float(np.mean([r["n_events"] > 0 for r in good])),
                     "first_attempt_success": float(np.mean([r["first_attempt_success"] for r in good])),
                     "p_success_given_event": (float(np.mean([r["success"] for r in good if r["n_events"]]))
                                               if any(r["n_events"] for r in good) else None),
                     "events_by_type": types})
    (d / "events_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
