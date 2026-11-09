"""ACR data collection (Active Causal Recovery PoC; design: experiments/PLAN.md "ACR"). TRAIN split only.

python -m retryvla.search.acr_search configs/exp/acr_poc.yaml acr.tier=B [acr.lane=i acr.num_lanes=N]
  acr.tier = calib | B | A
  acr.mode = collect (default) | postprobe (needs acr.decisions = runs/_analysis/acr/decisions_<tier>.json)

collect, per state (instances: data/acr/instances.json, restored bit-exactly from the RGR / M5b precheck snapshots):
  Tier B / calib  PLANNING (particles h = nominal + range ends of every hidden type, never the true magnitude):
                    every probe's response + risk under every particle; n_plan_var trials of every family under
                    every particle -> V_plan[h, f]                        (calib: planning only)
                  REAL (one case per hidden type, true magnitude drawn inside the range): every probe's response +
                    risk; the OUTCOME TABLE O[f, j] = trial j of family f (n_real_var variants, same variants and
                    continuation seeds in every case -> paired)
  Tier A          REAL only (nominal physics): probe responses, outcome table, privileged cause label
Every arm (analysis) is computed from these tables; postprobe re-runs the allocated trials of one arm from the TRUE
post-probe state (the probe's side effects included) as the honest check of the table-based numbers.
Outputs: <out_root>/<tier>/<instance_id>.json (one file per state; resumable: existing files are skipped).
"""

from __future__ import annotations

import json
import logging
import time
import zlib

import numpy as np

from retryvla import REPO_ROOT, tracking
from retryvla.causal import families as FAM
from retryvla.causal import hypotheses as HYP
from retryvla.causal import probes as PR
from retryvla.causal.causes import cause_label
from retryvla.causal.exec import run_seqs
from retryvla.config import cli_config, to_dict
from retryvla.envs.splits import assert_allowed, load_split, scenes_for
from retryvla.envs.vec import SimPool
from retryvla.eval.rollout import Job
from retryvla.policy import SmolVLAWrapper
from retryvla.search.compare_arms import load_known_precheck, make_safe_run, precheck, select_jobs, iid
from retryvla.utils import setup_logging

log = logging.getLogger("acr_search")


def _js(x):
    if isinstance(x, dict):
        return {k: _js(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_js(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def restore_instances(pool, policy, cfg, ids: list[str], srun):
    """Snapshots of the selected instances (restore only for known pi0 failures: same protocol / seeds as RGR)."""
    split = load_split(REPO_ROOT / cfg.split.file)
    scenes = scenes_for(split, "train", REPO_ROOT / cfg.recover.scene_root)
    assert_allowed([s.tag for s in scenes], split)
    mb = cfg.m5b
    jobs = []
    for sh in list(cfg.acr.shards):
        mb.shard = int(sh)
        jobs += select_jobs(scenes, cfg, mb)
    want = set(ids)
    jobs = [j for j in jobs if iid(j.meta) in want]
    missing = want - {iid(j.meta) for j in jobs}
    if missing:
        raise RuntimeError(f"{len(missing)} instances not reconstructed, e.g. {sorted(missing)[:2]}")
    insts, _ = precheck(pool, policy, jobs, set(), run=srun, known=load_known_precheck(cfg.acr.precheck_from))
    by = {i.instance_id: i for i in insts}
    return [by[i] for i in ids if i in by]


def seeds(base: int, kind: int, fi: int, j: int) -> tuple[np.random.Generator, int]:
    return np.random.default_rng([base, kind, fi, j]), int((base + 500_000 * (kind - 1) + 1000 * fi + j) % (2**31 - 1))


def trials(srun, policy, inst, snap, hiddens, g, fams, n_var, kind, base, cfg, desc) -> tuple[np.ndarray, np.ndarray]:
    """[len(hiddens), F, n_var] success (prefix or pi) and env steps of every family variant from `snap` under every
    hidden h (one batch). Variant j of family f has the same waypoints / continuation seed under every h (paired)."""
    a = cfg.acr
    jobs, own = [], []
    for hi, hidden in enumerate(hiddens):
        for fi, f in enumerate(fams):
            for j in range(n_var):
                rng, ps = seeds(base, kind, fi, j)
                wps = FAM.build(f, g, rng, float(a.perception_sigma))
                jobs.append(Job("restore_family", dict(bddl=inst.bddl, snap=snap, hidden=hidden, wps=wps),
                                task=inst.task, max_steps=int(a.cont_steps), policy_seed=ps))
                own.append((hi, fi, j))
    res = srun(policy, jobs, noise_scale=float(cfg.policy.noise_scale), desc=desc)
    S = np.zeros((len(hiddens), len(fams), n_var), bool)
    N = np.zeros((len(hiddens), len(fams), n_var), int)
    for (hi, fi, j), r in zip(own, res):
        S[hi, fi, j] = bool(r.success or r.load_out.get("prefix_success", False))
        N[hi, fi, j] = int(r.steps) + int(r.load_out.get("n_prefix", 0))
    return S, N


def probe_all(pool, inst, snap, hidden_list, lib) -> list[dict]:
    """For every hidden h in hidden_list and every probe: restore_hidden(h) -> probe. Returns [len(h) * len(lib)]."""
    items = [[("restore_hidden", (inst.bddl, snap, h), {}), ("probe", (lib[n],), {"snapshot": False})]
             for h in hidden_list for n in lib]
    out = run_seqs(pool, items)
    for r in out:
        r.pop("obs", None)
        r["risk"] = PR.risk(r["geom0"], r["geom1"])
    return out


def collect(cfg, pool, policy, srun, run):
    a = cfg.acr
    tier = str(a.tier)
    sel = json.loads((REPO_ROOT / a.instances).read_text())
    ids = sel[a.id_list] if a.get("id_list") else {"calib": sel["tier_b_calib"], "B": sel["tier_b"], "A": sel["tier_a"]}[tier]
    ids = ids[int(a.get("lane", 0)):: int(a.get("num_lanes", 1))]
    if int(a.get("max_states", 0)):
        ids = ids[: int(a.max_states)]
    out_dir = REPO_ROOT / a.out_root / tier
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (out_dir / f"{i}.json").exists()]
    log.info("tier %s: %d states (%d todo)", tier, len(ids), len(todo))
    if not todo:
        return
    insts = restore_instances(pool, policy, cfg, todo, srun)
    fams = list(a.families)
    types = {k: list(v) for k, v in to_dict(a.types).items()}
    parts = HYP.particles(types)
    t0 = time.time()
    for n, inst in enumerate(insts):
        ti = time.time()
        base = zlib.crc32(inst.instance_id.encode()) % (2**31)
        info = run_seqs(pool, [[("restore_hidden", (inst.bddl, inst.snapshot, None), {}), ("state_info", (), {})]])[0]
        g0 = np.asarray(info["geom"])
        lib = PR.library(PR.gripper_cmd(g0))
        rec = {"instance_id": inst.instance_id, "scene": inst.scene_tag, "task": inst.task, "tier": tier,
               "task_folder": inst.meta.get("task_folder"), "level": sel["level"].get(inst.instance_id),
               "single_solved": sel["single_solved"].get(inst.instance_id), "g0": g0,
               "clutter_dist": float(info["clutter_dist"]), "cause": cause_label(g0, float(info["clutter_dist"])),
               "families": fams, "probes": list(lib), "probe_len": {k: int(len(v)) for k, v in lib.items()},
               "feature_keys": list(PR.FEATS_ORACLE), "cases": []}
        if tier in ("calib", "B"):
            pr = probe_all(pool, inst, inst.snapshot, parts, lib)
            P, A_ = len(parts), len(lib)
            rec["particles"] = parts
            rec["plan_R"] = [[PR.feat_vec(pr[p * A_ + k]["feats"], "oracle") for k in range(A_)] for p in range(P)]
            rec["plan_risk"] = [[pr[p * A_ + k]["risk"] for k in range(A_)] for p in range(P)]
            S, N = trials(srun, policy, inst, inst.snapshot, parts, g0, fams, int(a.n_plan_var), 2, base, cfg, "plan")
            rec["plan_S"], rec["plan_steps"] = S, int(N.sum())
            rec["plan_succ"] = S.sum(2)
        if tier == "calib":
            hs = []
        elif tier == "B":
            hs = [HYP.draw_true(types, t, int(base % 100_000) * 10 + k) for k, t in enumerate(["nominal"] + list(types))]
        else:
            hs = [None]
        if hs:
            pr = probe_all(pool, inst, inst.snapshot, hs, lib)
            S, N = trials(srun, policy, inst, inst.snapshot, hs, g0, fams, int(a.n_real_var), 1, base, cfg, "real")
            A_ = len(lib)
            for k, h in enumerate(hs):
                p_ = pr[k * A_:(k + 1) * A_]
                rec["cases"].append({"h_true": h or {"type": "natural"},
                                     "probe_R": [PR.feat_vec(r["feats"], "oracle") for r in p_],
                                     "probe_risk": [r["risk"] for r in p_], "probe_aborted": [r["aborted"] for r in p_],
                                     "probe_success": [r["success"] for r in p_], "O": S[k], "O_steps": N[k]})
        rec["t_s"] = time.time() - ti
        (out_dir / f"{inst.instance_id}.json.tmp").write_text(json.dumps(_js(rec)))
        (out_dir / f"{inst.instance_id}.json.tmp").replace(out_dir / f"{inst.instance_id}.json")
        line = " ".join(f"{c['h_true']['type'][:4]} {np.mean(c['O']):.2f}" for c in rec["cases"])
        if "plan_succ" in rec:
            line += " | plan " + " ".join(f"{p['type'][:4]}{p['mag']:g}:{int(np.sum(s))}"
                                         for p, s in zip(parts, rec["plan_succ"])) + f" (per family x {a.n_plan_var})"
        log.info("[%d/%d] %s cause %s | %s | %.0fs (total %.0fs)", n + 1, len(insts), inst.instance_id[-40:],
                 rec["cause"], line, time.time() - ti, time.time() - t0)
        run.log({"acr/states_done": n + 1}, step=n)


def postprobe(cfg, pool, policy, srun, run):
    """Re-run the allocated trials of the decisions file from the TRUE post-probe state (probe side effects in)."""
    a = cfg.acr
    tier = str(a.tier)
    dec = json.loads((REPO_ROOT / a.decisions).read_text())
    out_p = REPO_ROOT / a.out_root / f"postprobe_{tier}.jsonl"
    done = set()
    if out_p.exists():
        done = {(json.loads(l)["instance_id"], json.loads(l)["case"], json.loads(l)["arm"]) for l in open(out_p)}
    rows = [d for d in dec["rows"] if (d["instance_id"], d["case"], d["arm"]) not in done]
    lane, nl = int(a.get("lane", 0)), int(a.get("num_lanes", 1))
    ids = sorted({d["instance_id"] for d in rows})[lane::nl]
    if not ids:
        return
    insts = {i.instance_id: i for i in restore_instances(pool, policy, cfg, ids, srun)}
    fams = list(a.families)
    for k, iid_ in enumerate(ids):
        inst = insts[iid_]
        base = zlib.crc32(iid_.encode()) % (2**31)
        rec = json.loads((REPO_ROOT / a.out_root / tier / f"{iid_}.json").read_text())
        lib = PR.library(PR.gripper_cmd(np.asarray(rec["g0"])))
        for d in [d for d in rows if d["instance_id"] == iid_]:
            h = rec["cases"][d["case"]]["h_true"]
            h = None if h["type"] == "natural" else h
            snap, g = inst.snapshot, np.asarray(rec["g0"])
            pinfo = {"n": 0, "success": False}
            if d.get("probe"):
                r = run_seqs(pool, [[("restore_hidden", (inst.bddl, inst.snapshot, h), {}),
                                     ("probe", (lib[d["probe"]],), {"snapshot": True})]])[0]
                snap, pinfo = r["snapshot"], {"n": r["n"], "success": r["success"]}
                g = np.asarray(run_seqs(pool, [[("restore_hidden", (inst.bddl, snap, h), {}),
                                                ("state_info", (), {})]])[0]["geom"])
            jobs = []
            for fi, j in d["alloc"]:
                rng, ps = seeds(base, 1, fi, j)
                wps = FAM.build(fams[fi], g, rng, float(a.perception_sigma))
                jobs.append(Job("restore_family", dict(bddl=inst.bddl, snap=snap, hidden=h, wps=wps), task=inst.task,
                                max_steps=int(a.cont_steps), policy_seed=ps))
            res = srun(policy, jobs, noise_scale=float(cfg.policy.noise_scale), desc="postprobe")
            succ = [bool(r.success or r.load_out.get("prefix_success", False)) for r in res]
            with open(out_p, "a") as f:
                f.write(json.dumps({"instance_id": iid_, "case": d["case"], "arm": d["arm"], "probe": d.get("probe"),
                                    "alloc": d["alloc"], "succ": succ, "probe_success": pinfo["success"],
                                    "probe_n": pinfo["n"],
                                    "steps": [int(r.steps) + int(r.load_out.get("n_prefix", 0)) for r in res]}) + "\n")
        log.info("postprobe [%d/%d] %s", k + 1, len(ids), iid_[-40:])


def main() -> None:
    setup_logging()
    cfg = cli_config()
    a = cfg.acr
    cfg.tag = f"{cfg.tag}_{a.tier}_{a.get('mode', 'collect')}_l{a.get('lane', 0)}of{a.get('num_lanes', 1)}"
    with tracking.start(cfg) as run:
        policy = SmolVLAWrapper(cfg.policy.path, cfg.policy.device, cfg.policy.n_action_steps)
        with SimPool(int(cfg.eval.n_workers), int(cfg.env.resolution)) as pool:
            srun = make_safe_run(pool, int(cfg.m5b.get("worker_retries", 2)), int(cfg.m5b.get("oom_retries", 3)),
                                 float(cfg.m5b.get("oom_wait_s", 120)))
            (postprobe if str(a.get("mode", "collect")) == "postprobe" else collect)(cfg, pool, policy, srun, run)
        run.summary["config"] = to_dict(cfg)


if __name__ == "__main__":
    main()
