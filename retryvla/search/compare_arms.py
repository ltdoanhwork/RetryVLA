"""M5b: scaled, PAIRED comparison of recovery-search arms at an equal simulator budget.

python -m retryvla.search.compare_arms configs/exp/m5b_compare_arms.yaml [m5b.shard=i m5b.num_shards=N]

Instances: fresh pi0 failures on TRAIN scenes with object-noise trials never used to train the WM / critic
(trial k in 2..9 of the official 5 close + 5 open protocol; WM/critic data used k in {0, 1}).
Per instance every arm in `m5b.arms` (see retryvla/search/arms.py) picks `budget` candidates from its own
imagination-scored pool; all picks are verified in the simulator in one batch (same continuation seeds
per slot across arms -> paired). Logged per (instance, arm): hits, solved, critic calibration, and
distribution metrics of the verified successes (jerk, roughness, distance to SmolVLA's manifold,
pairwise spread, number of strategy cells). Successes -> one LeRobotDataset per arm (for matched SFT).
Resumable: work goes to out_root/p<k>/ parts; a part is usable once it has retryvla/CLOSED.
"""

from __future__ import annotations

import json
import logging
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from retryvla import REPO_ROOT, tracking
from retryvla.config import cli_config, to_dict
from retryvla.data.recovery_writer import RecoveryWriter
from retryvla.envs.splits import assert_allowed, load_split, scenes_for
from retryvla.envs.vec import SimPool
from retryvla.eval.eval_recover import instance_jobs
from retryvla.eval.rollout import Job, run_jobs
from retryvla.policy import SmolVLAWrapper
from retryvla.search import arms as A
from retryvla.search.run_search import Proposals
from retryvla.search.scorers import Instance
from retryvla.search.wm_search import WMCriticRanker
from retryvla.utils import setup_logging
from retryvla.wm.collect import encode_record

log = logging.getLogger("compare_arms")


def select_jobs(scenes, cfg, mb) -> list[Job]:
    jobs = instance_jobs(scenes, cfg.recover, int(cfg.eval.seed), cfg.recover.instruction, part="train")
    excl = {int(k) for k in mb.exclude_trials}
    jobs = [j for j in jobs if int(j.meta["trial"]) not in excl]
    tf = list(mb.get("task_filter") or [])
    if tf:  # keep only scenes whose task folder contains one of these substrings
        jobs = [j for j in jobs if any(s in j.meta["task_folder"] for s in tf)]
    rng = np.random.default_rng(int(mb.select_seed))
    keep = sorted(rng.permutation(len(jobs))[: int(mb.n_select)])
    jobs = [jobs[i] for i in keep]
    return jobs[int(mb.shard):: int(mb.num_shards)]


def iid(meta: dict) -> str:
    return f"{meta['scene']}__{meta['gripper_init']}{meta['trial']}"


def precheck(pool, policy, jobs, skip: set[str]) -> tuple[list[Instance], list[dict]]:
    """One pi0 rollout per instance from the failure state; keep the ones pi0 fails (+ exact snapshot)."""
    todo = []
    for j in jobs:
        if iid(j.meta) in skip:
            continue
        jj = Job(**{**j.__dict__})
        jj.return_snapshot = True
        todo.append(jj)
    res = run_jobs(pool, policy, todo, desc="precheck")
    insts, rows = [], []
    s_by = {}
    for j in todo:
        s_by[iid(j.meta)] = j
    for r in res:
        i = iid(r.meta)
        rows.append({"instance_id": i, "pi0_success": bool(r.success), "task_folder": r.meta["task_folder"]})
        if not r.success:
            j = s_by[i]
            insts.append(Instance(i, r.meta["scene"], j.load_kwargs["bddl"], j.task, r.load_out["snapshot"],
                                  r.load_out["obs0"], meta=dict(r.meta)))
    return insts, rows


def _save_retry(obj, path, log, tries: int = 5) -> None:
    """/pfss occasionally fails writes (2026-10-21: all 4 shards at once). Retry, then skip the (optional) verif log."""
    for k in range(tries):
        try:
            tmp = Path(str(path) + ".tmp")
            torch.save(obj, tmp)
            tmp.replace(path)
            return
        except (RuntimeError, OSError) as e:
            log.warning("save %s failed (%s), retry %d/%d", path.name, str(e)[:80], k + 1, tries)
            time.sleep(30 * (k + 1))
    log.error("giving up on verif log %s", path.name)


def closed_done(root: Path) -> set[str]:
    done = set()
    for p in sorted(root.glob("p*")):
        if (p / "CLOSED").exists() and (p / "done.jsonl").exists():
            done |= {json.loads(l)["instance_id"] for l in open(p / "done.jsonl")}
    return done


def main() -> None:
    setup_logging()
    cfg = cli_config()
    mb, sc = cfg.m5b, cfg.search
    split = load_split(REPO_ROOT / cfg.split.file)
    scenes = scenes_for(split, "train", REPO_ROOT / cfg.recover.scene_root)
    assert_allowed([s.tag for s in scenes], split)
    cfg.tag = f"{cfg.tag}_s{mb.shard}of{mb.num_shards}"
    arms = list(mb.arms)
    H, NP, B = int(sc.horizon), int(mb.pool_size), int(mb.budget)
    cont, stride = int(sc.continuation_max_steps), int(cfg.wm_collect.stride)
    dev = torch.device(cfg.policy.device)
    wm_ck = (REPO_ROOT / mb.wm_link).read_text().strip()
    cr_ck = (REPO_ROOT / mb.critic_link).read_text().strip()
    root = REPO_ROOT / mb.out_root / f"s{mb.shard}"
    root.mkdir(parents=True, exist_ok=True)
    done = closed_done(root)
    part = root / f"p{len(list(root.glob('p*')))}"
    part.mkdir()
    vdir = REPO_ROOT / mb.verif_out
    vdir.mkdir(parents=True, exist_ok=True)
    jobs = select_jobs(scenes, cfg, mb)
    canon = {s.tag: s.language_canonical for s in scenes}
    goal_feats = None
    if any(a in ("wm_goal", "wm_goal_comp") for a in arms):
        goal_feats = torch.load(REPO_ROOT / mb.get("goal_feats", "data/wm/goal_feats.pt"), weights_only=False)
    with tracking.start(cfg) as run:
        run.summary.update({"wm_ckpt": wm_ck, "critic_ckpt": cr_ck, "part": str(part), "resumed_done": len(done)})
        policy = SmolVLAWrapper(cfg.policy.path, cfg.policy.device, cfg.policy.n_action_steps)
        ranker = WMCriticRanker(wm_ck, cr_ck, cfg.wm_collect.encoder, int(cfg.wm_collect.pool_grid), dev,
                                float(mb.lambda_unc), float(mb.beta_rough))
        ranker2 = None
        if any(a.endswith("_v2") for a in arms):  # second WM / critic (e.g. retrained on broad-action data)
            ranker2 = WMCriticRanker((REPO_ROOT / mb.wm_link2).read_text().strip(), (REPO_ROOT / mb.critic_link2).read_text().strip(),
                                     cfg.wm_collect.encoder, int(cfg.wm_collect.pool_grid), dev,
                                     float(mb.lambda_unc), float(mb.beta_rough))
        writers = {a: RecoveryWriter(part / a, overwrite=True) for a in arms}
        rows, t0 = [], time.time()
        with SimPool(int(cfg.eval.n_workers), int(cfg.env.resolution)) as pool:
            insts, pre = precheck(pool, policy, jobs, done)
            pd.DataFrame(pre).to_csv(run.path("precheck.csv"), index=False)
            log.info("shard %s: %d selected, %d already done, %d pi0 failures to search", mb.shard, len(jobs),
                     len(done), len(insts))
            proposals = Proposals(policy, sc, H, int(cfg.eval.seed) + 77 * int(mb.shard))
            for n, inst in enumerate(insts):
                seed = zlib.crc32(inst.instance_id.encode()) % (2**31)  # stable across processes
                rng = np.random.default_rng(seed)
                ti = time.time()
                pool_c, src = proposals(inst, NP)
                ref = pool_c[[s.startswith("vla_") for s in src]]
                z0, q0 = ranker.encode(inst.obs0)
                pool_s = ranker.score(z0, q0, inst.task, pool_c)
                picked, extra_steps = {}, {}
                goal = None
                if goal_feats is not None:
                    goal = ranker.norm.z(goal_feats[canon[inst.scene_tag]].to(dev).float())
                for a in arms:
                    extra_steps[a] = 0
                    ta = time.time()
                    if a == "random":
                        cand, idx, cs = (*A.arm_random(pool_c, pool_s, B, rng)[:2], pool_s)
                    elif a == "wm_diverse":
                        cand, idx, cs = A.arm_wm_diverse(ranker, z0, q0, inst.task, pool_c, pool_s, B, seed % 10007)
                    elif a == "wm_noise":
                        cand, idx, cs = A.arm_wm_noise(policy, ranker, inst.obs0, z0, q0, inst.task, H, B, seed % 10007,
                                                       n=int(mb.noise_n), iters=int(mb.noise_iters))
                    elif a == "wm_qd":
                        cand, idx, cs = A.arm_wm_qd(ranker, z0, q0, inst.task, pool_c, pool_s, B, seed % 10007)
                    elif a in ("wm_goal", "wm_goal_comp", "wm_top"):
                        mode = {"wm_goal": "goal", "wm_goal_comp": "goal_comp", "wm_top": "comp"}[a]
                        cand, idx, cs = A.arm_wm_obj(ranker, z0, q0, inst.task, pool_c, B, seed % 10007, mode,
                                                     goal=goal, diverse=(a != "wm_top"))
                    elif a == "random_diverse":
                        cand, idx, cs = pool_c, A.arm_random_diverse(pool_c, B, rng), pool_s
                    elif a == "sim_prefix_critic":
                        # real dynamics instead of the WM: simulate every pool segment (no continuation), critic on the
                        # REAL end state, diverse top-B. All prefix transitions are charged to this arm.
                        pj = [Job("restore_snapshot", dict(bddl=inst.bddl, snap=inst.snapshot), task=inst.task, max_steps=H,
                                  prefix=c, record="all", record_cfg={"stride": H}) for c in pool_c]
                        pr = run_jobs(pool, policy, pj, desc="prefix")
                        zs, qs = [], []
                        for r in pr:
                            e = encode_record(ranker.enc, r.record, H, H)
                            k_ = int(e["valid"].sum()) - 1
                            zs.append(e["z"][k_]); qs.append(torch.as_tensor(e["q"][k_]))
                            extra_steps[a] += int(r.steps)
                            r.record = None
                        zr = ranker.norm.z(torch.stack(zs).to(dev).float())
                        qr = ranker.norm.q(torch.stack(qs).to(dev).float())
                        V = ranker.score_real(zr, qr, inst.task)
                        R = A.roughness(pool_c) if hasattr(A, "roughness") else 0.0
                        s = V - float(mb.beta_rough) * R
                        zf = torch.cat([zr.mean(1), qr], 1).cpu().numpy()
                        cs = {"value": V, "score": s, "rough": R, "zfeat": zf}
                        cand, idx = pool_c, A.diverse_ranking(s, zf, B, seed=seed % 10007)
                    elif a == "wm_no_prim":  # same as wm_diverse but the pool has NO primitives / perturbations
                        vla_only = pool_c[[s.startswith("vla_") for s in src]]
                        extra = policy.sample_chunks(inst.obs0, inst.task, NP - len(vla_only), noise_scale=1.25,
                                                     seed=seed % 100003)[:, :H]
                        pc = np.clip(np.concatenate([vla_only, extra]), -1, 1).astype(np.float32)
                        cand, idx, cs = A.arm_wm_diverse(ranker, z0, q0, inst.task, pc, ranker.score(z0, q0, inst.task, pc),
                                                         B, seed % 10007)
                    elif a in ("wm_plan", "wm_plan_v2"):
                        rk = ranker2 if a.endswith("_v2") else ranker
                        za, qa = rk.encode(inst.obs0) if a.endswith("_v2") else (z0, q0)
                        cand, idx, cs = A.arm_wm_plan(rk, za, qa, inst.task, H, B, seed % 10007,
                                                      iters=int(mb.get("plan_iters", 5)))
                    elif a == "wm_two_stage":
                        cand, idx, cs = A.arm_wm_two_stage(ranker, z0, q0, inst.task, pool_c, pool_s, B, seed % 10007)
                    else:
                        raise ValueError(a)
                    picked[a] = (cand[idx], {k: v[idx] for k, v in cs.items() if k not in ("zfeat",) and np.ndim(v) == 1},
                                 len(cand), time.time() - ta)
                # one verification batch for all arms; slot j gets the same continuation seed in every arm
                vj, owner = [], []
                for a in arms:
                    for j, c in enumerate(picked[a][0]):
                        vj.append(Job("restore_snapshot", dict(bddl=inst.bddl, snap=inst.snapshot), task=inst.task,
                                      max_steps=len(c) + cont, prefix=c, record="all", policy_seed=(seed + j) % (2**31 - 1)))
                        owner.append((a, j))
                res = run_jobs(pool, policy, vj, noise_scale=float(cfg.policy.noise_scale), desc="verify")
                by = {a: [] for a in arms}
                for (a, j), r in zip(owner, res):
                    by[a].append(r)
                # robustness: re-verify each success under `robust_extra` more continuation seeds; a success is kept
                # for the dataset only if it succeeds in >= robust_min of (1 + robust_extra) runs
                n_extra = int(mb.get("robust_extra", 0))
                robust = {a: [bool(r.success) for r in by[a]] for a in arms}
                if n_extra:
                    rj, rown = [], []
                    for a in arms:
                        for j, r in enumerate(by[a]):
                            if r.success:
                                for e in range(n_extra):
                                    rj.append(Job("restore_snapshot", dict(bddl=inst.bddl, snap=inst.snapshot), task=inst.task,
                                                  max_steps=len(picked[a][0][j]) + cont, prefix=picked[a][0][j], record="none",
                                                  policy_seed=(seed + j + 7777 * (e + 1)) % (2**31 - 1)))
                                    rown.append((a, j))
                    cnt = {}
                    for (a, j), r in zip(rown, run_jobs(pool, policy, rj, noise_scale=float(cfg.policy.noise_scale),
                                                        desc="robust") if rj else []):
                        cnt[(a, j)] = cnt.get((a, j), 0) + int(r.success)
                    need = int(mb.get("robust_min", 2))
                    robust = {a: [bool(r.success) and 1 + cnt.get((a, j), 0) >= need for j, r in enumerate(by[a])] for a in arms}
                # verification log (critic self-improvement data), all arms
                enc = [encode_record(ranker.enc, r.record, H, stride) for r in res]
                _save_retry({
                    "instance_id": inst.instance_id, "task": inst.task, "scene": inst.scene_tag,
                    "z": torch.stack([e["z"] for e in enc]), "q": torch.as_tensor(np.stack([e["q"] for e in enc])),
                    "valid": torch.as_tensor(np.stack([e["valid"] for e in enc])),
                    "actions": (torch.as_tensor(np.concatenate([picked[a][0] for a in arms]))
                                if len({picked[a][0].shape[1] for a in arms}) == 1
                                else [torch.as_tensor(x) for a in arms for x in picked[a][0]]),
                    "success": torch.as_tensor([r.success for r in res]),
                    "success_step": torch.as_tensor([r.success_step if r.success else -1 for r in res]),
                    "prefix_success": torch.as_tensor([r.prefix_success for r in res]),
                    "source": [a for a, _ in owner],
                    "pred_value": torch.as_tensor(np.concatenate([picked[a][1]["value"] for a in arms])),
                    "arm": "m5b_all",
                }, vdir / f"{inst.instance_id}.pt", log)
                line = []
                for a in arms:
                    cands, cs, n_scored, t_arm = picked[a]
                    succ = np.array([r.success for r in by[a]], bool)
                    w = writers[a]
                    w.add_instance(inst.instance_id, {"scene": inst.scene_tag, "task": inst.task, "arm": a})
                    rob = np.array(robust[a], bool)
                    order = [k for k in np.argsort(-cs["score"]) if rob[k]][: int(mb.max_trajs_per_instance)]
                    for k in order:
                        w.add_trajectory(by[a][k].record, inst.task, inst.instance_id,
                                         {"score": float(cs["score"][k]), "pred_value": float(cs["value"][k]),
                                          "arm": a, "scene": inst.scene_tag,
                                          "success_step": -1 if by[a][k].success_step is None else int(by[a][k].success_step)})
                    m_all, m_succ = A.set_metrics(cands, ref), A.set_metrics(cands[succ], ref)
                    rows.append({
                        "instance_id": inst.instance_id, "task_folder": inst.meta["task_folder"],
                        "gripper_init": inst.meta["gripper_init"], "arm": a, "budget": len(cands),
                        "hits": int(succ.sum()), "solved": bool(succ.any()), "n_written": len(order),
                        "robust_hits": int(rob.sum()), "robust_solved": bool(rob.any()),
                        "pred_mean": float(cs["value"].mean()),
                        "pred_succ": float(cs["value"][succ].mean()) if succ.any() else np.nan,
                        "pred_fail": float(cs["value"][~succ].mean()) if (~succ).any() else np.nan,
                        "n_scored": n_scored, "t_arm_s": t_arm,
                        "sim_steps_verify": int(sum(int(r.steps) for r in by[a])), "sim_steps_extra": int(extra_steps[a]),
                        "order_success": [bool(x) for x in succ],
                        **{f"all_{k}": v for k, v in m_all.items()}, **{f"succ_{k}": v for k, v in m_succ.items()},
                        "succ_cells_list": sorted(set(A.descriptor(cands[succ]).tolist())) if succ.any() else [],
                    })
                    line.append(f"{a} {int(succ.sum())}/{len(cands)} (robust {int(rob.sum())})")
                for r in res:
                    r.record = None
                with open(part / "done.jsonl", "a") as f:
                    f.write(json.dumps({"instance_id": inst.instance_id}) + "\n")
                d = pd.DataFrame(rows)
                d.to_json(run.path("per_instance.jsonl"), orient="records", lines=True)
                g = d.groupby("arm")
                run.log({**{f"m5b/solve_{a}": float(v) for a, v in g.solved.mean().items()},
                         **{f"m5b/hit_{a}": float(v) for a, v in (g.hits.sum() / g.budget.sum()).items()}}, step=n)
                log.info("[%d/%d] %s | %s | %.0fs (total %.0fs)", n + 1, len(insts), inst.instance_id[-38:],
                         " | ".join(line), time.time() - ti, time.time() - t0)
        for w in writers.values():
            w.close()
        (part / "CLOSED").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        d = pd.DataFrame(rows)
        if len(d):
            g = d.groupby("arm")
            run.summary.update({
                "n_instances": int(d.instance_id.nunique()),
                "solve": {a: float(v) for a, v in g.solved.mean().items()},
                "hit_rate": {a: float(v) for a, v in (g.hits.sum() / g.budget.sum()).items()},
                "n_trajectories": {a: int(v) for a, v in g.n_written.sum().items()},
            })
        run.summary["config"] = to_dict(cfg)


if __name__ == "__main__":
    main()
