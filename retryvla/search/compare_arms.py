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
    with tracking.start(cfg) as run:
        run.summary.update({"wm_ckpt": wm_ck, "critic_ckpt": cr_ck, "part": str(part), "resumed_done": len(done)})
        policy = SmolVLAWrapper(cfg.policy.path, cfg.policy.device, cfg.policy.n_action_steps)
        ranker = WMCriticRanker(wm_ck, cr_ck, cfg.wm_collect.encoder, int(cfg.wm_collect.pool_grid), dev,
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
                picked = {}
                for a in arms:
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
                    else:
                        raise ValueError(a)
                    picked[a] = (cand[idx], {k: v[idx] for k, v in cs.items() if k != "zfeat"}, len(cand),
                                 time.time() - ta)
                # one verification batch for all arms; slot j gets the same continuation seed in every arm
                vj, owner = [], []
                for a in arms:
                    for j, c in enumerate(picked[a][0]):
                        vj.append(Job("restore_snapshot", dict(bddl=inst.bddl, snap=inst.snapshot), task=inst.task,
                                      max_steps=H + cont, prefix=c, record="all", policy_seed=(seed + j) % (2**31 - 1)))
                        owner.append((a, j))
                res = run_jobs(pool, policy, vj, noise_scale=float(cfg.policy.noise_scale), desc="verify")
                by = {a: [] for a in arms}
                for (a, j), r in zip(owner, res):
                    by[a].append(r)
                # verification log (critic self-improvement data), all arms
                enc = [encode_record(ranker.enc, r.record, H, stride) for r in res]
                torch.save({
                    "instance_id": inst.instance_id, "task": inst.task, "scene": inst.scene_tag,
                    "z": torch.stack([e["z"] for e in enc]), "q": torch.as_tensor(np.stack([e["q"] for e in enc])),
                    "valid": torch.as_tensor(np.stack([e["valid"] for e in enc])),
                    "actions": torch.as_tensor(np.concatenate([picked[a][0] for a in arms])),
                    "success": torch.as_tensor([r.success for r in res]),
                    "success_step": torch.as_tensor([r.success_step if r.success else -1 for r in res]),
                    "prefix_success": torch.as_tensor([r.prefix_success for r in res]),
                    "source": [a for a, _ in owner],
                    "pred_value": torch.as_tensor(np.concatenate([picked[a][1]["value"] for a in arms])),
                    "arm": "m5b_all",
                }, vdir / f"{inst.instance_id}.pt")
                line = []
                for a in arms:
                    cands, cs, n_scored, t_arm = picked[a]
                    succ = np.array([r.success for r in by[a]], bool)
                    w = writers[a]
                    w.add_instance(inst.instance_id, {"scene": inst.scene_tag, "task": inst.task, "arm": a})
                    order = [k for k in np.argsort(-cs["score"]) if succ[k]][: int(mb.max_trajs_per_instance)]
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
                        "pred_mean": float(cs["value"].mean()),
                        "pred_succ": float(cs["value"][succ].mean()) if succ.any() else np.nan,
                        "pred_fail": float(cs["value"][~succ].mean()) if (~succ).any() else np.nan,
                        "n_scored": n_scored, "t_arm_s": t_arm,
                        **{f"all_{k}": v for k, v in m_all.items()}, **{f"succ_{k}": v for k, v in m_succ.items()},
                        "succ_cells_list": sorted(set(A.descriptor(cands[succ]).tolist())) if succ.any() else [],
                    })
                    line.append(f"{a} {int(succ.sum())}/{len(cands)}")
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
