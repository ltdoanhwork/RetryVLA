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
from retryvla.envs.vec import SimPool, WorkerDied
from retryvla.eval.eval_recover import instance_jobs
from retryvla.eval.rollout import Job, run_jobs
from retryvla.policy import SmolVLAWrapper
from retryvla.search import arms as A
from retryvla.search.run_search import Proposals
from retryvla.search.scorers import Instance, ensemble_values
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


def load_known_precheck(globs) -> dict[str, bool]:
    """instance_id -> pi0_success from earlier runs' precheck.csv (same instances, same protocol seeds)."""
    import glob as _g

    known = {}
    for pat in list(globs or []):
        for f in sorted(_g.glob(str(REPO_ROOT / pat))):
            for r in pd.read_csv(f).itertuples():
                known[r.instance_id] = bool(r.pi0_success)
    return known


def precheck(pool, policy, jobs, skip: set[str], run=None, known: dict[str, bool] | None = None
             ) -> tuple[list[Instance], list[dict]]:
    """One pi0 rollout per instance from the failure state; keep the ones pi0 fails (+ exact snapshot).
    known (optional, m5b.precheck_from): instance_id -> pi0_success from an earlier precheck of the SAME instance;
    those are only restored (0 policy steps, exact snapshot) instead of re-rolled; known successes are dropped."""
    todo = []
    known = known or {}
    for j in jobs:
        if iid(j.meta) in skip or known.get(iid(j.meta)) is True:
            continue
        jj = Job(**{**j.__dict__})
        jj.return_snapshot = True
        if iid(j.meta) in known:
            jj.max_steps = 0
        todo.append(jj)
    res = (run or (lambda *a, **k: run_jobs(pool, *a, **k)))(policy, todo, desc="precheck")
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
        if (p / "skipped.jsonl").exists():  # deterministic sim-worker crashes: never retried on resume
            done |= {json.loads(l)["instance_id"] for l in open(p / "skipped.jsonl")}
    return done


def make_safe_run(pool: SimPool, retries: int, oom_retries: int = 0, oom_wait_s: float = 120.0):
    """run_jobs that survives sim-worker deaths (MuJoCo segfault in env rebuild, 2026-10-23): on WorkerDied
    respawn the pool and re-run the WHOLE batch (same jobs + seeds -> same paired results), up to `retries`
    times, then re-raise. retries=0 = old behaviour.
    oom_retries > 0 (M11 incident 2026-10-23 06:55): a CUDA OOM in the main process's batched policy call (shared
    80 GB GPU filled by other lanes) is transient, not instance-specific -> free the cache, wait oom_wait_s * k,
    respawn the pool (workers may be mid-batch, pipes must be clean) and re-run the whole batch (deterministic).
    oom_retries=0 = old behaviour (OOM propagates)."""
    def run(*a, **k):
        t_w = t_o = 0
        while True:
            try:
                return run_jobs(pool, *a, **k)
            except WorkerDied as e:
                if t_w == retries:
                    raise
                t_w += 1
                log.warning("%s; respawning sim pool and re-running batch '%s' (%d/%d)", e, k.get("desc", "?"),
                            t_w, retries)
                pool.respawn()
            except torch.OutOfMemoryError as e:
                if t_o == oom_retries:
                    raise
                t_o += 1
                torch.cuda.empty_cache()
                log.warning("CUDA OOM in batch '%s' (%s); waiting %.0fs, respawning sim pool, re-running (%d/%d)",
                            k.get("desc", "?"), str(e)[:120], oom_wait_s * t_o, t_o, oom_retries)
                time.sleep(oom_wait_s * t_o)
                pool.respawn()
    return run


def main() -> None:
    setup_logging()
    cfg = cli_config()
    mb, sc = cfg.m5b, cfg.search
    split = load_split(REPO_ROOT / cfg.split.file)
    scenes = scenes_for(split, "train", REPO_ROOT / cfg.recover.scene_root)
    assert_allowed([s.tag for s in scenes], split)
    cfg.tag = f"{cfg.tag}_s{mb.shard}of{mb.num_shards}"
    if int(mb.get("num_sub_shards", 1)) > 1:
        cfg.tag += f"_sub{int(mb.sub_shard)}of{int(mb.num_sub_shards)}"
    arms = list(mb.arms)
    H, NP, B = int(sc.horizon), int(mb.pool_size), int(mb.budget)
    cont, stride = int(sc.continuation_max_steps), int(cfg.wm_collect.stride)
    dev = torch.device(cfg.policy.device)
    wm_ck = (REPO_ROOT / mb.wm_link).read_text().strip()
    cr_ck = (REPO_ROOT / mb.critic_link).read_text().strip()
    root = REPO_ROOT / mb.out_root / f"s{mb.shard}"
    if int(mb.get("num_sub_shards", 1)) > 1:  # sub-shards of one shard run in parallel -> own part dirs
        root = root / f"sub{int(mb.sub_shard)}"
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
        if str(mb.get("ranker", "dino")) == "vjepa2_ac":  # foundation-WM test: V-JEPA 2-AC + C^pi on its latents
            from retryvla.wm.vjepa2_ac import VJRanker

            wm_ck = (REPO_ROOT / mb.vj_wm_link).read_text().strip()
            cr_ck = (REPO_ROOT / mb.vj_critic_link).read_text().strip()
            run.summary.update({"wm_ckpt": wm_ck, "critic_ckpt": cr_ck, "ranker": "vjepa2_ac"})
            ranker = VJRanker(wm_ck, cr_ck, dev, float(mb.lambda_unc), float(mb.beta_rough))
        else:
            ranker = WMCriticRanker(wm_ck, cr_ck, cfg.wm_collect.encoder, int(cfg.wm_collect.pool_grid), dev,
                                    float(mb.lambda_unc), float(mb.beta_rough))
        if mb.get("boot_heads") and not isinstance(ranker, WMCriticRanker):
            raise ValueError("boot_heads are DINO-WM heads; set m5b.boot_heads null for ranker vjepa2_ac")
        if mb.get("boot_heads"):  # RQ3: real ensemble (bootstrap heads on the frozen trunk; WM's own heads identical)
            from retryvla.wm.boot_heads import load_boot

            ranker.boot = load_boot(REPO_ROOT / mb.boot_heads, dev)
            ranker.mc_dropout = float(mb.get("mc_dropout", 0.0))
            run.summary.update({"boot_heads": str(mb.boot_heads), "mc_dropout": ranker.mc_dropout})
        elif not isinstance(ranker, WMCriticRanker):
            pass  # VJRanker.ensemble_values: U_ens = critic MC-dropout only (single predictor)
        elif any(A.unc_arm_lambda(a) is not None for a in arms) or mb.get("log_uens", False):
            raise ValueError("U_ens arms need m5b.boot_heads (the WM's own ensemble heads are identical -> U == 0)")
        ranker2 = None
        rk_vj = None
        if any(A.xwm_arm_lambda(a) is not None or a == "vj_unc_l0" for a in arms):  # cross-WM: WM v1 x V-JEPA 2-AC
            from retryvla.wm.vjepa2_ac import VJRanker

            vw = (REPO_ROOT / mb.vj_wm_link).read_text().strip()
            vc = (REPO_ROOT / mb.vj_critic_link).read_text().strip()
            rk_vj = VJRanker(vw, vc, dev, float(mb.lambda_unc), float(mb.beta_rough))
            run.summary.update({"vj_wm_ckpt": vw, "vj_critic_ckpt": vc})
        if any(a.endswith("_v2") for a in arms):  # second WM / critic (e.g. retrained on broad-action data)
            ranker2 = WMCriticRanker((REPO_ROOT / mb.wm_link2).read_text().strip(), (REPO_ROOT / mb.critic_link2).read_text().strip(),
                                     cfg.wm_collect.encoder, int(cfg.wm_collect.pool_grid), dev,
                                     float(mb.lambda_unc), float(mb.beta_rough))
        value_rk = {}  # P2: policy-agnostic critics on the SAME WM (optional keys, only loaded when the arm is used)
        for a_, key in A.VALUE_ARMS.items():
            if a_ in arms:
                ck_ = (REPO_ROOT / mb[key]).read_text().strip()
                value_rk[a_] = A.value_ranker_from(ranker, ck_)
                run.summary[f"{a_}_critic_ckpt"] = ck_
        writers = {a: RecoveryWriter(part / a, overwrite=True, write_retries=int(mb.get("write_retries", 0)))
                   for a in arms}
        rows, t0 = [], time.time()
        with SimPool(int(cfg.eval.n_workers), int(cfg.env.resolution)) as pool:
            wretries = int(mb.get("worker_retries", 0))  # 0 = old behaviour (a dead sim worker kills the run)
            srun = make_safe_run(pool, wretries, int(mb.get("oom_retries", 0)), float(mb.get("oom_wait_s", 120)))
            # paired_rng (opt-in): precheck ALL selected instances and draw every instance's proposal pool in order,
            # skipping already-done / other-sub-shard instances AFTER drawing -> pools identical to an un-split,
            # un-resumed run of this shard (the Proposals RNG is sequential). done = done.jsonl of every part.
            paired = bool(mb.get("paired_rng", False))
            nsub, sub = int(mb.get("num_sub_shards", 1)), int(mb.get("sub_shard", 0))
            if paired:
                for p_ in root.glob("p*/done.jsonl"):
                    done |= {json.loads(x)["instance_id"] for x in p_.read_text().splitlines() if x.strip()}
            insts, pre = precheck(pool, policy, jobs, set() if paired else done, run=srun,
                                  known=load_known_precheck(mb.get("precheck_from") or []))
            pd.DataFrame(pre).to_csv(run.path("precheck.csv"), index=False)
            log.info("shard %s: %d selected, %d already done, %d pi0 failures to search", mb.shard, len(jobs),
                     len(done), len(insts))
            proposals = Proposals(policy, sc, H, int(cfg.eval.seed) + 77 * int(mb.shard))
            for n, inst in enumerate(insts):
                try:
                    seed = zlib.crc32(inst.instance_id.encode()) % (2**31)  # stable across processes
                    rng = np.random.default_rng(seed)
                    ti = time.time()
                    pool_c, src = proposals(inst, NP)
                    if paired and (n % nsub != sub or inst.instance_id in done):
                        continue
                    ref = pool_c[[s.startswith("vla_") for s in src]]
                    z0, q0 = ranker.encode(inst.obs0)
                    pool_s = ranker.score(z0, q0, inst.task, pool_c)
                    picked, extra_steps = {}, {}
                    pool_ens = None  # RQ3: ensemble values of the shared pool, computed once per instance
                    pool_x = None    # cross-WM values of the shared pool (WM v1 + V-JEPA), computed once
                    zq_vj = rk_vj.encode(inst.obs0) if rk_vj is not None else None
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
                            pr = srun(policy, pj, desc="prefix")
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
                        elif a in A.VALUE_ARMS:
                            cand, idx, cs = A.arm_wm_value(value_rk[a], z0, q0, inst.task, pool_c, B, seed % 10007)
                        elif A.xwm_arm_lambda(a) is not None:  # cross-WM: J = mean(v1, v2) - lam |v1 - v2| - mu R
                            if pool_x is None:
                                pool_x = A.xwm_values(ranker, (z0, q0), rk_vj, zq_vj, inst.task, pool_c)
                            cand, idx, cs = A.arm_xwm(ranker, (z0, q0), rk_vj, zq_vj, inst.task, pool_c, B, seed % 10007,
                                                      A.xwm_arm_lambda(a), float(mb.get("mu_cost", mb.beta_rough)),
                                                      pool_cs=pool_x)
                        elif a == "vj_unc_l0":  # V-JEPA alone, same pipeline as wm_unc_l0 (= the vj_sweep arm)
                            cand, idx, cs = A.arm_wm_unc(rk_vj, *zq_vj, inst.task, pool_c, B, seed % 10007, 0.0,
                                                         float(mb.get("mu_cost", mb.beta_rough)))
                        elif A.unc_arm_lambda(a) is not None:  # RQ3: J = C^pi - lambda U_ens - mu R
                            if pool_ens is None:
                                pool_ens = ensemble_values(ranker, z0, q0, inst.task, pool_c)
                            lam_, mu_ = A.unc_arm_lambda(a), float(mb.get("mu_cost", mb.beta_rough))
                            if a.startswith("wm_bud_"):
                                cand, idx, cs = A.arm_wm_budget(ranker, z0, q0, inst.task, pool_c, seed % 10007, lam_, mu_,
                                                                budgets=tuple(int(x) for x in mb.get("curve_budgets", [8, 32, 128, 512])),
                                                                k=int(mb.get("curve_k", 4)), pool_cs=pool_ens)
                            else:
                                cand, idx, cs = A.arm_wm_unc(ranker, z0, q0, inst.task, pool_c, B, seed % 10007, lam_, mu_,
                                                             pool_cs=pool_ens)
                        else:
                            raise ValueError(a)
                        picked[a] = (cand[idx], {k: v[idx] for k, v in cs.items() if k not in ("zfeat",) and np.ndim(v) == 1},
                                     len(cand), time.time() - ta)
                        if a in value_rk:  # C^pi's prediction for the picks of a policy-agnostic arm (analysis only)
                            picked[a][1]["value_cpi"] = ranker.score(z0, q0, inst.task, cand[idx])["value"]
                        if mb.get("log_uens", False) and "u_ens" not in picked[a][1]:  # RQ3: U_ens of every arm's picks
                            ev = ensemble_values(ranker, z0, q0, inst.task, cand[idx])
                            picked[a][1]["u_ens"], picked[a][1]["v_ens"] = ev["u_ens"], ev["value"]
                        if rk_vj is not None and "v1" not in picked[a][1]:  # cross-WM: both verifiers on every pick
                            xv = A.xwm_values(ranker, (z0, q0), rk_vj, zq_vj, inst.task, cand[idx])
                            picked[a][1]["v1"], picked[a][1]["v2"] = xv["v1"], xv["v2"]
                    # one verification batch for all arms; slot j gets the same continuation seed in every arm
                    vj, owner = [], []
                    for a in arms:
                        for j, c in enumerate(picked[a][0]):
                            vj.append(Job("restore_snapshot", dict(bddl=inst.bddl, snap=inst.snapshot), task=inst.task,
                                          max_steps=len(c) + cont, prefix=c, record="all", policy_seed=(seed + j) % (2**31 - 1)))
                            owner.append((a, j))
                    res = srun(policy, vj, noise_scale=float(cfg.policy.noise_scale), desc="verify")
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
                        for (a, j), r in zip(rown, srun(policy, rj, noise_scale=float(cfg.policy.noise_scale),
                                                            desc="robust") if rj else []):
                            cnt[(a, j)] = cnt.get((a, j), 0) + int(r.success)
                        need = int(mb.get("robust_min", 2))
                        robust = {a: [bool(r.success) and 1 + cnt.get((a, j), 0) >= need for j, r in enumerate(by[a])] for a in arms}
                    # verification log (critic self-improvement data), all arms
                    slim = bool(mb.get("verif_slim", False))  # RQ3 runs: no latents in the verif log (/pfss quota)
                    enc = [] if slim else [encode_record(ranker.enc, r.record, H, stride) for r in res]
                    _save_retry({
                        "instance_id": inst.instance_id, "task": inst.task, "scene": inst.scene_tag,
                        **({"slimmed": True, "z0": z0.half().cpu(), "q0": q0.cpu()} if slim else {
                            "z": torch.stack([e["z"] for e in enc]), "q": torch.as_tensor(np.stack([e["q"] for e in enc])),
                            "valid": torch.as_tensor(np.stack([e["valid"] for e in enc]))}),
                        "actions": (torch.as_tensor(np.concatenate([picked[a][0] for a in arms]))
                                    if len({picked[a][0].shape[1] for a in arms}) == 1
                                    else [torch.as_tensor(x) for a in arms for x in picked[a][0]]),
                        "success": torch.as_tensor([r.success for r in res]),
                        "success_step": torch.as_tensor([r.success_step if r.success else -1 for r in res]),
                        "prefix_success": torch.as_tensor([r.prefix_success for r in res]),
                        "source": [a for a, _ in owner],
                        "pred_value": torch.as_tensor(np.concatenate([picked[a][1]["value"] for a in arms])),
                        **({"pred_uens": torch.as_tensor(np.concatenate([
                                picked[a][1].get("u_ens", np.full(len(picked[a][0]), np.nan)) for a in arms])),
                            "pick_budget": torch.as_tensor(np.concatenate([
                                picked[a][1].get("budget", np.full(len(picked[a][0]), -1)) for a in arms]))}
                           if any("u_ens" in picked[a][1] for a in arms) else {}),
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
                            **({"pred_cpi_mean": float(cs["value_cpi"].mean()),
                                "pred_cpi_succ": float(cs["value_cpi"][succ].mean()) if succ.any() else np.nan,
                                "pred_cpi_fail": float(cs["value_cpi"][~succ].mean()) if (~succ).any() else np.nan}
                               if "value_cpi" in cs else {}),
                            "n_scored": n_scored, "t_arm_s": t_arm,
                            "sim_steps_verify": int(sum(int(r.steps) for r in by[a])), "sim_steps_extra": int(extra_steps[a]),
                            "order_success": [bool(x) for x in succ],
                            **({"order_pred": [float(x) for x in cs["value"]],
                                "order_uens": [float(x) for x in cs["u_ens"]],
                                "order_budget": [int(x) for x in cs.get("budget", np.full(len(succ), -1))]}
                               if "u_ens" in cs else {}),
                            **({"order_pred": [float(x) for x in cs["value"]],
                                "order_v1": [float(x) for x in cs["v1"]], "order_v2": [float(x) for x in cs["v2"]]}
                               if "v1" in cs else {}),
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
                except WorkerDied as e:  # same instance kept killing a sim worker after `worker_retries` respawns
                    if not wretries:
                        raise
                    log.error("[%d/%d] SKIP %s: sim worker crashed %d times (%s)", n + 1, len(insts), inst.instance_id,
                              wretries + 1, e)
                    pool.respawn()
                    with open(part / "skipped.jsonl", "a") as f:
                        f.write(json.dumps({"instance_id": inst.instance_id, "reason": f"sim worker crash x{wretries + 1}: {e}",
                                            "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
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
