"""M1a + M1b: recovery search on training failure instances, writing verified recoveries.

python -m retryvla.search.run_search configs/exp/m1_oracle_cem_spatial.yaml [key=value ...]

Per failure instance (scene x gripper_init, fixed protocol seed):
  0. restore with the official protocol, keep an exact snapshot + failure obs
  1. (optional) pi0 pre-check rollouts; skip instances pi0 already solves
  2. proposal pool: SmolVLA chunk samples (several noise scales) + bounded perturbations
     + recovery primitives
  3. score with the configured scorer (oracle_sim: sim rollout of segment + pi0 continuation)
  4. method=cem: refit Gaussian to elites, resample (n_iters); method=random: fresh proposals
  5. every simulator-verified success is written to the recovery LeRobotDataset; every branch
     (success or not) goes to branches.parquet for critic / preference learning later.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from retryvla import REPO_ROOT, tracking
from retryvla.config import cli_config, to_dict
from retryvla.data.recovery_writer import RecoveryWriter
from retryvla.envs.splits import assert_allowed, load_split, scenes_for, split_sha
from retryvla.envs.vec import SimPool
from retryvla.eval.eval_recover import instance_jobs
from retryvla.eval.rollout import run_jobs
from retryvla.policy import SmolVLAWrapper
from retryvla.search import primitives
from retryvla.search.cem import CEM, CEMConfig
from retryvla.search.scorers import Instance, make_scorer
from retryvla.utils import setup_logging

log = logging.getLogger("run_search")


class Proposals:
    """Initial / random proposal distribution at a failure observation."""

    def __init__(self, policy, sc, horizon: int, seed: int):
        self.policy, self.sc, self.h = policy, sc, horizon
        self.rng = np.random.RandomState(seed)
        self.calls = 0

    def __call__(self, inst: Instance, n_total: int | None = None, fresh_only: bool = False
                 ) -> tuple[np.ndarray, list[str]]:
        """fresh_only: drop the deterministic fixed primitives (random-search iterations >= 1 would
        just re-evaluate identical sequences); their slots go to random primitive variants."""
        ic = self.sc.init
        n_vla, n_pert = int(ic.n_vla), int(ic.n_perturb)
        if n_total is not None:  # random-search: rescale the pool to the requested budget
            base = n_vla + n_pert + (16 if ic.use_primitives else 0)
            n_vla, n_pert = max(1, round(n_vla * n_total / base)), max(0, round(n_pert * n_total / base))
        cands, src = [], []
        scales = list(ic.vla_noise_scales)
        per = int(np.ceil(n_vla / len(scales)))
        for s in scales:
            ch = self.policy.sample_chunks(inst.obs0, inst.task, per, noise_scale=float(s),
                                           seed=self.rng.randint(1 << 30))[:, : self.h]
            cands.append(ch)
            src += [f"vla_s{s}"] * len(ch)
        vla = np.concatenate(cands, 0)[:n_vla]
        cands, src = [vla], src[:n_vla]
        if n_pert:
            base = vla[self.rng.randint(len(vla), size=n_pert)]
            from retryvla.search.cem import smooth_noise

            z = smooth_noise(self.rng, n_pert, self.h, 7, 5) * float(ic.perturb_std)
            z[..., 6] = 0.0
            pert = np.clip(base + z, -1, 1)
            flip = self.rng.rand(n_pert) < 0.25  # occasionally flip gripper intent
            pert[flip, :, 6] *= -1
            cands.append(pert)
            src += ["perturb"] * n_pert
        if ic.use_primitives:
            n_rand_prim = 8 if n_total is None else max(0, (n_total or 0) // 8)
            if fresh_only:
                n_rand_prim += 8
            prim, names = primitives.library(self.h, self.rng, n_random=n_rand_prim, include_fixed=not fresh_only)
            cands.append(prim)
            src += [f"prim_{n}" for n in names]
        out = np.concatenate(cands, 0).astype(np.float32)
        if n_total is not None:
            out, src = out[:n_total], src[:n_total]
        self.calls += 1
        return out, src


def build_instances(pool, policy, scenes, cfg, part) -> tuple[list[Instance], list[dict]]:
    """Restore each failure instance once (returning snapshot + obs) and run pi0 pre-check."""
    trials = list(cfg.search_on.trials_per_scene)
    jobs = instance_jobs(scenes, cfg.recover, int(cfg.eval.seed), cfg.recover.instruction, part=part,
                         trials=trials)
    k = int(cfg.search_on.precheck_rollouts)
    pre_jobs = []
    for j in jobs:
        for r in range(max(1, k)):
            jj = type(j)(**{**j.__dict__})
            jj.meta = {**j.meta, "precheck": r}
            jj.policy_seed = j.policy_seed + 7919 * r
            jj.return_snapshot = r == 0
            jj.max_steps = j.max_steps if k > 0 else 0
            pre_jobs.append(jj)
    res = run_jobs(pool, policy if k > 0 else None, pre_jobs, desc="precheck")
    by_inst: dict[str, list] = {}
    for r in res:
        iid = f"{r.meta['scene']}__{r.meta['gripper_init']}{r.meta['trial']}"
        by_inst.setdefault(iid, []).append(r)
    insts, pre_rows = [], []
    s_by_tag = {s.tag: s for s in scenes}
    for iid, rs in by_inst.items():
        r0 = next(r for r in rs if r.meta["precheck"] == 0)
        m = r0.meta
        s = s_by_tag[m["scene"]]
        pi0_sr = float(np.mean([r.success for r in rs])) if k > 0 else 0.0
        pre_rows.append({"instance_id": iid, "scene": s.tag, "gripper_init": m["gripper_init"], "pi0_sr": pi0_sr})
        insts.append(Instance(iid, s.tag, s.bddl, s.instruction(cfg.recover.instruction),
                              r0.load_out["snapshot"], r0.load_out["obs0"],
                              meta={**m, "pi0_precheck_sr": pi0_sr}))
    return insts, pre_rows


def completed_instances(roots) -> set[str]:
    """Instance ids fully searched in previous (possibly crashed) runs.
    Uses done.jsonl when present; older runs without it: every instance in instances.jsonl
    except the last one (which may have been interrupted mid-write)."""
    done: set[str] = set()
    for root in roots:
        side = REPO_ROOT / root / "retryvla"
        if (side / "done.jsonl").exists():
            done |= {json.loads(l)["instance_id"] for l in open(side / "done.jsonl")}
        elif (side / "instances.jsonl").exists():
            ids = [json.loads(l)["instance_id"] for l in open(side / "instances.jsonl")]
            done |= set(ids[:-1])
    return done


def search_instance(inst, scorer, proposals, sc, rng_seed) -> tuple[list[dict], list[dict]]:
    """Returns (branches rows, verified successes [{record, score, ...}])."""
    h = int(sc.horizon)
    ccfg = CEMConfig(n_samples=int(sc.n_samples), n_elites=int(sc.n_elites), min_std=float(sc.cem.min_std),
                     init_std=float(sc.cem.init_std), smooth_kernel=int(sc.cem.smooth_kernel),
                     alpha=float(sc.cem.alpha))
    cem = CEM(h, 7, ccfg, seed=rng_seed)
    branches, wins = [], []
    elite_c, elite_s = None, None
    for it in range(int(sc.n_iters)):
        if it == 0:
            cands, src = proposals(inst, None if sc.method == "cem" else int(sc.n_samples))
        elif sc.method == "cem" and str(sc.cem.get("mode", "gaussian")) == "elite_local":
            cands = cem.sample_local(elite_c, elite_s, it, temperature=float(sc.cem.get("temperature", 0.2)),
                                     decay=float(sc.cem.get("decay", 0.6)))
            src = ["cem_local"] * len(cands)
        elif sc.method == "cem":
            cands, src = cem.sample(), ["cem"] * int(sc.n_samples)
        else:
            cands, src = proposals(inst, int(sc.n_samples), fresh_only=True)
        out = scorer.score(inst, cands, it)
        for k in range(len(cands)):
            branches.append({
                "instance_id": inst.instance_id, "iteration": it, "k": k, "source": src[k],
                "score": float(out.scores[k]), "success": bool(out.success[k]),
                "success_step": int(out.success_step[k]), "progress": float(out.progress[k]),
                "prefix_success": bool(out.info.get("prefix_success", np.zeros(len(cands)))[k]),
                "actions": cands[k].astype(np.float32).tobytes(), "horizon": h,
            })
            if out.success[k] and out.records[k] is not None:
                wins.append({"record": out.records[k], "score": float(out.scores[k]), "iteration": it,
                             "source": src[k], "success_step": int(out.success_step[k]),
                             "prefix_success": bool(out.info.get("prefix_success", np.zeros(len(cands)))[k])})
        log.info("  %s it%d: %d/%d success, best %.3f", inst.instance_id, it, int(out.success.sum()), len(cands),
                 float(out.scores.max()))
        if len(wins) >= int(sc.stop_after_successes):
            break
        if sc.method == "cem":
            all_c = cands if elite_c is None else np.concatenate([elite_c, cands])
            all_s = out.scores if elite_s is None else np.concatenate([elite_s, out.scores])
            elite_c, idx = CEM.select_elites(all_c, all_s, ccfg.n_elites)
            elite_s = all_s[idx]
            cem.refit(elite_c)
    return branches, wins


def main() -> None:
    setup_logging()
    cfg = cli_config()
    sc = cfg.search
    split_path = REPO_ROOT / cfg.split.file
    split = load_split(split_path)
    part = cfg.search_on.split
    if part == "test":
        raise SystemExit("refusing to search on the test split")
    scenes = scenes_for(split, part, REPO_ROOT / cfg.recover.scene_root)
    assert_allowed([s.tag for s in scenes], split)
    if int(sc.get("max_instances", 0)) > 0:
        n_sc = int(np.ceil(int(sc.max_instances) / len(cfg.search_on.trials_per_scene)))
        scenes = scenes[:: max(1, len(scenes) // n_sc)][:n_sc]
    n_sh, sh = int(cfg.search_on.get("num_shards", 1)), int(cfg.search_on.get("shard", 0))
    only_ids = None
    if cfg.search_on.get("instance_ids_file"):
        only_ids = {l.strip() for l in open(REPO_ROOT / cfg.search_on.instance_ids_file) if l.strip()}
        keep_scenes = {i.rsplit("__", 1)[0] for i in only_ids}
        scenes = [s for s in scenes if s.tag in keep_scenes]
        log.info("instance list %s: %d instances over %d scenes", cfg.search_on.instance_ids_file, len(only_ids),
                 len(scenes))
    if n_sh > 1:  # scene-level sharding across GPUs; each shard writes its own dataset
        scenes = scenes[sh::n_sh]
        cfg.tag = f"{cfg.tag}_s{sh}of{n_sh}"
        log.info("shard %d/%d: %d scenes", sh, n_sh, len(scenes))

    with tracking.start(cfg) as run:
        run.summary["split_sha"] = split_sha(split_path)
        policy = SmolVLAWrapper(cfg.policy.path, cfg.policy.device, cfg.policy.n_action_steps)
        out_root = REPO_ROOT / cfg.dataset.out_root
        writer = RecoveryWriter(out_root, overwrite=bool(cfg.dataset.get("overwrite", True)))
        (out_root / "retryvla" / "config.json").write_text(json.dumps(to_dict(cfg), indent=1, default=str))
        t0 = time.time()
        with SimPool(int(cfg.eval.n_workers), int(cfg.env.resolution)) as pool:
            insts, pre_rows = build_instances(pool, policy, scenes, cfg, part)
            pd.DataFrame(pre_rows).to_csv(run.path("precheck.csv"), index=False)
            if cfg.search_on.only_pi0_failures:
                todo = [i for i in insts if i.meta["pi0_precheck_sr"] < 1.0]
            else:
                todo = insts
            done_ids = completed_instances(cfg.search_on.get("resume_from") or [])
            if only_ids is not None:
                todo = [i for i in todo if i.instance_id in only_ids]
            if done_ids:
                todo = [i for i in todo if i.instance_id not in done_ids]
                log.info("resume: skipping %d instances already searched", len(done_ids))
            log.info("%d instances, %d after pi0 pre-check filter", len(insts), len(todo))
            n_sub, sub = int(cfg.search_on.get("num_sub_shards", 1)), int(cfg.search_on.get("sub_shard", 0))
            if n_sub > 1:  # split the remaining instances of this shard over more GPUs
                todo = todo[sub::n_sub]
                log.info("sub-shard %d/%d: %d instances", sub, n_sub, len(todo))
            scorer = make_scorer(sc.scorer, pool=pool, policy=policy, objective=sc.objective,
                                 horizon=int(sc.horizon), continuation_max_steps=int(sc.continuation_max_steps),
                                 success_bonus=float(sc.score.success_bonus), time_weight=float(sc.score.time_weight),
                                 progress_weight=float(sc.score.progress_weight),
                                 smooth_cost=float(sc.score.smooth_cost), noise_scale=float(cfg.policy.noise_scale))
            proposals = Proposals(policy, sc, int(sc.horizon), int(cfg.eval.seed))
            all_branches, n_solved, n_traj = [], 0, 0
            per_inst = []
            for n, inst in enumerate(todo):
                writer.add_instance(inst.instance_id, {"scene": inst.scene_tag, "task": inst.task,
                                                       **{k: v for k, v in inst.meta.items() if k != "obs0"}})
                branches, wins = search_instance(inst, scorer, proposals, sc, int(cfg.eval.seed) + n)
                all_branches += branches
                wins = sorted(wins, key=lambda w: -w["score"])[: int(cfg.dataset.max_trajs_per_instance)]
                for w in wins:
                    writer.add_trajectory(w["record"], inst.task, inst.instance_id,
                                          {k: v for k, v in w.items() if k != "record"} | {"scene": inst.scene_tag},
                                          keep_continuation=bool(cfg.dataset.keep_continuation))
                n_traj += len(wins)
                n_solved += bool(wins)
                with open(out_root / "retryvla" / "done.jsonl", "a") as f:  # resume marker
                    f.write(json.dumps({"instance_id": inst.instance_id, "n_written": len(wins)}) + "\n")
                per_inst.append({"instance_id": inst.instance_id, "n_branches": len(branches),
                                 "n_success": int(sum(b["success"] for b in branches)), "n_written": len(wins),
                                 "pi0_precheck_sr": inst.meta["pi0_precheck_sr"]})
                run.log({"search/solved_frac": n_solved / (n + 1), "search/n_traj": n_traj,
                         "search/sim_episodes": scorer.sim_episodes}, step=n)
                log.info("[%d/%d] %s -> %d wins | solved %d/%d | %d traj | %.0fs", n + 1, len(todo),
                         inst.instance_id, len(wins), n_solved, n + 1, n_traj, time.time() - t0)
                if (n + 1) % 5 == 0 or n + 1 == len(todo):
                    pd.DataFrame(all_branches).to_parquet(out_root / "retryvla" / "branches.parquet")
                    pd.DataFrame(per_inst).to_csv(run.path("per_instance.csv"), index=False)
        writer.close()
        pd.DataFrame(all_branches).to_parquet(out_root / "retryvla" / "branches.parquet")
        n_br = len(all_branches)
        res = {
            "n_instances": len(insts), "n_searched": len(todo), "n_scenes_solved": n_solved,
            "verify_rate": n_solved / max(1, len(todo)), "n_trajectories": n_traj,
            "sim_episodes": scorer.sim_episodes, "sim_steps": scorer.sim_steps,
            "branch_success_rate": float(np.mean([b["success"] for b in all_branches])) if n_br else 0.0,
            "pi0_precheck_sr": float(np.mean([r["pi0_sr"] for r in pre_rows])) if pre_rows else float("nan"),
            "dataset": str(out_root), "wall_h": (time.time() - t0) / 3600,
            "by_source_success": pd.DataFrame(all_branches).groupby("source")["success"].mean().to_dict() if n_br else {},
        }
        run.summary.update(res)
        log.info("search done: %s", {k: v for k, v in res.items() if k != "by_source_success"})


if __name__ == "__main__":
    main()
