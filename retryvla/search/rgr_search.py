"""RGR search comparison: single-bridge vs recursive (chained) recovery, with / without the geometry verifier.

python -m retryvla.search.rgr_search configs/exp/rgr_sweep.yaml [m5b.shard=i] [rgr.arms=[...]]

Same instances / pools / continuation seeds as compare_arms (M5b train-split pi0 failures, precheck restored,
paired_rng pools), TRAIN split only. Arms (rgr.arms):
  single          pool + 2 elite-local latent-CEM rounds, J = C^pi - mu R, diverse top picks (= wm_unc_l0 / M5b)
  single_geo      same, J = C^pi - lam E_geom(head) - 1[invalid] - mu R
  rec             recursive: level-1 bridges scored by Q(b1) = max(J1, gamma max_b2 J2(b1, b2)) (imagined 2-step
                  lookahead); chosen b1 are EXECUTED in sim, the real state is re-encoded; forward correction:
                  if latent discrepancy > disc_thr the continuation is re-planned from the real state, else the
                  imagined plan is kept. depth K: repeat at the next level while C^pi(s_k) < tau.
  rec_geo         rec + geometry head (E_geom on imagined states, intermediate states must be geom-valid; head
                  on the observed state after each executed segment: invalid -> chain dropped / re-planned)
  rec_geo_oracle  rec_geo + privileged sim geometry of every executed segment (ORACLE upper bound)
Budgets per arm and instance: imagination <= rgr.imag_budget WM rollouts; sim <= budget x (H + cont) env steps
(worst case), counting segment executions; every final chain is verified from the ORIGINAL failure snapshot
(prefix = concatenated bridges, then pi), i.e. bit-exact re-execution, before it is written.
"""

from __future__ import annotations

import json
import logging
import time
import zlib

import numpy as np
import pandas as pd
import torch

from retryvla import REPO_ROOT, tracking
from retryvla.config import cli_config, to_dict
from retryvla.data.recovery_writer import RecoveryWriter
from retryvla.envs.splits import assert_allowed, load_split, scenes_for
from retryvla.envs.vec import SimPool
from retryvla.eval.rollout import Job
from retryvla.geom.exec import run_bridges
from retryvla.geom.features import geom_energy
from retryvla.policy import SmolVLAWrapper
from retryvla.search import primitives
from retryvla.search import rgr_core as R
from retryvla.search.compare_arms import closed_done, iid, load_known_precheck, make_safe_run, precheck, select_jobs
from retryvla.search.run_search import Proposals
from retryvla.search.wm_search import WMCriticRanker
from retryvla.utils import setup_logging

log = logging.getLogger("rgr_search")
ARMS = ("single", "single_geo", "rec", "rec_geo", "rec_geo_oracle")

class Ctx:
    """Per-instance shared state (same for every arm -> paired)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _np(cs: dict) -> dict:
    return {k: v for k, v in cs.items() if not torch.is_tensor(v)}


def arm_single(c: Ctx, geo: bool):
    rc = c.rc
    cand, cs = R.cem_rounds(c.ranker, c.head, c.z0, c.q0, c.task, c.pool_c, _np(c.base), int(rc.single_cem_iters),
                            c.lam, c.mu, geo, c.seed, g_ref=c.g0h)
    s = R.objective(cs, c.lam, c.mu, geo)
    picks = R.select_diverse(s, cs["zfeat"], int(c.B), c.seed)
    chains = [{"segs": [cand[i]], "pred": float(cs["value"][i]), "score": float(s[i]), "imag_valid":
               bool(cs["valid"][i]) if "valid" in cs else True} for i in picks]
    return chains, {"n_imag": int(len(cand)), "exec_steps": 0, "n_exec": 0}


def _replan(c: Ctx, obs, z, q, g_ref, geo: bool, cont: np.ndarray, rng, k: int):
    """Plan the next bridge from an OBSERVED (real) intermediate state: fresh SmolVLA samples at the real obs +
    primitives + the level-1 continuation set, scored in imagination from the real latent."""
    rc = c.rc
    vla = c.policy.sample_chunks(obs, c.task, int(rc.replan_vla), noise_scale=1.0,
                                 seed=int(rng.integers(1 << 30)))[:, : c.H]
    prim, _ = primitives.library(c.H, np.random.RandomState(int(rng.integers(1 << 30))), n_random=int(rc.replan_prim_random))
    p2 = np.clip(np.concatenate([vla, prim, cont]), -1, 1).astype(np.float32)
    cs = R.imagine(c.ranker, c.head, z, q, c.task, p2, g_ref)
    s = R.objective(cs, c.lam, c.mu, geo)
    idx = R.select_diverse(s, cs["zfeat"], k, c.seed)
    return p2[idx], cs["value"][idx], s[idx], len(p2)


def arm_rec(c: Ctx, geo: bool, oracle: bool, fc: bool = True, depth: int | None = None):
    rc = c.rc
    depth = int(depth or rc.depth)
    H, cont_steps, tau, gamma = c.H, c.cont, float(rc.tau), float(rc.gamma)
    rng = np.random.default_rng(c.seed + 17)
    cand1, cs1 = R.cem_rounds(c.ranker, c.head, c.z0, c.q0, c.task, c.pool_c, c.base, int(rc.rec_cem_iters),
                              c.lam, c.mu, geo, c.seed, g_ref=c.g0h, n=int(rc.rec_cem_n), keep=True)
    J1 = R.objective(_np(cs1), c.lam, c.mu, geo)
    top1 = R.select_diverse(J1, cs1["zfeat"], int(rc.n1), c.seed)
    # continuation set for the imagined lookahead (state-independent part of the library + random pool members)
    is_prim = np.array([s.startswith("prim_") for s in c.src])
    pi_ = np.nonzero(is_prim)[0][: int(rc.n2) // 2]
    rest = rng.choice(np.nonzero(~is_prim)[0], size=int(rc.n2) - len(pi_), replace=False)
    cont = c.pool_c[np.concatenate([pi_, rest])]
    Q, order, J2, V2, inB0 = R.lookahead(c.ranker, c.head, cs1, cand1, top1, cont, c.task, c.lam, c.mu, geo, gamma, tau)
    n_imag = len(cand1) + len(top1) * len(cont)
    rank = np.argsort(-Q)
    st = {"n_imag": 0, "exec_steps": 0, "n_exec": 0, "n_replan": 0, "n_dropped_invalid": 0, "disc": [],
          "v1_hat": [], "v1_real": [], "in_b0_hat": [], "in_b0_real": [], "geo_disc_cm": []}
    chains_by_pick = []
    k1 = int(rc.k1)
    if not fc:  # open-loop chain chosen entirely in imagination (ablation)
        for p in rank[:k1]:
            b1 = cand1[top1[p]]
            ch = []
            if J1[top1[p]] >= gamma * J2[p].max():
                ch.append({"segs": [b1], "pred": float(cs1["value"][top1[p]]), "score": float(Q[p])})
            for j in order[p, :2]:
                ch.append({"segs": [b1, cont[j]], "pred": float(V2[p, j]), "score": float(Q[p])})
            chains_by_pick.append(ch)
        st["n_imag"] = n_imag
        return _flatten(chains_by_pick), st
    budget = int(c.B) * (H + cont_steps)
    used = 0
    cand_order = list(rank)
    n_exec_max = int(rc.max_exec)
    flat: list[dict] = []

    def good_cost() -> int:  # worst-case verification cost of the chains already believed to reach B_0
        return sum(H * len(ch["segs"]) + cont_steps for ch in flat if ch["pred"] >= tau)

    while cand_order and st["n_exec"] < n_exec_max:
        # adaptive screening: a 50-step execution + re-observation is ~1/7 of a verification; keep executing first
        # bridges while the chains that (observed) reach B_0 do not yet fill the sim budget
        if st["n_exec"] and used + good_cost() + H * 2 > budget:
            break
        nb = k1 if st["n_exec"] == 0 else 2
        take = cand_order[: min(nb, n_exec_max - st["n_exec"])]
        cand_order = cand_order[len(take):]
        res = run_bridges(c.pool, [(c.bddl, c.snap, cand1[top1[p]]) for p in take], every=H)
        chains_by_pick = []
        for p, r in zip(take, res):
            st["n_exec"] += 1
            st["exec_steps"] += int(r["n"])
            used += int(r["n"])
            b1 = cand1[top1[p]]
            z1, q1 = c.ranker.encode(r["obs"])
            v1r = float(c.ranker.score_real(z1, q1, c.task)[0])
            d = float(R.latent_discrepancy(z1, cs1["zH"][top1[p]][None])[0])
            g1h = c.head.predict(z1, q1).cpu().numpy() if c.head is not None else None
            st["disc"].append(d); st["v1_hat"].append(float(cs1["value"][top1[p]])); st["v1_real"].append(v1r)
            st["in_b0_hat"].append(bool(inB0[p])); st["in_b0_real"].append(v1r >= tau)
            if g1h is not None and "g" in cs1:
                st["geo_disc_cm"].append(float(np.linalg.norm(g1h[0, :3] - cs1["g"][top1[p], :3]) * 100))
            if geo:
                g_from, g_to = (c.g0t, r["geom"][-1][None]) if oracle else (c.g0h, g1h)
                _, ok, _ = geom_energy(g_from, g_to)
                if not bool(np.asarray(ok).reshape(-1)[0]) and not r["success"]:
                    st["n_dropped_invalid"] += 1
                    continue
            ch = []
            if r["success"] or v1r >= tau:  # s_1 is (believed) in B_0: hand over to pi right after b1
                ch.append({"segs": [b1], "pred": 1.0 if r["success"] else v1r, "score": float(Q[p])})
                if r["success"]:
                    chains_by_pick.append(ch)
                    continue
            g_ref = (r["geom"][-1][None] if oracle else g1h)
            replan_cost = int(rc.replan_vla) + 16 + int(rc.replan_prim_random) + len(cont)
            want = (d > float(rc.disc_thr) or abs(v1r - float(cs1["value"][top1[p]])) > float(rc.value_disc_thr)
                    or (geo and st["geo_disc_cm"] and st["geo_disc_cm"][-1] > float(rc.geo_disc_thr_cm)))
            if want and n_imag + replan_cost <= int(rc.imag_budget):
                b2s, v2s, _, n_p = _replan(c, r["obs"], z1, q1, g_ref, geo, cont, rng, 2)
                n_imag += n_p
                st["n_replan"] += 1
            else:  # keep the imagined plan; its value is re-read from the OBSERVED s_1 (2 rollouts)
                b2s = cont[order[p, :2]]
                v2s = R.imagine(c.ranker, None, z1, q1, c.task, b2s)["value"]
                n_imag += len(b2s)
            if depth >= 3 and max(v2s) < tau:
                # depth 3: execute the best b2 from the real s_1 (exact snapshot), re-observe, plan b3 from real s_2
                r2 = run_bridges(c.pool, [(c.bddl, r["snapshot"], b2s[0])], every=H)[0]
                st["n_exec"] += 1; st["exec_steps"] += int(r2["n"]); used += int(r2["n"])
                z2, q2 = c.ranker.encode(r2["obs"])
                g2 = r2["geom"][-1][None] if oracle else (c.head.predict(z2, q2).cpu().numpy() if c.head else None)
                b3s, v3s, _, n_p = _replan(c, r2["obs"], z2, q2, g2, geo, cont, rng, 2)
                n_imag += n_p
                for b3, v3 in zip(b3s, v3s):
                    ch.append({"segs": [b1, b2s[0], b3], "pred": float(v3), "score": float(Q[p])})
            for b2, v2 in zip(b2s, v2s):
                ch.append({"segs": [b1, b2], "pred": float(v2), "score": float(Q[p])})
            chains_by_pick.append(ch)
        flat += [x for ch in chains_by_pick for x in ch]
    st["n_imag"] = n_imag
    # spend verifications by (observed / re-planned) predicted value; ties keep the level-1 Q order
    return sorted(flat, key=lambda ch: -ch["pred"]), st


def _flatten(chains_by_pick: list[list[dict]]) -> list[dict]:
    """Round robin over level-1 picks: every pick's first chain, then every pick's second, ..."""
    out = []
    for r in range(max([len(x) for x in chains_by_pick] or [0])):
        out += [x[r] for x in chains_by_pick if r < len(x)]
    return out


def allocate(chains: list[dict], budget_steps: int, used: int, H: int, cont: int) -> list[dict]:
    """Keep chains in priority order while the worst-case sim cost (segments + pi continuation) fits the budget."""
    out = []
    for ch in chains:
        cost = H * len(ch["segs"]) + cont
        if used + cost > budget_steps:
            continue
        used += cost
        out.append(ch)
    return out

def run_arm(c: Ctx, a: str):
    if a == "single":
        return arm_single(c, False)
    if a == "single_geo":
        return arm_single(c, True)
    if a == "rec":
        return arm_rec(c, False, False)
    if a == "rec_geo":
        return arm_rec(c, True, False)
    if a == "rec_geo_oracle":
        return arm_rec(c, True, True)
    if a == "rec_geo_nofc":
        return arm_rec(c, True, False, fc=False)
    if a == "rec3_geo":
        return arm_rec(c, True, False, depth=3)
    if a == "rec_nofc":
        return arm_rec(c, False, False, fc=False)
    raise ValueError(a)


def main() -> None:
    setup_logging()
    cfg = cli_config()
    mb, sc, rc = cfg.m5b, cfg.search, cfg.rgr
    split = load_split(REPO_ROOT / cfg.split.file)
    scenes = scenes_for(split, "train", REPO_ROOT / cfg.recover.scene_root)
    assert_allowed([s.tag for s in scenes], split)
    cfg.tag = f"{cfg.tag}_s{mb.shard}of{mb.num_shards}"
    nsub, sub = int(mb.get("num_sub_shards", 1)), int(mb.get("sub_shard", 0))
    if nsub > 1:
        cfg.tag += f"_sub{sub}of{nsub}"
    arms = list(rc.arms)
    H, NP, B, cont = int(sc.horizon), int(mb.pool_size), int(mb.budget), int(sc.continuation_max_steps)
    dev = torch.device(cfg.policy.device)
    wm_ck = (REPO_ROOT / mb.wm_link).read_text().strip()
    cr_ck = (REPO_ROOT / mb.critic_link).read_text().strip()
    gh_ck = (REPO_ROOT / rc.geom_head_link).read_text().strip()
    root = REPO_ROOT / rc.out_root / f"s{mb.shard}"
    if nsub > 1:
        root = root / f"sub{sub}"
    root.mkdir(parents=True, exist_ok=True)
    done = set()
    for p_ in root.glob("p*/done.jsonl"):
        done |= {json.loads(x)["instance_id"] for x in p_.read_text().splitlines() if x.strip()}
    part = root / f"p{len(list(root.glob('p*')))}"
    part.mkdir()
    jobs = select_jobs(scenes, cfg, mb)
    from retryvla.geom.head import load_head

    with tracking.start(cfg) as run:
        run.summary.update({"wm_ckpt": wm_ck, "critic_ckpt": cr_ck, "geom_head": gh_ck, "part": str(part)})
        policy = SmolVLAWrapper(cfg.policy.path, cfg.policy.device, cfg.policy.n_action_steps)
        ranker = WMCriticRanker(wm_ck, cr_ck, cfg.wm_collect.encoder, int(cfg.wm_collect.pool_grid), dev,
                                0.0, float(mb.beta_rough))
        head = load_head(gh_ck, dev)
        writers = {a: RecoveryWriter(part / a, overwrite=True, write_retries=int(mb.get("write_retries", 0)))
                   for a in arms}
        rows, t0 = [], time.time()
        with SimPool(int(cfg.eval.n_workers), int(cfg.env.resolution)) as pool:
            srun = make_safe_run(pool, int(mb.get("worker_retries", 0)), int(mb.get("oom_retries", 0)),
                                 float(mb.get("oom_wait_s", 120)))
            insts, pre = precheck(pool, policy, jobs, set(), run=srun,
                                  known=load_known_precheck(mb.get("precheck_from") or []))
            pd.DataFrame(pre).to_csv(run.path("precheck.csv"), index=False)
            log.info("shard %s: %d selected, %d done, %d pi0 failures", mb.shard, len(jobs), len(done), len(insts))
            proposals = Proposals(policy, sc, H, int(cfg.eval.seed) + 77 * int(mb.shard))
            for n, inst in enumerate(insts):
                seed = zlib.crc32(inst.instance_id.encode()) % (2**31)
                ti = time.time()
                pool_c, src = proposals(inst, NP)  # drawn for EVERY instance in order -> pools paired with compare_arms
                if n % nsub != sub or inst.instance_id in done:
                    continue
                if int(rc.get("max_instances", 0)) and len(rows) >= int(rc.max_instances) * len(arms):
                    break
                z0, q0 = ranker.encode(inst.obs0)
                g0h = head.predict(z0, q0).cpu().numpy()
                g0t = run_bridges(pool, [(inst.bddl, inst.snapshot, np.zeros((0, 7), np.float32))], every=H)[0]["geom"][:1]
                base = R.imagine(ranker, head, z0, q0, inst.task, pool_c, g0h, keep=True)
                c = Ctx(rc=rc, ranker=ranker, head=head, policy=policy, pool=pool, task=inst.task, bddl=inst.bddl,
                        snap=inst.snapshot, z0=z0, q0=q0, g0h=g0h, g0t=g0t, pool_c=pool_c, src=src, base=base,
                        seed=seed % 10007, B=B, H=H, cont=cont, lam=float(rc.lam_geom), mu=float(mb.get("mu_cost", 1.0)))
                picked, stats = {}, {}
                for a in arms:
                    ta = time.time()
                    chains, st = run_arm(c, a)
                    budget = B * (H + cont)
                    chains = allocate(chains, budget, int(st["exec_steps"]), H, cont)
                    st["t_arm_s"] = time.time() - ta
                    picked[a], stats[a] = chains, st
                # one verification batch for all arms: bit-exact re-execution of every chain from the ORIGINAL failure
                # snapshot (prefix = concatenated bridges) + pi continuation; slot j has the same seed in every arm
                vj, owner = [], []
                for a in arms:
                    for j, ch in enumerate(picked[a]):
                        pre_ = np.concatenate(ch["segs"]).astype(np.float32)
                        vj.append(Job("restore_snapshot", dict(bddl=inst.bddl, snap=inst.snapshot), task=inst.task,
                                      max_steps=len(pre_) + cont, prefix=pre_, record="all",
                                      policy_seed=(seed + j) % (2**31 - 1)))
                        owner.append((a, j))
                res = srun(policy, vj, noise_scale=float(cfg.policy.noise_scale), desc="verify")
                # measurement only (not charged): TRUE geometry at every segment boundary of every chain
                gm = []
                if bool(rc.get("measure_geom", True)) and vj:
                    gm = run_bridges(pool, [(inst.bddl, inst.snapshot, j_.prefix) for j_ in vj], every=H, snapshot=False)
                by = {a: [] for a in arms}
                for k_, ((a, j), r) in enumerate(zip(owner, res)):
                    tv = None
                    if gm:
                        g = gm[k_]["geom"]
                        _, ok, _ = geom_energy(g[:-1], g[1:])
                        tv = bool(np.all(ok))
                    by[a].append((r, tv))
                line = []
                for a in arms:
                    chs, st = picked[a], stats[a]
                    succ = np.array([r.success for r, _ in by[a]], bool)
                    w = writers[a]
                    w.add_instance(inst.instance_id, {"scene": inst.scene_tag, "task": inst.task, "arm": a})
                    order = [k for k in np.argsort([-ch["score"] for ch in chs]) if succ[k]][: int(mb.max_trajs_per_instance)]
                    for k in order:
                        w.add_trajectory(by[a][k][0].record, inst.task, inst.instance_id,
                                         {"score": float(chs[k]["score"]), "pred_value": float(chs[k]["pred"]),
                                          "arm": a, "scene": inst.scene_tag, "n_bridges": len(chs[k]["segs"])})
                    rows.append({
                        "instance_id": inst.instance_id, "scene": inst.scene_tag, "task_folder": inst.meta["task_folder"],
                        "gripper_init": inst.meta["gripper_init"], "arm": a, "budget": len(chs),
                        "hits": int(succ.sum()), "solved": bool(succ.any()), "n_written": len(order),
                        "order_success": [bool(x) for x in succ], "order_pred": [float(ch["pred"]) for ch in chs],
                        "order_len": [len(ch["segs"]) for ch in chs],
                        "order_true_valid": [tv for _, tv in by[a]],
                        "order_prefix_success": [bool(r.prefix_success) for r, _ in by[a]],
                        "sim_steps_verify": int(sum(int(r.steps) for r, _ in by[a])),
                        "sim_steps_exec": int(st["exec_steps"]), "n_exec": int(st["n_exec"]),
                        "n_imag": int(st["n_imag"]), "t_arm_s": float(st["t_arm_s"]),
                        **{k: (v if not isinstance(v, (np.floating, np.integer)) else v.item())
                           for k, v in st.items() if k not in ("exec_steps", "n_exec", "n_imag", "t_arm_s")},
                    })
                    line.append(f"{a} {int(succ.sum())}/{len(chs)}")
                for r in res:
                    r.record = None
                with open(part / "done.jsonl", "a") as f:
                    f.write(json.dumps({"instance_id": inst.instance_id}) + "\n")
                with open(part / "rows.jsonl", "a") as f:  # survives a crash (per_instance.jsonl is rewritten)
                    for row in rows[-len(arms):]:
                        f.write(json.dumps(row) + "\n")
                d = pd.DataFrame(rows)
                d.to_json(run.path("per_instance.jsonl"), orient="records", lines=True)
                g = d.groupby("arm")
                run.log({f"rgr/solve_{a}": float(v) for a, v in g.solved.mean().items()}, step=n)
                log.info("[%d/%d] %s | %s | %.0fs (total %.0fs)", n + 1, len(insts), inst.instance_id[-38:],
                         " | ".join(line), time.time() - ti, time.time() - t0)
        for w in writers.values():
            w.close()
        (part / "CLOSED").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        d = pd.DataFrame(rows)
        if len(d):
            g = d.groupby("arm")
            run.summary.update({"n_instances": int(d.instance_id.nunique()),
                                "solve": {a: float(v) for a, v in g.solved.mean().items()},
                                "hits": {a: int(v) for a, v in g.hits.sum().items()}})
        run.summary["config"] = to_dict(cfg)


if __name__ == "__main__":
    main()
