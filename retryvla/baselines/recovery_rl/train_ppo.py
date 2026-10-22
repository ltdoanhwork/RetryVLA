"""B3.3: PPO for the residual bridge controller from the M5b train failure states, at a matched sim budget.

python -m retryvla.baselines.recovery_rl.train_ppo configs/exp/b33_rl_sparse.yaml [rl.budget_fraction=0.25 ...]

- Instances: the SAME 256 pi0 failures the PoC searched in M5b (pi1's data source): `select_jobs` of
  compare_arms (all 8 shards) filtered to the instance ids in the M5b per_instance logs. Start state = the exact
  load_scene snapshot (what compare_arms' precheck returns), restored with `restore_snapshot` like its
  verification jobs. Precheck (finding the failures) is not re-run and not charged (shared by both methods).
- Episode: residual active for the first `rl.horizon` steps (decisions every n_act steps), then pi0 alone until
  success or `rl.max_steps`. ALL env steps (bridge + continuation) are charged to the budget.
- Reward per bridge decision: progress_coef * (progress after - before) - action_pen * mean(delta^2); the last
  decision gets + success_bonus * episode success (pi0's continuation outcome). gamma = 1, GAE(lambda).
- Learning curve: deterministic residual on `rl.eval.n_instances` train instances with HELD-OUT pi0 seeds
  (disjoint from training seeds) at transition 0 and every `rl.eval.every` transitions; eval transitions are
  logged separately and NOT charged. Checkpoints at every budget milestone (rl.milestones, absolute counts).
- Resumable: everything lives in rl.out_dir (ckpt_last.pt, curve.jsonl, iters.jsonl); /pfss writes retried.
"""

from __future__ import annotations

import collections
import glob
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from retryvla import REPO_ROOT, tracking
from retryvla.baselines.recovery_rl.bridge import ResidualBridge
from retryvla.baselines.recovery_rl.env import run_episodes
from retryvla.baselines.recovery_rl.policy import ActorCritic, FeatureEncoder, save_ckpt
from retryvla.config import cli_config, to_dict
from retryvla.envs.splits import assert_allowed, load_split, scenes_for
from retryvla.envs.vec import SimPool
from retryvla.eval.rollout import Job, run_jobs
from retryvla.policy import SmolVLAWrapper
from retryvla.search.compare_arms import iid, select_jobs
from retryvla.utils import setup_logging

log = logging.getLogger("b33_ppo")


def _retry(fn, what: str, tries: int = 6):
    for k in range(tries):
        try:
            return fn()
        except (OSError, RuntimeError) as e:
            log.warning("%s failed (%s), retry %d/%d", what, str(e)[:100], k + 1, tries)
            time.sleep(20 * (k + 1))
    raise RuntimeError(f"giving up: {what}")


def save_atomic(obj, path: Path) -> None:
    def go():
        tmp = Path(str(path) + ".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)
    _retry(go, f"save {path.name}")


def append_jsonl(path: Path, row: dict) -> None:
    def go():
        with open(path, "a") as f:
            f.write(json.dumps(row, default=float) + "\n")
    _retry(go, f"append {path.name}")


def build_instances(cfg, pool) -> list[dict]:
    rl = cfg.rl
    split = load_split(REPO_ROOT / cfg.split.file)
    scenes = scenes_for(split, "train", REPO_ROOT / cfg.recover.scene_root)
    assert_allowed([s.tag for s in scenes], split)
    mb = OmegaConf.merge(cfg.m5b, {"shard": 0, "num_shards": 1})
    jobs = select_jobs(scenes, cfg, mb)
    ids = set()
    for f in sorted(glob.glob(str(REPO_ROOT / rl.instances_from))):
        ids |= set(pd.read_json(f, lines=True).instance_id)
    jobs = [j for j in jobs if iid(j.meta) in ids]
    if int(rl.get("max_instances", 0)):
        jobs = jobs[: int(rl.max_instances)]
    log.info("%d failure instances (M5b ids found: %d)", len(jobs), len(ids))
    lj = [Job(**{**j.__dict__, "max_steps": 0, "return_snapshot": True}) for j in jobs]
    res = run_jobs(pool, None, lj, desc="snapshots")
    out = []
    for j, r in zip(jobs, res):
        out.append({"instance_id": iid(j.meta), "bddl": j.load_kwargs["bddl"], "snap": r.load_out["snapshot"],
                    "task": j.task, "task_folder": j.meta["task_folder"], "scene": j.meta["scene"]})
    return out


def rewards(ep: dict, rc) -> list[float]:
    d = ep["decisions"]
    r = [float(rc.progress_coef) * (x.get("prog1", x["prog0"]) - x["prog0"]) - float(rc.action_pen) * x["delta_sq"]
         for x in d]
    if r:
        r[-1] += float(rc.success_bonus) * float(ep["success"])
    return r


def batch_from(eps: list[dict], rl) -> dict:
    img, low, ac, ag, lp, adv, ret, val = [], [], [], [], [], [], [], []
    lam = float(rl.gae_lambda)
    for ep in eps:
        d = ep["decisions"]
        if not d:
            continue
        r = rewards(ep, rl.reward)
        v = [float(x["value"]) for x in d] + [0.0]
        g, a = 0.0, [0.0] * len(d)
        for k in reversed(range(len(d))):
            delta = r[k] + v[k + 1] - v[k]
            g = delta + lam * g
            a[k] = g
        for k, x in enumerate(d):
            img.append(x["img"]); low.append(x["low"]); ac.append(x["a_cont"]); ag.append(x["a_grip"])
            lp.append(x["logp"]); adv.append(a[k]); ret.append(a[k] + v[k]); val.append(v[k])
    if not img:
        return {}
    return {"img": torch.stack(img), "low": torch.stack(low), "a_cont": torch.stack(ac),
            "a_grip": torch.stack(ag).long(), "logp": torch.stack(lp).float(), "adv": torch.tensor(adv),
            "ret": torch.tensor(ret), "val": torch.tensor(val)}


def ppo_update(model, opt, b: dict, rl, dev) -> dict:
    n = len(b["adv"])
    adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8) if n > 1 else b["adv"]
    stats = collections.defaultdict(list)
    mb = int(rl.minibatch)
    for _ in range(int(rl.ppo_epochs)):
        perm = torch.randperm(n)
        for s in range(0, n, mb):
            idx = perm[s: s + mb]
            g = lambda k: b[k][idx].to(dev)  # noqa: E731
            logp, ent, v = model.evaluate(g("img"), g("low"), g("a_cont"), g("a_grip"))
            ratio = (logp - g("logp")).exp()
            A = adv[idx].to(dev)
            pg = -torch.min(ratio * A, ratio.clamp(1 - float(rl.clip), 1 + float(rl.clip)) * A).mean()
            vl = 0.5 * (v - g("ret")).pow(2).mean()
            loss = pg + float(rl.vf_coef) * vl - float(rl.ent_coef) * ent.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(rl.max_grad_norm))
            opt.step()
            with torch.no_grad():
                stats["pg"].append(float(pg)); stats["vl"].append(float(vl)); stats["ent"].append(float(ent.mean()))
                stats["kl"].append(float((g("logp") - logp).mean()))
                stats["clipfrac"].append(float(((ratio - 1).abs() > float(rl.clip)).float().mean()))
    return {k: float(np.mean(v)) for k, v in stats.items()}


def ep_summary(eps: list[dict]) -> dict:
    d = [x for e in eps for x in e["decisions"]]
    return {"sr": float(np.mean([e["success"] for e in eps])), "bridge_sr": float(np.mean([e["bridge_success"] for e in eps])),
            "steps": int(sum(e["steps"] for e in eps)),
            "delta_rms": float(np.sqrt(np.mean([x["delta_sq"] for x in d]))) if d else 0.0,
            "grip_override": float(np.mean([int(x["a_grip"]) != 0 for x in d])) if d else 0.0}


def main() -> None:
    setup_logging()
    cfg = cli_config()
    rl = cfg.rl
    dev = torch.device(cfg.policy.device)
    out = REPO_ROOT / rl.out_dir
    out.mkdir(parents=True, exist_ok=True)
    budget = int(round(float(rl.budget_transitions) * float(rl.budget_fraction)))
    milestones = sorted({int(m) for m in rl.milestones if int(m) <= budget} | {budget})
    torch.manual_seed(int(rl.seed))
    with tracking.start(cfg) as run:
        run.summary.update({"out_dir": str(out), "budget": budget, "budget_full": int(rl.budget_transitions)})
        base = SmolVLAWrapper(cfg.policy.path, cfg.policy.device, cfg.policy.n_action_steps)
        feats = FeatureEncoder(rl.encoder, int(rl.pool_grid), cfg.policy.device)
        n_act = base.n_action_steps
        n_dec = (int(rl.horizon) + n_act - 1) // n_act
        model = ActorCritic(feats.img_dim, n_act, n_dec, int(rl.hidden), float(rl.init_log_std)).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=float(rl.lr), eps=1e-5)
        rl_cfg = {k: to_dict(rl)[k] for k in ("horizon", "scale", "encoder", "pool_grid", "max_steps")}
        st = {"it": 0, "trans": 0, "trans_eval": 0, "ptr": 0, "evals_done": [], "ms_done": [], "eps": 0}
        rng = np.random.default_rng(int(rl.seed))
        gen = torch.Generator().manual_seed(int(rl.seed))
        last = out / "ckpt_last.pt"
        if last.exists():
            ck = torch.load(last, map_location=dev, weights_only=False)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); st = ck["state"]
            rng.bit_generator.state = ck["np_rng"]; gen.set_state(ck["torch_gen"])
            log.info("resumed from %s: it %d, %d transitions", last, st["it"], st["trans"])
        bridge = ResidualBridge(model, feats, int(rl.horizon), n_act, float(rl.scale), cfg.policy.device)
        with SimPool(int(cfg.eval.n_workers), int(cfg.env.resolution)) as pool:
            cache = out / "instances.pt"
            insts = _retry(lambda: torch.load(cache, weights_only=False), "load instances") if cache.exists() else None
            if insts is None:
                insts = build_instances(cfg, pool)
                save_atomic(insts, cache)
            n_i = len(insts)
            if "perm" not in st:
                st["perm"] = np.random.default_rng(int(rl.seed) + 1).permutation(n_i).tolist()
            ev_idx = sorted(np.random.default_rng(12345).permutation(n_i)[: min(n_i, int(rl.eval.n_instances))])
            run.summary["n_instances"] = n_i

            def save(tag=None):
                extra = {"rl_cfg": rl_cfg, "state": st, "opt": opt.state_dict(), "np_rng": rng.bit_generator.state,
                         "torch_gen": gen.get_state()}
                def go():
                    save_ckpt(str(last) + ".tmp", model, extra)
                    Path(str(last) + ".tmp").replace(last)
                _retry(go, "ckpt_last")
                if tag:
                    save_atomic({"model": model.state_dict(), "hparams": model.hparams, "rl_cfg": rl_cfg,
                                 "state": dict(st)}, out / f"{tag}.pt")

            def evaluate(label):
                model.eval()
                eps = [{**{k: insts[i][k] for k in ("bddl", "snap", "task", "instance_id")},
                        "policy_seed": int(rl.eval.seed_base) + i, "max_steps": int(rl.max_steps)} for i in ev_idx]
                t = time.time()
                res = run_episodes(pool, base, bridge, eps, sample=False, noise_scale=float(cfg.policy.noise_scale),
                                   desc=f"eval@{label}")
                s = ep_summary(res)
                st["trans_eval"] += s["steps"]
                row = {"transitions": st["trans"], "it": st["it"], "episodes": st["eps"], "eval_sr": s["sr"],
                       "eval_bridge_sr": s["bridge_sr"], "eval_delta_rms": s["delta_rms"],
                       "eval_grip_override": s["grip_override"], "n": len(res), "eval_transitions": s["steps"],
                       "wall_s": time.time() - t, "per_instance": {e["instance_id"]: bool(e["success"]) for e in res}}
                append_jsonl(out / "curve.jsonl", row)
                st["last_eval_trans"] = st["trans"]
                run.log({"rl/eval_sr": s["sr"], "rl/eval_bridge_sr": s["bridge_sr"]}, step=st["trans"])
                log.info("EVAL @%d transitions: SR %.3f (bridge %.3f) on %d held-out-seed episodes", st["trans"],
                         s["sr"], s["bridge_sr"], len(res))

            if 0 not in st["evals_done"] and int(rl.eval.n_instances) > 0:
                evaluate(0)
                st["evals_done"].append(0)
                save()
            ev_every = int(rl.eval.every)
            while st["trans"] < budget:
                ti = time.time()
                model.eval()
                eps = []
                G = max(1, int(rl.get("group_size", 1)))  # episodes per instance per iteration (fewer env rebuilds)
                for _ in range(max(1, int(rl.episodes_per_iter) // G)):
                    i = int(st["perm"][st["ptr"] % n_i])
                    st["ptr"] += 1
                    if st["ptr"] % n_i == 0:
                        st["perm"] = rng.permutation(n_i).tolist()
                    for _g in range(G):
                        eps.append({**{k: insts[i][k] for k in ("bddl", "snap", "task", "instance_id")},
                                    "policy_seed": int(rng.integers(0, 1_000_000_000)), "max_steps": int(rl.max_steps)})
                res = run_episodes(pool, base, bridge, eps, sample=True, gen=gen,
                                   noise_scale=float(cfg.policy.noise_scale), desc=f"it{st['it']}")
                s = ep_summary(res)
                st["trans"] += s["steps"]
                st["eps"] += len(res)
                b = batch_from(res, rl)
                model.train()
                u = ppo_update(model, opt, b, rl, dev) if b else {}
                st["it"] += 1
                row = {"it": st["it"], "transitions": st["trans"], "episodes": st["eps"], "train_sr": s["sr"],
                       "train_bridge_sr": s["bridge_sr"], "delta_rms": s["delta_rms"],
                       "grip_override": s["grip_override"], "n_dec": len(b.get("adv", [])),
                       "std": float(model.log_std.exp().mean()), "wall_s": time.time() - ti,
                       "steps_per_s": s["steps"] / max(1e-6, time.time() - ti), **u}
                append_jsonl(out / "iters.jsonl", row)
                run.log({f"rl/{k}": v for k, v in row.items() if isinstance(v, (int, float))}, step=st["trans"])
                log.info("it %d | %d/%d transitions (%.1f%%) | train SR %.3f bridge %.3f | |d| %.3f grip %.2f | "
                         "kl %.4f | %.0f steps/s", st["it"], st["trans"], budget, 100 * st["trans"] / budget, s["sr"],
                         s["bridge_sr"], s["delta_rms"], s["grip_override"], u.get("kl", 0), row["steps_per_s"])
                tag = None
                for m in milestones:
                    if st["trans"] >= m and m not in st["ms_done"]:
                        st["ms_done"].append(m)
                        tag = f"ckpt_m{m}"
                save(tag)
                k_ev = st["trans"] // ev_every if ev_every > 0 else 0
                if ev_every > 0 and k_ev not in st["evals_done"] and int(rl.eval.n_instances) > 0:
                    st["evals_done"].append(k_ev)
                    evaluate(k_ev)
                    save()
            if (f"final{budget}" not in st["evals_done"] and int(rl.eval.n_instances) > 0
                    and st.get("last_eval_trans") != st["trans"]):
                evaluate("final")
                st["evals_done"].append(f"final{budget}")
            save("final")
        run.summary.update({"transitions": st["trans"], "eval_transitions": st["trans_eval"], "iterations": st["it"],
                            "episodes": st["eps"], "final_ckpt": str(out / "final.pt"), "config": to_dict(cfg)})


if __name__ == "__main__":
    main()
