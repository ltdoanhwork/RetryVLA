"""M3: world-model-guided recovery search on HARD, WM-unseen failure instances.

python -m retryvla.search.wm_search configs/exp/m3_wm_search_hard.yaml [m3.shard=i m3.num_shards=N]

Per instance (same proposal distribution for both arms, same simulator budget M):
  1. restore the failure state, encode obs0 with frozen DINOv2
  2. candidate pool: N_pool proposals (SmolVLA samples, smooth perturbations, primitives)
  3. WM arm: optional latent CEM (elite-local refinement scored ONLY in imagination), then rank by
        S(A) = V_psi(imagined end state) - lambda * U_WM(A) - beta * roughness(A)
     and send the top-M to the simulator for verification
  4. random arm: M uniformly random candidates from the initial pool, verified the same way
Reports solve rate (>=1 verified success within M), verified success rate, and critic calibration on
the verified candidates. WM-found verified recoveries are written as a LeRobotDataset.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
import torch

from retryvla import REPO_ROOT, tracking
from retryvla.config import cli_config, to_dict
from retryvla.critic.train_critic import load_wm
from retryvla.critic.value_model import RecoveryValue
from retryvla.data.recovery_writer import RecoveryWriter
from retryvla.envs.splits import assert_allowed, load_split, scenes_for
from retryvla.envs.vec import SimPool
from retryvla.policy import SmolVLAWrapper
from retryvla.search.cem import CEM, CEMConfig, roughness
from retryvla.search.oracle_sim import OracleSimScorer
from retryvla.search.run_search import Proposals, build_instances
from retryvla.utils import setup_logging
from retryvla.wm.encoder_dino import DinoEncoder

log = logging.getLogger("wm_search")


class WMCriticRanker:
    def __init__(self, wm_ckpt: str, critic_ckpt: str, encoder: str, pool_grid: int, dev, lam_unc: float,
                 beta_rough: float):
        self.dev = dev
        self.wm, self.norm, _ = load_wm(wm_ckpt, dev)
        ck = torch.load(critic_ckpt, map_location="cpu", weights_only=False)
        c = ck["cfg"]
        self.vocab = ck["vocab"]
        self.critic = RecoveryValue(dim=ck["dim"], n_patch=ck["n_patch"], n_tasks=max(16, len(self.vocab)),
                                    width=int(c["width"]), depth=int(c["depth"]), dropout=0.0).to(dev).eval()
        self.critic.load_state_dict(ck["model"])
        self.enc = DinoEncoder(encoder, pool_grid=pool_grid, device=str(dev))
        self.lam, self.beta = lam_unc, beta_rough

    @torch.no_grad()
    def encode(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            za = self.enc.encode_images(torch.as_tensor(obs["image"], device=self.dev)[None])
            zw = self.enc.encode_images(torch.as_tensor(obs["image2"], device=self.dev)[None])
        z = torch.cat([za, zw], 1).float()
        return self.norm.z(z), self.norm.q(torch.as_tensor(obs["state"], device=self.dev)[None])

    @torch.no_grad()
    def score_real(self, zH, qH, task: str) -> np.ndarray:
        """Critic on REAL (simulated) end states (normalized latents) -> V [N]."""
        tid = self.vocab.get(task, 0)
        n = len(zH)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            v = torch.sigmoid(self.critic.logits(zH.float(), qH.float(),
                                                 torch.full((n,), tid, device=self.dev, dtype=torch.long)))
        return v.float().cpu().numpy()

    @torch.no_grad()
    def score(self, z0, q0, task: str, cands: np.ndarray, bs: int = 256, goal=None) -> dict[str, np.ndarray]:
        """goal: optional [G, P, D] normalized goal latents (final frames of successful demos of this task) ->
        adds "goal" = -min_g mean-squared latent distance of the imagined end state (DINO-WM-style objective)."""
        tid = self.vocab.get(task, 0)
        V, U, ZF, GD = [], [], [], []
        for i in range(0, len(cands), bs):
            a = torch.as_tensor(cands[i: i + bs], device=self.dev)
            n = len(a)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                zH, qH, unc, _, _ = self.wm.rollout(z0.expand(n, -1, -1), q0.expand(n, -1), a)
                v = torch.sigmoid(self.critic.logits(zH.float(), qH.float(),
                                                     torch.full((n,), tid, device=self.dev, dtype=torch.long)))
            V.append(v.float().cpu().numpy()); U.append(unc.float().cpu().numpy())
            ZF.append(torch.cat([zH.float().mean(1), qH.float()], 1).cpu().numpy())  # imagined end-state feature
            if goal is not None:
                d = torch.stack([((zH.float() - g[None]) ** 2).mean((1, 2)) for g in goal]).min(0).values
                GD.append((-d).cpu().numpy())
        V, U = np.concatenate(V), np.concatenate(U)
        U_n = U / (np.median(U) + 1e-8)
        R = roughness(cands)
        out = {"value": V, "unc": U, "score": V - self.lam * U_n * V.std() - self.beta * R, "rough": R,
               "zfeat": np.concatenate(ZF), "unc_n": U_n}
        if goal is not None:
            out["goal"] = np.concatenate(GD)
        return out


def diverse_ranking(score: np.ndarray, feat: np.ndarray, k: int, oversample: int = 4, seed: int = 0) -> np.ndarray:
    """Pick k candidates: k-means over the imagined end states of the top (oversample*k) by score,
    best candidate per cluster, ordered by score. Avoids spending the budget on 16 variants of one plan."""
    top = np.argsort(-score)[: max(k, oversample * k)]
    if len(top) <= k:
        return top
    x = feat[top].astype(np.float64)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(15):
        lab = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (lab == j).any():
                c[j] = x[lab == j].mean(0)
    picks = []
    for j in range(k):
        m = np.nonzero(lab == j)[0]
        if len(m):
            picks.append(top[m[np.argmax(score[top[m]])]])
    rest = [i for i in top if i not in set(picks)]  # fill empty clusters by score
    picks += rest[: k - len(picks)]
    return np.array(sorted(picks, key=lambda i: -score[i]))


def main() -> None:
    setup_logging()
    cfg = cli_config()
    m3 = cfg.m3
    sc = cfg.search
    split = load_split(REPO_ROOT / cfg.split.file)
    only = {l.strip() for l in open(REPO_ROOT / m3.instance_ids_file) if l.strip()}
    scenes = [s for s in scenes_for(split, "train", REPO_ROOT / cfg.recover.scene_root)
              if s.tag in {i.rsplit("__", 1)[0] for i in only}]
    assert_allowed([s.tag for s in scenes], split)
    scenes = scenes[int(m3.shard)::int(m3.num_shards)]
    cfg.tag = f"{cfg.tag}_s{m3.shard}of{m3.num_shards}"
    dev = torch.device(cfg.policy.device)
    wm_ck = (REPO_ROOT / m3.wm_link).read_text().strip()
    cr_ck = (REPO_ROOT / m3.critic_link).read_text().strip()
    H, M, NP = int(sc.horizon), int(m3.verify_budget), int(m3.pool_size)
    with tracking.start(cfg) as run:
        run.summary.update({"wm_ckpt": wm_ck, "critic_ckpt": cr_ck})
        policy = SmolVLAWrapper(cfg.policy.path, cfg.policy.device, cfg.policy.n_action_steps)
        ranker = WMCriticRanker(wm_ck, cr_ck, cfg.wm_collect.encoder, int(cfg.wm_collect.pool_grid), dev,
                                float(m3.lambda_unc), float(m3.beta_rough))
        out_root = REPO_ROOT / m3.out_root
        writer = RecoveryWriter(out_root, overwrite=True)
        rows, t0 = [], time.time()
        cfg.search_on.precheck_rollouts = 0
        with SimPool(int(cfg.eval.n_workers), int(cfg.env.resolution)) as pool:
            insts, _ = build_instances(pool, policy, scenes, cfg, "train")
            insts = [i for i in insts if i.instance_id in only]
            scorer = OracleSimScorer(pool, policy, objective="competence", horizon=H,
                                     continuation_max_steps=int(sc.continuation_max_steps),
                                     noise_scale=float(cfg.policy.noise_scale))
            proposals = Proposals(policy, sc, H, int(cfg.eval.seed) + 77 * int(m3.shard))
            for n, inst in enumerate(insts):
                rng = np.random.default_rng(hash(inst.instance_id) % (2**31))
                pool_c, src = proposals(inst, NP)
                z0, q0 = ranker.encode(inst.obs0)
                s0 = ranker.score(z0, q0, inst.task, pool_c)
                cand, cs, csrc = pool_c, s0, list(src)
                # latent CEM: refine around imagined elites (no simulator calls)
                cem = CEM(H, 7, CEMConfig(n_samples=NP, init_std=float(m3.latent_cem_std), smooth_kernel=5),
                          seed=n + 1)
                for it in range(1, int(m3.latent_cem_iters) + 1):
                    el = np.argsort(-cs["score"])[: int(m3.latent_elites)]
                    new = cem.sample_local(cand[el], cs["score"][el], it, temperature=0.05, decay=0.6)
                    ns = ranker.score(z0, q0, inst.task, new)
                    cand = np.concatenate([cand, new])
                    cs = {k: np.concatenate([cs[k], ns[k]]) for k in cs}
                    csrc += ["wm_cem"] * len(new)
                wm_pick = np.argsort(-cs["score"])[:M]
                rnd_pick = rng.choice(len(pool_c), size=M, replace=False)
                both = np.concatenate([cand[wm_pick], pool_c[rnd_pick]])
                res = scorer.score(inst, both, iteration=0)
                wm_s, rd_s = res.success[:M], res.success[M:]
                writer.add_instance(inst.instance_id, {"scene": inst.scene_tag, "task": inst.task})
                wins = [k for k in range(M) if wm_s[k] and res.records[k] is not None]
                for k in sorted(wins, key=lambda k: -cs["score"][wm_pick[k]])[: int(m3.max_trajs_per_instance)]:
                    writer.add_trajectory(res.records[k], inst.task, inst.instance_id,
                                          {"score": float(cs["score"][wm_pick[k]]), "source": csrc[wm_pick[k]],
                                           "pred_value": float(cs["value"][wm_pick[k]]), "scene": inst.scene_tag})
                row = {"instance_id": inst.instance_id,
                       "wm_hits": int(wm_s.sum()), "rand_hits": int(rd_s.sum()),
                       "wm_solved": bool(wm_s.any()), "rand_solved": bool(rd_s.any()),
                       "wm_pred_mean": float(cs["value"][wm_pick].mean()),
                       "pool_pred_mean": float(s0["value"].mean()),
                       "wm_rough": float(cs["rough"][wm_pick].mean()), "rand_rough": float(roughness(pool_c[rnd_pick]).mean()),
                       "wm_from_latent_cem": int(sum(csrc[i] == "wm_cem" for i in wm_pick)),
                       "pred_verified": [float(v) for v in cs["value"][wm_pick]], "verified": [bool(v) for v in wm_s]}
                rows.append(row)
                d = pd.DataFrame(rows)
                run.log({"m3/wm_solve": d.wm_solved.mean(), "m3/rand_solve": d.rand_solved.mean(),
                         "m3/wm_hit_rate": d.wm_hits.sum() / (M * len(d)), "m3/rand_hit_rate": d.rand_hits.sum() / (M * len(d))},
                        step=n)
                log.info("[%d/%d] %s | WM %d/%d  random %d/%d | solved so far WM %.2f vs random %.2f | %.0fs", n + 1,
                         len(insts), inst.instance_id[-45:], row["wm_hits"], M, row["rand_hits"], M,
                         d.wm_solved.mean(), d.rand_solved.mean(), time.time() - t0)
                d.to_json(run.path("per_instance.jsonl"), orient="records", lines=True)
        writer.close()
        d = pd.DataFrame(rows)
        run.summary.update({
            "n_instances": len(d), "verify_budget": M, "pool_size": NP,
            "wm_solve_rate": float(d.wm_solved.mean()), "rand_solve_rate": float(d.rand_solved.mean()),
            "wm_hit_rate": float(d.wm_hits.sum() / (M * len(d))), "rand_hit_rate": float(d.rand_hits.sum() / (M * len(d))),
            "only_wm": int((d.wm_solved & ~d.rand_solved).sum()), "only_rand": int((~d.wm_solved & d.rand_solved).sum()),
            "wm_rough": float(d.wm_rough.mean()), "rand_rough": float(d.rand_rough.mean()), "config": to_dict(cfg)})
        log.info("M3 done: %s", {k: v for k, v in run.summary.items() if k not in ("config", "meta")})


if __name__ == "__main__":
    main()
