"""Back on Track (arXiv 2610.19579) as a test-time wrapper around a frozen policy (black-box: only predict_chunk).

At every policy observation (chunk boundary):
  monitor  V = value(encode(obs)); V_tilde = mean of the last W = 3 values; trigger iff V_tilde > eta_t,
           eta_t = 1 / (1 + exp((t - t_c) / sigma)) (t = env steps since episode start, t_c 60, sigma 10, terminal 0)
  if triggered: query the policy ONCE for its nominal chunk, CEM in the latent WM around it with the PCA action basis
           (delta = s V_k Lambda_k^1/2 eps, s 0.6, gripper = nominal), objective J = sum_i V(imagined s_i) (minimised),
           256 samples x 5 iterations (= their 1,280 ten-step plans / cycle), execute the first h of the best plan,
           then control returns to the monitor (re-invoked at the next observation if still triggered)
  else: the policy's chunk runs unchanged (n_action_steps).
Value: "liveness" (retryvla.baselines.bot.liveness ensemble, high = far from completion under pi) or "cpi"
(our C^pi critic, mapped V = 1 - 2 C^pi so the same threshold / objective apply; labelled variant).

Episode identity: rollout.run_jobs calls ``begin_batch(keys, n_calls)`` (opt-in hook) before each predict_chunk.
Cost accounting per episode: VLA / monitor / planning wall time (batched calls attributed evenly), #triggers,
#WM forward rows; ``bench()`` measures single-episode planning latency.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

SEED_MULT = 100_003


@dataclass
class BoTCfg:
    value: str = "liveness"       # liveness | cpi
    bot_ckpt: str = ""            # bot.pt (liveness ensemble + PCA basis + wm path)
    critic_ckpt: str = ""         # cpi variant
    wm_ckpt: str = ""             # default: the one stored in bot.pt
    horizon: int = 10             # H env steps planned (multiple of the WM chunk k = 5)
    execute: int = 5              # h env steps executed before control returns to the monitor
    n_samples: int = 256
    n_iters: int = 5
    n_elites: int = 32
    scale: float = 0.6
    min_std: float = 0.05
    window: int = 3
    t_c: float = 60.0
    sigma: float = 10.0
    eta_final: float = 0.0
    include_nominal: bool = True
    seed: int = 0
    encoder: str = "facebook/dinov2-small"
    pool_grid: int = 6
    wm_bs: int = 4096

    @classmethod
    def from_dict(cls, d: dict | None) -> "BoTCfg":
        d = dict(d or {})
        return cls(**{k: type(getattr(cls, k))(v) for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class _Ep:
    t: int = 0
    hist: deque = field(default_factory=lambda: deque(maxlen=3))
    n_obs: int = 0
    n_trig: int = 0
    wm_rows: int = 0
    vla_s: float = 0.0
    mon_s: float = 0.0
    plan_s: float = 0.0
    v_trace: list = field(default_factory=list)
    trig_t: list = field(default_factory=list)


class BackOnTrackPolicy:
    needs_pca = True

    def __init__(self, base, cfg: BoTCfg | dict, device: str = "cuda"):
        from retryvla.critic.train_critic import load_wm
        from retryvla.wm.encoder_dino import DinoEncoder

        self.base = base
        self.cfg = cfg if isinstance(cfg, BoTCfg) else BoTCfg.from_dict(cfg)
        c = self.cfg
        self.dev = torch.device(device)
        bk = torch.load(c.bot_ckpt, map_location="cpu", weights_only=False) if c.bot_ckpt else None
        self.bk = bk
        wm_path = c.wm_ckpt or (bk["wm_ckpt"] if bk else "")
        self.wm, self.norm, _ = load_wm(wm_path, self.dev)
        self.k = int(self.wm.k)
        assert c.horizon % self.k == 0, "planner horizon must be a multiple of the WM chunk"
        self.enc = DinoEncoder(c.encoder, pool_grid=c.pool_grid, device=str(self.dev))
        if c.value == "liveness":
            from retryvla.baselines.bot.liveness import LivenessEnsemble

            self.ens = LivenessEnsemble(n_tok=bk["n_tok"], dim=bk["dim"], n_members=bk["n_members"]).to(self.dev)
            self.ens.load_state_dict(bk["ensemble"])
            self.ens.eval()
            self.vocab = bk["vocab"]
        elif c.value == "progress":
            from retryvla.baselines.bot.liveness import ProgressHead

            self.prog = ProgressHead(bk["n_tok"], bk["dim"]).to(self.dev)
            self.prog.load_state_dict(bk["progress"])
            self.prog.eval()
            self.vocab = bk["vocab"]
        elif c.value == "cpi":
            from retryvla.critic.value_model import RecoveryValue

            ck = torch.load(c.critic_ckpt, map_location="cpu", weights_only=False)
            cc = ck["cfg"]
            self.vocab = ck["vocab"]
            self.critic = RecoveryValue(dim=ck["dim"], n_patch=ck["n_patch"], n_tasks=max(16, len(self.vocab)),
                                        width=int(cc["width"]), depth=int(cc["depth"]), dropout=0.0).to(self.dev).eval()
            self.critic.load_state_dict(ck["model"])
        else:
            raise ValueError(c.value)
        if bk is not None and "pca_V" in bk and int(bk["horizon"]) == c.horizon:
            V, lam = bk["pca_V"], bk["pca_lam"]
            self.B = torch.as_tensor(V * np.sqrt(lam)[None], device=self.dev)      # [H*6, k]
            self.kpca = self.B.shape[1]
        elif self.needs_pca:
            raise ValueError(f"bot.pt PCA basis horizon {bk and bk['horizon']} != planner horizon {c.horizon}")
        else:
            self.B, self.kpca = None, 0
        self.gen = torch.Generator(device=self.dev).manual_seed(c.seed)
        self.eps: dict = {}
        self._keys = None
        self._done: list[dict] = []

    # ------------------------------------------------------------------------------------------------ plumbing
    @property
    def n_action_steps(self) -> int:
        return self.base.n_action_steps

    def __getattr__(self, k):
        return getattr(self.base, k)

    def begin_batch(self, keys, n_calls) -> None:
        self._keys = list(keys)
        for kk, n in zip(keys, n_calls):
            if n == 0:
                self.eps[kk] = _Ep(hist=deque(maxlen=self.cfg.window))

    def pop_episode_stats(self, key) -> dict:
        e = self.eps.pop(key, None)
        if e is None:
            return {}
        d = {"bot_n_obs": e.n_obs, "bot_n_trig": e.n_trig, "bot_wm_rows": e.wm_rows, "bot_vla_s": e.vla_s,
             "bot_mon_s": e.mon_s, "bot_plan_s": e.plan_s, "bot_steps": e.t,
             "bot_trig_t": e.trig_t[:200], "bot_v": [round(v, 3) for v in e.v_trace[:200]]}
        self._done.append({k: v for k, v in d.items() if not isinstance(v, list)})
        return d

    def stats(self) -> dict:
        if not self._done:
            return {}
        keys = [k for k in self._done[0]]
        out = {f"mean_{k}": float(np.mean([d[k] for d in self._done])) for k in keys}
        out["episodes"] = len(self._done)
        out["frac_triggered"] = float(np.mean([d["bot_n_trig"] > 0 for d in self._done]))
        return out

    # ------------------------------------------------------------------------------------------------- value
    def _tid(self, tasks):
        """Unknown task -> -1: the value is undefined there (e.g. Goal C^pi has no data for the 3 tasks without audit
        failures) -> V = -1 ("on track"): never triggers, Commit / the nominal plan wins all ties."""
        return torch.as_tensor([self.vocab.get(t, -1) for t in tasks], device=self.dev, dtype=torch.long)

    @torch.no_grad()
    def value(self, z, q, tid) -> torch.Tensor:
        """normalized z [N, 72, D], q [N, 8] -> V in [-1, 1], high = bad."""
        unk = tid < 0
        tid = tid.clamp_min(0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            if self.cfg.value == "liveness":
                from retryvla.baselines.bot.liveness import pool_tokens

                v = self.ens(pool_tokens(z.float(), self.cfg.pool_grid), q.float(), tid)
            elif self.cfg.value == "progress":
                from retryvla.baselines.bot.liveness import pool_tokens

                v = 1.0 - 2.0 * self.prog(pool_tokens(z.float(), self.cfg.pool_grid), q.float(), tid)
            else:
                v = 1.0 - 2.0 * torch.sigmoid(self.critic.logits(z.float(), q.float(), tid))
        return torch.where(unk, -torch.ones_like(v.float()), v.float())

    @torch.no_grad()
    def encode(self, obs: Sequence[dict]):
        im = torch.as_tensor(np.stack([o["image"] for o in obs]), device=self.dev)
        iw = torch.as_tensor(np.stack([o["image2"] for o in obs]), device=self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            z = torch.cat([self.enc.encode_images(im), self.enc.encode_images(iw)], 1).float()
        q = torch.as_tensor(np.stack([o["state"] for o in obs]), device=self.dev, dtype=torch.float32)
        return self.norm.z(z), self.norm.q(q)

    def eta(self, t: int) -> float:
        c = self.cfg
        x = (t - c.t_c) / c.sigma
        s = 1.0 / (1.0 + math.exp(min(60.0, x)))
        return c.eta_final + (1.0 - c.eta_final) * s

    # -------------------------------------------------------------------------------------------------- plan
    @torch.no_grad()
    def plan(self, z0, q0, tid, a_nom: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Batched CEM for T triggered episodes. z0 [T,P,D], q0 [T,8], a_nom [T,H,7] -> best [T,H,7], WM rows."""
        c = self.cfg
        T, H = a_nom.shape[0], c.horizon
        n, K = c.n_samples, self.kpca
        mu = torch.zeros(T, K, device=self.dev)
        sd = torch.ones(T, K, device=self.dev)
        best_J = torch.full((T,), float("inf"), device=self.dev)
        best_a = a_nom.clone()
        rows = 0
        for it in range(c.n_iters):
            eps = torch.randn(T, n, K, device=self.dev, generator=self.gen)
            coef = mu[:, None] + sd[:, None] * eps
            if it == 0 and c.include_nominal:
                coef[:, 0] = 0.0
            delta = c.scale * torch.einsum("tnk,dk->tnd", coef, self.B).view(T, n, H, 6)
            a = a_nom[:, None].expand(T, n, H, 7).clone()
            a[..., :6] = (a[..., :6] + delta).clamp(-1, 1)
            J = self._score(z0, q0, tid, a.view(T * n, H, 7)).view(T, n)
            rows += T * n * (H // self.k)
            jb, ib = J.min(1)
            upd = jb < best_J
            best_J = torch.where(upd, jb, best_J)
            best_a[upd] = a[upd, ib[upd]]
            el = J.topk(c.n_elites, dim=1, largest=False).indices                   # [T, E]
            ce = torch.gather(coef, 1, el[..., None].expand(-1, -1, K))
            mu, sd = ce.mean(1), ce.std(1).clamp_min(c.min_std)
        return best_a, rows

    @torch.no_grad()
    def _score(self, z0, q0, tid, a):
        n_per = a.shape[0] // z0.shape[0]
        Z = z0.repeat_interleave(n_per, 0)
        Q = q0.repeat_interleave(n_per, 0)
        Tt = tid.repeat_interleave(n_per, 0)
        out = []
        for i in range(0, len(a), self.cfg.wm_bs):
            sl = slice(i, i + self.cfg.wm_bs)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                _, _, _, zs, qs = self.wm.rollout(Z[sl], Q[sl], a[sl])
            J = sum(self.value(zz.float(), qq.float(), Tt[sl]) for zz, qq in zip(zs, qs))
            out.append(J)
        return torch.cat(out)

    # ----------------------------------------------------------------------------------------------- control
    def predict_chunk(self, obs: Sequence[dict], tasks: Sequence[str], seeds: Sequence[int] | None = None,
                      noise_scale: float = 1.0):
        if len(obs) == 0:
            return self.base.predict_chunk(obs, tasks, seeds, noise_scale)
        c, Bn = self.cfg, len(obs)
        keys = self._keys if self._keys is not None and len(self._keys) == Bn else \
            [int(s) // SEED_MULT for s in (seeds or range(Bn))]
        self._keys = None
        for kk in keys:
            if kk not in self.eps:
                self.eps[kk] = _Ep(hist=deque(maxlen=c.window))
        E = [self.eps[kk] for kk in keys]
        sync = (lambda: torch.cuda.synchronize()) if self.dev.type == "cuda" else (lambda: None)
        t0 = time.time()
        z, q = self.encode(obs)
        tid = self._tid(tasks)
        v = self.value(z, q, tid).cpu().numpy()
        trig = []
        for b, e in enumerate(E):
            e.hist.append(float(v[b])); e.v_trace.append(float(v[b])); e.n_obs += 1
            if float(np.mean(e.hist)) > self.eta(e.t):
                trig.append(b)
        sync()
        t1 = time.time()
        chunks = self.base.predict_chunk(obs, tasks, seeds, noise_scale)
        sync()
        t2 = time.time()
        out = [np.asarray(ch, np.float32) for ch in chunks]
        if trig:
            ti = torch.as_tensor(trig, device=self.dev)
            a_nom = torch.as_tensor(np.stack([np.clip(out[b][: c.horizon], -1, 1) for b in trig]), device=self.dev)
            best, rows = self.plan(z[ti], q[ti], tid[ti], a_nom)
            best = best[:, : c.execute].float().cpu().numpy()
            for j, b in enumerate(trig):
                out[b] = best[j]
                E[b].n_trig += 1
                E[b].trig_t.append(E[b].t)
                E[b].wm_rows += rows // len(trig)
            sync()
        t3 = time.time()
        for b, e in enumerate(E):
            e.mon_s += (t1 - t0) / Bn
            e.vla_s += (t2 - t1) / Bn
            if b in trig:
                e.plan_s += (t3 - t2) / len(trig)
            e.t += min(len(out[b]), self.n_action_steps)
        return out

    def bench(self, obs: dict, task: str, reps: int = 5) -> dict:
        z, q = self.encode([obs])
        tid = self._tid([task])
        a = torch.zeros(1, self.cfg.horizon, 7, device=self.dev)
        self.plan(z, q, tid, a)
        ts = []
        for _ in range(reps):
            if self.dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            z, q = self.encode([obs])
            self.value(z, q, tid)
            t1 = time.time()
            self.plan(z, q, tid, a)
            if self.dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append((t1 - t0, time.time() - t1))
        t0 = time.time()
        for _ in range(reps):
            self.base.predict_chunk([obs], [task], [0])
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        tv = (time.time() - t0) / reps
        return {"monitor_s": float(np.median([x[0] for x in ts])), "plan_s": float(np.median([x[1] for x in ts])),
                "vla_s": tv, "plans_per_cycle": self.cfg.n_samples * self.cfg.n_iters}


@dataclass
class ForkCastCfg(BoTCfg):
    """ForkCast-style selector (2610.18650), adapted: no detector, Commit always in the set, argmax at a common
    horizon. Candidates = the policy's own chunk (Commit) + library primitives (Replan analogue; no Reobserve:
    LIBERO cameras are fixed)."""
    primitives: str = "retreat20,regrasp,reposition_0,reposition_180"
    first_decision: int = 60     # env steps (ForkCast: first decision at step 64)
    horizon: int = 50            # common horizon H_f (env steps; = chunk size and primitive length)
    tie_margin: float = 0.0      # Commit wins unless a primitive beats it by more than this (in V units)


class ForkCastPolicy(BackOnTrackPolicy):
    needs_pca = False

    def __init__(self, base, cfg, device: str = "cuda"):
        cfg = cfg if isinstance(cfg, ForkCastCfg) else ForkCastCfg.from_dict(cfg)
        super().__init__(base, cfg, device)
        from retryvla.search.primitives import library

        seqs, names = library(cfg.horizon, np.random.RandomState(0), n_random=0)
        want = [p for p in cfg.primitives.split(",") if p]
        self.prim_names = want
        self.prims = np.stack([seqs[names.index(p)] for p in want]).astype(np.float32)          # [P, H, 7]
        # active length = up to the last step with arm motion or a gripper change (padding = hold)
        self.prim_len = []
        for s in self.prims:
            mov = np.nonzero(np.abs(s[:, :6]).sum(1) > 0)[0]
            self.prim_len.append(int(mov[-1] + 1) if len(mov) else 10)
        self.pending: dict = {}
        self.choice_counts = {n: 0 for n in ["commit"] + want}

    @torch.no_grad()
    def plan(self, z0, q0, tid, a_nom):
        """Score Commit + primitives at the common horizon; returns (index of the winner per episode [T], rows)."""
        T, H = a_nom.shape[0], self.cfg.horizon
        P = torch.as_tensor(self.prims, device=self.dev)[None].expand(T, -1, -1, -1)
        cand = torch.cat([a_nom[:, None], P], 1)                                         # [T, 1+P, H, 7]
        n = cand.shape[1]
        Z = z0.repeat_interleave(n, 0)
        Q = q0.repeat_interleave(n, 0)
        Tt = tid.repeat_interleave(n, 0)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            zH, qH, _, _, _ = self.wm.rollout(Z, Q, cand.reshape(T * n, H, 7))
        V = self.value(zH.float(), qH.float(), Tt).view(T, n)
        best = V.argmin(1)
        commit_ok = V[:, 0] <= V.min(1).values + self.cfg.tie_margin
        best = torch.where(commit_ok, torch.zeros_like(best), best)
        return best, T * n * (H // self.k)

    def predict_chunk(self, obs, tasks, seeds=None, noise_scale: float = 1.0):
        if len(obs) == 0:
            return self.base.predict_chunk(obs, tasks, seeds, noise_scale)
        c, Bn = self.cfg, len(obs)
        keys = self._keys if self._keys is not None and len(self._keys) == Bn else \
            [int(s) // SEED_MULT for s in (seeds or range(Bn))]
        self._keys = None
        for kk in keys:
            if kk not in self.eps:
                self.eps[kk] = _Ep(hist=deque(maxlen=c.window))
                self.pending.pop(kk, None)
        E = [self.eps[kk] for kk in keys]
        out: list = [None] * Bn
        for b, kk in enumerate(keys):                  # an executing primitive keeps control (hand back after it)
            q_ = self.pending.get(kk)
            if q_ is not None and len(q_):
                out[b] = q_[: self.n_action_steps]
                self.pending[kk] = q_[self.n_action_steps:]
        need = [b for b in range(Bn) if out[b] is None]
        sync = (lambda: torch.cuda.synchronize()) if self.dev.type == "cuda" else (lambda: None)
        if need:
            t0 = time.time()
            chunks = self.base.predict_chunk([obs[b] for b in need], [tasks[b] for b in need],
                                             None if seeds is None else [seeds[b] for b in need], noise_scale)
            sync()
            t1 = time.time()
            dec = [j for j, b in enumerate(need) if E[b].t >= c.first_decision]
            for j, b in enumerate(need):
                out[b] = np.asarray(chunks[j], np.float32)
                E[b].vla_s += (t1 - t0) / len(need)
            if dec:
                bs = [need[j] for j in dec]
                z, q = self.encode([obs[b] for b in bs])
                tid = self._tid([tasks[b] for b in bs])
                sync()
                t2 = time.time()
                a_nom = torch.as_tensor(np.stack([np.clip(chunks[j][:c.horizon], -1, 1) for j in dec]),
                                        device=self.dev)
                win, rows = self.plan(z, q, tid, a_nom)
                sync()
                t3 = time.time()
                for i, b in enumerate(bs):
                    e = E[b]
                    e.n_obs += 1
                    e.mon_s += (t2 - t1) / len(bs)
                    e.plan_s += (t3 - t2) / len(bs)
                    e.wm_rows += rows // len(bs)
                    w = int(win[i])
                    name = "commit" if w == 0 else self.prim_names[w - 1]
                    self.choice_counts[name] += 1
                    if w > 0:
                        p = self.prims[w - 1][: self.prim_len[w - 1]]
                        out[b] = p[: self.n_action_steps]
                        self.pending[keys[b]] = p[self.n_action_steps:]
                        e.n_trig += 1
                        e.trig_t.append(e.t)
                        e.v_trace.append(float(w))
        for b, e in enumerate(E):
            e.t += min(len(out[b]), self.n_action_steps)
        return out

    def stats(self) -> dict:
        s = super().stats()
        s["choice_counts"] = dict(self.choice_counts)
        return s


def make_wrapper(base, cfg: dict, device: str = "cuda"):
    cfg = dict(cfg)
    arm = cfg.pop("arm", "bot")
    return (ForkCastPolicy if arm == "forkcast" else BackOnTrackPolicy)(base, cfg, device)
