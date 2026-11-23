"""Policy-conditioned discounted liveness value V_pi (arXiv 2605.11479, used as monitor + planning objective by
Back on Track, arXiv 2610.19579), re-implemented on RetryVLA's latent space.

Bellman operator (per env step):  V(s) = (1 - g) + g * min{ l(s), V(s') },  l = -1 at a goal state, +1 otherwise.
Values in [-1, 1]; LOW = close to completion under pi (1 - 2 g^k for a state k steps before the goal).

Two-stage algorithm (2605.11479 Alg. 1):
  stage 1  well-defined anchor values on successful episodes (closed form 1 - 2 g^k), +1 on every state of a
           timed-out episode; an ensemble is regressed on them -> V_bar (in a continuous state space the
           regression is what lets "similar" states of failed episodes inherit success values, their Remark 1)
  stage 2  bootstrapped target l_bar(s) = V_bar(s) (success-episode states keep the exact anchors), then fitted
           min-Bellman iteration over the dataset with a target network + prioritized replay (alpha 0.6,
           beta 0.4 -> 1), ensemble of 5 independently initialised members (Back on Track Sec. IV-A).
Frames are stored every `stride` env steps; one frame transition = `stride` Bellman steps with l held constant:
  V(s_f) = 1 - g^stride * (1 - min{ l_bar(s_f), V(s_{f+1}) }).
Goal frame (last frame of a successful episode) = -1; last frame of a timed-out episode = l_bar(s_T) (no successor).

Input = our WM latent so imagined WM states can be scored: normalized DINOv2 6x6 patch tokens per camera, average
pooled to 2x2 (8 tokens x 384) + normalized proprio (8) + task one-hot (16) = 3096 dims; MLP 5 x 512, GELU,
LayerNorm (their Appx. C architecture; their input is SigLIP2 embeddings + proprio, 1544 dims).
"""

from __future__ import annotations

import glob
import logging
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

log = logging.getLogger("bot.liveness")

N_TASK = 16


def pool_tokens(z: torch.Tensor, grid: int = 6, out: int = 2) -> torch.Tensor:
    """[..., 2*grid*grid, D] (two cameras, row-major grid) -> [..., 2*out*out, D] average-pooled."""
    sh = z.shape
    D = sh[-1]
    x = z.reshape(-1, 2, grid, grid, D).permute(0, 1, 4, 2, 3).reshape(-1, 2 * D, grid, grid)
    x = Fn.adaptive_avg_pool2d(x.float(), out)                      # [M, 2D, out, out]
    x = x.reshape(-1, 2, D, out * out).permute(0, 1, 3, 2).reshape(*sh[:-2], 2 * out * out, D)
    return x


class LivenessMLP(nn.Module):
    def __init__(self, in_dim: int, width: int = 512, depth: int = 5):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, width), nn.LayerNorm(width), nn.GELU()]
            d = width
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LivenessEnsemble(nn.Module):
    """Ensemble mean V_bar(s). Input: normalized pooled tokens zp [N, 8, D], normalized q [N, 8], task idx [N]."""

    def __init__(self, n_tok: int = 8, dim: int = 384, n_members: int = 5, width: int = 512, depth: int = 5):
        super().__init__()
        self.in_dim = n_tok * dim + 8 + N_TASK
        self.members = nn.ModuleList([LivenessMLP(self.in_dim, width, depth) for _ in range(n_members)])

    @staticmethod
    def features(zp, q, tid):
        oh = Fn.one_hot(tid.long().clamp_max(N_TASK - 1), N_TASK).float()
        return torch.cat([zp.float().flatten(1), q.float(), oh], 1)

    def all_members(self, x):
        return torch.stack([m(x) for m in self.members])             # [E, N]

    def forward(self, zp, q, tid):
        return self.all_members(self.features(zp, q, tid)).mean(0).clamp(-1, 1)


# ----------------------------------------------------------------------------------------------------- data
@dataclass
class FrameSet:
    x_zp: torch.Tensor      # [M, 8, D] raw pooled tokens (fp16)
    q: torch.Tensor         # [M, 8]
    tid: torch.Tensor       # [M]
    ep: torch.Tensor        # [M] episode index
    nxt: torch.Tensor       # [M] index of next frame (-1 = terminal)
    goal: torch.Tensor      # [M] bool: goal frame (terminal of a successful episode)
    anchor: torch.Tensor    # [M] stage-1 anchor target (1 - 2 g^k on success episodes, +1 on failures)
    succ_ep: torch.Tensor   # [M] bool: frame belongs to a successful episode
    ep_scene: list
    ep_success: np.ndarray
    vocab: dict


def load_frames(roots, gamma: float, vocab: dict | None = None) -> FrameSet:
    files = sorted(f for r in roots for f in glob.glob(f"{r}/part_*.pt"))
    if not files:
        raise FileNotFoundError(f"no rollout files under {roots}")
    eps = []
    for f in files:
        d = torch.load(f, weights_only=False)
        eps += d["episodes"] if "episodes" in d else [d]
    if vocab is None:
        vocab = {t: i for i, t in enumerate(sorted({e["task"] for e in eps}))}
    Z, Q, T, E, N, G, A, S, scenes, succ = [], [], [], [], [], [], [], [], [], []
    off = 0
    for i, e in enumerate(eps):
        zp, q, tf = e["zp"], torch.as_tensor(e["q"]).float(), np.asarray(e["t"])
        F = len(tf)
        ok = bool(e["success"])
        Z.append(zp.half()); Q.append(q)
        T.append(torch.full((F,), vocab.get(e["task"], 0)))
        E.append(torch.full((F,), i))
        nx = torch.arange(off + 1, off + F + 1)
        nx[-1] = -1
        N.append(nx)
        g = torch.zeros(F, dtype=torch.bool)
        if ok:
            g[-1] = True
            k = float(e["steps"]) - tf.astype(np.float64)                  # env steps to the goal
            A.append(torch.as_tensor(1.0 - 2.0 * gamma ** k, dtype=torch.float32))
        else:
            A.append(torch.ones(F))
        G.append(g)
        S.append(torch.full((F,), ok))
        scenes.append(e.get("scene", "")); succ.append(ok)
        off += F
    return FrameSet(torch.cat(Z), torch.cat(Q), torch.cat(T), torch.cat(E), torch.cat(N), torch.cat(G),
                    torch.cat(A), torch.cat(S), scenes, np.asarray(succ), vocab)


# ------------------------------------------------------------------------------------------------- training
class ProgressHead(nn.Module):
    """ForkCast-style progress head (2610.18650: mean-pooled tokens, LayerNorm, 2-layer MLP), here on the pooled
    WM tokens + proprio + task; label rho = t / T on successful on-policy episodes (time progress; ForkCast's label is a
    privileged object-to-goal distance, unavailable for most LIBERO-Goal predicates). Output in [0, 1]."""

    def __init__(self, n_tok: int = 8, dim: int = 384, width: int = 512):
        super().__init__()
        self.in_dim = 2 * dim + 8 + N_TASK
        self.net = nn.Sequential(nn.LayerNorm(self.in_dim), nn.Linear(self.in_dim, width), nn.GELU(),
                                 nn.Linear(width, 1))

    def features(self, zp, q, tid):
        P = zp.shape[1] // 2
        cams = torch.cat([zp[:, :P].float().mean(1), zp[:, P:].float().mean(1)], 1)
        oh = Fn.one_hot(tid.long().clamp_max(N_TASK - 1), N_TASK).float()
        return torch.cat([cams, q.float(), oh], 1)

    def forward(self, zp, q, tid):
        return torch.sigmoid(self.net(self.features(zp, q, tid)).squeeze(-1))


def train_progress(fs: "FrameSet", norm, dev, steps: int = 6000, lr: float = 3e-4, bs: int = 512, seed: int = 0,
                   val_eps: set | None = None) -> tuple[ProgressHead, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    zp, q = norm.z(fs.x_zp.float()), norm.q(fs.q)
    # label: t / T within successful episodes (frame index / last frame index of that episode)
    ep = fs.ep.numpy()
    lab = np.zeros(len(ep), np.float32)
    for e in np.unique(ep):
        m = np.nonzero(ep == e)[0]
        lab[m] = np.arange(len(m)) / max(1, len(m) - 1)
    use = fs.succ_ep.numpy() & ~np.isin(ep, list(val_eps or []))
    val = fs.succ_ep.numpy() & np.isin(ep, list(val_eps or []))
    head = ProgressHead(fs.x_zp.shape[1], fs.x_zp.shape[2]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    Z, Q, T, Y = zp.to(dev), q.to(dev), fs.tid.to(dev), torch.as_tensor(lab, device=dev)
    idx = np.nonzero(use)[0]
    if len(idx) == 0:  # no successful episode (smoke runs only): fall back to every frame
        idx = np.arange(len(ep))
    for s in range(steps):
        b = torch.as_tensor(rng.choice(idx, bs), device=dev)
        loss = Fn.mse_loss(head(Z[b], Q[b], T[b]), Y[b])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    head.eval()
    out = {}
    with torch.no_grad():
        for name, m in (("train", use), ("val", val)):
            if m.any():
                ii = torch.as_tensor(np.nonzero(m)[0], device=dev)
                p = head(Z[ii], Q[ii], T[ii]).cpu().numpy()
                out[f"progress_{name}_mae"] = float(np.abs(p - lab[m]).mean())
        f = ~fs.succ_ep.numpy()
        if f.any():
            ii = torch.as_tensor(np.nonzero(f)[0], device=dev)
            out["progress_fail_mean"] = float(head(Z[ii], Q[ii], T[ii]).mean())
    return head.cpu(), out


def _member_inputs(fs: FrameSet, norm, dev):
    zp = norm.z(fs.x_zp.float())                                             # per-dim affine: pool(norm) = norm(pool)
    q = norm.q(fs.q)
    return LivenessEnsemble.features(zp, q, fs.tid).to(dev)


def train_liveness(fs: FrameSet, norm, dev, gamma: float, stride: int, n_members: int = 5, lr: float = 3e-4,
                   bs: int = 512, steps1: int = 8000, steps2: int = 20000, target_every: int = 500,
                   per_alpha: float = 0.6, per_beta0: float = 0.4, val_eps: set | None = None, seed: int = 0,
                   log_every: int = 2000) -> tuple[LivenessEnsemble, dict]:
    torch.manual_seed(seed)
    X = _member_inputs(fs, norm, dev)
    M = len(X)
    ep = fs.ep.numpy()
    va = np.isin(ep, list(val_eps or []))
    tr_idx = np.nonzero(~va)[0]
    nxt, goal = fs.nxt.to(dev), fs.goal.to(dev)
    anchor = fs.anchor.to(dev)
    succ = fs.succ_ep.to(dev)
    gs = gamma ** stride
    ens = LivenessEnsemble(n_tok=fs.x_zp.shape[1], dim=fs.x_zp.shape[2], n_members=n_members).to(dev)
    hist = {}
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for mi, net in enumerate(ens.members):
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        # stage 1: anchors
        for s in range(steps1):
            b = torch.as_tensor(rng.choice(tr_idx, bs), device=dev)
            loss = Fn.mse_loss(net(X[b]), anchor[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            vbar = torch.cat([net(X[i:i + 8192]) for i in range(0, M, 8192)]).clamp(-1, 1)
        lbar = torch.where(succ, anchor, torch.minimum(vbar, torch.ones_like(vbar)))
        lbar = torch.where(goal, -torch.ones_like(lbar), lbar)
        # stage 2: fitted min-Bellman with target net + PER
        tgt_net = LivenessMLP(ens.in_dim).to(dev)
        tgt_net.load_state_dict(net.state_dict())
        prio = np.ones(M, np.float64)
        prio[va] = 0.0
        for s in range(steps2):
            if s % target_every == 0:
                tgt_net.load_state_dict(net.state_dict())
            p = prio ** per_alpha
            p /= p.sum()
            bi = rng.choice(M, bs, p=p)
            beta = per_beta0 + (1.0 - per_beta0) * s / max(1, steps2 - 1)
            w = (M * p[bi]) ** (-beta)
            w = torch.as_tensor(w / w.max(), device=dev, dtype=torch.float32)
            b = torch.as_tensor(bi, device=dev)
            with torch.no_grad():
                nb = nxt[b]
                has = nb >= 0
                vn = torch.ones(bs, device=dev)
                if has.any():
                    vn[has] = tgt_net(X[nb[has]]).clamp(-1, 1)
                m = torch.minimum(lbar[b], vn)
                y = torch.where(has, 1.0 - gs * (1.0 - m), lbar[b])
                y = torch.where(goal[b], -torch.ones_like(y), y)
            pred = net(X[b])
            td = pred - y
            loss = (w * td ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            prio[bi] = np.abs(td.detach().cpu().numpy()) + 1e-3
            if (s + 1) % log_every == 0:
                log.info("member %d stage2 step %d loss %.4f | %.0fs", mi, s + 1, float(loss), time.time() - t0)
    ens.eval()
    with torch.no_grad():
        V = torch.cat([ens.all_members(X[i:i + 8192]).mean(0) for i in range(0, M, 8192)]).clamp(-1, 1).cpu()
    hist.update(metrics(V.numpy(), fs, gamma, va))
    hist["train_s"] = time.time() - t0
    return ens, hist


def metrics(V: np.ndarray, fs: FrameSet, gamma: float, va_mask: np.ndarray, Ns: int = 520) -> dict:
    """2605.11479 success / failure metrics (on val frames): steps-to-goal k = log_g((1 - V) / 2)."""
    k = np.log(np.clip((1 - V) / 2, 1e-9, 1)) / math.log(gamma)
    out = {}
    for name, m in (("val", va_mask), ("train", ~va_mask)):
        s = m & fs.succ_ep.numpy()
        f = m & ~fs.succ_ep.numpy()
        # success metric: frames of successful episodes whose estimated k < remaining episode length (anchor-implied)
        kr = np.log(np.clip((1 - fs.anchor.numpy()) / 2, 1e-9, 1)) / math.log(gamma)
        out[f"{name}_success_metric"] = float((k[s] <= kr[s] + 1e-6).mean()) if s.any() else float("nan")
        out[f"{name}_failure_metric"] = float((k[f] > Ns).mean()) if f.any() else float("nan")
        # simple separability: mean V per episode outcome
        out[f"{name}_V_succ_mean"] = float(V[s].mean()) if s.any() else float("nan")
        out[f"{name}_V_fail_mean"] = float(V[f].mean()) if f.any() else float("nan")
        # AUC of -V for episode outcome at frame level (diagnostic, not in the paper)
        if s.any() and f.any():
            from retryvla.critic.value_model import auc
            out[f"{name}_frame_auc"] = auc(-V[m], fs.succ_ep.numpy()[m])
    return out
