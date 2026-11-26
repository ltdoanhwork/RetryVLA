"""E1 (PLAN 'E1 ... run plan and deviations', docs/research/WM_3D_FOUNDATION.md sec. 4): WM-DAgger x 3D / contact
auxiliary heads. Same recipe as M2 `train_wm_v1` (LatentWM 6x384, AdamW 3e-4, wd 0.01, OneCycle 5 %, bs 64, 20k
steps, bf16, lam_open 1), trained from scratch; only the data and the auxiliary loss differ per arm.

python -m retryvla.wm.train_wm_e1 --arm A2 --seed 1 --out runs/e1/wm/A2_s1 [--steps 20000]

arms (collect_v1 = WM v1's data, its M2 group split: same train / val episodes as WM v1):
  A0    collect_v1
  A1    + 2,896 verif_m5b `random` bridges of the DAgger-pool instances (volume control)
  A2    + 2,896 search-selected bridges (wm_diverse / wm_noise / wm_qd; per instance 16 of 48, stratified, by rank)
  A2x3  + all 8,688 search-selected bridges
  A3    A0 + aux heads;  A4  A2 + aux heads
DAgger pool / held-out split: configs/exp/e1_split.json (group-disjoint; held-out instances are never loaded).
Normalizer = WM v1's (C^pi v2 reads WM v1-normalised latents). Aux heads (D4): one 1-query attention pool + MLP
over (z, q) -> 25-d sym (critic3d, t = 0..50 / 5; binary dims BCE, rest standardised MSE, pads masked) and 15-d
point-cloud stats (per-class centroid robot / obj / goal / other + obj xyz spread, t = 0..50 / 10), applied to the
TRUE latents and to the open-loop imagined latents (gradient into the dynamics), weight 0.1 + 0.1.
Writes <out>/wm.pt (load_wm-compatible), <out>/summary.json, <out>/DONE.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from retryvla import REPO_ROOT
from retryvla.critic.train_critic import load_wm
from retryvla.utils import setup_logging
from retryvla.wm.data import group_split, load_latents, save_ckpt
from retryvla.wm.dynamics import LatentWM, horizon_errors

log = logging.getLogger("train_wm_e1")
SPLIT = REPO_ROOT / "configs/exp/e1_split.json"
BIN = [12, 14, 16, 17, 18, 19, 21, 22, 23]
CONT = [i for i in range(25) if i not in BIN]
SEL = ("wm_diverse", "wm_noise", "wm_qd")
ARMS = {"A0": (None, False), "A1": ("random", False), "A2": ("sel1", False), "A2x3": ("sel3", False),
        "A3": (None, True), "A4": ("sel1", True)}


def pc_stats(pc: torch.Tensor, pcl: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """pc [K, T, 512, 3] fp16, pcl [K, T, 512] uint8 -> stats [K, T, 15] (4 class centroids + obj spread), mask."""
    x = pc.float()
    out, msk = [], []
    for c in range(4):
        m = (pcl == c).float()[..., None]
        n = m.sum(2)
        cen = (x * m).sum(2) / n.clamp_min(1)
        out.append(cen); msk.append((n > 0).expand(-1, -1, 3))
        if c == 1:
            sp = (((x - cen[:, :, None]) ** 2) * m).sum(2) / n.clamp_min(1)
            obj_sp, obj_m = sp.sqrt(), (n > 0).expand(-1, -1, 3)
    out.append(obj_sp); msk.append(obj_m)
    return torch.cat(out, -1), torch.cat(msk, -1)


def load_aux(dirname: str, stems: list[str], sel: list[np.ndarray | None]):
    S, ST, SM = [], [], []
    for st, ix in zip(stems, sel):
        b = torch.load(REPO_ROOT / "data/critic3d" / dirname / f"{st}.pt", weights_only=False)
        sym, pc, pcl = b["sym"], b["pc"], b["pcl"]
        if ix is not None:
            sym, pc, pcl = sym[ix], pc[ix], pcl[ix]
        s, m = pc_stats(pc, pcl)
        S.append(sym.float()); ST.append(s); SM.append(m)
    return torch.cat(S), torch.cat(ST), torch.cat(SM)


def load_verif(stems: list[str], mode: str, seed_rank: int = 0):
    """DAgger-pool candidates of verif_m5b. mode random | sel1 | sel3. Returns tensors + per-file index lists."""
    Z, Q, V, A, IX = [], [], [], [], []
    for n, st in enumerate(stems):
        d = torch.load(REPO_ROOT / "data/wm/verif_m5b" / f"{st}.pt", weights_only=False)
        src = list(d["source"])
        if mode == "random":
            ix = [k for k, s in enumerate(src) if s == "random"]
        else:
            per = {a: [k for k, s in enumerate(src) if s == a] for a in SEL}   # logged order = pick rank
            if mode == "sel3":
                ix = sum(per.values(), [])
            else:   # 16 of 48: 6 / 5 / 5, the arm that gets 6 rotates with the instance index
                quota = [5, 5, 5]
                quota[n % 3] = 6
                ix = sum([per[a][:q] for a, q in zip(SEL, quota)], [])
        ix = np.asarray(sorted(ix))
        Z.append(d["z"][ix]); Q.append(d["q"][ix].float()); V.append(d["valid"][ix].bool()); A.append(d["actions"][ix].float())
        IX.append(ix)
    return torch.cat(Z), torch.cat(Q), torch.cat(V), torch.cat(A), IX


class AuxHead(nn.Module):
    def __init__(self, dim=384, width=256, n_sym=25, n_pc=15):
        super().__init__()
        self.zp, self.qp = nn.Linear(dim, width), nn.Linear(8, width)
        self.query = nn.Parameter(torch.zeros(1, 1, width))
        self.att = nn.MultiheadAttention(width, 4, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, n_sym + n_pc))
        self.n_sym = n_sym

    def forward(self, z, q):
        x = torch.cat([self.zp(z), self.qp(q)[:, None]], 1)
        y, _ = self.att(self.query.expand(len(x), -1, -1), x, x)
        o = self.mlp(y[:, 0])
        return o[:, : self.n_sym], o[:, self.n_sym:]


def aux_terms(head, z, q, sym_t, w_t, stat_t, sm_t, sym_mu, sym_sd, st_mu, st_sd, has_pc):
    """One frame: sym loss + (if has_pc) pc-stat loss, both masked by w_t [B] (valid frame)."""
    ps, pp = head(z, q)
    ps, pp = ps.float(), pp.float()
    w = w_t.float()
    yb = sym_t[:, BIN]
    mb = (yb >= 0).float() * w[:, None]
    lb = (Fn.binary_cross_entropy_with_logits(ps[:, BIN], yb.clamp(0, 1), reduction="none") * mb).sum() / (mb.sum() + 1e-6)
    yc = (sym_t[:, CONT] - sym_mu) / sym_sd
    lc = (((ps[:, CONT] - yc) ** 2).mean(1) * w).sum() / (w.sum() + 1e-6)
    ls = lb + lc
    lp = z.new_zeros(()).float()
    if has_pc:
        m = sm_t.float() * w[:, None]
        lp = ((((pp - (stat_t - st_mu) / st_sd) ** 2) * m).sum()) / (m.sum() + 1e-6)
    return ls, lp


def loss_fn(model, head, z, q, a, m, aux, lam_open=1.0, lam_q=1.0, w_sym=0.1, w_pc=0.1):
    """M2 wm_loss (teacher-forced 1-step over the ensemble + open-loop multistep) + aux heads on true and imagined."""
    B, F = z.shape[:2]
    k = model.k
    l1 = z.new_zeros(()); lq = z.new_zeros(()); lo = z.new_zeros(())
    ls_ = z.new_zeros(()); lp_ = z.new_zeros(())
    n = 0
    zo, qo = z[:, 0], q[:, 0]
    if aux is not None:
        sym, stat, sm, nrm = aux
        s0, p0 = aux_terms(head, z[:, 0], q[:, 0], sym[:, 0], m[:, 0], stat[:, 0], sm[:, 0], *nrm, True)
        ls_, lp_ = ls_ + s0, lp_ + p0
    for f in range(F - 1):
        tgt = m[:, f + 1] & m[:, f]
        if not tgt.any():
            break
        ac = a[:, f * k:(f + 1) * k]
        _, _, _, (ze, qe) = model.step(z[:, f], q[:, f], ac)
        E, P = ze.shape[0], ze.shape[2]
        w = tgt.float()
        l1 = l1 + (((ze - z[None, :, f + 1]) ** 2).mean(-1).sum(-1) * w[None]).sum() / (w.sum() * E * P + 1e-6)
        lq = lq + (((qe - q[None, :, f + 1]) ** 2).mean(-1) * w[None]).sum() / (w.sum() * E + 1e-6)
        zo, qo, _, _ = model.step(zo, qo, ac)
        lo = lo + (((zo - z[:, f + 1]) ** 2).mean(-1).sum(-1) * w).sum() / (w.sum() * P + 1e-6)
        lo = lo + lam_q * (((qo - q[:, f + 1]) ** 2).mean(-1) * w).sum() / (w.sum() + 1e-6)
        if aux is not None:
            t = f + 1
            hp = t % 2 == 0
            ti = t // 2
            st_t = stat[:, ti] if hp else None
            sm_t = sm[:, ti] if hp else None
            s1, p1 = aux_terms(head, z[:, t], q[:, t], sym[:, t], tgt, st_t, sm_t, *nrm, hp)      # true (teacher)
            s2, p2 = aux_terms(head, zo, qo, sym[:, t], tgt, st_t, sm_t, *nrm, hp)                # imagined
            ls_, lp_ = ls_ + s1 + s2, lp_ + p1 + p2
        n += 1
    n = max(n, 1)
    loss = l1 / n + lam_q * lq / n + lam_open * lo / n
    out = {"z1": (l1 / n).detach(), "q1": (lq / n).detach(), "open": (lo / n).detach()}
    if aux is not None:
        loss = loss + w_sym * ls_ / n + w_pc * lp_ / n
        out.update(sym=(ls_ / n).detach(), pc=(lp_ / n).detach())
    out["loss"] = loss
    return out


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--max_pool", type=int, default=0, help="smoke: only the first N DAgger-pool instances")
    a = ap.parse_args()
    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        log.info("DONE exists, skip"); return
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    extra, use_aux = ARMS[a.arm]
    wm1 = (REPO_ROOT / "data/wm/latest_wm.txt").read_text().strip()
    _, norm, ck1 = load_wm(wm1, "cpu")
    c1 = ck1["cfg"]
    ls = load_latents(["data/wm/collect_v1"])
    tr, va = group_split(ls, float(c1["val_frac"]), int(c1["seed"]))   # = WM v1's train / val episodes
    split = json.loads(SPLIT.read_text())
    held = set(split["heldout"])
    assert not held & set(ls.inst), "held-out instance in collect_v1"
    Z, Q, V, A = ls.z[tr], ls.q[tr], ls.valid[tr], ls.actions[tr]
    aux_parts = []
    if use_aux:   # collect_v1 aux labels, same per-file order as load_latents (sorted files, all candidates)
        stems = sorted({p.stem for p in (REPO_ROOT / "data/wm/collect_v1").glob("*.pt")})
        s, st, sm = load_aux("collect_v1", stems, [None] * len(stems))
        aux_parts.append((s[tr], st[tr], sm[tr]))
    n_extra = 0
    if extra:
        pool = [s_ for s_ in split["train_pool"]][: a.max_pool or None]
        assert not held & set(pool)
        mode = {"random": "random", "sel1": "sel1", "sel3": "sel3"}[extra]
        z2, q2, v2, a2, IX = load_verif(pool, mode)
        Z, Q, V, A = torch.cat([Z, z2]), torch.cat([Q, q2]), torch.cat([V, v2]), torch.cat([A, a2])
        n_extra = len(z2)
        if use_aux:
            aux_parts.append(load_aux("verif_m5b", pool, IX))
    N = len(Z)
    log.info("arm %s seed %d: %d train trajectories (%d collect_v1 + %d DAgger-pool), aux %s", a.arm, a.seed, N,
             len(tr), n_extra, use_aux)
    # everything on the GPU (<= ~9 GB fp16)
    Zg, Qg, Vg, Ag = Z.to(dev), Q.to(dev), V.to(dev), A.to(dev)
    zm, zs = norm.z_mean.to(dev), norm.z_std.to(dev)
    P, D = Z.shape[2], Z.shape[3]
    k = A.shape[1] // (Z.shape[1] - 1)
    model = LatentWM(dim=D, n_patch=P, k=k, width=int(c1["width"]), depth=int(c1["depth"]), heads=int(c1["heads"]),
                     n_ensemble=int(c1["n_ensemble"])).to(dev)
    head = AuxHead(D).to(dev) if use_aux else None
    params = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=float(c1["weight_decay"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    if use_aux:
        SY = torch.cat([p[0] for p in aux_parts]).to(dev)
        STt = torch.cat([p[1] for p in aux_parts]).to(dev)
        SMt = torch.cat([p[2] for p in aux_parts]).to(dev)
        vm = Vg
        sc = SY[..., CONT][vm]
        sym_mu, sym_sd = sc.mean(0), sc.std(0).clamp_min(1e-3)
        stv = STt[SMt.all(-1)] if SMt.all(-1).any() else STt.flatten(0, 1)
        st_mu, st_sd = STt.flatten(0, 1).mean(0), STt.flatten(0, 1).std(0).clamp_min(1e-3)
        nrm = (sym_mu, sym_sd, st_mu, st_sd)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    perm, pi = rng.permutation(N), 0
    hist = []
    for step in range(1, a.steps + 1):
        if pi + a.bs > N:
            perm, pi = rng.permutation(N), 0
        j = torch.as_tensor(perm[pi: pi + a.bs], device=dev)
        pi += a.bs
        z = (Zg[j].float() - zm) / zs
        q = (Qg[j] - norm.q_mean.to(dev)) / norm.q_std.to(dev)
        aux = (SY[j], STt[j], SMt[j], nrm) if use_aux else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            L = loss_fn(model, head, z, q, Ag[j], Vg[j], aux)
        opt.zero_grad(set_to_none=True)
        L["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            msg = {k_: round(float(v), 4) for k_, v in L.items()}
            log.info("step %d/%d %s | %.1f it/s", step, a.steps, msg, step / (time.time() - t0))
            hist.append({"step": step, **msg})
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            he = evaluate(model, ls, norm, va, dev)
            model.train()
            log.info("val open-loop ratio: %s", {h: round(v["ratio"], 3) for h, v in he.items()})
    ckp = out / "wm.pt"
    cfg = dict(c1, seed=a.seed, arm=a.arm, steps=a.steps)
    save_ckpt(ckp, model=model.state_dict(), norm=norm.state_dict(), P=P, D=D, k=k, cfg=cfg,
              aux_head=(head.state_dict() if head is not None else None), e1_arm=a.arm)
    summ = {"arm": a.arm, "seed": a.seed, "n_train": N, "n_collect": len(tr), "n_extra": n_extra, "aux": use_aux,
            "horizon_errors": he, "wall_s": time.time() - t0, "hist": hist, "checkpoint": str(ckp)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    (out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    log.info("saved %s (%.0f s)", ckp, time.time() - t0)


@torch.no_grad()
def evaluate(model, ls, norm, idx, dev, bs=256):
    agg: dict[int, list] = {}
    for i in range(0, len(idx), bs):
        j = torch.as_tensor(idx[i: i + bs])
        z, q = norm.z(ls.z[j].to(dev)), norm.q(ls.q[j].to(dev))
        he = horizon_errors(model, z, q, ls.actions[j].to(dev), ls.valid[j].to(dev))
        for h, v in he.items():
            agg.setdefault(h, []).append(v)
    out = {}
    for h, vs in sorted(agg.items()):
        n = sum(v["n"] for v in vs)
        out[h] = {k: sum(v[k] * v["n"] for v in vs) / n for k in ("z_mse", "copy_baseline", "q_mse")}
        out[h]["ratio"] = out[h]["z_mse"] / max(out[h]["copy_baseline"], 1e-9)
        out[h]["n"] = n
    return out


if __name__ == "__main__":
    main()
