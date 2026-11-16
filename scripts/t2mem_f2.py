#!/usr/bin/env python
"""T2Mem follow-up F2 (pre-registered in NOTES, 2026-11-16; train only, 0 new sim).

python -m scripts.t2mem_f2 [--epochs 40] [--seeds 3]

Answers review experiments/reviews/T2MEM_review.md follow-ups 1-5:
  base      logistic recalibrator on [l_wm, h_wm] (fold-trained), used from round 1
  mem_fixed delta-rule memory on top of the FROZEN base: residual on the CURRENT score + live latent read-out
  mem_ss    latent channel only;  mem_pos  writes successes only
  gp_ucb    GP residual surrogate (kernel on h_wm);  knn_pos  bonus near verified successes
  regimes   K4 x R4, K1 x R16, K4 x R8
  prequential recalibrator trained only on its own picks;  budget reallocation (Hyperband-style)
All p-values: sign flips over failure-episode clusters.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from scripts.t2mem_poc1 import REPO, boot_ci, folds, load

log = logging.getLogger("t2mem_f2")


def csignflip(d, groups, n=20000, seed=0):
    """Sign-flip test with whole failure-episode clusters flipped together (H0: mean diff = 0)."""
    d = np.asarray(d, float)
    g = np.asarray(groups)
    u = sorted(set(g))
    s = np.array([d[g == x].sum() for x in u])
    if not np.any(s):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(s.sum())
    fl = rng.choice([-1, 1], size=(n, len(s)))
    return float((np.abs(fl @ s) >= obs - 1e-9).mean())


def feats(x):
    return torch.cat([x["lw"][:, None], x["hw"]], 1).cpu().numpy()


def fit_recal(insts, idx_lists=None, C=0.05):
    """Logistic recalibrator. idx_lists: per-instance candidate indices whose labels are visible (None = all 64)."""
    X, Y = [], []
    for j, x in enumerate(insts):
        f, y = feats(x), x["y"].cpu().numpy()
        if idx_lists is not None:
            f, y = f[idx_lists[j]], y[idx_lists[j]]
        X.append(f); Y.append(y)
    X, Y = np.concatenate(X), np.concatenate(Y)
    if Y.min() == Y.max():
        return None
    return LogisticRegression(C=C, max_iter=2000).fit(X, Y)


def recal_logit(m, x):
    if m is None:
        return x["lw"].clone()
    return torch.as_tensor(m.decision_function(feats(x)), dtype=torch.float32, device=x["lw"].device)


def replay(score0, rescore, K, R):
    """score0 [64]; rescore(picked_so_far, new_idx) -> new scores [64]. Returns per-round index tensors."""
    s = score0.detach()
    picked = torch.zeros(64, dtype=torch.bool, device=s.device)
    tb = torch.arange(64, device=s.device) * 1e-6
    rounds = []
    for r in range(R):
        sc = s.detach().clone() - tb
        sc[picked] = -1e9
        idx = torch.topk(sc, K).indices
        picked[idx] = True
        rounds.append(idx)
        if r < R - 1 and rescore is not None:
            s = rescore(torch.cat(rounds), idx)
    return rounds


def hits(x, rounds):
    return [float(x["y"][i].sum()) for i in rounds]


def set_base(x, m):
    """Attach the frozen linear recalibrator (w0 * l_wm + w_h . h + b) to an instance."""
    dev = x["lw"].device
    if m is None:
        x["bw0"], x["bwh"], x["bb"] = torch.tensor(1.0, device=dev), torch.zeros(256, device=dev), torch.tensor(0., device=dev)
    else:
        c = torch.as_tensor(m.coef_[0], dtype=torch.float32, device=dev)
        x["bw0"], x["bwh"], x["bb"] = c[0], c[1:], torch.tensor(float(m.intercept_[0]), device=dev)
    x["base"] = x["bw0"] * x["lw"] + x["hw"] @ x["bwh"] + x["bb"]


class ResMem(nn.Module):
    """Fast-weight memory ON TOP of the frozen base. W starts at 0 every instance (no learned W0, so no hidden
    static recalibration). Write = delta rule on (residual of the CURRENT score, latent correction h_true - h_wm)."""

    def __init__(self, mode, dk=32):
        super().__init__()
        self.mode, self.dk = mode, dk
        self.ln = nn.LayerNorm(256)
        self.q, self.k, self.v = (nn.Linear(256, dk, bias=False) for _ in range(3))
        self.o = nn.Linear(dk, 256, bias=False)
        nn.init.normal_(self.o.weight, std=0.01)              # live latent read-out (review Q2.2b)
        self.lab_out = nn.Parameter(torch.tensor(1.0))
        self.log_eta = nn.Parameter(torch.tensor(0.0))

    def score(self, x, W):
        Rd = F.normalize(self.q(self.ln(x["hw"])), dim=-1) @ W
        s = x["base"] + self.o(Rd[:, : self.dk]) @ x["bwh"]
        if self.mode != "ss":
            s = s + self.lab_out * Rd[:, self.dk]
        return s

    def write(self, x, W, new, cur):
        if self.mode == "pos":
            new = new[x["y"][new] > 0]
            if len(new) == 0:
                return W
        kk = F.normalize(self.k(self.ln(x["hw"][new])), dim=-1)
        v_lat = self.v(x["ht"][new] - x["hw"][new])
        v_lab = (x["y"][new] - torch.sigmoid(cur[new]).detach())[:, None]   # surprise vs CURRENT score (Q2.2a)
        if self.mode == "ss":
            v_lab = torch.zeros_like(v_lab)
        V = torch.cat([v_lat, v_lab], 1)
        return W - self.log_eta.exp() * 2 * kk.T @ (kk @ W - V) / len(new)

    def run(self, x, K, R, train=False):
        W = torch.zeros(self.dk, self.dk + 1, device=x["lw"].device)
        s = x["base"]
        picked = torch.zeros(64, dtype=torch.bool, device=s.device)
        tb = torch.arange(64, device=s.device) * 1e-6
        rounds, loss = [], 0.0
        for r in range(R):
            sc = s.detach().clone() - tb
            sc[picked] = -1e9
            idx = torch.topk(sc, K).indices
            picked[idx] = True
            rounds.append(idx)
            if r < R - 1:
                W = self.write(x, W, idx, s)
                s = self.score(x, W)
                if train:
                    loss = loss + F.binary_cross_entropy_with_logits(s[~picked], x["y"][~picked])
        return rounds, loss


def train_resmem(insts, mode, K, R, epochs, seed, dev):
    torch.manual_seed(seed)
    m = ResMem(mode).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    for ep in range(epochs):
        perm = np.random.default_rng(seed + ep).permutation(len(insts))
        for b in range(0, len(perm), 16):
            opt.zero_grad()
            bl = perm[b: b + 16]
            loss = sum(m.run(insts[i], K, R, train=True)[1] for i in bl) / len(bl)
            loss.backward()
            opt.step()
    log.info("resmem %s K%d R%d seed %d eta %.3f lab_out %.3f", mode, K, R, seed, float(m.log_eta.exp()),
             float(m.lab_out))
    return m.eval()


def kern(x, a, b, bw):
    e = F.normalize(x["hw"], dim=-1)
    return torch.exp((e[a] @ e[b].T - 1) / bw)


def make_gp(x, bw, noise, beta):
    p0 = torch.sigmoid(x["base"])
    allc = torch.arange(64, device=p0.device)

    def f(obs, new):
        Koo = kern(x, obs, obs, bw) + noise * torch.eye(len(obs), device=p0.device)
        Kco = kern(x, allc, obs, bw)
        A = torch.linalg.solve(Koo, torch.cat([(x["y"][obs] - p0[obs])[:, None], Kco.T], 1))
        mu = Kco @ A[:, 0]
        var = (1 - (Kco * A[:, 1:].T).sum(1)).clamp_min(0)
        return p0 + mu + beta * var.sqrt()
    return f


def make_knn_pos(x, lam, bw):
    p0 = torch.sigmoid(x["base"])
    allc = torch.arange(64, device=p0.device)

    def f(obs, new):
        succ = obs[x["y"][obs] > 0]
        if len(succ) == 0:
            return p0
        return p0 + lam * kern(x, allc, succ, bw).max(1).values
    return f


REGIMES = [(4, 4), (1, 16), (4, 8)]
GP_GRID = [dict(bw=b, noise=n, beta=be) for b, n, be in itertools.product([0.03, 0.1, 0.3], [0.1, 0.3], [0.0, 0.1, 0.3])]
KNN_GRID = [dict(lam=l, bw=b) for l, b in itertools.product([0.1, 0.3, 1.0, 3.0], [0.03, 0.1, 0.3])]


def run_fold(a, insts, fl, dev):
    f = a.fold
    tr = [insts[i] for i in range(len(insts)) if fl[i] != f]
    te_i = [i for i in range(len(insts)) if fl[i] == f]
    te = [insts[i] for i in te_i]
    m_all = fit_recal(tr)
    for x in te:
        set_base(x, m_all)
    gtr = sorted({x["group"] for x in tr})
    rng = np.random.default_rng(f)
    rng.shuffle(gtr)
    inner = {g: j % 3 for j, g in enumerate(gtr)}
    for j in range(3):                                         # cross-fitted base for meta-training instances
        m_j = fit_recal([x for x in tr if inner[x["group"]] != j])
        for x in tr:
            if inner[x["group"]] == j:
                set_base(x, m_j)
    out = {"iids": [x["iid"] for x in te], "groups": [x["group"] for x in te], "fold": f,
           "base_logit": [x["base"].cpu().tolist() for x in te], "res": {}}
    mems = {}
    for (K, R) in [(4, 4), (1, 16)]:
        for mode in ["fixed", "ss", "pos"]:
            mems[(mode, K, R)] = [train_resmem(tr, mode, K, R, a.epochs, 100 * f + sd, dev) for sd in range(a.seeds)]
    for (K, R) in REGIMES:
        mk = (4, 4) if (K, R) == (4, 8) else (K, R)
        res = {}
        with torch.no_grad():
            res["static"] = [hits(x, replay(x["lw"], None, K, R)) for x in te]
            res["base"] = [hits(x, replay(x["base"], None, K, R)) for x in te]
            for mode in ["fixed", "ss", "pos"]:
                res[f"mem_{mode}"] = [np.mean([hits(x, m.run(x, K, R)[0]) for m in mems[(mode, *mk)]], 0).tolist()
                                      for x in te]
            for name, grid, maker in [("gp_ucb", GP_GRID, make_gp), ("knn_pos", KNN_GRID, make_knn_pos)]:
                best, bp = -1e9, None
                for p in grid:
                    v = np.mean([sum(hits(x, replay(torch.sigmoid(x["base"]), maker(x, **p), K, R))[1:]) for x in tr])
                    if v > best:
                        best, bp = v, p
                res[name] = [hits(x, replay(torch.sigmoid(x["base"]), maker(x, **bp), K, R)) for x in te]
                out.setdefault("chosen", {})[f"{name}_K{K}R{R}"] = bp
        out["res"][f"K{K}R{R}"] = res
        log.info("fold %d regime K%d R%d done", f, K, R)
    p = REPO / a.out / f"f2_fold{f}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w"), default=float)


def prequential(insts, n_orders=3, refit_every=10, C=0.05):
    """Recalibrator trained ONLY on its own 16 picks of earlier instances (realistic label access)."""
    groups = sorted({x["group"] for x in insts})
    out = []
    for o in range(n_orders):
        rng = np.random.default_rng(100 + o)
        gord = list(rng.permutation(groups))
        order = sorted(range(len(insts)), key=lambda i: (gord.index(insts[i]["group"]), insts[i]["iid"]))
        seen, picks, m = [], [], None
        rows = []
        for t, i in enumerate(order):
            x = insts[i]
            if t % refit_every == 0 and seen:
                m = fit_recal([insts[j] for j in seen], picks, C) or m
            s = recal_logit(m, x)
            rd = replay(s, None, 4, 4)
            st = replay(x["lw"], None, 4, 4)
            rows.append({"i": i, "t": t, "group": x["group"], "rec": hits(x, rd), "static": hits(x, st)})
            seen.append(i)
            picks.append(torch.cat(rd).cpu().numpy())
        out.append(rows)
    return out


def budget_realloc(insts, logits, cap_miss=8, K=4):
    """Uniform 16 verifications vs: r1-miss instances stop at cap_miss, saved budget goes (in chunks of K) to r1-hit
    instances round-robin. Same total. Ranking fixed (no memory)."""
    ranks = [torch.argsort(l - torch.arange(64, device=l.device) * 1e-6, descending=True) for l in logits]
    ys = [x["y"][r].cpu().numpy() for x, r in zip(insts, ranks)]
    uni = [y[:16] for y in ys]
    r1hit = [y[:K].sum() > 0 for y in ys]
    alloc = [16 if h else cap_miss for h in r1hit]
    saved = sum(16 - a_ for a_ in alloc)
    hit_idx = [i for i, h in enumerate(r1hit) if h]
    j = 0
    while saved >= K and hit_idx:
        i = hit_idx[j % len(hit_idx)]
        if alloc[i] + K <= 64:
            alloc[i] += K
            saved -= K
        j += 1
        if j > 100000:
            break
    re = [y[:a_] for y, a_ in zip(ys, alloc)]
    tot_u, tot_r = sum(len(u) for u in uni), sum(len(r) for r in re)
    return {"uniform": {"verif": tot_u, "hits": float(sum(u.sum() for u in uni)), "solved": float(np.mean([u.sum() > 0 for u in uni]))},
            "realloc": {"verif": tot_r, "hits": float(sum(r.sum() for r in re)), "solved": float(np.mean([r.sum() > 0 for r in re]))}}


def merge(a, insts, dev):
    folds_ = [json.load(open(REPO / a.out / f"f2_fold{f}.json")) for f in range(4)]
    by_iid = {x["iid"]: x for x in insts}
    L = ["# T2Mem follow-up F2 (review follow-ups 1-5; train only; p = failure-episode-clustered sign flip)", ""]
    decision = {}
    for (K, R) in REGIMES:
        key = f"K{K}R{R}"
        res, groups = {}, []
        for fd in folds_:
            groups += fd["groups"]
            for arm, v in fd["res"][key].items():
                res.setdefault(arm, []).extend(v)
        L += [f"## Regime {key} ({K * R} verifications; 'after r1' = rounds 2-{R})", "",
              "| arm | hits r1 | hits after r1 | hits total | solved | d after r1 vs base [95% CI] | p |",
              "|---|---|---|---|---|---|---|"]
        base_after = np.array([sum(h[1:]) for h in res["base"]])
        base_solved = np.mean([sum(h) > 0 for h in res["base"]])
        for arm, v in res.items():
            after = np.array([sum(h[1:]) for h in v])
            d = after - base_after
            lo, hi = boot_ci(d)
            solved = np.mean([sum(h) > 0 for h in v])
            L.append(f"| {arm} | {np.mean([h[0] for h in v]):.3f} | {after.mean():.3f} | {np.mean([sum(h) for h in v]):.3f} "
                     f"| {solved:.3f} | {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {csignflip(d, groups):.3g} |")
            if arm.startswith("mem_") or arm in ("gp_ucb", "knn_pos"):
                ok = d.mean() >= 0.10 and csignflip(d, groups) < 0.10 and solved >= base_solved
                decision[f"{arm}_{key}"] = bool(ok)
        L.append("")
    prim = decision.get("mem_fixed_K4R4") or decision.get("mem_fixed_K1R16")
    sec = [k for k, v in decision.items() if v and not k.startswith("mem_fixed")]
    L += [f"**Memory rule (primary mem_fixed in K4R4 or K1R16): {'CONTINUE' if prim else 'FAIL'}**; "
          f"secondary arms passing: {sec or 'none'}.", ""]
    # prequential recalibrator
    pq = prequential(insts)
    L += ["## Recalibrator with realistic labels (prequential, own picks only, K4R4)", "",
          "| order | 2nd-half d hits r2-4 vs static | p | 2nd-half d hits r1-4 | full-stream d hits r2-4 |", "|---|---|---|---|---|"]
    gains = []
    for o, rows in enumerate(pq):
        h2 = [r for r in rows if r["t"] >= len(rows) // 2]
        d = np.array([sum(r["rec"][1:]) - sum(r["static"][1:]) for r in h2])
        dt = np.array([sum(r["rec"]) - sum(r["static"]) for r in h2])
        df = np.array([sum(r["rec"][1:]) - sum(r["static"][1:]) for r in rows])
        p = csignflip(d, [r["group"] for r in h2])
        gains.append((d.mean(), p))
        L.append(f"| {o} | {d.mean():+.3f} | {p:.3g} | {dt.mean():+.3f} | {df.mean():+.3f} |")
    lead = all(g >= 0.15 and p < 0.10 for g, p in gains)
    L += ["", f"**Lead rule (all orders: 2nd-half gain >= +0.15 and p < 0.10): {'CONTINUE' if lead else 'FAIL'}**", ""]
    # budget reallocation
    base_logits = {}
    for fd in folds_:
        for iid, bl in zip(fd["iids"], fd["base_logit"]):
            base_logits[iid] = torch.tensor(bl, device=dev)
    L += ["## Budget reallocation (r1-miss instances capped at 8, same total)", "",
          "| ranking | policy | verifications | hits | hits / verif | solved |", "|---|---|---|---|---|---|"]
    bud = {}
    for name, logits in [("static", [x["lw"] for x in insts]), ("base", [base_logits[x["iid"]] for x in insts])]:
        b = budget_realloc(insts, logits)
        bud[name] = b
        for pol, v in b.items():
            L.append(f"| {name} | {pol} | {v['verif']} | {v['hits']:.0f} | {v['hits'] / v['verif']:.4f} | {v['solved']:.3f} |")
    br = {k: (v["realloc"]["hits"] / v["realloc"]["verif"]) / (v["uniform"]["hits"] / v["uniform"]["verif"]) - 1
          for k, v in bud.items()}
    bl_ok = {k: br[k] >= 0.20 and bud[k]["uniform"]["solved"] - bud[k]["realloc"]["solved"] <= 0.02 for k in bud}
    L += ["", f"Hits / verification gain: " + ", ".join(f"{k} {v:+.1%}" for k, v in br.items())
          + f" -> **budget rule: {bl_ok}**"]
    (REPO / a.out / "f2.md").write_text("\n".join(L) + "\n")
    json.dump({"decision": decision, "memory_continue": bool(prim), "lead_continue": bool(lead), "budget": bud,
               "budget_ok": bl_ok, "prequential": pq}, open(REPO / a.out / "f2.json", "w"), default=float)
    print("\n".join(L))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/_analysis/t2mem/cache_verif_m5b.pt")
    ap.add_argument("--out", default="runs/_analysis/t2mem")
    ap.add_argument("--fold", type=int, default=-1, help="run one fold (0-3); -1 = merge")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    insts, _ = load(a.cache, dev)
    fl = folds(insts)
    if a.fold >= 0:
        run_fold(a, insts, fl, dev)
    else:
        merge(a, insts, dev)


if __name__ == "__main__":
    main()
