#!/usr/bin/env python
"""T2Mem PoC 1, stage 2 (0 new sim): does instance-level TEST-TIME MEMORY of verified outcomes improve recovery search?

python -m scripts.t2mem_poc1 [--cache runs/_analysis/t2mem/cache_verif_m5b.pt] [--out runs/_analysis/t2mem]

Offline replay of the verification loop on the 256 TRAIN instances of verif_m5b (64 sim-verified candidates each, every
outcome and true end state known). Budget = the real one: R = 4 rounds x k = 4 picks = 16 verifications / instance.
Round 1 is identical for every method (top-4 by C^pi v2 on the WM v1-imagined end state). After each round the method
sees, for the picked candidates only, what a real verification returns: the TRUE end-state embedding and the outcome,
and re-scores the remaining candidates. Methods:
  static        no memory (C^pi on imagination; = the current pipeline's ranking)
  knn_label     explicit memory: kernel-weighted residual (y - p) of observed picks, kernel on imagined embeddings
  knn_latent    explicit memory, self-supervised: kernel-weighted (h_true - h_wm) correction, re-scored by the critic
  ttt_plain     TTT without meta-learning: linear fast weight W (init 0) fit by GD on (h_wm -> h_true - h_wm)
  t2m_ss        T2Mem-style, self-supervised: learned q / k / v / o projections, fast-weight init W0 and step size,
                inner loss || W k(h_wm) - v(h_true - h_wm) ||^2, outer loss = BCE of the critic (frozen head) on the
                not-yet-verified candidates after each write (meta-learned on the train folds)
  t2m_label     t2m_ss + a label channel in the value (y - p): verification outcome is also written
  t2m_*_static  controls: same architecture and outer training, inner step size forced to 0 (no memory write);
                any gain of t2m_* over its static control is attributable to the test-time memory itself
  oracle_true   rank by C^pi on the TRUE end state of every candidate (upper bound for latent-only correction)
Cross-validation: 4 folds by failure episode (leakage unit); hyper-parameters of knn / ttt_plain picked on the train
folds by grid; meta-learned models trained on the train folds. Everything is on the TRAIN split (verif_m5b).

Metric: hits in rounds 2-4 (12 verifications that can use memory), solved@16, per-round precision. Paired sign-flip
test over instances. PRE-REGISTERED decision (written before running): primary = t2m_label. CONTINUE to an online
PoC only if (1) t2m_label - static >= +0.03 hits / verification on rounds 2-4 with sign-flip p < 0.10, (2) t2m_label
> best explicit baseline (knn_label, knn_latent, ttt_plain), and (3) t2m_label > t2m_label_static with p < 0.10.
Otherwise: no online PoC; report as negative / redesign.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
log = logging.getLogger("t2mem_poc1")
R, K = 4, 4


# ----------------------------------------------------------------------------------------------------------- data
def load(cache: str, dev):
    c = torch.load(REPO / cache, weights_only=False)
    head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 1))
    head.load_state_dict(c["critic_head"])
    head = head.to(dev).eval()
    for p in head.parameters():
        p.requires_grad_(False)
    insts = []
    for r in c["instances"]:
        if len(r["success"]) != 64:
            continue
        insts.append({"iid": r["instance_id"], "group": r["instance_id"].split("__step")[0], "task": r["task"],
                      "hw": r["h_wm"].float().to(dev), "ht": r["h_true"].float().to(dev),
                      "lw": r["l_wm"].float().to(dev), "lt": r["l_true"].float().to(dev),
                      "y": r["success"].float().to(dev), "arm": r["arm"]})
    return insts, head


def folds(insts, n=4):
    groups = sorted({x["group"] for x in insts})
    rng = np.random.default_rng(0)
    rng.shuffle(groups)
    gf = {g: i % n for i, g in enumerate(groups)}
    return [gf[x["group"]] for x in insts]


# -------------------------------------------------------------------------------------------------------- replay
def replay(inst, rescore, state0=None):
    """Generic R x K verification loop. rescore(state, inst, obs_idx) -> (new scores [64], state). Returns picks list."""
    s = inst["lw"].detach()
    picked = torch.zeros(64, dtype=torch.bool, device=s.device)
    rounds, state = [], state0
    tb = torch.arange(64, device=s.device) * 1e-6               # deterministic tie-break
    for r in range(R):
        sc = s.detach().clone() - tb
        sc[picked] = -1e9
        idx = torch.topk(sc, K).indices
        picked[idx] = True
        rounds.append(idx)
        if r < R - 1:
            s, state = rescore(state, inst, torch.cat(rounds))
    return rounds


def hits_of(inst, rounds):
    y = inst["y"]
    per = [float(y[i].sum()) for i in rounds]
    return per


# ----------------------------------------------------------------------------------------------- explicit methods
def make_knn_label(lam, bw, tau):
    def f(state, inst, obs):
        e = F.normalize(inst["hw"], dim=-1)
        kmat = torch.exp((e @ e[obs].T - 1) / bw)                # [64, n_obs]
        res = inst["y"][obs] - torch.sigmoid(inst["lw"][obs])
        corr = (kmat * res).sum(1) / (kmat.sum(1) + tau)
        return inst["lw"] + lam * corr, state
    return f


def make_knn_latent(head, bw, tau):
    def f(state, inst, obs):
        e = F.normalize(inst["hw"], dim=-1)
        kmat = torch.exp((e @ e[obs].T - 1) / bw)
        d = inst["ht"][obs] - inst["hw"][obs]
        h = inst["hw"] + (kmat @ d) / (kmat.sum(1, keepdim=True) + tau)
        return head(h).squeeze(-1), state
    return f


def make_ttt_plain(head, eta, steps):
    def f(state, inst, obs):
        k = F.layer_norm(inst["hw"], (256,))
        W = torch.zeros(256, 256, device=k.device)
        ko, vo = k[obs], inst["ht"][obs] - inst["hw"][obs]
        for _ in range(steps):
            W = W - eta * (ko.T @ (ko @ W - vo)) * 2 / len(obs)   # GD on mean ||k W - v||^2
        h = inst["hw"] + k @ W
        return head(h).squeeze(-1), state
    return f


# --------------------------------------------------------------------------------------------- T2Mem-style memory
class FastMem(nn.Module):
    """Linear fast-weight memory f_W(x) = x W over learned projections; one GD write per round on the new picks."""

    def __init__(self, head, dk=64, label=False, static=False):
        super().__init__()
        self.head, self.label, self.static = head, label, static
        self.ln = nn.LayerNorm(256)
        self.q = nn.Linear(256, dk, bias=False)
        self.k = nn.Linear(256, dk, bias=False)
        self.v = nn.Linear(256, dk, bias=False)
        self.o = nn.Linear(dk, 256, bias=False)
        nn.init.zeros_(self.o.weight)
        dv = dk + (1 if label else 0)
        self.W0 = nn.Parameter(torch.zeros(dk, dv))
        self.log_eta = nn.Parameter(torch.tensor(math.log(0.5)))
        if label:
            self.lab_scale = nn.Parameter(torch.tensor(1.0))     # value scale of the outcome channel
            self.lab_out = nn.Parameter(torch.tensor(1.0))       # readout -> logit
        self.dk = dk

    def score(self, inst, W):
        x = self.ln(inst["hw"])
        R_ = F.normalize(self.q(x), dim=-1) @ W                  # read
        h = inst["hw"] + self.o(R_[:, : self.dk])
        logit = self.head(h).squeeze(-1)
        if self.label:
            logit = logit + self.lab_out * R_[:, self.dk]
        return logit

    def write(self, inst, W, new):
        x = self.ln(inst["hw"][new])
        kk = F.normalize(self.k(x), dim=-1)
        v = self.v(inst["ht"][new] - inst["hw"][new])
        if self.label:
            p = torch.sigmoid(inst["lw"][new]).detach()
            v = torch.cat([v, (self.lab_scale * (inst["y"][new] - p))[:, None]], 1)
        if self.static:
            return W
        eta = self.log_eta.exp()
        g = kk.T @ (kk @ W - v) * 2 / len(new)                   # d/dW mean ||kW - v||^2
        return W - eta * g

    def run(self, inst, train=False):
        """Returns rounds (picks) and the outer loss (BCE on not-yet-verified candidates after each write)."""
        W = self.W0
        s = inst["lw"]
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
                W = self.write(inst, W, idx)
                s = self.score(inst, W)
                if train:
                    m = ~picked
                    loss = loss + F.binary_cross_entropy_with_logits(s[m], inst["y"][m])
        return rounds, loss


def train_mem(head, insts, label, static, epochs, seed, dev):
    torch.manual_seed(seed)
    m = FastMem(head, label=label, static=static).to(dev)
    opt = torch.optim.AdamW([p for n, p in m.named_parameters() if not n.startswith("head.")], lr=2e-3,
                            weight_decay=1e-4)
    for ep in range(epochs):
        perm = np.random.default_rng(seed + ep).permutation(len(insts))
        tot = 0.0
        for b in range(0, len(perm), 16):
            opt.zero_grad()
            loss = sum(m.run(insts[i], train=True)[1] for i in perm[b: b + 16]) / len(perm[b: b + 16])
            loss.backward()
            opt.step()
            tot += float(loss) * len(perm[b: b + 16])
        if ep % 20 == 0 or ep == epochs - 1:
            log.info("mem label=%d static=%d ep %d loss %.4f eta %.3f", label, static, ep, tot / len(insts),
                     float(m.log_eta.exp()))
    return m.eval()


# ------------------------------------------------------------------------------------------------------- stats
def signflip(d, n=20000, seed=0):
    d = np.asarray(d, float)
    if not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    s = rng.choice([-1, 1], size=(n, len(d)))
    return float((np.abs((s * d).mean(1)) >= obs - 1e-12).mean())


def boot_ci(d, n=10000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(per):  # per: [n_inst, R] hits
    per = np.asarray(per, float)
    return {"hits_r1": per[:, 0].mean(), "hits_r24": per[:, 1:].sum(1).mean(),
            "p_per_verif_r24": per[:, 1:].sum(1).mean() / (K * (R - 1)), "solved16": (per.sum(1) > 0).mean(),
            "prec_round": (per.mean(0) / K).tolist()}


# -------------------------------------------------------------------------------------------------------- main
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/_analysis/t2mem/cache_verif_m5b.pt")
    ap.add_argument("--out", default="runs/_analysis/t2mem")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    insts, head = load(a.cache, dev)
    fl = folds(insts)
    log.info("%d instances, %d groups, base success rate %.3f", len(insts), len({x['group'] for x in insts}),
             float(torch.stack([x["y"] for x in insts]).mean()))

    grids = {
        "knn_label": [dict(lam=l, bw=b, tau=t) for l, b, t in itertools.product([1, 2, 4, 8, 16, 32],
                                                                                [0.03, 0.05, 0.1, 0.2, 0.4],
                                                                                [0.1, 0.5])],
        "knn_latent": [dict(bw=b, tau=t) for b, t in itertools.product([0.03, 0.05, 0.1, 0.2, 0.4], [0.03, 0.1, 0.5,
                                                                                                      2.0])],
        "ttt_plain": [dict(eta=e, steps=s) for e, s in itertools.product([0.001, 0.003, 0.01, 0.03, 0.1],
                                                                         [1, 5, 20, 50])],
    }
    makers = {"knn_label": lambda p: make_knn_label(**p), "knn_latent": lambda p: make_knn_latent(head, **p),
              "ttt_plain": lambda p: make_ttt_plain(head, **p)}
    res = {m: [None] * len(insts) for m in ["static", "oracle_true", *grids, "t2m_ss", "t2m_ss_static", "t2m_label",
                                            "t2m_label_static"]}
    chosen = {}
    with torch.no_grad():
        for i, x in enumerate(insts):
            res["static"][i] = hits_of(x, replay(x, lambda st, inst, obs: (inst["lw"], st)))
            res["oracle_true"][i] = hits_of(x, replay(x, lambda st, inst, obs: (inst["lt"], st)))
    for f in range(4):
        tr = [i for i in range(len(insts)) if fl[i] != f]
        te = [i for i in range(len(insts)) if fl[i] == f]
        with torch.no_grad():
            for name, grid in grids.items():
                best, bp = -1, None
                for p in grid:
                    fn = makers[name](p)
                    v = np.mean([sum(hits_of(insts[i], replay(insts[i], fn))[1:]) for i in tr])
                    if v > best:
                        best, bp = v, p
                chosen.setdefault(name, []).append(bp)
                fn = makers[name](bp)
                for i in te:
                    res[name][i] = hits_of(insts[i], replay(insts[i], fn))
        for name, label, static in [("t2m_ss", False, False), ("t2m_ss_static", False, True),
                                    ("t2m_label", True, False), ("t2m_label_static", True, True)]:
            acc = {i: [] for i in te}
            for sd in range(a.seeds):
                m = train_mem(head, [insts[i] for i in tr], label, static, a.epochs, 1000 * f + sd, dev)
                with torch.no_grad():
                    for i in te:
                        acc[i].append(hits_of(insts[i], m.run(insts[i])[0]))
            for i in te:
                res[name][i] = np.mean(acc[i], 0).tolist()       # mean over meta-training seeds
        log.info("fold %d done", f)

    S = {m: summarize(v) for m, v in res.items()}
    base = np.array([sum(h[1:]) for h in res["static"]])
    lines = ["# T2Mem PoC 1: instance-level test-time memory in recovery search (offline replay, verif_m5b train)", "",
             f"{len(insts)} instances x 64 verified candidates, R = {R} rounds x K = {K} picks (16 verifications). "
             "Round 1 identical for all methods. Hits r2-4 = successes among the 12 picks made after >= 1 memory write.",
             "", "| method | hits r1 | hits r2-4 | prec / verif r2-4 | solved@16 | prec per round | d hits r2-4 vs static "
             "[95% CI] | sign-flip p |", "|---|---|---|---|---|---|---|---|"]
    for m, s in S.items():
        d = np.array([sum(h[1:]) for h in res[m]]) - base
        lo, hi = boot_ci(d)
        lines.append(f"| {m} | {s['hits_r1']:.3f} | {s['hits_r24']:.3f} | {s['p_per_verif_r24']:.3f} | "
                     f"{s['solved16']:.3f} | {' / '.join(f'{v:.3f}' for v in s['prec_round'])} | "
                     f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {signflip(d):.3g} |")

    def dpair(a_, b_):
        d = np.array([sum(h[1:]) for h in res[a_]]) - np.array([sum(h[1:]) for h in res[b_]])
        return d.mean() / (K * (R - 1)), signflip(d)

    expl = max(["knn_label", "knn_latent", "ttt_plain"], key=lambda m: S[m]["hits_r24"])
    c1v, c1p = dpair("t2m_label", "static")
    c2v, c2p = dpair("t2m_label", expl)
    c3v, c3p = dpair("t2m_label", "t2m_label_static")
    ok = (c1v >= 0.03 and c1p < 0.10) and c2v > 0 and (c3v > 0 and c3p < 0.10)
    lines += ["", "Pre-registered decision (primary = t2m_label; per-verification differences on rounds 2-4):", "",
              f"1. vs static: {c1v:+.4f} / verif, p {c1p:.3g} (need >= +0.03 and p < 0.10) -> "
              f"{'PASS' if c1v >= 0.03 and c1p < 0.10 else 'FAIL'}",
              f"2. vs best explicit baseline ({expl}): {c2v:+.4f} / verif, p {c2p:.3g} (need > 0) -> "
              f"{'PASS' if c2v > 0 else 'FAIL'}",
              f"3. vs its static control: {c3v:+.4f} / verif, p {c3p:.3g} (need > 0 and p < 0.10) -> "
              f"{'PASS' if c3v > 0 and c3p < 0.10 else 'FAIL'}",
              "", f"**Decision: {'CONTINUE to online PoC 2' if ok else 'NO online PoC (negative / redesign)'}**", "",
              "Secondary (not part of the decision): t2m_ss vs static " + "%+.4f / verif, p %.3g" % dpair("t2m_ss", "static")
              + "; t2m_ss vs t2m_ss_static " + "%+.4f, p %.3g" % dpair("t2m_ss", "t2m_ss_static")
              + "; knn_label vs static " + "%+.4f, p %.3g" % dpair("knn_label", "static")
              + "; oracle_true vs static " + "%+.4f, p %.3g" % dpair("oracle_true", "static") + ".",
              "", "Chosen hyper-parameters per fold: " + json.dumps(chosen)]
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "poc1.md").write_text("\n".join(lines) + "\n")
    json.dump({"summary": S, "per_instance": {m: v for m, v in res.items()}, "iids": [x["iid"] for x in insts],
               "folds": fl, "chosen": chosen, "decision": ok}, open(out / "poc1.json", "w"), indent=1, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
