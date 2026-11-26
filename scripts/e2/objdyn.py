#!/usr/bin/env python
"""E2 (PLAN E1/E2 run plan): ObjDyn, a small object-centric rigid 3D dynamics model on privileged sim state, as a
partner imagination whose errors should be LESS correlated with WM v1 than V-JEPA's. GPU < 1 h, offline.

python -m scripts.e2.objdyn [--steps 8000] [--out runs/_analysis/e2]

Data (data/critic3d + data/wm, same bridges): per bridge, input = 4 per-class point-cloud centroids (robot / obj /
goal / other) + obj spread + 25-d sym at t = 0, proprio q0 (8), actions [50, 7]; targets = sym at t = 5..50 and
pc stats at t = 10..50. Train: collect_v1 + verif_r1 + verif_m5b DAgger-pool instances (configs/exp/e1_split.json);
held-out: the 75 held-out instances' verif_m5b candidates (16 random + 48 WM v1-picked).
C_sym = small MLP success critic on TRUE end sym (+ task id), same training bridges.
Tests (pre-registered, WM_3D_FOUNDATION.md E2):
  (a) ObjDyn end-centroid error on WM v1-picked / random held-out candidates <= 1.2
  (b) P(fail | C_v1(imag) > 0.5 and C_sym(ObjDyn) > 0.5) <= 0.50 on WM v1-picked held-out candidates
  (c) FP@top4 per instance with J = min(C_v1, C_sym(ObjDyn)) <= FP of x-WM min (0.702 RESULTS xwm) - 0.05 = 0.652
      with success retention >= 70 % (true successes among C_v1 > 0.5 candidates kept by the veto)
  (d) precision at equal retention better than the RGR GeomHead E_geom veto (GeomValid on WM v1 imagined latents)
WM v1 imagined C^pi from runs/_analysis/wm_optimism/cache/rows.parquet (dir verif_m5b).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from retryvla import REPO_ROOT
from retryvla.utils import setup_logging
from retryvla.wm.train_wm_e1 import BIN, CONT, pc_stats

log = logging.getLogger("e2_objdyn")


def load_bank(name: str, stems: list[str]):
    X0, Q0, A, SY, ST, SM, S, TASK, IID, SRC, K = [], [], [], [], [], [], [], [], [], [], []
    for st in stems:
        fw = REPO_ROOT / "data/wm" / name / f"{st}.pt"
        fc = REPO_ROOT / "data/critic3d" / name / f"{st}.pt"
        if not fw.exists() or not fc.exists():
            continue
        w = torch.load(fw, weights_only=False); c = torch.load(fc, weights_only=False)
        if isinstance(w["actions"], list) or w["actions"].shape[1] != 50 or "q" not in w:
            continue
        n = len(c["success"])
        s, m = pc_stats(c["pc"], c["pcl"])
        X0.append(torch.cat([s[:, 0], c["sym"][:, 0].float()], -1)); Q0.append(w["q"][:, 0].float())
        A.append(w["actions"].float()); SY.append(c["sym"].float()); ST.append(s); SM.append(m)
        S.append(c["success"].bool()); TASK += [w["task"]] * n; IID += [w["instance_id"]] * n
        SRC += list(w["source"]); K += list(range(n))
    return (torch.cat(X0), torch.cat(Q0), torch.cat(A), torch.cat(SY), torch.cat(ST), torch.cat(SM), torch.cat(S),
            TASK, IID, SRC, K)


class ObjDyn(nn.Module):
    """tokens: [state(t=0) token, q0 token, 10 action-chunk tokens] -> transformer -> per chunk: sym(t) + stats."""

    def __init__(self, n_in=40, width=256, depth=4):
        super().__init__()
        self.si, self.qi, self.ai = nn.Linear(n_in, width), nn.Linear(8, width), nn.Linear(35, width)
        self.pos = nn.Parameter(torch.zeros(1, 12, width))
        L = nn.TransformerEncoderLayer(width, 4, 4 * width, batch_first=True, norm_first=True, activation="gelu")
        self.tr = nn.TransformerEncoder(L, depth)
        self.out = nn.Linear(width, 25 + 15)

    def forward(self, x0, q0, a):
        B = len(x0)
        tok = torch.cat([self.si(x0)[:, None], self.qi(q0)[:, None], self.ai(a.view(B, 10, 35))], 1) + self.pos
        h = self.tr(tok)[:, 2:]
        o = self.out(h)
        return o[..., :25], o[..., 25:]       # [B, 10, 25] = t 5..50, [B, 10, 15]


class CSym(nn.Module):
    def __init__(self, n_tasks=64):
        super().__init__()
        self.t = nn.Embedding(n_tasks, 32)
        self.m = nn.Sequential(nn.Linear(25 + 32, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 128), nn.GELU(),
                               nn.Linear(128, 1))

    def forward(self, s, t):
        return self.m(torch.cat([s, self.t(t)], -1))[:, 0]


def sym_feat(sym_raw_bin_prob, sym_cont, mu, sd):
    """critic input: binary dims as probabilities, continuous dims standardised."""
    s = torch.zeros(sym_cont.shape[0], 25, device=sym_cont.device)
    s[:, BIN] = sym_raw_bin_prob
    s[:, CONT] = sym_cont
    return s


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--out", default="runs/_analysis/e2")
    a = ap.parse_args()
    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    dev = torch.device("cuda")
    sp = json.loads((REPO_ROOT / "configs/exp/e1_split.json").read_text())
    held = set(sp["heldout"])
    hgrp = {"__".join(i.split("__")[:2]) for i in held}
    stem = lambda d: sorted(Path(f).stem for f in glob.glob(str(REPO_ROOT / "data/critic3d" / d / "*.pt")))
    tr_parts = [load_bank("collect_v1", stem("collect_v1")), load_bank("verif_m5b", sp["train_pool"])]
    if (REPO_ROOT / "data/critic3d/verif_r1").exists():
        tr_parts.append(load_bank("verif_r1", [s for s in stem("verif_r1") if "__".join(s.split("__")[:2]) not in hgrp]))
    te = load_bank("verif_m5b", sorted(held))
    cat = lambda i: torch.cat([p[i] for p in tr_parts])
    X0, Q0, A, SY, ST, SM, S = (cat(i) for i in range(7))
    TASK = sum([p[7] for p in tr_parts], [])
    vocab = {t: i for i, t in enumerate(sorted(set(TASK) | set(te[7])))}
    T = torch.as_tensor([vocab[t] for t in TASK])
    log.info("train %d bridges, held-out %d", len(X0), len(te[0]))
    # normalisers
    xm, xs = X0.mean(0), X0.std(0).clamp_min(1e-3)
    qm, qs = Q0.mean(0), Q0.std(0).clamp_min(1e-3)
    cm, cs_ = SY[..., CONT].flatten(0, 1).mean(0), SY[..., CONT].flatten(0, 1).std(0).clamp_min(1e-3)
    stm, sts = ST.flatten(0, 1).mean(0), ST.flatten(0, 1).std(0).clamp_min(1e-3)
    g = lambda x: x.to(dev)
    X0n, Q0n, Ag = g((X0 - xm) / xs), g((Q0 - qm) / qs), g(A)
    SYg, STg, SMg, Sg, Tg = g(SY), g(ST), g(SM), g(S), g(T)
    model = ObjDyn().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.steps, pct_start=0.05)
    N = len(X0n)
    rng = np.random.default_rng(0)
    for step in range(1, a.steps + 1):
        j = torch.as_tensor(rng.integers(0, N, 256), device=dev)
        ps, pst = model(X0n[j], Q0n[j], Ag[j])
        y = SYg[j, 1:]
        yb = y[..., BIN]; mb = (yb >= 0).float()
        lb = (Fn.binary_cross_entropy_with_logits(ps[..., BIN], yb.clamp(0, 1), reduction="none") * mb).sum() / mb.sum()
        lc = ((ps[..., CONT] - (y[..., CONT] - g(cm)) / g(cs_)) ** 2).mean()
        st_t = STg[j, 1:]; sm = SMg[j, 1:].float()     # pc stats at t = 10..50 -> chunks 2, 4, ..., 10 (idx 1,3,..,9)
        pp = pst[:, 1::2]
        lp = ((((pp - (st_t - g(stm)) / g(sts)) ** 2) * sm).sum() / sm.sum())
        loss = lb + lc + lp
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        if step % 1000 == 0:
            log.info("objdyn step %d loss %.4f (bin %.4f cont %.4f pc %.4f)", step, float(loss), float(lb), float(lc), float(lp))
    # C_sym on TRUE end sym of the training bridges
    yend = SYg[:, -1]
    feat_true = sym_feat(yend[:, BIN].clamp(0, 1), (yend[:, CONT] - g(cm)) / g(cs_), None, None)
    csym = CSym(len(vocab)).to(dev)
    o2 = torch.optim.AdamW(csym.parameters(), lr=1e-3, weight_decay=0.01)
    pw = (1 - Sg.float().mean()) / Sg.float().mean().clamp_min(1e-3)
    for step in range(1, 4001):
        j = torch.as_tensor(rng.integers(0, N, 512), device=dev)
        l = Fn.binary_cross_entropy_with_logits(csym(feat_true[j], Tg[j]), Sg[j].float(), pos_weight=pw)
        o2.zero_grad(); l.backward(); o2.step()
    model.eval(); csym.eval()
    # ---- held-out evaluation
    X0t, Q0t, At, SYt, STt, SMt, St, TASKt, IIDt, SRCt, Kt = te
    Tt = torch.as_tensor([vocab[t] for t in TASKt], device=dev)
    with torch.no_grad():
        ps, pst = model(g((X0t - xm) / xs), g((Q0t - qm) / qs), g(At))
        fe = sym_feat(torch.sigmoid(ps[:, -1, BIN]), ps[:, -1, CONT], None, None)
        c_sym_imag = torch.sigmoid(csym(fe, Tt)).cpu().numpy()
        ye = g(SYt[:, -1])
        c_sym_true = torch.sigmoid(csym(sym_feat(ye[:, BIN].clamp(0, 1), (ye[:, CONT] - g(cm)) / g(cs_), None, None), Tt)).cpu().numpy()
        pend = (pst[:, -1] * g(sts) + g(stm)).cpu()
    st_end, sm_end = STt[:, -1], SMt[:, -1]
    obj_err = ((pend[:, 3:6] - st_end[:, 3:6]).norm(dim=-1)).numpy()          # obj centroid error (m)
    obj_disp = ((st_end[:, 3:6] - STt[:, 0, 3:6]).norm(dim=-1)).numpy()
    copy_err = obj_disp
    df = pd.DataFrame({"instance_id": IIDt, "k": Kt, "source": SRCt, "success": St.numpy(),
                       "c_sym_imag": c_sym_imag, "c_sym_true": c_sym_true, "obj_err": obj_err, "copy_err": copy_err,
                       "grasped_true": SYt[:, -1, 17].numpy(), "grasped_pred": torch.sigmoid(ps[:, -1, 17]).cpu().numpy(),
                       "obj_valid": sm_end[:, 3].numpy()})
    # WM v1 imagined C^pi from the wm_optimism cache (same candidates: file stem = instance, k)
    rows = pd.read_parquet(REPO_ROOT / "runs/_analysis/wm_optimism/cache/rows.parquet")
    rows = rows[rows.dir == "verif_m5b"][["file", "k", "c_wm", "c_true", "fam"]].rename(columns={"file": "instance_id"})
    df = df.merge(rows, on=["instance_id", "k"], how="left")
    # RGR GeomHead veto on WM v1 imagined latents
    df["geom_valid"] = geom_veto(df, dev)
    df.to_parquet(out / "rows.parquet")
    report(df, out)


@torch.no_grad()
def geom_veto(df: pd.DataFrame, dev) -> np.ndarray:
    from retryvla.critic.train_critic import load_wm
    from retryvla.geom.features import geom_energy
    from retryvla.geom.head import load_head

    hp = (REPO_ROOT / "data/geom/latest_geom_head.txt")
    hp = hp.read_text().strip() if hp.exists() else str(REPO_ROOT / "runs/20261108-173840_RGR_geom_head/geom_head.pt")
    head = load_head(hp, dev)
    wm, norm, _ = load_wm((REPO_ROOT / "data/wm/latest_wm.txt").read_text().strip(), dev)
    out = np.ones(len(df), bool)
    for iid, g_ in df.groupby("instance_id"):
        v = torch.load(REPO_ROOT / "data/wm/verif_m5b" / f"{iid}.pt", weights_only=False)
        z, q = norm.z(v["z"].float().to(dev)), norm.q(v["q"].float().to(dev))
        lv = v["valid"].long().cumsum(1).argmax(1).to(dev)
        _, _, _, zs, qs = wm.rollout(z[:, 0], q[:, 0], v["actions"].float().to(dev))
        Zs, Qs = torch.stack(zs, 1), torch.stack(qs, 1)
        ar = torch.arange(len(z), device=dev)
        hi = (lv - 1).clamp_min(0)
        g0 = head.predict(z[:, 0], q[:, 0]).cpu().numpy()
        g1 = head.predict(Zs[ar, hi], Qs[ar, hi]).cpu().numpy()
        _, valid, _ = geom_energy(g0, g1)
        ks = g_.k.values
        out[g_.index.values] = valid[ks]
    return out


def report(df: pd.DataFrame, out: Path) -> None:
    L = ["# E2: ObjDyn (object-centric rigid 3D dynamics on privileged state) as a partner imagination\n"]
    pk = df.source != "random"
    rnd = ~pk
    ok = df.obj_valid
    e_pk, e_rn = df.obj_err[pk & ok].mean(), df.obj_err[rnd & ok].mean()
    L.append(f"(a) obj end-centroid error: WM v1-picked {e_pk * 100:.2f} cm (n {int((pk & ok).sum())}), random "
             f"{e_rn * 100:.2f} cm (n {int((rnd & ok).sum())}), ratio {e_pk / e_rn:.2f} (rule <= 1.2) -> "
             f"{'PASS' if e_pk / e_rn <= 1.2 else 'FAIL'}; copy-start baseline picked {df.copy_err[pk & ok].mean() * 100:.2f} cm")
    L.append(f"    C_sym on TRUE end sym AUC (held-out) {auc(df.c_sym_true, df.success):.3f}; on ObjDyn end sym "
             f"{auc(df.c_sym_imag, df.success):.3f}; grasped AUC (ObjDyn vs true) {auc(df.grasped_pred, df.grasped_true > 0.5):.3f}")
    corr = np.corrcoef(df.c_wm[pk] - df.c_true[pk], df.c_sym_imag[pk] - df.c_sym_true[pk])[0, 1]
    L.append(f"    error correlation on picked: corr(C_v1 imag - true, C_sym imag - true) = {corr:+.3f}")
    a1 = pk & (df.c_wm > 0.5)
    both = a1 & (df.c_sym_imag > 0.5)
    pf = 1 - df.success[both].mean()
    L.append(f"(b) P(fail | C_v1 > 0.5 & ObjDyn agrees) = {pf:.3f} (n {int(both.sum())}; C_v1 alone {1 - df.success[a1].mean():.3f}, "
             f"n {int(a1.sum())}) rule <= 0.50 -> {'PASS' if pf <= 0.5 else 'FAIL'}")
    # (c) FP@top4 per instance by J
    def fp_top4(J):
        fps, nk = [], []
        for _, g_ in df[pk].assign(J=J[pk]).groupby("instance_id"):
            t = g_.sort_values("J", ascending=False).head(4)
            t = t[t.J > 0.5]
            if len(t):
                fps.append((~t.success).sum()); nk.append(len(t))
        return sum(fps) / max(1, sum(nk)), sum(nk)
    J0 = df.c_wm.values
    J1 = np.minimum(df.c_wm.values, df.c_sym_imag.values)
    f0, n0 = fp_top4(J0); f1, n1 = fp_top4(J1)
    ret = (both & df.success).sum() / max(1, (a1 & df.success).sum())
    L.append(f"(c) FP@top4 (pred > 0.5) C_v1 {f0:.3f} (n {n0}) -> min(C_v1, C_sym(ObjDyn)) {f1:.3f} (n {n1}); "
             f"success retention {ret:.2f}; rule FP <= 0.652 and retention >= 0.70 -> "
             f"{'PASS' if f1 <= 0.652 and ret >= 0.70 else 'FAIL'}")
    gv = a1 & df.geom_valid
    prec_obj = df.success[both].mean(); prec_geo = df.success[gv].mean()
    ret_geo = (gv & df.success).sum() / max(1, (a1 & df.success).sum())
    # equal retention: threshold C_sym to match the geometry veto's retention
    ths = np.linspace(0, 1, 201)
    best = None
    for th in ths:
        m = a1 & (df.c_sym_imag > th)
        r = (m & df.success).sum() / max(1, (a1 & df.success).sum())
        if r <= ret_geo + 1e-9:
            best = (th, df.success[m].mean(), r); break
    L.append(f"(d) precision among C_v1 > 0.5: none {df.success[a1].mean():.3f}; ObjDyn agree {prec_obj:.3f} (ret {ret:.2f}); "
             f"GeomHead valid {prec_geo:.3f} (ret {ret_geo:.2f}); ObjDyn at equal retention "
             f"{'n/a' if best is None else f'{best[1]:.3f} (th {best[0]:.2f}, ret {best[2]:.2f})'} -> "
             f"{'PASS' if best is not None and best[1] > prec_geo else 'FAIL'}")
    by = df[pk].assign(gap=df.c_wm - df.c_true, gap_s=df.c_sym_imag - df.c_sym_true).groupby("fam")[["gap", "gap_s"]].mean()
    L.append("\nPer family (WM v1-picked held-out candidates): mean imag - true gap, WM v1 + C^pi vs ObjDyn + C_sym\n")
    L.append("| fam | WM v1 | ObjDyn |\n|---|---|---|")
    for f, r in by.iterrows():
        L.append(f"| {f} | {r.gap:+.3f} | {r.gap_s:+.3f} |")
    (out / "e2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def auc(s, y) -> float:
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = pd.Series(s).rank().values
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


if __name__ == "__main__":
    main()
