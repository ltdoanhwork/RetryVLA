#!/usr/bin/env python
"""RQ3 analysis: how optimistic is the WM + C^pi verifier when search exploits it?

python scripts/analyze_verifier.py [--rescored runs/_analysis/verifier/rescored.parquet] [--out runs/_analysis/verifier]

Inputs (0 new sim for the offline part):
  rescored.parquet   scripts/verifier_rescore.py: every sim-verified candidate of the M5b-instance logs, re-scored
                     with WM v1 + C^pi v2 (+ bootstrap-head / MC-dropout U_ens)
  RQ3 sweep runs     runs/*_RQ3_*/per_instance.jsonl (compare_arms with wm_unc_l<lam> / wm_bud_l<lam> arms)
Outputs: summary.md, curve.csv, sweep.csv, *.png.
Definitions (unit = verified candidate unless stated):
  false positive = verifier says success (C^pi > 0.5) but the sim continuation fails;  FP rate = FP / predicted pos.
  ECE            = 10-bin expected calibration error of C^pi vs sim success.
  precision@k    = fraction of the top-k (by the arm's objective) that the sim verifies.
  exploitation curve: imagination budget N (8 / 32 / 128 / 512) -> top-k precision, mean C^pi of the top-k, gap.
"""

from __future__ import annotations

import argparse
import glob
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
# imagined (WM-scored) candidates behind each existing arm's picks (pool 256 + CEM rounds; README of each arm)
IMAG = {"random": 1, "random_diverse": 1, "wm_diverse": 768, "wm_top": 768, "wm_qd": 768, "wm_noise": 768,
        "wm_goal": 768, "wm_goal_comp": 768, "wm_no_prim": 768, "wm_plan": 1536, "wm_plan_v2": 1536}
CPI_ARMS = ["random", "random_diverse", "wm_diverse", "wm_top", "wm_qd", "wm_noise", "wm_goal_comp", "wm_no_prim",
            "wm_plan"]  # ranked with C^pi v2 + WM v1 (wm_goal ranks by goal distance, value arms by other critics)


def ece(p, y, bins=10) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def auc(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = pd.Series(p).rank().to_numpy()
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def calib(p, y) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, bool)
    pos = p > 0.5
    return {"n": int(len(p)), "real_sr": float(y.mean()) if len(y) else np.nan, "pred_mean": float(p.mean()) if len(p) else np.nan,
            "gap": float(p.mean() - y.mean()) if len(p) else np.nan, "pred_pos": int(pos.sum()),
            "fp_rate": float((~y[pos]).mean()) if pos.any() else np.nan,
            "fn_rate": float(y[~pos].mean()) if (~pos).any() else np.nan,
            "ece": ece(p, y), "auc": auc(p, y)}


def boot_ci(x, n=4000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    s = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def mcnemar(a, b) -> tuple[int, int, float]:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    p = sum(comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n
    return n10, n01, float(min(1.0, 2 * p))


def signflip(d, n=10000, seed=0) -> float:
    d = np.asarray(d, float)
    if not len(d) or not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    s = np.abs((d[None] * rng.choice([-1, 1], size=(n, len(d)))).mean(1))
    return float(max((s >= obs).mean(), 1 / n))


# ---------------------------------------------------------------- offline (existing verification logs, 0 new sim)
def arm_label(r) -> str:
    return r["arm"] + ("@b32" if r["dir"] == "verif_m9_b32" else "")


def prec_at(df: pd.DataFrame, ks=(1, 4, 8, 16), by="rank") -> dict:
    """precision@k over instances: mean success of the candidates with rank < k (rank 0 = arm's top pick)."""
    out = {}
    for k in ks:
        s = df[df[by] < k]
        if len(s) and s[by].max() >= k - 1:
            out[f"p@{k}"] = float(s.success.mean())
    return out


def offline(d: pd.DataFrame, out: Path, lams=(0, 0.5, 1, 2, 4, 8), mu=1.0) -> tuple[list[str], dict]:
    md, js = [], {}
    d = d.copy()
    d["label"] = d.apply(arm_label, axis=1)
    cp = d[d.arm.isin(CPI_ARMS)]
    # ---- A. calibration / optimism per arm
    rows = []
    for lab, g in cp.groupby("label"):
        arm = lab.split("@")[0]
        rows.append({"arm": lab, "imagined": IMAG.get(arm, np.nan), "n_inst": g.instance_id.nunique(),
                     **calib(g.value, g.success), **prec_at(g)})
    A = pd.DataFrame(rows).sort_values(["imagined", "arm"])
    A.to_csv(out / "offline_arms.csv", index=False)
    js["offline_arms"] = A.to_dict("records")
    md += ["## A. Verifier calibration on existing sim-verified picks (offline, 0 new sim)", "",
           "C^pi v2 on WM v1 imagination (re-scored identically for every arm) vs the real pi0-continuation outcome. "
           "`imagined` = WM-scored candidates behind the picks (random = pool draws, no selection). FP rate = P(sim fail | "
           "C^pi > 0.5); gap = mean C^pi - real success; p@k = success of the arm's top-k picks.", "",
           "| arm | imagined | inst | n | real SR | mean C^pi | gap | pred pos | FP rate | ECE | AUC | p@1 | p@4 | p@16 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in A.to_dict("records"):
        md.append(f"| {r['arm']} | {r['imagined']:.0f} | {r['n_inst']} | {r['n']} | {r['real_sr']:.3f} | {r['pred_mean']:.3f} | "
                  f"{r['gap']:+.3f} | {r['pred_pos']} | {r['fp_rate']:.3f} | {r['ece']:.3f} | {r['auc']:.3f} | "
                  f"{r.get('p@1', np.nan):.3f} | {r.get('p@4', np.nan):.3f} | {r.get('p@16', np.nan):.3f} |")
    # sim_prefix_critic: SAME critic on the REAL end state of each segment (no WM) -> isolates critic vs WM optimism
    sp = d[d.arm == "sim_prefix_critic"]
    if len(sp):
        c_real, c_im = calib(sp.pred_logged, sp.success), calib(sp.value, sp.success)
        js["sim_prefix"] = {"critic_on_real_end_state": c_real, "critic_on_wm_imagination": c_im}
        md += ["", f"Control `sim_prefix_critic` ({len(sp)} picks, chosen by C^pi on the REAL simulated end state): C^pi on the "
                   f"real end state mean {c_real['pred_mean']:.3f}, FP rate {c_real['fp_rate']:.3f}, ECE {c_real['ece']:.3f}; the "
                   f"SAME candidates through WM imagination: mean {c_im['pred_mean']:.3f}, FP rate {c_im['fp_rate']:.3f}, ECE "
                   f"{c_im['ece']:.3f}; real SR {c_real['real_sr']:.3f}."]
    # ---- B. best-of-N exploitation curve from uniform pool draws (random arms: 16 / 32 verified per instance)
    rng = np.random.default_rng(0)
    curve = []
    for lab, Ns in (("random", (1, 2, 4, 8, 16)), ("random@b32", (1, 2, 4, 8, 16, 32))):
        g = cp[cp.label == lab]
        if g.empty:
            continue
        for lam in (0.0, 1.0, 2.0):
            for N in Ns:
                ps, ss, fp = [], [], []
                for _, gi in g.groupby("instance_id"):
                    v, s, u, r_ = gi.value.to_numpy(), gi.success.to_numpy(), gi.u_ens.to_numpy(), gi.rough.to_numpy()
                    if len(v) < N:
                        continue
                    J = v - lam * u - mu * r_
                    for _ in range(64 if N < len(v) else 1):
                        sub = rng.choice(len(v), N, replace=False)
                        b = sub[np.argmax(J[sub])]
                        ps.append(v[b]); ss.append(s[b]); fp.append(v[b] > 0.5 and not s[b])
                curve.append({"source": lab, "lam": lam, "N": N, "top1_pred": float(np.mean(ps)),
                              "top1_real": float(np.mean(ss)), "top1_fp": float(np.mean(fp)),
                              "gap": float(np.mean(ps) - np.mean(ss))})
    for lab in ("wm_top", "wm_diverse", "wm_plan", "wm_diverse@b32"):
        g = cp[(cp.label == lab) & (cp["rank"] == 0)]
        if len(g):
            curve.append({"source": lab, "lam": 0.0, "N": IMAG.get(lab.split("@")[0]), "top1_pred": float(g.value.mean()),
                          "top1_real": float(g.success.mean()), "top1_fp": float(((g.value > 0.5) & ~g.success).mean()),
                          "gap": float(g.value.mean() - g.success.mean())})
    C = pd.DataFrame(curve)
    C.to_csv(out / "offline_curve.csv", index=False)
    js["offline_curve"] = C.to_dict("records")
    md += ["", "## B. Best-of-N exploitation curve from uniform pool draws (offline)", "",
           "Per instance, N candidates drawn uniformly from the verified random picks (pool draws); the verifier keeps the "
           "argmax of J = C^pi - lambda U_ens - R; top-1 real = sim success of that pick. The 768 / 1536 rows are the top "
           "(rank-0) pick of the optimizing arms (pool + latent CEM, lambda_WM 0.5 x U_WM which is 0, see NOTES).", "",
           "| source | lambda | N imagined | top-1 C^pi | top-1 real | gap | top-1 FP |", "|---|---|---|---|---|---|---|"]
    for r in C.to_dict("records"):
        md.append(f"| {r['source']} | {r['lam']:g} | {r['N']} | {r['top1_pred']:.3f} | {r['top1_real']:.3f} | {r['gap']:+.3f} | "
                  f"{r['top1_fp']:.3f} |")
    # ---- C. does U flag the false positives?  (predicted positives of the optimizing C^pi arms)
    opt = ["wm_diverse", "wm_top", "wm_qd", "wm_noise", "wm_goal_comp", "wm_no_prim", "wm_plan"]
    pp = cp[(cp.value > 0.5) & cp.label.str.split("@").str[0].isin(opt)]
    U = {}
    for u in ("u_ens", "u_wmb", "u_crit", "u_wm", "rough"):
        if u not in pp:
            continue
        U[u] = {"auc_minus_u_for_success": auc(-pp[u], pp.success) if pp[u].std() > 0 else np.nan,
                "mean_u_tp": float(pp[u][pp.success].mean()), "mean_u_fp": float(pp[u][~pp.success].mean()),
                "median_u": float(pp[u].median())}
    js["u_flags_fp"] = {"n_pred_pos": int(len(pp)), "fp_rate": float((~pp.success).mean()) if len(pp) else np.nan, **U}
    md += ["", f"## C. Does the uncertainty flag the false positives? ({len(pp)} predicted-positive picks of the optimizing "
               f"C^pi arms, FP rate {(~pp.success).mean():.3f})", "",
           "AUC of -U for real success among predicted positives (0.5 = U says nothing about which positives are false).", "",
           "| U | AUC(-U -> success) | mean U true pos | mean U false pos | median U |", "|---|---|---|---|---|"]
    for u, v in U.items():
        md.append(f"| {u} | {v['auc_minus_u_for_success']:.3f} | {v['mean_u_tp']:.4f} | {v['mean_u_fp']:.4f} | {v['median_u']:.4f} |")
    # ---- D. offline lambda re-ranking within each instance's pooled verified candidates (selection-biased!)
    pool = cp[cp.dir.isin(["verif_m5b", "verif_m9_mech"])]
    R = []
    for lam in lams:
        hits = {k: [] for k in (1, 4, 8, 16)}
        for _, gi in pool.groupby("instance_id"):
            J = gi.value.to_numpy() - lam * gi.u_ens.to_numpy() - mu * gi.rough.to_numpy()
            s = gi.success.to_numpy()[np.argsort(-J)]
            for k in hits:
                if len(s) >= k:
                    hits[k].append(s[:k].mean())
        R.append({"lam": lam, **{f"p@{k}": float(np.mean(v)) for k, v in hits.items()}, "n_inst": len(hits[1])})
    R = pd.DataFrame(R)
    R.to_csv(out / "offline_rerank.csv", index=False)
    js["offline_rerank"] = R.to_dict("records")
    md += ["", "## D. Offline lambda re-ranking of each instance's pooled verified picks (M5b + M9 C^pi arms)", "",
           "Selection-biased (only candidates some arm already picked), so this only says whether U_ens re-orders KNOWN "
           "outcomes usefully; the online sweep (E) is the real test.", "", "| lambda | p@1 | p@4 | p@8 | p@16 | instances |",
           "|---|---|---|---|---|---|"]
    for r in R.to_dict("records"):
        md.append(f"| {r['lam']:g} | {r['p@1']:.3f} | {r['p@4']:.3f} | {r['p@8']:.3f} | {r['p@16']:.3f} | {r['n_inst']} |")
    return md, js


# ---------------------------------------------------------------- online RQ3 sweep (compare_arms wm_unc / wm_bud arms)
def load_runs(pattern: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(REPO / "runs" / pattern / "per_instance.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    d = pd.DataFrame(rows)
    return d.drop_duplicates(["instance_id", "arm"], keep="last") if len(d) else d


def explode(d: pd.DataFrame) -> pd.DataFrame:
    """One row per verified pick: arm, instance, rank inside the arm (or inside its budget), C^pi, U, budget, success."""
    out = []
    for r in d.to_dict("records"):
        if not isinstance(r.get("order_pred"), list):
            continue
        bud = r.get("order_budget") or [-1] * len(r["order_pred"])
        seen = {}
        for j, (s, p, u, b) in enumerate(zip(r["order_success"], r["order_pred"], r["order_uens"], bud)):
            rk = seen.get(b, 0); seen[b] = rk + 1
            out.append({"instance_id": r["instance_id"], "arm": r["arm"], "rank": rk, "success": bool(s), "value": p,
                        "u_ens": u, "budget": int(b)})
    return pd.DataFrame(out)


def online(out: Path, pattern: str) -> tuple[list[str], dict]:
    d = load_runs(pattern)
    if d.empty:
        return ["", "## E. Online sweep: no RQ3 runs yet"], {}
    ref = load_runs("*_M5_m5b_compare_arms_s*")
    md, js = [], {}
    P = explode(d)
    # ---- exploitation curve (wm_bud_l*: top-k per imagination budget)
    cur = []
    for (arm, N), g in P[P.budget > 0].groupby(["arm", "budget"]):
        cur.append({"arm": arm, "N": N, "n_inst": g.instance_id.nunique(), **calib(g.value, g.success),
                    **prec_at(g, ks=(1, 2, 4))})
    CU = pd.DataFrame(cur)
    if len(CU):
        CU.to_csv(out / "curve.csv", index=False)
        js["curve"] = CU.to_dict("records")
        md += ["", "## E1. Exploitation curve (online, new sims): verifier top-k vs imagination budget N", "",
               "Per instance and N: N/2 pool draws (nested) + one elite-local latent-CEM round of N/2, picks = top-4 by "
               "J = C^pi - lambda U_ens - R, all verified in sim (same continuation seeds per slot).", "",
               "| arm | N | inst | mean C^pi top-4 | real SR top-4 (= p@4) | gap | p@1 | p@2 | FP rate | ECE |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for r in CU.to_dict("records"):
            md.append(f"| {r['arm']} | {r['N']} | {r['n_inst']} | {r['pred_mean']:.3f} | {r['real_sr']:.3f} | {r['gap']:+.3f} | "
                      f"{r.get('p@1', np.nan):.3f} | {r.get('p@2', np.nan):.3f} | {r['fp_rate']:.3f} | {r['ece']:.3f} |")
    # ---- lambda sweep arms (wm_unc_l*), paired
    sw = d[d.arm.str.startswith("wm_unc_l")]
    rows, pair_rows = [], []
    if len(sw):
        base_ids = set.intersection(*[set(g.instance_id) for _, g in sw.groupby("arm")])
        both = sw[sw.instance_id.isin(base_ids)]
        refs = ref[ref.instance_id.isin(base_ids) & ref.arm.isin(["random", "wm_diverse"])] if len(ref) else ref
        allr = pd.concat([both, refs]) if len(refs) else both
        for arm, g in allr.groupby("arm"):
            pg = P[(P.arm == arm) & P.instance_id.isin(base_ids)]
            c = calib(pg.value, pg.success) if len(pg) else {}
            lo, hi = boot_ci(g.solved.astype(float))
            rows.append({"arm": arm, "n_inst": g.instance_id.nunique(), "solved": float(g.solved.mean()), "solved_lo": lo,
                         "solved_hi": hi, "hits_per_inst": float(g.hits.mean()), "hits_per_sim": float(g.hits.mean()) / 16.0,
                         "n_traj": int(g.n_written.sum()), **{f"cal_{k}": v for k, v in c.items()},
                         **(prec_at(pg, ks=(1, 4, 8, 16)) if len(pg) else {})})
        S = pd.DataFrame(rows)
        S.to_csv(out / "sweep.csv", index=False)
        js["sweep"] = S.to_dict("records")
        piv = allr.pivot_table(index="instance_id", columns="arm", values="solved", aggfunc="first")
        pih = allr.pivot_table(index="instance_id", columns="arm", values="hits", aggfunc="first")
        arms = sorted(sw.arm.unique(), key=lambda a: float(a[len("wm_unc_l"):]))
        for a in arms:
            for b in [x for x in arms if x != a and arms.index(x) < arms.index(a)] + [x for x in ("wm_diverse", "random") if x in piv]:
                m = piv[[a, b]].dropna()
                h = pih[[a, b]].dropna()
                n10, n01, p = mcnemar(m[a].astype(bool), m[b].astype(bool))
                dh = (h[a] - h[b]).to_numpy(float)
                pair_rows.append({"a": a, "b": b, "n": len(m), "solved_a": float(m[a].mean()), "solved_b": float(m[b].mean()),
                                  "only_a": n10, "only_b": n01, "mcnemar_p": p, "d_hits": float(dh.mean()),
                                  "signflip_p": signflip(dh)})
        PR = pd.DataFrame(pair_rows)
        PR.to_csv(out / "sweep_pairs.csv", index=False)
        js["sweep_pairs"] = PR.to_dict("records")
        md += ["", f"## E2. Lambda sweep (online, new sims; {len(base_ids)} train-split instances, 16 verified / arm)", "",
               "wm_unc_l<lambda> = wm_diverse pipeline (pool 256 + 2 latent-CEM rounds, diverse top-16) with objective "
               "J = C^pi - lambda U_ens - R. wm_diverse / random = M5b arms on the SAME instances (paired).", "",
               "| arm | inst | solved [95% CI] | hits / sim | p@1 | p@4 | p@16 | mean C^pi | FP rate | ECE | traj written |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in S.to_dict("records"):
            md.append(f"| {r['arm']} | {r['n_inst']} | {r['solved']:.3f} [{r['solved_lo']:.3f}, {r['solved_hi']:.3f}] | "
                      f"{r['hits_per_sim']:.3f} | {r.get('p@1', np.nan):.3f} | {r.get('p@4', np.nan):.3f} | "
                      f"{r.get('p@16', np.nan):.3f} | {r.get('cal_pred_mean', np.nan):.3f} | {r.get('cal_fp_rate', np.nan):.3f} | "
                      f"{r.get('cal_ece', np.nan):.3f} | {r['n_traj']} |")
        md += ["", "| A vs B | n | solved A / B | only A / only B | McNemar p | d hits / inst | sign-flip p |",
               "|---|---|---|---|---|---|---|"]
        for r in PR.to_dict("records"):
            md.append(f"| {r['a']} vs {r['b']} | {r['n']} | {r['solved_a']:.3f} / {r['solved_b']:.3f} | {r['only_a']} / "
                      f"{r['only_b']} | {r['mcnemar_p']:.3g} | {r['d_hits']:+.2f} | {r['signflip_p']:.3g} |")
    return md, js


# ---------------------------------------------------------------- figures
def figures(out: Path, js: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    C = pd.DataFrame(js.get("offline_curve", []))
    CU = pd.DataFrame(js.get("curve", []))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    if len(C):
        for (src, lam), g in C[C.source.str.startswith("random")].groupby(["source", "lam"]):
            if lam not in (0.0, 2.0):
                continue
            g = g.sort_values("N")
            ls = "-" if lam == 0 else ":"
            ax[0].plot(g.N, g.top1_pred, "o" + ls, c="tab:red", alpha=0.6 if src == "random" else 1,
                       label=f"C^pi of top-1 ({src}, lam {lam:g})")
            ax[0].plot(g.N, g.top1_real, "s" + ls, c="tab:blue", alpha=0.6 if src == "random" else 1,
                       label=f"sim success of top-1 ({src}, lam {lam:g})")
        opt = C[~C.source.str.startswith("random")]
        ax[0].scatter(opt.N, opt.top1_pred, marker="*", s=120, c="tab:red", label="optimizing arms: C^pi (rank-0 pick)")
        ax[0].scatter(opt.N, opt.top1_real, marker="*", s=120, c="tab:blue", label="optimizing arms: sim success")
        for r in opt.itertuples():
            ax[0].annotate(r.source, (r.N, r.top1_real), fontsize=7, xytext=(3, -10), textcoords="offset points")
    ax[0].set_xscale("log", base=2); ax[0].set_xlabel("candidates scored by the verifier (N)")
    ax[0].set_ylabel("top-1"); ax[0].set_ylim(0, 1.02); ax[0].set_title("Offline: best-of-N on existing verified picks")
    ax[0].legend(fontsize=6.5, loc="center right"); ax[0].grid(alpha=0.3)
    if len(CU):
        for arm, g in CU.groupby("arm"):
            g = g.sort_values("N")
            ax[1].plot(g.N, g.pred_mean, "o-", label=f"{arm}: mean C^pi of top-4")
            ax[1].plot(g.N, g.real_sr, "s--", label=f"{arm}: sim precision@4")
        ax[1].set_title("Online: verifier top-4 vs imagination budget (new sims)")
    else:
        ax[1].text(0.5, 0.5, "online curve: pending", ha="center", transform=ax[1].transAxes)
    ax[1].set_xscale("log", base=2); ax[1].set_xlabel("imagination budget N (8 / 32 / 128 / 512)")
    ax[1].set_ylim(0, 1.02); ax[1].legend(fontsize=6.5); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = out / "exploitation_curve.png"
    fig.savefig(p, dpi=140); plt.close(fig); made.append(p.name)
    # calibration (reliability) of C^pi on the existing picks
    A = pd.DataFrame(js.get("offline_arms", []))
    if len(A):
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.scatter(A.pred_mean, A.real_sr, s=30 + 0.05 * A.n)
        for r in A.itertuples():
            ax.annotate(r.arm, (r.pred_mean, r.real_sr), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set_xlabel("mean C^pi of the arm's verified picks (imagined)"); ax.set_ylabel("sim success of the same picks")
        ax.set_title("Verifier optimism per arm"); ax.grid(alpha=0.3); fig.tight_layout()
        p = out / "optimism_per_arm.png"
        fig.savefig(p, dpi=140); plt.close(fig); made.append(p.name)
    S = pd.DataFrame(js.get("sweep", []))
    if len(S):
        s = S[S.arm.str.startswith("wm_unc_l")].copy()
        s["lam"] = s.arm.str[len("wm_unc_l"):].astype(float)
        s = s.sort_values("lam")
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.errorbar(s.lam, s.solved, yerr=[s.solved - s.solved_lo, s.solved_hi - s.solved], fmt="o-", label="solved (>= 1 of 16)")
        ax.plot(s.lam, s.hits_per_sim, "s-", label="verified yield (hits / sim)")
        if "p@1" in s:
            ax.plot(s.lam, s["p@1"], "^-", label="precision@1")
        for ref, c in (("wm_diverse", "gray"), ("random", "lightgray")):
            r = S[S.arm == ref]
            if len(r):
                ax.axhline(float(r.solved.iloc[0]), c=c, ls=":", label=f"{ref} solved")
        ax.set_xlabel("lambda (J = C^pi - lambda U_ens - R)"); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        ax.set_title("Lambda sweep (train split)"); fig.tight_layout()
        p = out / "lambda_sweep.png"
        fig.savefig(p, dpi=140); plt.close(fig); made.append(p.name)
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescored", default="runs/_analysis/verifier/rescored.parquet")
    ap.add_argument("--out", default="runs/_analysis/verifier")
    ap.add_argument("--runs", default="*_RQ3_*")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    md = ["# RQ3: verifier optimism under search exploitation (WM v1 + C^pi v2)", "",
          "`python scripts/analyze_verifier.py` (offline part: `python -m scripts.verifier_rescore` first). Unit = verified "
          "candidate unless stated. U_ens = std over 4 bootstrap WM heads (frozen trunk) x C^pi MC-dropout (p 0.1) of "
          "C^pi(z_H^e); u_wmb = WM members only, u_crit = critic dropout only, u_wm = the WM's own head variance "
          "(identically 0: the 4 trained heads are bit-identical).", ""]
    js = {}
    rp = REPO / a.rescored
    if rp.exists():
        m, j = offline(pd.read_parquet(rp), out)
        md += m; js.update(j)
    m, j = online(out, a.runs)
    md += m; js.update(j)
    md += ["", "Figures: " + ", ".join(f"`{x}`" for x in figures(out, js))]
    for name, txt in (("summary.md", "\n".join(md) + "\n"), ("summary.json", json.dumps(js, indent=1, default=float))):
        tmp = out / (name + ".tmp")
        tmp.write_text(txt)
        tmp.replace(out / name)
    print("\n".join(md))


if __name__ == "__main__":
    main()
