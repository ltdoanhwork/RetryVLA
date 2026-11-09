"""ACR analysis: every arm computed from the stored planning / outcome tables (paired by construction), gates G0-G2
(pre-registered in experiments/PLAN.md "ACR"), decisions for the post-probe honest check.

python scripts/analyze_acr.py [--root data/acr/v1] [--fset proprio] [--out runs/_analysis/acr]
  -> <out>/acr.md, rows_<tier>.parquet, decision.json, decisions_<tier>.json (postprobe input)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from retryvla import REPO_ROOT
from retryvla.causal import belief as Bf
from retryvla.causal.causes import CAUSES
from retryvla.causal.probes import FEATS_ORACLE, FEATURE_SETS

FLOOR = {"grip_min": 0.002, "grip_end": 0.002, "grip_std_closed": 0.001, "eef_disp_max": 0.003,
         "eef_lift_max": 0.003, "eef_return_err": 0.003, "eef_lag": 0.003, "contact_frac": 0.1,
         "obj_disp_max": 0.003, "obj_disp_end": 0.003, "obj_lift_max": 0.003}
LAM_RISK, BETA_COST, RISK_MAX = 1.0, 0.001, 0.2
ARMS = ["no_probe", "extra_trial", "passive", "random_probe", "infogain_probe", "evi_probe", "oracle_cause",
        "hindsight"]
BUDGETS = (2, 4, 8)
B_PRIMARY = 4


def fidx(fset):
    return [FEATS_ORACLE.index(k) for k in FEATURE_SETS[fset]]


def outcome(O: np.ndarray, Ns: np.ndarray, n: np.ndarray) -> tuple[bool, int, int, list]:
    """Allocation n [F] -> (solved, hits, env steps, [(f, j)]) by look-up of variants j = 0..n[f]-1."""
    alloc = [(f, j) for f in range(len(n)) for j in range(int(n[f]))]
    hits = int(sum(O[f][j] for f, j in alloc))
    steps = int(sum(Ns[f][j] for f, j in alloc))
    return hits > 0, hits, steps, alloc


def score_case(arm, n, probe, O, Ns, case, plen, base_alloc, probes=None):
    s, h, st, alloc = outcome(O, Ns, n)
    psucc = bool(probe is not None and case["probe_success"][list(plen).index(probe)])
    st += plen.get(probe, 0) if probe is not None else 0
    return {"arm": arm, "solved": bool(s or psucc), "hits": h, "steps": st, "probe": probe, "alloc": alloc,
            "probe_success": psucc, "changed": alloc != base_alloc}


def tier_b(recs, fset, B):
    """One row per (state, case, arm) at budget B."""
    rows, idx = [], fidx(fset)
    for r in recs:
        parts = r["particles"]
        types = np.array([p["type"] for p in parts])
        P, F = len(parts), len(r["families"])
        V = Bf.beta_mean(np.asarray(r["plan_succ"], float), np.full((P, F), len(r["plan_S"][0][0])))
        R = np.asarray(r["plan_R"], float)[:, :, idx]                       # [P, A, D]
        prisk = np.asarray(r["plan_risk"], float).mean(0)                   # [A] mean over particles
        probes = r["probes"]
        floor = np.array([FLOOR[FEATS_ORACLE[i]] for i in idx])
        b0 = np.full(P, 1.0 / P)
        cap = len(r["cases"][0]["O"][0])
        allowed = [k for k, a in enumerate(probes) if a != "hold" and prisk[k] <= RISK_MAX]
        ev = {}
        for k in range(len(probes)):
            sc = Bf.scales(R[:, k], floor)
            evi, eig = Bf.evi_particles(b0, V, R[:, k], sc, B, cap)
            ev[k] = (evi, eig, sc)
        k_evi = max(allowed, key=lambda k: ev[k][0] - LAM_RISK * prisk[k] - BETA_COST * r["probe_len"][probes[k]])
        evi_obj = ev[k_evi][0] - LAM_RISK * prisk[k_evi] - BETA_COST * r["probe_len"][probes[k_evi]]
        k_ig = max(allowed, key=lambda k: ev[k][1])
        k_hold = probes.index("hold")
        n0 = Bf.allocate(b0, V, B, cap)
        for ci, c in enumerate(r["cases"]):
            O, Ns = np.asarray(c["O"], bool), np.asarray(c["O_steps"], int)
            rr = np.asarray(c["probe_R"], float)[:, idx]
            base_alloc = outcome(O, Ns, n0)[3]
            rng = np.random.default_rng(abs(hash((r["instance_id"], ci))) % (2**32))
            k_rand = int(rng.choice(allowed))

            def post_alloc(k):
                lik = Bf.kernel_lik(rr[k], R[:, k], ev[k][2])
                post = Bf.posterior(b0, lik)
                return Bf.allocate(post, V, B, cap), post

            out = {}
            out["no_probe"] = score_case("no_probe", n0, None, O, Ns, c, r["probe_len"], base_alloc)
            out["extra_trial"] = score_case("extra_trial", Bf.allocate(b0, V, B + 1, cap),
                                            None, O, Ns, c, r["probe_len"], base_alloc)
            map_ok = {}
            for arm, k in (("passive", k_hold), ("random_probe", k_rand), ("infogain_probe", k_ig),
                           ("evi_probe", k_evi)):
                if arm == "evi_probe" and evi_obj <= 0:
                    out[arm] = score_case(arm, n0, None, O, Ns, c, r["probe_len"], base_alloc)
                    map_ok[arm] = np.nan
                    continue
                n, post = post_alloc(k)
                out[arm] = score_case(arm, n, probes[k], O, Ns, c, r["probe_len"], base_alloc)
                tp = {t: post[types == t].sum() for t in np.unique(types)}
                map_ok[arm] = float(max(tp, key=tp.get) == c["h_true"]["type"])
                out[arm]["risk_real"] = float(c["probe_risk"][k])
                out[arm]["aborted"] = bool(c["probe_aborted"][k])
            bt = (types == c["h_true"]["type"]).astype(float)
            out["oracle_cause"] = score_case("oracle_cause", Bf.allocate(bt / bt.sum(), V, B, cap), None, O, Ns, c,
                                             r["probe_len"], base_alloc)
            Vt = Bf.beta_mean(O.sum(1, keepdims=True).T.astype(float), np.full((1, F), O.shape[1]))
            out["hindsight"] = score_case("hindsight", Bf.allocate(np.ones(1), Vt, B, cap), None, O, Ns, c,
                                          r["probe_len"], base_alloc)
            for arm, o in out.items():
                rows.append({"instance_id": r["instance_id"], "case": ci, "h_type": c["h_true"]["type"],
                             "h_mag": c["h_true"].get("mag"), "level": r.get("level"), "B": B,
                             "map_ok": map_ok.get(arm, np.nan), "evi_best": float(ev[k_evi][0]),
                             "evi_probe_name": probes[k_evi], "ig_probe_name": probes[k_ig], **o})
    return pd.DataFrame(rows)


def groups_of(recs):
    return np.array(["__".join(r["scene"].split("__")[:2]) for r in recs])


def tier_a(recs, fset, B, n_folds=5, seed=0):
    """Cross-fitted (group k-fold by failure episode) V(cause, f), b0 and response -> cause classifiers."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    idx = fidx(fset)
    C = len(CAUSES)
    y = np.array([CAUSES.index(r["cause"]) for r in recs])
    g = groups_of(recs)
    ug = np.random.default_rng(seed).permutation(np.unique(g))
    fold = {gg: i % n_folds for i, gg in enumerate(ug)}
    fo = np.array([fold[x] for x in g])
    probes = recs[0]["probes"]
    A = len(probes)
    O = [np.asarray(r["cases"][0]["O"], bool) for r in recs]
    Ns = [np.asarray(r["cases"][0]["O_steps"], int) for r in recs]
    proprio0 = np.array([np.asarray(r["g0"])[[7, 8, 9, 10]] for r in recs])     # eef xyz + gripper opening
    X = [np.concatenate([proprio0, np.array([np.asarray(r["cases"][0]["probe_R"])[k, idx] for r in recs])], 1)
         for k in range(A)]
    F = O[0].shape[0]
    cap = O[0].shape[1]

    def fit_V(tr):
        s, n = np.zeros((C, F)), np.zeros((C, F))
        for i in tr:
            s[y[i]] += O[i].sum(1)
            n[y[i]] += O[i].shape[1]
        b0 = np.bincount(y[tr], minlength=C) + 0.5
        return Bf.beta_mean(s, n), b0 / b0.sum()

    def clf_post(tr, te, k):
        cl = sorted(set(y[tr]))
        P = np.zeros((len(te), C))
        if len(cl) == 1:
            P[:, cl[0]] = 1
            return P
        sc = StandardScaler().fit(X[k][tr])
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[k][tr]), y[tr])
        P[:, m.classes_] = m.predict_proba(sc.transform(X[k][te]))
        return P

    rows = []
    for f_ in range(n_folds):
        te, tr = np.nonzero(fo == f_)[0], np.nonzero(fo != f_)[0]
        V, b0 = fit_V(tr)
        n0 = Bf.allocate(b0, V, B, cap)
        # inner cross-fitted posteriors of the TRAINING instances (EVI predictive distribution)
        inner = {k: np.zeros((len(tr), C)) for k in range(A)}
        fo_tr = fo[tr]
        for f2 in np.unique(fo_tr):
            a_, b_ = np.nonzero(fo_tr != f2)[0], np.nonzero(fo_tr == f2)[0]
            for k in range(A):
                inner[k][b_] = clf_post(tr[a_], tr[b_], k)
        w = np.full(len(tr), 1.0 / len(tr))
        ev = {k: Bf.evi_samples(b0, V, inner[k], y[tr], w, B, cap) for k in range(A)}
        prisk = np.array([np.mean([recs[i]["cases"][0]["probe_risk"][k] for i in tr]) for k in range(A)])
        plen = recs[0]["probe_len"]
        allowed = [k for k in range(A) if probes[k] != "hold" and prisk[k] <= RISK_MAX]
        k_evi = max(allowed, key=lambda k: ev[k][0] - LAM_RISK * prisk[k] - BETA_COST * plen[probes[k]])
        evi_obj = ev[k_evi][0] - LAM_RISK * prisk[k_evi] - BETA_COST * plen[probes[k_evi]]
        k_ig = max(allowed, key=lambda k: ev[k][1])
        k_hold = probes.index("hold")
        posts = {k: clf_post(tr, te, k) for k in {k_evi, k_ig, k_hold} | set(allowed)}
        for t_i, i in enumerate(te):
            c = recs[i]["cases"][0]
            base_alloc = outcome(O[i], Ns[i], n0)[3]
            rng = np.random.default_rng(abs(hash((recs[i]["instance_id"], 7))) % (2**32))
            k_rand = int(rng.choice(allowed))
            out = {"no_probe": score_case("no_probe", n0, None, O[i], Ns[i], c, plen, base_alloc),
                   "extra_trial": score_case("extra_trial", Bf.allocate(b0, V, B + 1, cap), None, O[i], Ns[i], c,
                                             plen, base_alloc)}
            map_ok = {}
            for arm, k in (("passive", k_hold), ("random_probe", k_rand), ("infogain_probe", k_ig),
                           ("evi_probe", k_evi)):
                if arm == "evi_probe" and evi_obj <= 0:
                    out[arm] = score_case(arm, n0, None, O[i], Ns[i], c, plen, base_alloc)
                    map_ok[arm] = np.nan
                    continue
                p = posts[k][t_i]
                out[arm] = score_case(arm, Bf.allocate(p, V, B, cap), probes[k], O[i], Ns[i], c, plen, base_alloc)
                map_ok[arm] = float(int(np.argmax(p)) == y[i])
                out[arm]["risk_real"] = float(c["probe_risk"][k])
                out[arm]["aborted"] = bool(c["probe_aborted"][k])
            bt = np.eye(C)[y[i]]
            out["oracle_cause"] = score_case("oracle_cause", Bf.allocate(bt, V, B, cap), None, O[i], Ns[i], c, plen,
                                             base_alloc)
            Vt = Bf.beta_mean(O[i].sum(1, keepdims=True).T.astype(float), np.full((1, F), O[i].shape[1]))
            out["hindsight"] = score_case("hindsight", Bf.allocate(np.ones(1), Vt, B, cap), None, O[i], Ns[i], c,
                                          plen, base_alloc)
            for arm, o in out.items():
                rows.append({"instance_id": recs[i]["instance_id"], "case": 0, "h_type": recs[i]["cause"],
                             "level": recs[i].get("level"), "B": B, "fold": f_, "map_ok": map_ok.get(arm, np.nan),
                             "single_solved": recs[i].get("single_solved"), "evi_best": float(ev[k_evi][0]),
                             "evi_probe_name": probes[k_evi], "ig_probe_name": probes[k_ig], **o})
    return pd.DataFrame(rows)


def paired(d: pd.DataFrame, a: str, b: str, col: str = "solved") -> dict:
    """Paired McNemar (exact binomial) on solved + sign-flip on hits; unit = (instance, case)."""
    x = d[d.arm == a].set_index(["instance_id", "case"])
    y = d[d.arm == b].set_index(["instance_id", "case"]).loc[x.index]
    n01 = int(((x[col]) & (~y[col])).sum())
    n10 = int(((~x[col]) & (y[col])).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    dh = (x.hits - y.hits).to_numpy(float)
    rng = np.random.default_rng(0)
    obs = abs(dh.mean()) if len(dh) else 0.0
    flips = np.abs((dh[None] * rng.choice([-1, 1], size=(4000, len(dh)))).mean(1)) if len(dh) else np.zeros(1)
    return {"a": a, "b": b, "sa": float(x[col].mean()), "sb": float(y[col].mean()), "n01": n01, "n10": n10,
            "p": float(p), "d_hits": float(dh.mean()) if len(dh) else 0.0, "p_hits": float((flips >= obs - 1e-12).mean())}


def arm_table(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("arm")
    t = pd.DataFrame({
        "solved": g.solved.mean(), "hits": g.hits.mean(), "steps": g.steps.mean(),
        "hits_per_1k": g.hits.sum() / g.steps.sum() * 1000,
        "probe_rate": g.probe.apply(lambda s: s.notna().mean()),
        "changed": g.changed.mean(), "map_acc": g.map_ok.mean(),
        "risk_real": g.apply(lambda x: x["risk_real"].mean() if "risk_real" in x else np.nan, include_groups=False),
        "n": g.size()})
    return t.reindex([a for a in ARMS if a in t.index])


def change_gain(d: pd.DataFrame, arm: str) -> tuple[float, float, int]:
    x = d[d.arm == arm].set_index(["instance_id", "case"])
    y = d[d.arm == "no_probe"].set_index(["instance_id", "case"]).loc[x.index]
    ch = x.changed.to_numpy(bool)
    if not ch.any():
        return 0.0, float("nan"), 0
    gain = (x.solved.to_numpy(float) - y.solved.to_numpy(float))[ch]
    return float(ch.mean()), float(gain.mean()), int(ch.sum())


def decision_relevance(recs) -> float:
    """Tier B: fraction of states where argmax_f V(h, f) differs across hypothesis TYPES (planning table)."""
    out = []
    for r in recs:
        types = np.array([p["type"] for p in r["particles"]])
        S = np.asarray(r["plan_succ"], float)
        best = set()
        for t in np.unique(types):
            v = S[types == t].mean(0)
            if v.max() > 0:
                best.add(int(np.argmax(v)))
        out.append(len(best) > 1)
    return float(np.mean(out)) if out else float("nan")


def load(root: Path, tier: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((root / tier).glob("*.json"))]


def md_table(t: pd.DataFrame) -> str:
    cols = list(t.columns)
    lines = ["| arm | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for a, row in t.iterrows():
        lines.append(f"| {a} | " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/acr/v1")
    ap.add_argument("--fset", default="proprio", choices=list(FEATURE_SETS))
    ap.add_argument("--out", default="runs/_analysis/acr")
    args = ap.parse_args()
    root, out = REPO_ROOT / args.root, REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    md, dec = [f"# ACR PoC analysis (feature set: {args.fset}; root {args.root})", ""], {"fset": args.fset}
    for tier in ("B", "A"):
        recs = load(root, tier)
        if tier == "A" and len(recs) < 10:
            md.append(f"Tier A: {len(recs)} states, too few for cross-fitting (skip)")
            continue
        if not recs:
            continue
        ds = []
        for B in BUDGETS:
            ds.append(tier_b(recs, args.fset, B) if tier == "B" else tier_a(recs, args.fset, B))
        d = pd.concat(ds, ignore_index=True)
        d.to_parquet(out / f"rows_{tier}_{args.fset}.parquet")
        n_cases = d[(d.B == B_PRIMARY) & (d.arm == "no_probe")].shape[0]
        md += [f"## Tier {tier}: {len(recs)} states, {n_cases} cases", ""]
        if tier == "B":
            dr = decision_relevance(recs)
            md.append(f"Decision relevance (argmax_f V differs across hidden types, planning table): {dr:.3f}")
            dec["B_decision_relevance"] = dr
        else:
            md.append("Causes: " + ", ".join(f"{k} {v}" for k, v in pd.Series([r['cause'] for r in recs]).value_counts().items()))
        for B in BUDGETS:
            dB = d[d.B == B]
            md += ["", f"### B = {B}", "", md_table(arm_table(dB)), ""]
            tests = [paired(dB, "oracle_cause", "no_probe")] + [paired(dB, "evi_probe", b) for b in
                     ("no_probe", "passive", "random_probe", "extra_trial", "infogain_probe")] + \
                    [paired(dB, a, "no_probe") for a in ("passive", "random_probe", "infogain_probe", "extra_trial",
                                                         "hindsight")]
            md += ["| a vs b | solved a / b | n01 / n10 | McNemar p | d hits | sign-flip p |", "|---|---|---|---|---|---|"]
            md += [f"| {t['a']} vs {t['b']} | {t['sa']:.3f} / {t['sb']:.3f} | {t['n01']} / {t['n10']} | {t['p']:.3f} | "
                   f"{t['d_hits']:+.2f} | {t['p_hits']:.3f} |" for t in tests]
            cr, cg, nc = change_gain(dB, "evi_probe")
            md.append(f"\nevi_probe decision-change rate {cr:.3f} ({nc}), mean solved gain on changed {cg:+.3f}")
            per = dB.pivot_table(index="h_type", columns="arm", values="solved", aggfunc="mean")
            md += ["", "Per hidden type / cause (solved):", "", md_table(per.reindex(columns=[a for a in ARMS if a in per]).astype(float).round(3))]
            if B == B_PRIMARY:
                o = tests[0]
                g = {"oracle_gain_pt": 100 * (o["sa"] - o["sb"]), "oracle_p": o["p"]}
                ev = {t["b"]: t for t in tests[1:6]}
                g["G1"] = bool(all(ev[b]["sa"] > ev[b]["sb"] for b in ("passive", "random_probe", "extra_trial"))
                               and cr > 0 and (cg == cg and cg > 0))
                g["G2"] = bool(ev["infogain_probe"]["sa"] >= ev["infogain_probe"]["sb"] and
                               dB[dB.arm == "evi_probe"].steps.mean() <= dB[dB.arm == "infogain_probe"].steps.mean() + 1e-9)
                if tier == "B":
                    g["G0"] = bool(dec["B_decision_relevance"] >= 0.30 and g["oracle_gain_pt"] >= 10 and o["p"] < 0.10)
                g["tests"] = tests
                dec[f"tier_{tier}"] = g
                dec[f"rows_{tier}"] = [{"instance_id": r.instance_id, "case": int(r.case), "arm": r.arm,
                                        "probe": r.probe, "alloc": [list(map(int, x)) for x in r.alloc]}
                                       for r in dB[dB.arm.isin(["evi_probe", "no_probe"])].itertuples()]
                (out / f"decisions_{tier}.json").write_text(json.dumps({"rows": dec[f"rows_{tier}"]}))
        md.append("")
    (out / f"acr_{args.fset}.md").write_text("\n".join(md))
    (out / f"decision_{args.fset}.json").write_text(json.dumps({k: v for k, v in dec.items() if not k.startswith("rows_")},
                                                               indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
