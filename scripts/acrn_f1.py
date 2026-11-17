"""ACR-N F1 (selector shoot-out) + F1b (probe features), offline, cross-fitted. Pre-registration: experiments/PLAN.md
"ACR-N follow-ups F1 / F1b / F2". Same table / executor / folds as scripts/analyze_acrn.py.

python scripts/acrn_f1.py [--root data/acr/n1] [--out runs/_analysis/acrn_f1] [--boot 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_acrn import FSETS, execute, folds, load, mcnemar, merged_causes, obs_x  # noqa: E402

from retryvla import REPO_ROOT  # noqa: E402

SCRIPTED = ("hover", "grasp_slow", "grasp_firm", "push_goal", "retreat", "place_goal", "pull_free")
SUBS = ("lift", "near", "none", "na")


def worst_costs(opts):
    return np.array([350 if o.startswith("wm_") else (300 if o == "vla" else 440) for o in opts], float)


def fail_cost(tr, F):
    out = []
    for o in range(F):
        st = np.concatenate([r["O_steps"][o][r["O"][o] == 0] for r in tr] + [np.array([400])])
        out.append(float(st.mean()))
    return np.array(out)


def vtable(tr, lab, C, F, nv, strength):
    """V[c, o, k] = P(variant k succeeds | variants < k of o failed, label c), Beta(strength) toward the marginal."""
    V = np.zeros((len(C), F, nv))
    for o in range(F):
        for k in range(nv):
            ok = [bool((r["O"][o, :k] <= 0).all() and r["O"][o, k] >= 0) for r in tr]
            s_all = sum(int(r["O"][o, k] == 1) for r, t in zip(tr, ok) if t)
            m = (s_all + 0.5) / (sum(ok) + 1.0)
            for ci, c in enumerate(C):
                sel = [t and lab[i] == c for i, t in enumerate(ok)]
                s = sum(int(tr[i]["O"][o, k] == 1) for i, t in enumerate(sel) if t)
                V[ci, o, k] = (s + strength * m) / (sum(sel) + strength)
    return V


def plan_belief(b, V, cost, worst, budget):
    b = b.copy()
    n = np.zeros(V.shape[1], int)
    used, seq = 0.0, []
    while True:
        sc = [(float(b @ V[:, o, n[o]]) / cost[o]) if (n[o] < V.shape[2] and used + worst[o] <= budget) else -1
              for o in range(V.shape[1])]
        o = int(np.argmax(sc))
        if sc[o] < 0:
            return seq
        seq.append(o)
        b = b * (1 - V[:, o, n[o]])
        b = b / b.sum() if b.sum() > 0 else b
        n[o] += 1
        used += cost[o]


def subgoal(r, o, j):
    name = r["options"][o]
    if name not in SCRIPTED:
        return "na"
    s = np.asarray(r["O_sub"], float)[o, j]
    if np.isnan(s).all():
        return "na"
    if s[0] >= 0.03:
        return "lift"
    if s[1] < 0.04:
        return "near"
    return "none"


def run_seq_policy(r, score0, decay, cost, worst, budget, mult=None):
    """Outcome-informed sequential execution. score0 [F] = initial per-option success prob; decay [F, nv];
    mult: dict (o, s) -> [F] multiplier applied to the other options after a failed trial of o with sub-goal s."""
    F, nv = r["O"].shape
    n = np.zeros(F, int)
    m = np.ones(F)
    used, trials = 0, []
    first = None
    while True:
        sc = np.array([(score0[o] * decay[o, n[o]] * m[o] / cost[o]) if (n[o] < nv and r["O"][o, n[o]] >= 0 and
                       used + worst[o] <= budget) else -1 for o in range(F)])
        o = int(np.argmax(sc))
        if sc[o] < 0:
            return {"solved": False, "steps": used, "trials": trials, "first": first}
        j = n[o]
        n[o] += 1
        used += int(r["O_steps"][o, j])
        trials.append((o, j))
        first = o if first is None else first
        if r["O"][o, j] == 1:
            return {"solved": True, "steps": used, "trials": trials, "first": first}
        if mult is not None:
            mm = mult.get((o, subgoal(r, o, j)))
            if mm is not None:
                m = m * mm
                m[o] = 1.0 * m[o] / mm[o] if mm[o] > 0 else m[o]


def latents(recs, root):
    Z = []
    for r in recs:
        p = root / "latents" / f"{r['instance_id']}.npy"
        if not p.exists():
            return None
        Z.append(np.load(p).reshape(-1))
    return np.stack(Z)


def probe_mat(recs, fset):
    return np.stack([np.concatenate([[float(r["probes"][a]["feats"][k]) for k in FSETS[fset]]
                                     for a in sorted(r["probes"])]) for r in recs])


class Predictor:
    """Per-option L2 logistic on per-trial labels; features standardised, extra blocks PCA-16 within the fold."""

    def __init__(self, tr_idx, X0, blocks, O, C=0.3):
        self.blocks = []
        Xs = [X0[tr_idx]]
        for B in blocks:
            sc = StandardScaler().fit(B[tr_idx])
            pca = PCA(n_components=min(16, len(tr_idx) - 1), random_state=0).fit(sc.transform(B[tr_idx]))
            self.blocks.append((B, sc, pca))
            Xs.append(pca.transform(sc.transform(B[tr_idx])))
        Xtr = np.concatenate(Xs, 1)
        self.sx = StandardScaler().fit(Xtr)
        self.X0 = X0
        Xtr = self.sx.transform(Xtr)
        F, nv = O.shape[1], O.shape[2]
        self.models = []
        for o in range(F):
            y = O[tr_idx, o, :]
            rows = np.repeat(np.arange(len(tr_idx)), nv)
            yy = y.reshape(-1)
            keep = yy >= 0
            if yy[keep].min() == yy[keep].max():
                self.models.append(float(yy[keep].mean()))
            else:
                self.models.append(LogisticRegression(C=C, max_iter=3000).fit(Xtr[rows[keep]], yy[keep]))

    def feats(self, i):
        x = [self.X0[i][None]] + [pca.transform(sc.transform(B[i][None])) for B, sc, pca in self.blocks]
        return self.sx.transform(np.concatenate(x, 1))

    def predict(self, i):
        x = self.feats(i)
        return np.array([m if isinstance(m, float) else m.predict_proba(x)[0, 1] for m in self.models])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/acr/n1")
    ap.add_argument("--out", default="runs/_analysis/acrn_f1")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    root = REPO_ROOT / args.root
    recs = load(root)
    causes = merged_causes(recs)
    C = sorted(set(causes))
    opts = recs[0]["options"]
    F, nv = recs[0]["O"].shape
    fo = folds(recs)
    O = np.stack([r["O"] for r in recs])
    X0 = np.stack([obs_x(r) for r in recs])
    Z = latents(recs, root)
    lat_note = "DINOv2 failure-latent PCA-16 included" if Z is not None else "NO latents found (obs_x only)"
    lat_blocks = [Z] if Z is not None else []
    worst = worst_costs(opts)
    fsets = ("proprio", "obs", "tactile", "oracle")
    PM = {fs: probe_mat(recs, fs) for fs in fsets}
    rows, preds = [], {k: np.zeros((len(recs), F)) for k in ("learned",) + tuple(f"probe_{fs}" for fs in fsets)}
    for budget in (2800, 1400):
        for k in sorted(set(fo)):
            tri = np.nonzero(fo != k)[0]
            tr = [recs[i] for i in tri]
            lab = [causes[i] for i in tri]
            cost = fail_cost(tr, F)
            # (a) fixed option chosen on training folds
            fr = [np.mean([execute(r, [o] * nv, budget, worst)["solved"] for r in tr]) for o in range(F)]
            o_fix = int(np.argmax(fr))
            Vc = vtable(tr, lab, C, F, nv, 20.0)
            Vm = vtable(tr, ["all"] * len(tr), ["all"], F, nv, 2.0)[0]
            decay = Vm / Vm[:, :1]
            top2 = list(np.argsort(-np.array(fr))[:2])
            pr = Predictor(tri, X0, lat_blocks, O)
            prb = {fs: Predictor(tri, X0, lat_blocks + [PM[fs]], O) for fs in fsets}
            # outcome multipliers m[o, s, o']
            base = O[tri].clip(0).mean((0, 2)) + 1e-3
            acc = {}
            for i in tri:
                r = recs[i]
                rate = np.array([np.mean(r["O"][o][r["O"][o] >= 0] == 1) for o in range(F)])
                for o in range(F):
                    for j in range(nv):
                        if r["O"][o, j] == 0:
                            key = (o, subgoal(r, o, j))
                            s, c = acc.get(key, (np.zeros(F), 0))
                            acc[key] = (s + rate, c + 1)
            mult = {key: ((s + 20 * base) / (c + 20)) / base for key, (s, c) in acc.items()}
            for i in np.nonzero(fo == k)[0]:
                r = recs[i]
                ci = C.index(causes[i])
                out = {"instance_id": r["instance_id"], "set": r["set"], "level": r["level"], "cause": causes[i],
                       "budget": budget, "fixed_option": opts[o_fix]}
                res = {"fixed": execute(r, [o_fix] * nv, budget, worst)}
                pm = np.full(len(C), 1e-9)
                pm[ci] = 1.0
                res["cause_s20"] = execute(r, plan_belief(pm / pm.sum(), Vc, cost, worst, budget), budget, worst)
                a_, b_ = top2 if Vc[ci, top2[0], 0] >= Vc[ci, top2[1], 0] else top2[::-1]
                res["cause_tie"] = execute(r, [a_, b_] * nv, budget, worst)
                p = pr.predict(i)
                preds["learned"][i] = p if budget == 2800 else preds["learned"][i]
                res["learned"] = run_seq_policy(r, p, decay, cost, worst, budget)
                res["learned_sub"] = run_seq_policy(r, p, decay, cost, worst, budget, mult)
                for fs in fsets:
                    pb = prb[fs].predict(i)
                    if budget == 2800:
                        preds[f"probe_{fs}"][i] = pb
                    res[f"probe_{fs}"] = run_seq_policy(r, pb, decay, cost, worst, budget)
                # ceilings
                succ = r["O"] == 1
                trm = O[tri].clip(0)
                own = succ[:, :4].sum(1) + 1e-3 * trm[:, :, :4].mean((0, 2))
                out["splithalf_oracle"] = bool(succ[int(np.argmax(own)), 4:].any())
                out["splithalf_fixed"] = bool(succ[int(np.argmax(trm[:, :, :4].mean((0, 2)))), 4:].any())
                kmax = (budget // worst).astype(int)
                out["hindsight"] = bool(any(succ[o, :kmax[o]].any() for o in range(F)))
                for name, e in res.items():
                    out[name] = bool(e["solved"])
                    out[f"{name}__steps"] = int(e["steps"])
                    f1 = e["trials"][0][0] if e["trials"] else -1
                    out[f"{name}__first"] = opts[f1] if f1 >= 0 else None
                rows.append(out)
    d = pd.DataFrame(rows)
    outd = REPO_ROOT / args.out
    outd.mkdir(parents=True, exist_ok=True)
    d.to_json(outd / "rows.jsonl", orient="records", lines=True)
    arms = ["fixed", "cause_s20", "cause_tie", "learned", "learned_sub"] + [f"probe_{fs}" for fs in fsets]
    md = [f"# ACR-N F1 / F1b ({len(recs)} instances; {lat_note}; 5 episode folds, cross-fitted)", ""]
    dec = {}
    for budget in (2800, 1400):
        db = d[d.budget == budget]
        md += [f"## Budget {budget}", "", "| arm | U solved | McNemar vs fixed (U) n01/n10, p | S solved | U+S |",
               "|---|---|---|---|---|"]
        u, s = db[db.set == "U"], db[db.set == "S"]
        for a in arms + ["splithalf_fixed", "splithalf_oracle", "hindsight"]:
            m = mcnemar(u.fixed.values.astype(bool), u[a].values.astype(bool))
            md.append(f"| {a} | {u[a].mean():.3f} ({int(u[a].sum())}) | {m[0]}/{m[1]}, {m[2]:.3f} | "
                      f"{s[a].mean():.3f} | {db[a].mean():.3f} |")
        md += ["", "fixed option chosen per fold: " + json.dumps(db.fixed_option.value_counts().to_dict()),
               "learned_sub first options (U): " + json.dumps(u.learned_sub__first.value_counts().to_dict()), ""]
        m = mcnemar(u.fixed.values.astype(bool), u.learned_sub.values.astype(bool))
        dec[str(budget)] = {"d_minus_a_U": float(u.learned_sub.mean() - u.fixed.mean()), "mcnemar": list(m),
                            "d_S": float(s.learned_sub.mean()),
                            "cause_best_U": float(max(u.cause_s20.mean(), u.cause_tie.mean())),
                            "learned_U": float(u.learned.mean())}
        for fs in fsets:
            dec[str(budget)][f"probe_{fs}_minus_learned_U"] = float(u[f"probe_{fs}"].mean() - u.learned.mean())
    p2 = dec["2800"]
    dec["selection_matters"] = bool(p2["d_minus_a_U"] >= 0.05 and p2["mcnemar"][2] < 0.10 and p2["d_S"] >= 0.95
                                    and p2["mcnemar"][0] > p2["mcnemar"][1])
    dec["cause_reopened"] = bool(all(dec[b]["cause_best_U"] >= dec[b]["learned_U"] for b in ("2800", "1400")))
    # F1b AUC ladder (pooled cells, cross-fitted predictions from the 2800 pass), cluster bootstrap over instances
    yi = [(i, o, j) for i in range(len(recs)) for o in range(F) for j in range(nv) if O[i, o, j] >= 0]
    inst = np.array([t[0] for t in yi])
    y = np.array([O[t] for t in yi])

    def auc(name, idx=None):
        p = np.array([preds[name][i, o] for i, o, _ in yi])
        sel = np.ones(len(y), bool) if idx is None else np.isin(inst, idx)
        return roc_auc_score(y[sel], p[sel])

    rng = np.random.default_rng(0)
    md += ["## F1b option-outcome AUC (pooled (instance, option, variant) cells, cross-fitted)", "",
           "| predictor | AUC all | AUC U | dAUC vs learned (all), 90 % cluster-bootstrap CI | U solved 2800 / 1400 vs learned |",
           "|---|---|---|---|---|"]
    Uidx = [i for i, r in enumerate(recs) if r["set"] == "U"]
    base_auc = auc("learned")
    ids = np.arange(len(recs))
    for name in ["learned"] + [f"probe_{fs}" for fs in fsets]:
        diffs = []
        if name != "learned":
            for _ in range(args.boot):
                b = rng.choice(ids, len(ids), replace=True)
                cnt = np.bincount(b, minlength=len(recs))
                w = cnt[inst]
                pm_ = np.array([preds[name][i, o] for i, o, _ in yi])
                pl_ = np.array([preds["learned"][i, o] for i, o, _ in yi])
                keep = w > 0
                try:
                    diffs.append(roc_auc_score(y[keep], pm_[keep], sample_weight=w[keep])
                                 - roc_auc_score(y[keep], pl_[keep], sample_weight=w[keep]))
                except ValueError:
                    pass
        a_all, a_u = auc(name), auc(name, Uidx)
        ci = (np.percentile(diffs, 5), np.percentile(diffs, 95)) if diffs else (0.0, 0.0)
        sol = " / ".join(f"{d[(d.budget == b) & (d.set == 'U')][name].mean() - d[(d.budget == b) & (d.set == 'U')].learned.mean():+.3f}"
                         for b in (2800, 1400))
        md.append(f"| {name} | {a_all:.3f} | {a_u:.3f} | {a_all - base_auc:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {sol} |")
        if name != "learned":
            fs = name.split("_", 1)[1]
            dec[f"auc_{fs}"] = {"auc": a_all, "d_auc": a_all - base_auc, "ci90": list(ci)}
    dec["probing_continues"] = bool(any(
        dec[f"auc_{fs}"]["d_auc"] >= 0.05 and dec[f"auc_{fs}"]["ci90"][0] > 0 and
        max(dec[b][f"probe_{fs}_minus_learned_U"] for b in ("2800", "1400")) >= 0.05 for fs in fsets if fs != "oracle"))
    md += ["", "## Pre-registered decisions", "", "```", json.dumps(dec, indent=1), "```"]
    (outd / "f1.md").write_text("\n".join(md))
    (outd / "decision.json").write_text(json.dumps(dec, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
