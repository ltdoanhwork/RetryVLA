"""ACR-N analysis (design + pre-registered rules: experiments/PLAN.md "ACR-N"). Offline; no simulation.

python scripts/analyze_acrn.py [--root data/acr/n1] [--out runs/_analysis/acrn] [--write-decisions]

Inputs: one JSON per instance (retryvla.search.acr_search tier N): oracle cause measurements, every probe's response
(observation-only proprio + visual, tactile, oracle perception), the outcome table O[option, variant] (+ steps).
Everything learned (decision table V, belief b0, probe likelihoods, probe risk) is CROSS-FITTED: 5 group folds by
failure episode; the test fold never contributes to its own model.

Sequential recovery policy (identical for every arm; arms differ only in the belief): trial k picks the option
maximising sum_c b(c) V[c, o, n_o] / cost(o) (V[c, o, k] = P(variant k of option o succeeds | its earlier variants
failed, cause c), cost = mean failed-trial steps); after a failure b(c) <- b(c) (1 - V[c, o, n_o]). A trial starts only
if used + worst-case cost(o) <= budget; actual table steps are charged; stop at the first success.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from retryvla import REPO_ROOT

PREFIX_MAX = 140
LAM_RISK = 1.0
RISK_MAX = 0.2
PROPRIO = ("grip_min", "grip_end", "grip_std_closed", "eef_disp_max", "eef_lift_max", "eef_return_err", "eef_lag")
VISUAL = ("lat_agent_end", "lat_wrist_end", "lat_agent_mid", "lat_wrist_mid", "cpi_end")
FSETS = {"obs": PROPRIO + VISUAL, "proprio": PROPRIO, "tactile": PROPRIO + VISUAL + ("contact_frac",),
         "oracle": PROPRIO + VISUAL + ("contact_frac", "obj_disp_max", "obj_disp_end", "obj_lift_max")}
MERGE = {"tipped": "contact_free", "held_ok": "held", "held_slip": "held", "stuck": "blocked",
         "obstructed": "blocked", "contact_free": "contact_free", "free": "free"}
HEAD_IDX = (2, 6, 10, 11, 12, 15)   # geometry head: obj_z, obj_upz, grip_open, d_grip_obj, finger_obj_contact, obj_lift


def load(root: Path) -> list[dict]:
    recs = [json.loads(p.read_text()) for p in sorted((root / "N").glob("*.json"))]
    for r in recs:
        r["O"] = np.asarray(r["O"], int)
        r["O_steps"] = np.asarray(r["O_steps"], int)
        r["episode"] = r["instance_id"].split("__step")[0]
    return recs


def merged_causes(recs: list[dict]) -> list[str]:
    raw = pd.Series([r["cause"] for r in recs])
    cnt = raw.value_counts()
    return [c if cnt.get(c, 0) >= 4 else MERGE[c] for c in raw]


def folds(recs, k=5, seed=0):
    eps = sorted({r["episode"] for r in recs})
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(eps))
    fid = {e: i % k for i, e in enumerate(perm)}
    return np.array([fid[r["episode"]] for r in recs])


def obs_x(r) -> np.ndarray:
    st = np.asarray(r["obs_feats"]["state"], float)
    hd = np.asarray(r["obs_feats"]["head"], float)
    return np.concatenate([st[[2]], st[6:8], [r["obs_feats"]["cpi0"]], hd[list(HEAD_IDX)]])


def probe_r(r, a, fset) -> np.ndarray:
    f = r["probes"][a]["feats"]
    return np.array([float(f[k]) for k in FSETS[fset]], float)


class Model:
    """Everything fitted on the TRAINING folds."""

    def __init__(self, tr: list[dict], causes: list[str], C: list[str], options: list[str], nv: int, probes: list[str],
                 fset: str):
        self.C, self.opts, self.nv, self.probes, self.fset = C, options, nv, probes, fset
        ci = np.array([C.index(c) for c in causes])
        self.ci = ci
        F = len(options)
        # V[c, o, k]: P(variant k succeeds | variants < k of o failed, cause c); Beta prior on the option marginal
        V = np.zeros((len(C), F, nv))
        for o in range(F):
            for k in range(nv):
                tot = [(r["O"][o, :k] <= 0).all() and r["O"][o, k] >= 0 for r in tr]
                s_all = sum(int(r["O"][o, k] == 1) for r, t in zip(tr, tot) if t)
                n_all = sum(tot)
                m = (s_all + 0.5) / (n_all + 1.0)
                for c in range(len(C)):
                    sel = [t and ci[i] == c for i, t in enumerate(tot)]
                    s = sum(int(tr[i]["O"][o, k] == 1) for i, t in enumerate(sel) if t)
                    n = sum(sel)
                    V[c, o, k] = (s + 2 * m) / (n + 2)
        self.V = V
        fail_cost = []
        for o in range(F):
            st = [r["O_steps"][o][r["O"][o] == 0] for r in tr]
            st = np.concatenate([x for x in st if len(x)] or [np.array([400])])
            fail_cost.append(float(st.mean()))
        self.cost = np.array(fail_cost)
        self.worst = np.array([350 if o.startswith("wm_") else (300 if o == "vla" else PREFIX_MAX + 300)
                               for o in options], float)
        X = np.stack([obs_x(r) for r in tr])
        self.sx = StandardScaler().fit(X)
        self.present = sorted(set(ci))
        self.clf = LogisticRegression(C=0.3, max_iter=2000).fit(self.sx.transform(X), ci) if len(self.present) > 1 else None
        self.prior = np.bincount(ci, minlength=len(C)) / len(ci)
        # probe likelihoods: diagonal Gaussian per cause on standardized responses, variance shrunk to pooled
        self.lik = {}
        self.risk = {}
        for a in probes:
            R = np.stack([probe_r(r, a, fset) for r in tr])
            mu0, sd0 = R.mean(0), R.std(0) + 1e-6
            Z = (R - mu0) / sd0
            mus = np.zeros((len(C), Z.shape[1]))
            var = np.ones((len(C), Z.shape[1]))
            for c in range(len(C)):
                zc = Z[ci == c]
                if len(zc) == 0:
                    continue
                mus[c] = zc.mean(0) * len(zc) / (len(zc) + 1.0)
                vc = zc.var(0) if len(zc) > 1 else np.ones(Z.shape[1])
                var[c] = np.maximum(0.5 * vc + 0.5 * 1.0, 0.05)
            self.lik[a] = (mu0, sd0, mus, var, Z)
            self.risk[a] = float(np.mean([r["probes"][a]["risk"] > 0 or r["probes"][a]["aborted"] for r in tr]))
        self.temp = 1.0 / np.sqrt(len(FSETS[fset]))

    def b0(self, r) -> np.ndarray:
        b = np.full(len(self.C), 1e-3)
        if self.clf is None:
            b[self.present[0]] = 1.0
        else:
            b[self.clf.classes_] += self.clf.predict_proba(self.sx.transform(obs_x(r)[None]))[0]
        return b / b.sum()

    def post(self, b0, a, z) -> np.ndarray:
        mu0, sd0, mus, var, _ = self.lik[a]
        ll = -0.5 * (((z[None] - mus) ** 2) / var + np.log(var)).sum(1) * self.temp
        p = b0 * np.exp(ll - ll.max())
        return p / p.sum()

    def zresp(self, r, a) -> np.ndarray:
        mu0, sd0 = self.lik[a][:2]
        return (probe_r(r, a, self.fset) - mu0) / sd0

    def plan(self, b, budget: float) -> list[int]:
        """Deterministic trial sequence (options) under belief b (failures only update b)."""
        b = b.copy()
        n = np.zeros(len(self.opts), int)
        used, seq = 0.0, []
        while True:
            best, bo = -1.0, -1
            for o in range(len(self.opts)):
                if n[o] >= self.nv or used + self.worst[o] > budget:
                    continue
                v = float(b @ self.V[:, o, n[o]]) / self.cost[o]
                if v > best + 1e-12:
                    best, bo = v, o
            if bo < 0:
                return seq
            seq.append(bo)
            b = b * (1 - self.V[:, bo, n[bo]])
            b = b / b.sum() if b.sum() > 0 else b
            n[bo] += 1
            used += self.cost[bo]

    def value(self, seq: list[int], c: int) -> float:
        n = np.zeros(len(self.opts), int)
        f = 1.0
        for o in seq:
            f *= 1 - self.V[c, o, n[o]]
            n[o] += 1
        return 1 - f

    def evi(self, r, a, budget: float, cost_a: int) -> tuple[float, float]:
        """(EVI, expected entropy reduction) of probe a for test instance r: responses sampled from the TRAINING
        instances' standardized responses of each cause."""
        b0 = self.b0(r)
        base = sum(b0[c] * self.value(self.plan(b0, budget), c) for c in range(len(self.C)))
        Z = self.lik[a][4]
        H0 = -(b0 * np.log(b0)).sum()
        tot, eig = 0.0, 0.0
        for c in range(len(self.C)):
            idx = np.nonzero(self.ci == c)[0]
            if b0[c] < 1e-6 or len(idx) == 0:
                continue
            vs, es = [], []
            for i in idx:
                p = self.post(b0, a, Z[i])
                vs.append(self.value(self.plan(p, budget - cost_a), c))
                es.append(H0 + (p * np.log(p)).sum())
            tot += b0[c] * float(np.mean(vs))
            eig += b0[c] * float(np.mean(es))
        return tot - base, eig


def execute(r, seq: list[int], budget: float, worst: np.ndarray) -> dict:
    """Walk the trial sequence in the instance's outcome table."""
    n = np.zeros(len(r["options"]), int)
    used, trials = 0, []
    for o in seq:
        j = n[o]
        if j >= r["O"].shape[1] or r["O"][o, j] < 0 or used + worst[o] > budget:
            continue
        n[o] += 1
        used += int(r["O_steps"][o, j])
        trials.append((int(o), int(j)))
        if r["O"][o, j] == 1:
            return {"solved": True, "steps": used, "trials": trials}
    return {"solved": False, "steps": used, "trials": trials}


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    n01 = int(((~a) & b).sum())
    n10 = int((a & ~b).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, float(p)


def run_arms(recs, causes, budget: float, fset: str = "obs", extra_fsets=("proprio", "tactile", "oracle")) -> pd.DataFrame:
    C = sorted(set(causes))
    fo = folds(recs)
    options = recs[0]["options"]
    nv = recs[0]["O"].shape[1]
    probes = list(recs[0]["probes"])
    plen = {a: int(recs[0]["probe_len"][a]) for a in probes}
    rows = []
    for k in sorted(set(fo)):
        tr = [r for r, f in zip(recs, fo) if f != k]
        trc = [c for c, f in zip(causes, fo) if f != k]
        models = {fs: Model(tr, trc, C, options, nv, probes, fs) for fs in (fset,) + tuple(extra_fsets)}
        M = models[fset]
        safe = [a for a in probes if M.risk[a] <= RISK_MAX]
        for i in np.nonzero(fo == k)[0]:
            r, c = recs[i], C.index(causes[i])
            b0 = M.b0(r)
            out = {"instance_id": r["instance_id"], "set": r["set"], "level": r["level"], "cause_raw": r["cause"],
                   "cause": causes[i], "fold": int(k), "single": bool(r["single_solved"])}
            p0 = M.plan(b0, budget)
            res = {"no_probe": (execute(r, p0, budget, M.worst), None, b0)}
            ws = [options.index("wm_single")] * nv
            res["wm_only"] = (execute(r, ws, budget, M.worst), None, b0)
            res["extra_budget"] = (execute(r, M.plan(b0, budget + 45), budget + 45, M.worst), None, b0)

            def probed(a, model=M, name=None):
                if a is None:
                    return (execute(r, p0, budget, M.worst), None, b0)
                z = model.zresp(r, a)
                b = model.post(model.b0(r), a, z)
                bud = budget - plen[a]
                e = execute(r, model.plan(b, bud), bud, M.worst)
                if r["probes"][a]["success"]:
                    e = {**e, "solved": True}
                e["steps"] += plen[a]
                return (e, a, b)

            res["passive"] = probed("hold")
            rng = np.random.default_rng(zlib.crc32(r["instance_id"].encode()) + 20261116)
            res["random_probe"] = probed(str(rng.choice(safe)) if safe else None)
            ev = {a: M.evi(r, a, budget, plen[a]) for a in safe}
            obj = {a: ev[a][0] - LAM_RISK * M.risk[a] for a in safe}
            a_evi = max(obj, key=obj.get) if obj and max(obj.values()) > 0 else None
            res["evi_probe"] = probed(a_evi)
            a_ig = max(ev, key=lambda a: ev[a][1]) if ev else None
            res["infogain_probe"] = probed(a_ig)
            for fs in extra_fsets:
                Mf = models[fs]
                evf = {a: Mf.evi(r, a, budget, plen[a])[0] - LAM_RISK * Mf.risk[a] for a in safe}
                af = max(evf, key=evf.get) if evf and max(evf.values()) > 0 else None
                res[f"evi_{fs}"] = probed(af, Mf)
            pm = np.full(len(C), 1e-9)
            pm[c] = 1.0
            pm /= pm.sum()
            cost5 = plen[a_evi] if a_evi else 0
            e5 = execute(r, M.plan(pm, budget - cost5), budget - cost5, M.worst)
            e5["steps"] += cost5
            res["oracle_cause"] = (e5, a_evi, pm)
            res["oracle_cause_free"] = (execute(r, M.plan(pm, budget), budget, M.worst), None, pm)
            # option-set ceilings (hindsight, not arms)
            best_single = False
            for o in range(len(options)):
                kmax = int(budget // M.worst[o])
                v = r["O"][o, :kmax]
                best_single |= bool((v == 1).any())
            out["hindsight_best"] = best_single
            out["coverage_any"] = bool((r["O"] == 1).any())
            out["evi_value"] = float(ev[a_evi][0]) if a_evi else 0.0
            first0 = res["no_probe"][0]["trials"][0][0] if res["no_probe"][0]["trials"] else -1
            for name, (e, a, b) in res.items():
                out[name] = bool(e["solved"])
                out[f"{name}__steps"] = int(e["steps"])
                out[f"{name}__probe"] = a
                out[f"{name}__map_ok"] = bool(int(np.argmax(b)) == c)
                f1 = e["trials"][0][0] if e["trials"] else -1
                out[f"{name}__first"] = options[f1] if f1 >= 0 else None
                out[f"{name}__changed"] = bool(f1 != first0)
                out[f"{name}__trials"] = e["trials"]
                out[f"{name}__risk"] = bool(a is not None and (r["probes"][a]["risk"] > 0 or r["probes"][a]["aborted"]))
            rows.append(out)
    return pd.DataFrame(rows)


ARMS = ["single", "no_probe", "wm_only", "extra_budget", "passive", "random_probe", "evi_probe", "infogain_probe",
        "evi_proprio", "evi_tactile", "evi_oracle", "oracle_cause", "oracle_cause_free", "hindsight_best", "coverage_any"]


def table(d: pd.DataFrame, title: str) -> str:
    lines = [f"### {title} (n = {len(d)})", "",
             "| arm | solved | McNemar vs no_probe n01/n10, p | vs wm_only n01/n10, p | mean steps | probe used | "
             "probe risk | decision changed | gain on changed | cause MAP acc |", "|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        if a not in d:
            continue
        s = d[a].astype(bool).values
        m1 = mcnemar(d["no_probe"].values.astype(bool), s)
        m2 = mcnemar(d["wm_only"].values.astype(bool), s)
        st = d.get(f"{a}__steps")
        pu = d.get(f"{a}__probe")
        ch = d.get(f"{a}__changed")
        gain = ""
        if ch is not None and ch.any():
            gain = f"{(d[a][ch].astype(int) - d['no_probe'][ch].astype(int)).mean():+.2f} (n {int(ch.sum())})"
        row = f"| {a} | {s.mean():.3f} ({int(s.sum())}) | {m1[0]}/{m1[1]}, {m1[2]:.3f} | {m2[0]}/{m2[1]}, {m2[2]:.3f} | "
        if st is not None:
            row += (f"{st.mean():.0f} | {pu.notna().mean():.2f} | {d[f'{a}__risk'].mean():.2f} | {ch.mean():.2f} | "
                    f"{gain} | {d[f'{a}__map_ok'].mean():.2f} |")
        else:
            row += "- | - | - | - | - | - |"
        lines.append(row)
    return "\n".join(lines)


def diagnostics(recs, causes) -> str:
    options = recs[0]["options"]
    C = sorted(set(causes))
    out = ["### Cause distribution (oracle)", "", "| raw cause | U | S | merged |", "|---|---|---|---|"]
    raw = pd.DataFrame({"raw": [r["cause"] for r in recs], "set": [r["set"] for r in recs], "m": causes})
    for c, g in raw.groupby("raw"):
        out.append(f"| {c} | {int((g.set == 'U').sum())} | {int((g.set == 'S').sum())} | {g.m.iloc[0]} |")
    out += ["", "### Decision table (all instances pooled, NOT cross-fitted: description only): per-trial success "
            "of each option (variant 0..7 mean) by merged cause", "",
            "| cause (n) | " + " | ".join(options) + " | best |", "|---|" + "---|" * (len(options) + 1)]
    for c in C:
        rs = [r for r, cc in zip(recs, causes) if cc == c]
        v = np.array([[np.mean(r["O"][o][r["O"][o] >= 0] == 1) for o in range(len(options))] for r in rs]).mean(0)
        out.append(f"| {c} ({len(rs)}) | " + " | ".join(f"{x:.2f}" for x in v) + f" | {options[int(np.argmax(v))]} |")
    allv = np.array([[np.mean(r["O"][o][r["O"][o] >= 0] == 1) for o in range(len(options))] for r in recs]).mean(0)
    out.append("| all | " + " | ".join(f"{x:.2f}" for x in allv) + f" | {options[int(np.argmax(allv))]} |")
    # mutual information between cause and the per-instance best option (hindsight)
    best = []
    for r in recs:
        v = np.array([np.mean(r["O"][o][r["O"][o] >= 0] == 1) for o in range(len(options))])
        best.append(options[int(np.argmax(v))] if v.max() > 0 else "none")
    ct = pd.crosstab(pd.Series(causes, name="cause"), pd.Series(best, name="best"))
    p = ct.values / ct.values.sum()
    pc, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    mi = float((p[nz] * np.log2(p[nz] / (pc @ pb)[nz])).sum())
    hb = float(-(pb[pb > 0] * np.log2(pb[pb > 0])).sum())
    out += ["", f"MI(cause; hindsight-best option) = {mi:.3f} bits (H(best) = {hb:.3f} bits)", "",
            "Cause x hindsight-best option (counts):", "", ct.to_markdown()]
    # prefix sub-goal vs final success (is the pi continuation the bottleneck?)
    out += ["", "### Prefix sub-goal reached vs final success (scripted options, all variants)", "",
            "| option | sub-goal def | sub-goal rate | final success | P(final | sub-goal) |", "|---|---|---|---|---|"]
    for o, defn in (("grasp_slow", "obj lifted >= 3 cm"), ("grasp_firm", "obj lifted >= 3 cm"),
                    ("place_goal", "obj-goal xy < 4 cm"), ("push_goal", "obj-goal xy < 4 cm"),
                    ("pull_free", "-"), ("hover", "-")):
        oi = options.index(o)
        sg, fin = [], []
        for r in recs:
            sub = np.asarray(r["O_sub"], float)[oi]
            for j in range(sub.shape[0]):
                if r["O"][oi, j] < 0:
                    continue
                if "lift" in defn:
                    sg.append(sub[j, 0] >= 0.03)
                elif "xy" in defn:
                    sg.append(sub[j, 1] < 0.04)
                else:
                    sg.append(np.nan)
                fin.append(r["O"][oi, j] == 1)
        sg, fin = np.array(sg, float), np.array(fin, bool)
        if np.isnan(sg).all():
            out.append(f"| {o} | - | - | {fin.mean():.3f} | - |")
        else:
            m = sg > 0
            out.append(f"| {o} | {defn} | {np.nanmean(sg):.3f} | {fin.mean():.3f} | "
                       f"{fin[m].mean() if m.any() else float('nan'):.3f} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/acr/n1")
    ap.add_argument("--out", default="runs/_analysis/acrn")
    ap.add_argument("--write-decisions", action="store_true")
    args = ap.parse_args()
    recs = load(REPO_ROOT / args.root)
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    causes = merged_causes(recs)
    md = [f"# ACR-N analysis ({len(recs)} instances: U {sum(r['set'] == 'U' for r in recs)}, "
          f"S {sum(r['set'] == 'S' for r in recs)})", "", diagnostics(recs, causes), ""]
    dec = {}
    for budget in (2800, 1400):
        d = run_arms(recs, causes, budget)
        d.to_json(out / f"rows_b{budget}.jsonl", orient="records", lines=True)
        md += [f"## Budget S = {budget} env steps", ""]
        for name, sub in (("U (single-unsolved, primary)", d[d.set == "U"]), ("S (single-solved control)", d[d.set == "S"]),
                          ("U + S", d)):
            if len(sub):
                md += [table(sub, name), ""]
        u = d[d.set == "U"]
        if len(u):
            md += ["Per level (U): " + ", ".join(
                f"{lv} n {len(g)}: no_probe {g.no_probe.mean():.2f} / evi {g.evi_probe.mean():.2f} / oracle "
                f"{g.oracle_cause.mean():.2f}" for lv, g in u.groupby("level")), "",
                   "Per cause (U): " + ", ".join(
                f"{c} n {len(g)}: no_probe {g.no_probe.mean():.2f} / evi {g.evi_probe.mean():.2f} / oracle "
                f"{g.oracle_cause.mean():.2f} / hindsight {g.hindsight_best.mean():.2f}" for c, g in u.groupby("cause")),
                   "", "EVI probe choices: " + json.dumps(d.evi_probe__probe.fillna("none").value_counts().to_dict()), ""]
            n01, n10, p = mcnemar(u.no_probe.values.astype(bool), u.oracle_cause.values.astype(bool))
            g0 = (u.oracle_cause.mean() - u.no_probe.mean() >= 0.08) and p < 0.10 and n01 > n10
            n01e, n10e, pe = mcnemar(u.no_probe.values.astype(bool), u.evi_probe.values.astype(bool))
            beats = all(u.evi_probe.mean() - u[x].mean() >= 0.05 for x in ("no_probe", "passive", "random_probe",
                                                                             "extra_budget"))
            g1 = bool(g0 and beats and pe < 0.10 and n01e > n10e)
            dec[str(budget)] = {"G0": bool(g0), "G0_stats": [n01, n10, p], "G1_table": g1,
                                "G1_stats": [n01e, n10e, pe]}
        if budget == 2800 and args.write_decisions:
            rows = []
            for _, x in d.iterrows():
                for arm in ("evi_probe", "random_probe"):
                    rows.append({"instance_id": x.instance_id, "arm": arm, "probe": x[f"{arm}__probe"],
                                 "seq": x[f"{arm}__trials"]})
            (out / "decisions.json").write_text(json.dumps({"budget": budget, "rows": rows}))
    md += ["## Pre-registered gates (PLAN ACR-N)", "", "```", json.dumps(dec, indent=1), "```"]
    (out / "acrn.md").write_text("\n".join(md))
    (out / "decision.json").write_text(json.dumps(dec, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
