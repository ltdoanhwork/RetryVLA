"""Goal G2 / P2-3 table vs the pre-registered rules (PLAN "Goal campaign G2 ... P2-3", 2026-11-24).

python scripts/analyze_goal_g2.py -> runs/goal_rsi/g2.{json,md}
Per arm (3 seeds): nominal val successes, Δ vs Original (scene x seed bootstrap), fixed / broken vs Original and vs M0,
failure incidence + first-attempt success (detector E-v1, only for runs with events), RSR_E8 and RSR_E8on
(entry x seed bootstrap vs Original), per-task means. Rules: G2 "curation fixes the nominal cost", "matches theirs",
"beats F0"; P2-3 "prevention helps" / "prevention data = more recovery data".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / "runs/goal_rsi"


def load(run, kind):
    c = R / run / "ckpt_02000" if run != "m0" else R / "m0_s7/ckpt_04000"
    ev = c / "eval_val/events.jsonl"
    if not ev.exists():   # Original / M0: events from the extra val pass with saved actions (eval_goal_only.sh)
        ev = c / "eval_val_ev/events.jsonl"
    f = {"val": c / "eval_val/episodes.jsonl", "e8": c / "eval_entries_e8/episodes.jsonl",
         "e8on": c / "eval_entries_e8on/episodes.jsonl", "ev": ev}[kind]
    if not f.exists():
        return None
    key = "entry_id" if kind in ("e8", "e8on") else "scene"
    return {json.loads(l)[key]: json.loads(l) for l in open(f)}


def arm(prefix, kind):
    out = {}
    for s in (7, 8, 9):
        x = load(f"{prefix}_s{s}", kind)
        if x:
            out[s] = x
    return out


def boot(A, B, field="success", n=4000):
    seeds = sorted(set(A) & set(B))
    if not seeds:
        return None
    keys = sorted(set.intersection(*[set(A[s]) & set(B[s]) for s in seeds]))
    X = np.array([[float(A[s][k][field]) for k in keys] for s in seeds])
    Y = np.array([[float(B[s][k][field]) for k in keys] for s in seeds])
    rng = np.random.default_rng(0)
    bs = [(X[si][:, ci] - Y[si][:, ci]).mean() for si, ci in
          ((rng.integers(len(seeds), size=len(seeds)), rng.integers(len(keys), size=len(keys))) for _ in range(n))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"delta": float((X - Y).mean() * 100), "ci": [float(lo * 100), float(hi * 100)], "seeds": seeds}


def fb(A, B):
    f = b = 0
    for s in set(A) & set(B):
        for k in A[s]:
            f += (not B[s][k]["success"]) and A[s][k]["success"]
            b += B[s][k]["success"] and not A[s][k]["success"]
    return f, b


def main():
    arms = {"Original": "orig", "F0_wm": "rec_wm", "F1_wm": "f1_wm", "D5_wm (old bank + cont, 36)": "d5_wm",
            "G2_wm (curated onset + cont, 36)": "g2_wm", "G2phys_wm (phys survivors = 18)": "g2phys_wm",
            "G2_wm at 18 (count-matched to G2phys)": "g2_wm18",
            "G2physns_wm (phys filter w/o seam, 36)": "g2physns_wm", "G2_wm at 5 (anchor-matched to G2phys)": "g2_wm5",
            "G2_random": "g2_rand",
            "P2-3 A (curated D_R, 36 + n_P)": "p23_A", "P2-3 B (curated D_R 36 + D_P n_P)": "p23_B"}
    V = {k: arm(p, "val") for k, p in arms.items()}
    V["D5_wm (old bank + cont, 36)"].update({7: load("diag_f1wm_full_s7", "val")} if load("diag_f1wm_full_s7", "val") else {})
    E = {k: arm(p, "e8") for k, p in arms.items()}
    EO = {k: arm(p, "e8on") for k, p in arms.items()}
    EV = {k: arm(p, "ev") for k, p in arms.items()}
    m0 = load("m0", "val")
    res = {"arms": {}}
    O = V["Original"]
    for k in arms:
        if not V[k]:
            continue
        r = {"val": {s: sum(x["success"] for x in v.values()) for s, v in V[k].items()}}
        r["val_mean"] = float(np.mean(list(r["val"].values())))
        r["d_vs_orig"] = boot(V[k], O) if k != "Original" else None
        r["fixed_broken_vs_orig"] = fb(V[k], O) if k != "Original" else None
        r["fixed_broken_vs_m0"] = fb(V[k], {s: m0 for s in V[k]})
        if E[k]:
            r["rsr_e8"] = {s: float(np.mean([x["success"] for x in v.values()])) for s, v in E[k].items()}
            r["rsr_e8_d_vs_orig"] = boot(E[k], E["Original"]) if k != "Original" and E["Original"] else None
        if EO[k]:
            r["rsr_e8on"] = {s: float(np.mean([x["success"] for x in v.values()])) for s, v in EO[k].items()}
        if EV[k]:
            r["failure_incidence"] = float(np.mean([x["n_events"] > 0 for v in EV[k].values() for x in v.values()]))
            r["first_attempt_success"] = float(np.mean([x["first_attempt_success"] for v in EV[k].values()
                                                        for x in v.values()]))
        tasks = sorted({x["task"] for v in V[k].values() for x in v.values()})
        r["per_task"] = {t: float(np.mean([sum(x["success"] for x in v.values() if x["task"] == t) for v in V[k].values()]))
                         for t in tasks}
        res["arms"][k] = r
    a = res["arms"]
    g2 = a.get("G2_wm (curated onset + cont, 36)")
    rules = {}
    if g2 and g2.get("d_vs_orig"):
        d, de = g2["d_vs_orig"], g2.get("rsr_e8_d_vs_orig")
        rules["G2_curation_fixes_nominal_cost"] = bool(d["delta"] >= -2 and d["ci"][0] > -5 and de and de["delta"] >= 5
                                                       and de["ci"][0] > 0)
        rules["G2_matches_theirs"] = bool(d["delta"] >= 5 and d["ci"][0] > 0)
        f0 = a.get("F0_wm", {})
        rules["G2_beats_F0"] = bool(np.mean(list(g2.get("rsr_e8", {0: 0}).values())) >= 0.282 - 0.03
                                    and g2["val_mean"] >= f0.get("val_mean", 1e9) + 8)
    gp = a.get("G2phys_wm (phys survivors = 18)")
    if gp and g2:
        dn_ = boot(V["G2phys_wm (phys survivors = 18)"], V["G2_wm (curated onset + cont, 36)"])
        res["G2phys_minus_G2_nominal"] = dn_
        ie = None
        if EV["G2phys_wm (phys survivors = 18)"] and EV["G2_wm (curated onset + cont, 36)"]:
            ie = boot({s: {k: {"success": v[k]["n_events"] > 0} for k in v} for s, v in EV["G2phys_wm (phys survivors = 18)"].items()},
                      {s: {k: {"success": v[k]["n_events"] > 0} for k in v} for s, v in EV["G2_wm (curated onset + cont, 36)"].items()})
        er = boot(E["G2phys_wm (phys survivors = 18)"], E["G2_wm (curated onset + cont, 36)"])
        rules["G2phys_helps"] = bool((er is None or er["delta"] >= -3) and ((dn_["delta"] >= 3 and dn_["ci"][0] > 0) or
                                     (ie is not None and ie["delta"] <= -5 and ie["ci"][1] < 0)))
    A_, B_ = "P2-3 A (curated D_R, 36 + n_P)", "P2-3 B (curated D_R 36 + D_P n_P)"
    if V[A_] and V[B_]:
        dn = boot(V[B_], V[A_])
        res["P2-3_B_minus_A_nominal"] = dn
        inc = None
        if EV[A_] and EV[B_]:
            inc = boot({s: {k: {"success": v[k]["n_events"] > 0} for k in v} for s, v in EV[B_].items()},
                       {s: {k: {"success": v[k]["n_events"] > 0} for k in v} for s, v in EV[A_].items()})
            res["P2-3_B_minus_A_incidence"] = inc
        de = boot(E[B_], E[A_]) if E[A_] and E[B_] else None
        res["P2-3_B_minus_A_rsr_e8"] = de
        ni = de is None or de["delta"] >= -3
        rules["P2-3_prevention_helps"] = bool(ni and ((dn["delta"] >= 3 and dn["ci"][0] > 0) or
                                                     (inc is not None and inc["delta"] <= -5 and inc["ci"][1] < 0)))
        rules["P2-3_prevention_eq_more_recovery"] = bool(abs(dn["delta"]) < 3 and (inc is None or abs(inc["delta"]) < 5))
    res["rules"] = rules
    res["paper"] = {"Original": 78, "Recovery": 88}
    (R / "g2.json").write_text(json.dumps(res, indent=1, default=str))
    f = lambda d: "-" if not d else f"{d['delta']:+.1f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}]"
    md = ["| arm | val s7 / s8 / s9 | mean | Δ vs Original | fixed / broken vs Orig | vs M0 | RSR_E8 | Δ RSR_E8 | "
          "RSR_E8on | fail. incidence | first-attempt |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in a.items():
        md.append(f"| {k} | {' / '.join(str(r['val'].get(s, '-')) for s in (7, 8, 9))} | {r['val_mean']:.1f} | "
                  f"{f(r['d_vs_orig'])} | {r['fixed_broken_vs_orig'] or '-'} | {r['fixed_broken_vs_m0']} | "
                  f"{np.mean(list(r['rsr_e8'].values())):.3f} | " if r.get("rsr_e8") else
                  f"| {k} | {' / '.join(str(r['val'].get(s, '-')) for s in (7, 8, 9))} | {r['val_mean']:.1f} | "
                  f"{f(r['d_vs_orig'])} | {r['fixed_broken_vs_orig'] or '-'} | {r['fixed_broken_vs_m0']} | - | ")
        md[-1] += (f"{f(r.get('rsr_e8_d_vs_orig'))} | " if r.get("rsr_e8") else "- | ")
        md[-1] += (f"{np.mean(list(r['rsr_e8on'].values())):.3f} | " if r.get("rsr_e8on") else "- | ")
        md[-1] += (f"{r['failure_incidence']:.2f} | {r['first_attempt_success']:.2f} |" if "failure_incidence" in r else "- | - |")
    md += ["", f"P2-3 B − A: nominal {f(res.get('P2-3_B_minus_A_nominal'))}, incidence "
           f"{f(res.get('P2-3_B_minus_A_incidence'))}, RSR_E8 {f(res.get('P2-3_B_minus_A_rsr_e8'))}",
           f"rules: {rules}", "paper (cross-paper reference): Original 78 -> Recovery 88 (+10)"]
    tasks = sorted({t for r in a.values() for t in r["per_task"]})
    md += ["", "| task | " + " | ".join(a) + " |", "|---|" + "---|" * len(a)]
    for t in tasks:
        md.append(f"| {t} | " + " | ".join(f"{a[k]['per_task'].get(t, float('nan')):.1f}" for k in a) + " |")
    (R / "g2.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
