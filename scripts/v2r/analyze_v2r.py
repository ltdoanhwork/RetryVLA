#!/usr/bin/env python3
"""V2R analysis (PLAN "V2R"): RSR + official levels, nominal + guard, per-task regression, E-v1s events, jerk,
completion steps, McNemar / scene x seed bootstrap vs R0 and vs each arm's direct control; search + bank funnels.

python scripts/v2r/analyze_v2r.py   -> runs/_analysis/v2r/{v2r.md, v2r.json}
Only finished evals are used; missing cells are listed as pending.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
from stats_cluster import compare, mcnemar  # noqa: E402

from retryvla.envs.splits import load_levels  # noqa: E402

OUT = REPO / "runs/_analysis/v2r"
SP = REPO / "data/assets/scenes/scenes/selected_bddl_scenes_spatial"
PI1NOM, PI0 = 0.796, 0.656
R0_TAGS = [("M5_m5c_sft_wm_diverse_rr_recover_test", "M5_m5c_sft_wm_diverse_rr_nominal"),
           ("M8_seed_pi1_s1_recover_test", "M8_seed_pi1_s1_nominal"),
           ("M8_seed_pi1_s2_recover_test", "M8_seed_pi1_s2_nominal")]
ARMS = ["r0", "r1", "r2", "r3", "r4"]
CTRL = {"r1": "r0", "r2": "r1", "r3": "r2", "r4": None}


def newest(tag, need=None):
    for d in sorted(glob.glob(str(REPO / "runs" / f"*_{tag}")), reverse=True):
        try:
            s = json.load(open(Path(d) / "summary.json"))
        except Exception:
            continue
        if s.get("status") != "done":
            continue
        if need and not (Path(d) / need).exists():
            continue
        return Path(d)
    return None


def eps(d):
    return [json.loads(l) for l in open(d / "episodes.jsonl")]


def cell(arm, s):
    """-> dict(rec=episodes, nom=episodes, ev=events_summary nominal, rj=recover jerk summary)"""
    if arm == "r0":
        rt, nt = R0_TAGS[s]
        dr, dn = newest(rt), newest(nt)
        de = newest(f"V2R_v2r_r0_s{s}_nominal", "events_summary.json")
    else:
        dr = newest(f"V2R_v2r_{arm}_s{s}_recover_test", "events_summary.json")
        dn = de = newest(f"V2R_v2r_{arm}_s{s}_nominal", "events_summary.json")
    out = {}
    if dr:
        out["rec"] = eps(dr)
        if (dr / "events_summary.json").exists():
            out["rj"] = json.load(open(dr / "events_summary.json"))
    if dn:
        out["nom"] = eps(dn)
    if de:
        out["ev"] = json.load(open(de / "events_summary.json"))
        out["ev_rows"] = [json.loads(l) for l in open(de / "events.jsonl")]
        out["ev_sr"] = float(np.mean([r["success"] for r in out["ev_rows"]]))
    return out


def ed(r):
    return {(x["scene"], x["gripper_init"], x["trial"]): float(x["success"]) for x in r}


def nom_tasks(r):
    d = {}
    for x in r:
        d.setdefault(x["task_id"], []).append(float(x["success"]))
    return np.array([np.mean(d[k]) for k in sorted(d)])


def fmt(v, k=3):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{k}f}"


def boot_rate(a, b, rng, n=5000):
    """a, b: [seeds, units] -> diff, CI (unit x seed bootstrap)"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    bs = []
    for _ in range(n):
        u = rng.integers(0, a.shape[1], a.shape[1])
        bs.append(a[rng.integers(0, len(a), len(a))][:, u].mean() - b[rng.integers(0, len(b), len(b))][:, u].mean())
    return a.mean() - b.mean(), *np.percentile(bs, [2.5, 97.5])


def main():
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)
    lv = load_levels(SP, official_only=True)
    C = {a: {s: cell(a, s) for s in range(3)} for a in ARMS}
    J = {"arms": {}}
    md = ["# V2R: RetryVLA v2 levers on LIBERO-Recover Spatial (test = dev split 56 x 10; nominal 10 x 50)", "",
          "Pre-registration: experiments/PLAN.md \"V2R\" (a547a63). R0 = pi1 (registered evals). ± = sd over SFT seeds. "
          "Δ = scene x seed bootstrap (10k) 95 % CI, p = paired sign-flip over scenes; McNemar pooled over seed pairs "
          "(seed k vs seed k). Nominal guard: mean >= 0.776. Events = E-v1s on nominal episodes (replay-checked).", "",
          "| arm | seeds | RSR | per seed | ΔRSR vs R0 [CI], p | McNemar vs R0 n01/n10, p | Δ vs control [CI] | L1 (1) | L2 (11) | L3 (16) | nominal | guard | incidence | first-attempt | compl. steps | jerk nom / rec |",
          "|" + "---|" * 16]
    for a in ARMS:
        cs = C[a]
        R = [cs[s]["rec"] for s in range(3) if "rec" in cs[s]]
        N = [cs[s]["nom"] for s in range(3) if "nom" in cs[s]]
        E = [cs[s]["ev"] for s in range(3) if "ev" in cs[s]]
        RJ = [cs[s]["rj"] for s in range(3) if "rj" in cs[s]]
        row = {"seeds_rec": len(R), "seeds_nom": len(N)}
        if not R:
            md.append(f"| {a.upper()} | 0 | pending |" + " |" * 14)
            J["arms"][a] = row
            continue
        rs = [np.mean([x["success"] for x in r]) for r in R]
        row.update(rsr=float(np.mean(rs)), rsr_seeds=[float(x) for x in rs])
        dv, pv, mc, dctl = "(ref)", "(ref)", "(ref)", "-"
        R0 = [C["r0"][s]["rec"] for s in range(3) if "rec" in C["r0"][s]]
        if a != "r0":
            d, ci, p, *_ = compare([ed(r) for r in R], [ed(r) for r in R0], rng)
            dv = f"{d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}], p {p:.3f}"
            ks = sorted(ed(R[0]))
            A_ = np.concatenate([[ed(cs[s]["rec"])[k] > 0.5 for k in ks] for s in range(3) if "rec" in cs[s] and "rec" in C["r0"][s]])
            B_ = np.concatenate([[ed(C["r0"][s]["rec"])[k] > 0.5 for k in ks] for s in range(3) if "rec" in cs[s] and "rec" in C["r0"][s]])
            n01, n10, pm = mcnemar(A_, B_)
            mc = f"{n01}/{n10}, {pm:.2g}"
            row.update(d_r0=float(d), d_r0_ci=[float(ci[0]), float(ci[1])], p_r0=float(p), mcnemar_r0=[n01, n10, pm])
            ctl = CTRL[a] or (open(REPO / "runs/v2r/r4_base.txt").read().strip() if (REPO / "runs/v2r/r4_base.txt").exists() else None)
            if ctl and ctl != "r0":
                Rc = [C[ctl][s]["rec"] for s in range(3) if "rec" in C[ctl][s]]
                if Rc:
                    d2, ci2, p2, *_ = compare([ed(r) for r in R], [ed(r) for r in Rc], rng)
                    dctl = f"vs {ctl.upper()} {d2:+.3f} [{ci2[0]:+.3f}, {ci2[1]:+.3f}]"
                    row.update(ctl=ctl, d_ctl=float(d2), d_ctl_ci=[float(ci2[0]), float(ci2[1])])
        levs = {}
        for L in ("L1", "L2", "L3"):
            v = [np.mean([x["success"] for x in r if lv.get(x["scene"]) == L]) for r in R]
            levs[L] = float(np.mean(v))
        row["levels"] = levs
        nm = [nom_tasks(r).mean() for r in N]
        nomv = float(np.mean(nm)) if nm else None
        row["nominal"] = nomv
        guard = "-" if nomv is None else ("pass" if nomv >= PI1NOM - 0.02 else "FAIL")
        inc = [e["failure_incidence"] for e in E]
        fa = [e["first_attempt_success"] for e in E]
        cst = [e["completion_steps_mean"] for e in E]
        jn = [e["jerk_mean"] for e in E]
        jr = [e["jerk_mean"] for e in RJ]
        row.update(incidence=float(np.mean(inc)) if inc else None, first_attempt=float(np.mean(fa)) if fa else None,
                   completion=float(np.mean(cst)) if cst else None, jerk_nom=float(np.mean(jn)) if jn else None,
                   jerk_rec=float(np.mean(jr)) if jr else None,
                   ev_replay_ok=[e["n_replay_ok"] for e in E])
        sdv = f" ± {np.std(rs, ddof=1):.3f}" if len(rs) > 1 else ""
        md.append(f"| {a.upper()} | {len(R)}/{len(N)} | {np.mean(rs):.3f}{sdv} | {' / '.join(f'{x:.3f}' for x in rs)} | {dv} | {mc} | {dctl} | "
                  f"{fmt(levs['L1'])} | {fmt(levs['L2'])} | {fmt(levs['L3'])} | {fmt(nomv)} | {guard} | "
                  f"{fmt(row['incidence'])} | {fmt(row['first_attempt'])} | {fmt(row['completion'], 1)} | "
                  f"{fmt(row['jerk_nom'])} / {fmt(row['jerk_rec'])} |")
        J["arms"][a] = row
    # nominal event / per-task deltas vs control (paired per episode init) + per-task regression vs pi1-nominal
    md += ["", "## Nominal: events vs control and per-task regression", "",
           "| arm | Δ nominal vs control [CI] | Δ incidence vs control [CI] | Δ first-attempt vs control [CI] | tasks < pi1-nominal − 6 pt | tasks < pi0 |",
           "|---|---|---|---|---|---|"]
    pn = newest("V2R_v2r_pi1nom_nominal", "events_summary.json")
    pn_t = nom_tasks(eps(pn)) if pn else None
    if pn:
        J["pi1nom_rerun"] = json.load(open(pn / "events_summary.json"))
    for a in ARMS[1:]:
        ctl = CTRL[a] or J["arms"].get(a, {}).get("ctl")
        if not ctl:
            ctl = "r0"
        cs, cc = C[a], C[ctl]
        S = [s for s in range(3) if "ev_rows" in cs[s] and "ev_rows" in cc[s]]
        if not S:
            md.append(f"| {a.upper()} | pending | | | | |")
            continue

        def mat(c, key):
            out = []
            for s in S:
                rows = {(r["task_id"], r["episode"]): r for r in c[s]["ev_rows"]}
                ks = sorted(rows)
                if key == "success":
                    out.append([float(rows[k]["success"]) for k in ks])
                elif key == "inc":
                    out.append([float(rows[k].get("n_events", 0) > 0) if rows[k].get("replay_ok", True) else np.nan for k in ks])
                else:
                    out.append([float(rows[k].get("first_attempt_success", False)) if rows[k].get("replay_ok", True) else np.nan for k in ks])
            return np.nan_to_num(np.array(out), nan=0.0)
        res = []
        for key in ("success", "inc", "fa"):
            d, lo, hi = boot_rate(mat(cs, key), mat(cc, key), rng)
            res.append(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        Nt = np.mean([nom_tasks(cs[s]["nom"]) for s in range(3) if "nom" in cs[s]], 0)
        reg = [int(i) for i in np.nonzero(Nt < (pn_t if pn_t is not None else PI1NOM) - 0.06)[0]] if pn_t is not None else []
        below0 = "-"
        md.append(f"| {a.upper()} (vs {ctl.upper()}) | {res[0]} | {res[1]} | {res[2]} | {reg} | {below0} |")
    # banks / funnel
    md += ["", "## Banks and search", ""]
    for name, pat in (("R1 search (critic swap)", "data/recovery/v2r_r1/s*/p*/wm_diverse"),
                      ("M5b wm_diverse (shipped C^pi)", "data/recovery/m5b/s*/p*/wm_diverse")):
        n = sum(sum(1 for _ in open(f)) for f in glob.glob(str(REPO / pat / "retryvla/trajectories.jsonl")))
        md.append(f"- {name}: {n} verified trajectories")
    try:
        import pandas as pd
        a1 = pd.concat([pd.read_json(f, lines=True) for f in glob.glob(str(REPO / "runs/*_V2R_v2r_r1_search_s*of8/per_instance.jsonl"))])
        a0 = pd.concat([pd.read_json(f, lines=True) for f in glob.glob(str(REPO / "runs/*_M5_m5b_compare_arms_s*of8/per_instance.jsonl"))])
        a0 = a0[a0.arm == "wm_diverse"].drop_duplicates("instance_id", keep="last").set_index("instance_id")
        a1 = a1.drop_duplicates("instance_id", keep="last").set_index("instance_id")
        com = a1.index.intersection(a0.index)
        s1, s0 = a1.loc[com, "solved"].astype(bool).values, a0.loc[com, "solved"].astype(bool).values
        n01, n10, p = mcnemar(s1, s0)
        h1 = a1.loc[com, "hits"].sum() / a1.loc[com, "budget"].sum()
        h0 = a0.loc[com, "hits"].sum() / a0.loc[com, "budget"].sum()
        md.append(f"- search, {len(com)} paired instances: solved base-critic {s1.mean():.3f} vs shipped {s0.mean():.3f} "
                  f"(McNemar {n01}/{n10}, p {p:.3g}); hits / verification {h1:.3f} vs {h0:.3f}; mean pred on picks "
                  f"{a1.loc[com, 'pred_mean'].mean():.3f} vs {a0.loc[com, 'pred_mean'].mean():.3f}")
        J["search"] = {"n": int(len(com)), "solved_r1": float(s1.mean()), "solved_m5b": float(s0.mean()),
                       "mcnemar": [n01, n10, p], "hits_r1": float(h1), "hits_m5b": float(h0)}
    except Exception as e:  # noqa: BLE001
        md.append(f"- search comparison pending ({str(e)[:80]})")
    for f, lab in (("data/recovery/v2r_r1/phys_summary_wm_diverse.json", "R2 plausibility scan of the R1 bank"),
                   ("data/recovery/v2r_dp/phys_summary_dp.json", "plausibility scan of D_P")):
        p_ = REPO / f
        if p_.exists():
            s = json.load(open(p_))
            md.append(f"- {lab}: kept {s['kept']} / {s['n']} ({s['kept_instances']} / {s['instances']} instances); "
                      f"reasons {s['reject_reasons']}; thresholds {json.dumps({k: round(v, 4) for k, v in s['thresholds'].items()})}")
            J[f"phys_{Path(f).stem}"] = s
    pr = [json.loads(l) for f in glob.glob(str(REPO / "data/recovery/v2r_dp/sh*/onset.jsonl")) for l in open(f) if l.strip()]
    if pr:
        last = {}
        for r in pr:
            last[r["group"]] = r
        pr = list(last.values())
        nl = sum(r["t_last"] is not None for r in pr)
        nf = sum(r.get("has_failure", False) for r in pr)
        nw = sum(len(r.get("written", [])) for r in pr)
        ng = sum(bool(r.get("written")) for r in pr)
        pm = [r["p_minus"] for r in pr if "p_minus" in r]
        md.append(f"- D_P funnel: {len(pr)} train source episodes -> {nl} with t_last (>= 2/8) -> {nf} s^- with a pi0 "
                  f"failure -> {ng} with >= 1 verified pair -> {nw} D_P trajectories; mean p(a-) {np.mean(pm) if pm else float('nan'):.2f}; "
                  f"t_last distribution {sorted([r['t_last'] for r in pr if r['t_last'] is not None])}; "
                  f"sim steps {sum(r.get('sim_steps', 0) for r in pr):,}")
        J["dp"] = {"episodes": len(pr), "t_last": nl, "s_minus": nf, "with_pair": ng, "n_dp": nw}
    (OUT / "v2r.md").write_text("\n".join(md) + "\n")
    (OUT / "v2r.json").write_text(json.dumps(J, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
