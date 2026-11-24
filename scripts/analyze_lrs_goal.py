"""LRS-G analysis vs the pre-registered rules (PLAN "LRS-G", 2026-11-24).

python scripts/analyze_lrs_goal.py -> runs/goal_rsi/lrs_goal.{json,md}
(1) collection: status counts, accepted / entry, LLM calls / s per entry, path lengths, windows / anchors / share;
(2) judge (EXTRA vs the paper): every hand-off test (accepted or rejected) vs the sim ground truth of the same student
    run continued to <= 300 steps -> precision P(GT | accept), recall P(accept | GT), Wilson CIs; also per entry
    "solved" (>= 1 hand-off with GT success) vs goal_search wm_diverse solved (McNemar, paired entries);
(3) SFT: arms lrs / lrs_all vs Original (3 seeds) with the campaign metrics (scripts/analyze_goal_g2 helpers).
"""
from __future__ import annotations

import json
import sys
from math import comb, sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_goal_g2 import arm, boot, fb, load  # noqa: E402

R = ROOT / "runs/goal_rsi"
D = ROOT / "data/recovery/lrs_goal"


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def collection():
    rows = {}
    for p in sorted(D.glob("a*/l*/rows.jsonl")):
        for r in map(json.loads, p.read_text().splitlines()):
            rows.setdefault(r["instance_id"], []).append(r)
    first = {i: v[0] for i, v in rows.items()}
    st = {}
    for r in first.values():
        st[r["status"]] = st.get(r["status"], 0) + 1
    H = [(h["accept"], h["gt_success"]) for v in rows.values() for r in v for s in r["stages"]
         if (h := s.get("handoff"))]
    tp = sum(a and g for a, g in H)
    na, ng = sum(a for a, _ in H), sum(g for _, g in H)
    acc = [r for r in first.values() if r["status"] == "accepted"]
    out = {"n_entries": len(first), "status": st,
           "accepted_rate": len(acc) / max(1, len(first)),
           "n_handoffs": len(H), "judge_accepts": na, "gt_successes": ng,
           "precision": tp / na if na else None, "precision_ci": wilson(tp, na),
           "recall": tp / ng if ng else None, "recall_ci": wilson(tp, ng),
           "accepted_path_gt_success": sum(bool(r["accepted_gt"]) for r in acc) / max(1, len(acc)),
           "llm_calls_mean": float(np.mean([r["llm_calls"] for r in first.values()])),
           "llm_s_mean": float(np.mean([r["llm_s"] for r in first.values()])),
           "t_s_mean": float(np.mean([r["t_s"] for r in first.values()])),
           "path_len_accepted_mean": float(np.mean([r["path_len"] for r in acc])) if acc else None,
           "teacher_actions_accepted": int(sum(r["path_len"] for r in acc)),
           "stages_mean": float(np.mean([r["n_stages"] for r in first.values()])),
           "teacher_self_success": sum(any(s.get("teacher_success") for s in r["stages"]) for r in first.values())}
    # solved (GT) per entry vs goal_search wm_diverse
    solved_lrs = {i: any(s.get("handoff", {}).get("gt_success") for r in v for s in r["stages"]) or
                  any(s.get("teacher_success") for r in v for s in r["stages"]) for i, v in rows.items()}
    wm = set()
    for s in range(3):
        for l in open(ROOT / f"data/recovery/goal_search/s{s}/p0/wm_diverse/retryvla/trajectories.jsonl"):
            wm.add(json.loads(l)["instance_id"])
    keys = sorted(solved_lrs)
    b = sum(solved_lrs[k] and k not in wm for k in keys)
    c = sum((k in wm) and not solved_lrs[k] for k in keys)
    out["solved_gt_lrs"] = float(np.mean([solved_lrs[k] for k in keys]))
    out["solved_wm_diverse_same_entries"] = float(np.mean([k in wm for k in keys]))
    out["solved_mcnemar"] = {"lrs_only": b, "wm_only": c, "p": mcnemar(b, c)}
    per_task = {}
    for r in first.values():
        t = per_task.setdefault(r["task_folder"], [0, 0, 0])
        t[0] += 1
        t[1] += r["status"] == "accepted"
        t[2] += bool(r.get("accepted_gt"))
    out["per_task_entries_accepted_acceptedGT"] = per_task
    for name in ("lrs_goal_ds", "lrs_goal_ds_all"):
        p = ROOT / "data/recovery" / name / "retryvla/trajectories.jsonl"
        if p.exists():
            m = [json.loads(l) for l in open(p)]
            fr = sum(x["i1"] - x["i0"] for x in m)
            anc = sum(max(0, x["i1"] - x["i0"] - 4) for x in m)
            out[name] = {"windows": len(m), "paths": len({x["instance_id"] for x in m}), "frames": fr,
                         "anchors_lmin5": anc, "share_of_anchors": anc / (anc + 38240)}
    return out


def sft():
    arms = {"Original": "orig", "LRS (their volume, ~1.6k anchors)": "lrs", "LRS all windows": "lrs_all",
            "F0_wm": "rec_wm"}
    V = {k: arm(p, "val") for k, p in arms.items()}
    E = {k: arm(p, "e8") for k, p in arms.items()}
    EO = {k: arm(p, "e8on") for k, p in arms.items()}
    EV = {k: arm(p, "ev") for k, p in arms.items()}
    m0 = load("m0", "val")
    O = V["Original"]
    res = {}
    for k in arms:
        if not V[k]:
            continue
        r = {"val": {s: sum(x["success"] for x in v.values()) for s, v in V[k].items()}}
        r["val_mean"] = float(np.mean(list(r["val"].values())))
        if k != "Original":
            r["d_vs_orig"] = boot(V[k], O)
            r["fixed_broken_vs_orig"] = fb(V[k], O)
            r["mcnemar_per_seed"] = {s: mcnemar(*fb({s: V[k][s]}, {s: O[s]})) for s in V[k] if s in O}
        r["fixed_broken_vs_m0"] = fb(V[k], {s: m0 for s in V[k]})
        if E[k]:
            r["rsr_e8"] = float(np.mean([x["success"] for v in E[k].values() for x in v.values()]))
            if k != "Original" and E["Original"]:
                r["rsr_e8_d_vs_orig"] = boot(E[k], E["Original"])
        if EO[k]:
            r["rsr_e8on"] = float(np.mean([x["success"] for v in EO[k].values() for x in v.values()]))
        if EV[k]:
            r["failure_incidence"] = float(np.mean([x["n_events"] > 0 for v in EV[k].values() for x in v.values()]))
            r["first_attempt_success"] = float(np.mean([x["first_attempt_success"] for v in EV[k].values()
                                                        for x in v.values()]))
        tasks = sorted({x["task"] for v in V[k].values() for x in v.values()})
        r["per_task"] = {t: float(np.mean([sum(x["success"] for x in v.values() if x["task"] == t)
                                           for v in V[k].values()])) for t in tasks}
        res[k] = r
    return res


def main():
    c = collection()
    s = sft()
    rules = {}
    L = s.get("LRS (their volume, ~1.6k anchors)")
    if L and L.get("d_vs_orig") and len(L["val"]) == 3:
        d = L["d_vs_orig"]
        rules["reproduces_their_gain"] = bool(d["delta"] >= 5 and d["ci"][0] > 0)
        rules["hurts"] = bool(d["delta"] <= -3 and d["ci"][1] < 0)
        rules["harmless_no_gain"] = not rules["reproduces_their_gain"] and not rules["hurts"]
    if c["precision"] is not None:
        rules["judge_over_accepts"] = bool(c["precision"] < 0.6 and c["precision_ci"][1] < 0.75)
    out = {"collection": c, "sft": s, "rules": rules, "paper": {"M0": 77, "Original": 78, "Recovery": 88}}
    (R / "lrs_goal.json").write_text(json.dumps(out, indent=1, default=str))
    f = lambda d: "-" if not d else f"{d['delta']:+.1f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}]"
    md = ["## collection", "```", json.dumps(c, indent=1, default=str), "```", "## SFT",
          "| arm | val s7 / s8 / s9 | mean | Δ vs Original | fixed / broken vs Orig | McNemar / seed | RSR_E8 | "
          "Δ RSR_E8 | RSR_E8on | fail. incidence | first-attempt |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in s.items():
        md.append(f"| {k} | {' / '.join(str(r['val'].get(sd, '-')) for sd in (7, 8, 9))} | {r['val_mean']:.1f} | "
                  f"{f(r.get('d_vs_orig'))} | {r.get('fixed_broken_vs_orig', '-')} | "
                  f"{ {sd: round(p, 3) for sd, p in r.get('mcnemar_per_seed', {}).items()} or '-'} | "
                  f"{r.get('rsr_e8', float('nan')):.3f} | {f(r.get('rsr_e8_d_vs_orig'))} | "
                  f"{r.get('rsr_e8on', float('nan')):.3f} | {r.get('failure_incidence', float('nan')):.2f} | "
                  f"{r.get('first_attempt_success', float('nan')):.2f} |")
    md += ["", f"rules: {rules}", "paper (cross-paper): M0 77, Original 78, Recovery 88"]
    tasks = sorted({t for r in s.values() for t in r["per_task"]})
    md += ["", "| task | " + " | ".join(s) + " |", "|---|" + "---|" * len(s)]
    for t in tasks:
        md.append(f"| {t} | " + " | ".join(f"{s[k]['per_task'].get(t, float('nan')):.1f}" for k in s) + " |")
    (R / "lrs_goal.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
