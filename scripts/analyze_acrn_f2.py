"""ACR-N F2 deployability analysis (pre-registration: PLAN "ACR-N follow-ups F1 / F1b / F2").

python scripts/analyze_acrn_f2.py [--root data/acr/n1] [--out runs/_analysis/acrn_f2]
Inputs: data/acr/n1/F2/<id>.json (head-pose grasp_slow / hover / grasp_firm x 8), data/acr/n1/N (GT-pose table),
data/recovery/acrn_single_fresh/s*/p*/rows.jsonl (fresh-seed `single`), runs/_analysis/acrn/rows_b2800.jsonl (wm_only).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_acrn import execute, folds, load, mcnemar  # noqa: E402

from retryvla import REPO_ROOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/acr/n1")
    ap.add_argument("--out", default="runs/_analysis/acrn_f2")
    args = ap.parse_args()
    root = REPO_ROOT / args.root
    gt = {r["instance_id"]: r for r in load(root)}
    f2 = {}
    for p in sorted((root / "F2").glob("*.json")):
        r = json.loads(p.read_text())
        r["O"], r["O_steps"] = np.asarray(r["O"], int), np.asarray(r["O_steps"], int)
        f2[r["instance_id"]] = r
    sf = {}
    for p in glob.glob(str(REPO_ROOT / "data/recovery/acrn_single_fresh/s*/**/rows.jsonl"), recursive=True):
        for ln in open(p):
            x = json.loads(ln)
            if x["arm"] == "single":
                sf[x["instance_id"]] = bool(x["solved"])
    ids = [i for i in gt if i in f2]
    recs = [gt[i] for i in ids]
    fo = folds(recs)
    W3 = np.full(3, 440.0)
    WN = np.array([350 if o.startswith("wm_") else (300 if o == "vla" else 440) for o in recs[0]["options"]], float)
    opts3 = f2[ids[0]]["options"]
    wm_rows = pd.read_json(REPO_ROOT / "runs/_analysis/acrn/rows_b2800.jsonl", lines=True).set_index("instance_id")
    rows = []
    for budget in (2800, 1400):
        for k in sorted(set(fo)):
            tr = [ids[i] for i in np.nonzero(fo != k)[0]]
            rate = [np.mean([execute(f2[t], [o] * 8, budget, W3)["solved"] for t in tr]) for o in range(3)]
            o_cf = int(np.argmax(rate))
            for i in np.nonzero(fo == k)[0]:
                iid = ids[i]
                g, h = gt[iid], f2[iid]
                gi = {o: g["options"].index(o) for o in opts3}
                row = {"instance_id": iid, "set": g["set"], "level": g["level"], "cause": g["cause"], "budget": budget,
                       "obj_err_cm": 100 * h["obj_err_m"], "goal_err_cm": 100 * h["goal_err_m"],
                       "single_hist": bool(g["single_solved"]), "single_fresh": sf.get(iid),
                       "wm_only": bool(wm_rows.loc[iid, "wm_only"]) if budget == 2800 else None,
                       "head_cf_option": opts3[o_cf],
                       "head_cf_fixed": execute(h, [o_cf] * 8, budget, W3)["solved"]}
                for oi, o in enumerate(opts3):
                    row[f"head_{o}"] = execute(h, [oi] * 8, budget, W3)["solved"]
                    row[f"gt_{o}"] = execute(g, [gi[o]] * 8, budget, WN)["solved"]
                    row[f"head_rate_{o}"] = float((h["O"][oi] == 1).mean())
                    row[f"gt_rate_{o}"] = float((g["O"][gi[o]] == 1).mean())
                rows.append(row)
    d = pd.DataFrame(rows)
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    d.to_json(out / "rows.jsonl", orient="records", lines=True)
    md = [f"# ACR-N F2: observation-only pose ({len(ids)} instances with F2 data; fresh single on "
          f"{sum(v is not None for v in d[d.budget == 2800].single_fresh)})", ""]
    e = d[d.budget == 2800]
    md += [f"Head pose error at the failure state: object xy median {e.obj_err_cm.median():.1f} cm, p90 "
           f"{e.obj_err_cm.quantile(0.9):.1f} cm; goal xy median {e.goal_err_cm.median():.1f} cm.", ""]
    dec = {}
    for budget in (2800, 1400):
        db = d[d.budget == budget]
        md += [f"## Budget {budget}", "", "| arm | U solved | S solved | U+S | McNemar vs fresh single (U) n01/n10, p |",
               "|---|---|---|---|---|"]
        arms = ["single_hist", "single_fresh"] + (["wm_only"] if budget == 2800 else []) + \
            [f"gt_{o}" for o in opts3] + [f"head_{o}" for o in opts3] + ["head_cf_fixed"]
        u = db[db.set == "U"]
        uf = u[u.single_fresh.notna()]
        for a in arms:
            m = mcnemar(uf.single_fresh.values.astype(bool), uf[a].values.astype(bool))
            s = db[db.set == "S"][a].dropna().astype(float)
            md.append(f"| {a} | {u[a].dropna().astype(float).mean():.3f} ({int(u[a].dropna().astype(float).sum())}) | "
                      f"{s.mean():.3f} | {db[a].dropna().astype(float).mean():.3f} | {m[0]}/{m[1]}, {m[2]:.3f} |")
        md += ["", "per-trial success rate GT -> head pose (U / S): " + ", ".join(
            f"{o} {u[f'gt_rate_{o}'].mean():.3f} -> {u[f'head_rate_{o}'].mean():.3f} / "
            f"{db[db.set == 'S'][f'gt_rate_{o}'].mean():.3f} -> {db[db.set == 'S'][f'head_rate_{o}'].mean():.3f}"
            for o in opts3),
               "head_cf option per fold: " + json.dumps(db.head_cf_option.value_counts().to_dict()),
               "U head grasp_slow by object pose error: " + ", ".join(
                   f"{lab} n {len(g_)}: head {g_.head_grasp_slow.mean():.2f} / gt {g_.gt_grasp_slow.mean():.2f}"
                   for lab, g_ in u.groupby(pd.cut(u.obj_err_cm, [0, 3, 6, 1000]), observed=True)), ""]
        m = mcnemar(uf.single_fresh.values.astype(bool), uf.head_grasp_slow.values.astype(bool))
        dec[str(budget)] = {"head_grasp_slow_U": float(u.head_grasp_slow.mean()),
                            "gt_grasp_slow_U": float(u.gt_grasp_slow.mean()),
                            "single_fresh_U": float(uf.single_fresh.astype(float).mean()) if len(uf) else None,
                            "head_minus_fresh_U": float(uf.head_grasp_slow.mean() - uf.single_fresh.astype(float).mean())
                            if len(uf) else None, "mcnemar_vs_fresh": list(m),
                            "head_grasp_slow_S": float(db[db.set == "S"].head_grasp_slow.mean())}
    p = dec["2800"]
    dec["F2_pass"] = bool(p["head_grasp_slow_U"] >= 0.22 and p["head_minus_fresh_U"] is not None and
                          p["head_minus_fresh_U"] >= 0.08 and p["mcnemar_vs_fresh"][2] < 0.10 and
                          p["mcnemar_vs_fresh"][0] > p["mcnemar_vs_fresh"][1])
    md += ["## Pre-registered decision (F2)", "", "```", json.dumps(dec, indent=1), "```"]
    (out / "f2.md").write_text("\n".join(md))
    (out / "decision.json").write_text(json.dumps(dec, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
