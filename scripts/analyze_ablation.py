#!/usr/bin/env python3
"""6-arm ablation (plan todo `ablation`, RQ2 + Part 2): Spatial test RSR (56 scenes x 10) + nominal (10 x 50), 3 seeds.

python scripts/analyze_ablation.py        # -> runs/_analysis/ablation/{ablation.md, ablation.json}

Arms (same init pi1-nominal, 6k steps, bs 32, 50 % nominal replay, same eval seeds / protocol; only the recovery /
future mix and the SFT seed differ):
  1 demo-only   2 action-grounded (pi1)   3 action-free C_pi-filtered   4 action-free unfiltered (random futures)
  5 hybrid (2 + 3)   6 Backtrack-distill
Stats: per-seed RSR mean +- sd (ddof 1); vs pi1: scene x seed bootstrap 95 % CI of the difference + paired sign-flip p
over scenes (stats_cluster.compare), exact McNemar on the paired episodes pooled over seed pairs (seed k vs pi1 seed k;
same perturbations); per-level on official LIBERO-Recovery labels only; nominal: mean +- sd over seeds, margin vs
pi1-nominal 0.796 (SFT init, pre-registered eps 2 pt) with a 10-task x seed bootstrap.
Only finished evals (summary.json status done) are used; missing seeds are listed as pending.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
from analyze_b32 import nom_delta  # noqa: E402
from stats_cluster import compare, mcnemar  # noqa: E402

from retryvla.envs.splits import load_levels  # noqa: E402

OUT = REPO / "runs/_analysis/ablation"
SP = REPO / "data/assets/scenes/scenes/selected_bddl_scenes_spatial"
PI1_NOM_SR = 0.796
EPS = 0.02
# arm -> list over seeds 0..2 of (recover tag, nominal tag)
ARMS = {
    "1 demo-only": [("M9_sft_demo_only_recover_test", "M9_sft_demo_only_nominal"),
                    ("M9_sft_demo_only_s1_recover_test", "M9_sft_demo_only_s1_nominal"),
                    ("M9_sft_demo_only_s2_recover_test", "M9_sft_demo_only_s2_nominal")],
    "2 action-grounded (pi1)": [("M5_m5c_sft_wm_diverse_rr_recover_test", "M5_m5c_sft_wm_diverse_rr_nominal"),
                                ("M8_seed_pi1_s1_recover_test", "M8_seed_pi1_s1_nominal"),
                                ("M8_seed_pi1_s2_recover_test", "M8_seed_pi1_s2_nominal")],
    **{name: [(f"P4_{stem}_s{k}_recover_test", f"P4_{stem}_s{k}_nominal") for k in range(3)]
       for name, stem in (("3 action-free, C_pi-filtered", "af_arm3_free_filtered"),
                          ("4 action-free, unfiltered", "af_arm4_free_unfiltered"),
                          ("5 hybrid (2 + 3)", "af_arm5_hybrid"))},
    "6 Backtrack-distill": [("BT_sft_bt_distill_recover_test", "BT_sft_bt_distill_nominal"),
                            ("BT_sft_bt_distill_s1_recover_test", "BT_sft_bt_distill_s1_nominal"),
                            ("BT_sft_bt_distill_s2_recover_test", "BT_sft_bt_distill_s2_nominal")],
}
REF = "2 action-grounded (pi1)"
PI1_NOM_TAG = "M1_pi1nom_nominal"


def rows(tag: str, n_min: int):
    """episodes of the newest FINISHED run of tag (summary status done, >= n_min episodes) or None"""
    for d in sorted(glob.glob(str(REPO / "runs" / f"*_{tag}")), reverse=True):
        try:
            if json.load(open(Path(d, "summary.json"))).get("status") != "done":
                continue
            r = [json.loads(l) for l in open(Path(d, "episodes.jsonl"))]
        except Exception:
            continue
        if len(r) >= n_min:
            return r
    return None


def ep_dict(r):
    return {(x["scene"], x["gripper_init"], x["trial"]): float(x["success"]) for x in r}


def nom_vec(r):
    d = {}
    for x in r:
        d.setdefault(x["task_id"], []).append(float(x["success"]))
    return [np.mean(d[k]) for k in sorted(d)]


def sd(v):
    return f" ± {np.std(v, ddof=1):.3f}" if len(v) > 1 else ""


def main() -> None:
    rng = np.random.default_rng(0)
    lv = load_levels(SP, official_only=True)
    data = {}
    for name, seeds in ARMS.items():
        R = {k: rows(rt, 560) for k, (rt, _) in enumerate(seeds)}
        N = {k: rows(nt, 500) for k, (_, nt) in enumerate(seeds)}
        data[name] = ({k: v for k, v in R.items() if v}, {k: v for k, v in N.items() if v})
    pi1n = rows(PI1_NOM_TAG, 500)
    pi1n = np.array([nom_vec(pi1n)]) if pi1n else None
    refR = data[REF][0]
    md = ["# 6-arm ablation (Spatial test = dev split, 56 scenes x 10 episodes; nominal LIBERO-Spatial 10 x 50)", "",
          "Same init (pi1-nominal), 6k steps, bs 32, 50 % nominal replay, same eval seeds; arms differ only in the "
          "recovery / future mix. ± = sd over SFT seeds. vs pi1: scene x seed bootstrap 95 % CI + sign-flip p over "
          "scenes; McNemar = exact, paired episodes pooled over seed pairs (seed k vs pi1 seed k). Nominal margin: "
          f"mean nominal >= pi1-nominal {PI1_NOM_SR:.3f} - 2 pt (10-task x seed bootstrap of the difference).", "",
          "| arm | seeds (RSR / nom) | test RSR | per seed | close / open | dRSR vs pi1 [95% CI], p | McNemar vs pi1 "
          "n01/n10, p | nominal | dNSR vs pi1-nominal [95% CI] | margin |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    lv_md = ["", "## Per level (official LIBERO-Recovery labels; 28 of 56 test scenes: 1 L1 / 11 L2 / 16 L3, no L4)", "",
             "Rates pool the episodes of the official-label scenes over seeds; ± = sd of the per-seed rate.", "",
             "| arm | seeds | official 28 scenes | L1 (1) | L2 (11) | L3 (16) |", "|---|---|---|---|---|---|"]
    js = {}
    for name, (R, N) in data.items():
        if not R:
            md.append(f"| {name} | 0 / {len(N)} | pending | | | | | | | |")
            lv_md.append(f"| {name} | 0 | pending | | | |")
            continue
        ks = sorted(R)
        rs = [np.mean([x["success"] for x in R[k]]) for k in ks]
        allr = [x for k in ks for x in R[k]]
        co = " / ".join(f"{np.mean([x['success'] for x in allr if x['gripper_init'] == g]):.3f}" for g in ("close", "open"))
        E = [ep_dict(R[k]) for k in ks]
        if name == REF:
            dcell, mcell = "(ref)", "(ref)"
            dd = None
        else:
            d, ci, p, *_ = compare(E, [ep_dict(refR[k]) for k in sorted(refR)], rng)
            dcell = f"{d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}], p {'< 1e-4' if p == 0 else f'{p:.2g}'}"
            a_, b_ = [], []
            for k in ks:
                if k in refR:
                    ea, eb = ep_dict(R[k]), ep_dict(refR[k])
                    keys = sorted(set(ea) & set(eb))
                    a_ += [ea[q] > 0.5 for q in keys]
                    b_ += [eb[q] > 0.5 for q in keys]
            n01, n10, pm = mcnemar(np.array(a_), np.array(b_))
            mcell = f"{n01}/{n10}, p {pm:.2g}"
            dd = {"diff": d, "ci": list(ci), "p_flip": p, "mcnemar": [n01, n10, pm]}
        if N:
            nv = np.array([nom_vec(N[k]) for k in sorted(N)])
            ns = nv.mean(1)
            nd, lo, hi = nom_delta(nv, pi1n, rng) if pi1n is not None else (np.nan,) * 3
            ncell = f"{ns.mean():.3f}{sd(ns)}"
            dncell = f"{nd:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            mg = "pass" if ns.mean() >= PI1_NOM_SR - EPS else "**FAIL**"
        else:
            ns, ncell, dncell, mg = np.array([]), "pending", "", ""
        md.append(f"| {name} | {len(ks)} / {len(N)} | {np.mean(rs):.3f}{sd(rs)} | "
                  f"{', '.join(f's{k} {r:.3f}' for k, r in zip(ks, rs))} | {co} | {dcell} | {mcell} | {ncell} | {dncell} | {mg} |")
        lvc, lvj = [], {}
        for L in (None, "L1", "L2", "L3"):
            per = [np.mean([x["success"] for x in R[k] if (lv.get(x["scene"]) == L if L else x["scene"] in lv)])
                   for k in ks]
            pool = np.mean([x["success"] for x in allr if (lv.get(x["scene"]) == L if L else x["scene"] in lv)])
            lvc.append(f"{pool:.3f}{sd(per)}")
            lvj[L or "official"] = per
        lv_md.append(f"| {name} | {len(ks)} | " + " | ".join(lvc) + " |")
        js[name] = {"seeds": ks, "rsr": rs, "close_open": co, "vs_pi1": dd, "nominal": list(map(float, ns)),
                    "nominal_seeds": sorted(N), "levels": lvj}
    md += lv_md
    OUT.mkdir(parents=True, exist_ok=True)
    for fn, body in (("ablation.md", "\n".join(md) + "\n"), ("ablation.json", json.dumps(js, indent=1, default=float))):
        tmp = OUT / (fn + ".tmp")
        tmp.write_text(body)
        tmp.replace(OUT / fn)
    print("\n".join(md))


if __name__ == "__main__":
    main()
