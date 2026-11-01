#!/usr/bin/env python3
"""Part 2: Backtrack baseline (no WM) vs pi0 / pi1, Spatial test (dev) + Object test (held-out).

python scripts/analyze_backtrack.py [--out runs/_analysis/backtrack]

Unit = test scene (10 trials averaged); several training seeds -> scene x seed bootstrap + paired sign-flip p
(stats_cluster.compare). Per-level = official LIBERO-Recovery labels only (configs/env/levels_*.yaml, src=official).
Nominal (distill only): LIBERO-Spatial 10 x 50, dNSR vs pi1-nominal (= SFT init), 10-task bootstrap; eps = 2 pt.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
from analyze_b32 import nom_delta, nom_tasks  # noqa: E402
from stats_cluster import compare  # noqa: E402

from retryvla.envs.splits import load_levels  # noqa: E402

SP = REPO / "data/assets/scenes/scenes/selected_bddl_scenes_spatial"
OB = REPO / "data/assets/scenes/scenes/selected_bddl_scenes_object"
PI1 = ["M5_m5c_sft_wm_diverse_rr_recover_test", "M8_seed_pi1_s1_recover_test", "M8_seed_pi1_s2_recover_test"]
PI1_NOM = ["M5_m5c_sft_wm_diverse_rr_nominal", "M8_seed_pi1_s1_nominal", "M8_seed_pi1_s2_nominal"]
ROWS_SP = [  # name, recover tags, nominal tags, test-time module, sim budget note
    ("pi0 (SmolVLA base)", ["@pi0"], ["M0_baseline_spatial_nominal"], "none"),
    ("pi1-nominal (demo SFT)", ["M1_pi1nom_recover_test"], ["M1_pi1nom_nominal"], "none"),
    ("demo-only (+6k steps)", ["M9_sft_demo_only_recover_test"], ["M9_sft_demo_only_nominal"], "none"),
    ("**pi1 RetryVLA (3 seeds)**", PI1, PI1_NOM, "none"),
    ("Backtrack single on pi0", ["BT_bt_single_recover_test"], [], "retreat at t=0"),
    ("Backtrack retry on pi0", ["BT_bt_retry_recover_test"], [], "retreat + <=3 retraces (stall / timeout)"),
    ("Backtrack-distill SFT (1 seed)", ["BT_sft_bt_distill_recover_test"], ["BT_sft_bt_distill_nominal"], "none"),
]
ROWS_OB = [
    ("[Object] pi0", ["M10_obj_pi0_recover_test"], [], "none"),
    ("[Object] pi1 (competence SFT)", ["M10_obj_comp_recover_test"], [], "none"),
    ("[Object] Backtrack retry on pi0", ["BT_obj_bt_retry_recover_test"], [], "retreat + <=3 retraces"),
]


def eps(tag, n_min, part=None):
    f = sorted(glob.glob(str(REPO / "runs" / f"*_{tag}")))
    if not f or not Path(f[-1], "episodes.jsonl").exists():
        return None, None
    rows = [json.loads(l) for l in open(Path(f[-1], "episodes.jsonl"))]
    if part:
        rows = [r for r in rows if r.get("part") == part]
    if len(rows) < n_min:
        return None, None
    return {(r["scene"], r["gripper_init"], r["trial"]): float(r["success"]) for r in rows}, rows


def load(tags, n_min):
    E, R = [], []
    for t in tags:
        e, r = eps("M0_recover_bddl", n_min, "test") if t == "@pi0" else eps(t, n_min)
        if e:
            E.append(e)
            R.append(r)
    return E, R


def fmt(d, ci, p):
    return f"{d:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}], p {'< 1e-4' if p == 0 else f'{p:.2g}'}"


def table(rows_def, ref_names, root, n_min, rng):
    lv = load_levels(root, official_only=True)
    refs = {k: load(v, n_min)[0] for k, v in ref_names.items()}
    L = ["| method | seeds | test RSR | close / open | L1 / L2 / L3 (official) | mean #backtracks | "
         + " | ".join(f"dRSR vs {k}" for k in refs) + " | nominal | dNSR vs pi1-nominal |",
         "|---|---|---|---|---|---|" + "---|" * len(refs) + "---|---|"]
    js = {}
    pi1nom = nom_tasks(["M1_pi1nom_nominal"])
    for name, rt, nt, _ in rows_def:
        E, R = load(rt, n_min)
        if not E:
            L.append(f"| {name} | - | pending |" + " |" * (4 + len(refs) + 2))
            continue
        rs = [np.mean(list(e.values())) for e in E]
        rsr = f"{np.mean(rs):.3f}" + (f" ± {np.std(rs, ddof=1):.3f}" if len(rs) > 1 else "")
        allr = [r for x in R for r in x]
        co = "/".join(f"{np.mean([r['success'] for r in allr if r['gripper_init'] == g]):.3f}" for g in ("close", "open"))
        lvs = []
        for k in ("L1", "L2", "L3"):
            v = [r["success"] for r in allr if lv.get(r["scene"]) == k]
            lvs.append(f"{np.mean(v):.3f}" if v else "-")
        nbt = [r["n_bt"] for r in allr if "n_bt" in r]
        cells = []
        for k, ref in refs.items():
            if not ref or rt == ref_names[k]:
                cells.append("(ref)" if ref else "-")
                continue
            d, ci, p, *_ = compare(E, ref, rng)
            cells.append(fmt(d, ci, p))
        nn = nom_tasks(nt) if nt and all(glob.glob(str(REPO / "runs" / f"*_{t}" / "episodes.jsonl")) for t in nt) else None
        if nn is not None:
            d, lo, hi = nom_delta(nn, pi1nom, rng)
            nom, dn = f"{nn.mean():.3f}", f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}] ({'pass' if d >= -0.02 else 'FAIL'})"
        else:
            nom, dn = ("pending" if nt else "-"), ""
        L.append(f"| {name} | {len(E)} | {rsr} | {co} | {' / '.join(lvs)} | "
                 f"{np.mean(nbt):.2f} | " if nbt else f"| {name} | {len(E)} | {rsr} | {co} | {' / '.join(lvs)} | - | ")
        L[-1] += " | ".join(cells) + f" | {nom} | {dn} |"
        js[name] = {"rsr": rs, "close_open": co, "levels": lvs, "n_bt": float(np.mean(nbt)) if nbt else None,
                    "nominal": None if nn is None else float(nn.mean())}
    return L, js


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/_analysis/backtrack")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    md = ["# Backtrack baseline (Part 2)", "", "## Spatial test (dev split, 56 scenes x 10)", ""]
    L, js = table(ROWS_SP, {"pi0": ["@pi0"], "pi1 (3 seeds)": PI1}, SP, 560, rng)
    md += L + ["", "## Object test (held-out, 26 scenes x 10; recipe frozen from Spatial)", ""]
    L2, js2 = table(ROWS_OB, {"pi0": ["M10_obj_pi0_recover_test"], "pi1": ["M10_obj_comp_recover_test"]}, OB, 260, rng)
    md += L2
    for name, body in (("backtrack.md", "\n".join(md) + "\n"), ("backtrack.json", json.dumps({"spatial": js, "object": js2},
                                                                                              indent=1, default=float))):
        tmp = out / (name + ".tmp")
        tmp.write_text(body)
        tmp.replace(out / name)
    print("\n".join(md))


if __name__ == "__main__":
    main()
