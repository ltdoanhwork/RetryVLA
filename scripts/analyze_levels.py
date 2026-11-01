#!/usr/bin/env python3
"""Per-level (L1-L4) RSR for existing evals, recomputed from their episodes.jsonl (no new rollouts).

python scripts/analyze_levels.py [--out runs/_analysis/levels]

Levels: configs/env/levels_{spatial,object}.yaml (scripts/build_levels.py; official LIBERO-Recovery v1.0.0 VLM labels).
Two views: `official` = only scenes annotated upstream (spatial: the 28 step_0180 test scenes; object: 17 / 26 test
scenes), `all` = + scenes whose label is inherited from the same failure episode 20 steps earlier (weak, secondary).
Multi-seed rows: rate pooled over seeds (n = episodes x seeds), plus the per-seed sd of the per-level rate.
Unit for the CI: Wilson on pooled episodes (episodes of one scene are correlated -> optimistic; read as descriptive).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from retryvla.envs.splits import load_levels  # noqa: E402
from retryvla.eval.metrics import LEVELS, level_table, wilson  # noqa: E402

SP = "data/assets/scenes/scenes/selected_bddl_scenes_spatial"
OB = "data/assets/scenes/scenes/selected_bddl_scenes_object"
# name, suite root, run-dir suffixes (one per training seed), filter part (pi0 M0 run also has pilot)
MODELS = [
    ("pi0", SP, ["M0_recover_bddl"], "test"),
    ("pi1-nominal", SP, ["M1_pi1nom_recover_test"], None),
    ("demo-only", SP, ["M9_sft_demo_only_recover_test"], None),
    ("goal-data SFT (3 seeds)", SP, ["M9_sft_goal_recover_test", "M9_sft_goal_seed1_recover_test",
                                     "M9_sft_goal_seed2_recover_test"], None),
    ("progress-data SFT", SP, ["B2_sft_progress_recover_test"], None),
    ("competence SFT n=66 (3 seeds)", SP, ["M9_sft_comp_recover_test", "M9_sft_comp_seed1_recover_test",
                                           "M9_sft_comp_seed2_recover_test"], None),
    ("pi1 RetryVLA (3 seeds)", SP, ["M5_m5c_sft_wm_diverse_rr_recover_test", "M8_seed_pi1_s1_recover_test",
                                    "M8_seed_pi1_s2_recover_test"], None),
    ("M7 base (3 seeds)", SP, ["M7_sft_base_recover_test", "M8_seed_base_s1_recover_test",
                               "M8_seed_base_s2_recover_test"], None),
    ("Backtrack single (pi0)", SP, ["BT_bt_single_recover_test"], None),
    ("Backtrack retry (pi0)", SP, ["BT_bt_retry_recover_test"], None),
    ("Backtrack-distill SFT", SP, ["BT_sft_bt_distill_recover_test"], None),
    ("[Object] pi0", OB, ["M10_obj_pi0_recover_test"], None),
    ("[Object] pi1-nominal", OB, ["M10_obj_pi1nom_recover_test"], None),
    ("[Object] demo-only", OB, ["M10_obj_demo_only_recover_test"], None),
    ("[Object] random SFT", OB, ["M10_obj_random_recover_test"], None),
    ("[Object] goal SFT", OB, ["M10_obj_goal_recover_test"], None),
    ("[Object] competence SFT (pi1)", OB, ["M10_obj_comp_recover_test"], None),
    ("[Object] Backtrack retry (pi0)", OB, ["BT_obj_bt_retry_recover_test"], None),
]


def rows_of(tag: str, part: str | None, n_min: int) -> list[dict] | None:
    f = sorted(glob.glob(str(REPO / "runs" / f"*_{tag}")))
    if not f or not Path(f[-1], "episodes.jsonl").exists():
        return None
    rows = [json.loads(l) for l in open(Path(f[-1], "episodes.jsonl"))]
    if part:
        rows = [r for r in rows if r.get("part") == part]
    return rows if len(rows) >= n_min else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/_analysis/levels")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    maps = {r: {"official": load_levels(REPO / r, official_only=True), "all": load_levels(REPO / r)} for r in (SP, OB)}
    js, md = {}, ["# Per-level RSR (L1-L4, official LIBERO-Recovery v1.0.0 labels), recomputed from episodes.jsonl", ""]
    for view in ("official", "all"):
        md += [f"## view: {view}", "",
               "| model | seeds | all (this view) | " + " | ".join(f"{lv} rate (k/n)" for lv in LEVELS) + " |",
               "|---|---|---|" + "---|" * len(LEVELS)]
        for name, root, tags, part in MODELS:
            n_min = 260 if root == OB else 560
            seeds = [rows_of(t, part, n_min) for t in tags]
            seeds = [s for s in seeds if s]
            if not seeds:
                md.append(f"| {name} | - | pending |" + " |" * len(LEVELS))
                continue
            lv = maps[root][view]
            per = [level_table([r for r in s if r["scene"] in lv], lv) for s in seeds]
            pooled = level_table([r for s in seeds for r in s if r["scene"] in lv], lv)
            allk = sum(v["k"] for k, v in pooled.items() if k != "unk")
            alln = sum(v["n"] for k, v in pooled.items() if k != "unk")
            cells = []
            for L in LEVELS:
                v = pooled[L]
                if v["n"] == 0:
                    cells.append("-")
                    continue
                sd = np.std([p[L]["rate"] for p in per], ddof=1) if len(per) > 1 else None
                cells.append(f"{v['rate']:.3f}" + (f" ± {sd:.3f}" if sd is not None else "") + f" ({v['k']}/{v['n']})")
            lo, hi = wilson(allk, alln)
            md.append(f"| {name} | {len(seeds)} | {allk / max(alln, 1):.3f} [{lo:.2f}, {hi:.2f}] | " + " | ".join(cells) + " |")
            js.setdefault(view, {})[name] = {"seeds": len(seeds), "pooled": pooled,
                                             "per_seed": per, "all": [allk, alln]}
        n_sc = {r: {L: sum(1 for t, v in maps[r][view].items() if v == L) for L in LEVELS} for r in (SP, OB)}
        md += ["", f"Scenes per level in this view (train + test): spatial {n_sc[SP]}, object {n_sc[OB]}.", ""]
    tmp = out / "levels.json.tmp"
    tmp.write_text(json.dumps(js, indent=1, default=float))
    tmp.replace(out / "levels.json")
    tmp = out / "levels.md.tmp"
    tmp.write_text("\n".join(md) + "\n")
    tmp.replace(out / "levels.md")
    print("\n".join(md))


if __name__ == "__main__":
    main()
