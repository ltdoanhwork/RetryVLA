#!/usr/bin/env python
"""M11 arm A: can the WM plan recovery bridges itself, without the hand-made proposal library?

python scripts/analyze_m11.py [--out runs/_analysis/m11]

Arms (all 16 verified sims / instance, paired continuation seeds, same pi0 failures):
  M11 a : wm_no_prim (WM+critic ranking of VLA samples only), wm_plan (CEM in imagination from broad noise)
  M5b   : random, wm_diverse, wm_qd, wm_noise
  M9    : wm_goal, wm_top, random_diverse, sim_prefix_critic
Rows are unioned across run dirs / resumed parts per shard (dedup on instance_id x arm, last wins); the paired
set = instances present for EVERY arm. Unit of all tests = failure instance: exact McNemar on `solved`,
instance bootstrap 95% CI of the solve-rate difference, sign-flip permutation test on hits.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_m5b import boot_ci, mcnemar, signflip  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SRC = {"M11": "*_M11_plan_search_a_s*", "M5b": "*_M5_m5b_compare_arms_s*", "M9": "*_M9_mech_search_s*"}
ARMS = {"M11": ["wm_no_prim", "wm_plan"], "M5b": ["random", "wm_diverse", "wm_qd", "wm_noise"],
        "M9": ["wm_goal", "wm_top", "random_diverse", "sim_prefix_critic"]}
# arms whose candidates come from the hand-made proposal library (VLA samples + primitives + perturbations)
LIBRARY = ["random", "wm_diverse", "wm_qd", "wm_goal", "wm_top", "random_diverse", "sim_prefix_critic"]
ORDER = ["random", "random_diverse", "wm_diverse", "wm_qd", "wm_top", "wm_goal", "sim_prefix_critic", "wm_noise",
         "wm_no_prim", "wm_plan"]


def load() -> pd.DataFrame:
    out = []
    for src, pat in SRC.items():
        for f in sorted(glob.glob(str(REPO / "runs" / pat / "per_instance.jsonl"))):
            shard = int(re.search(r"_s(\d+)of8", f).group(1))
            for l in open(f):
                r = json.loads(l)
                if r["arm"] in ARMS[src]:
                    out.append({**r, "src": src, "shard": shard, "run": Path(f).parent.name})
    d = pd.DataFrame(out)
    return d.drop_duplicates(["instance_id", "arm"], keep="last")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/_analysis/m11")
    ap.add_argument("--s3_total", type=int, default=0, help="pi0 failures in shard 3 (0 = unknown)")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    d = load()
    m11 = d[d.src == "M11"]
    per_shard = m11[m11.arm == "wm_plan"].groupby("shard").instance_id.nunique().to_dict()
    arms = [x for x in ORDER if x in set(d.arm)]
    sets = [set(d[d.arm == x].instance_id) for x in arms]
    paired = sorted(set.intersection(*sets))
    d = d[d.instance_id.isin(paired)]
    n = len(paired)
    W = {k: d.pivot(index="instance_id", columns="arm", values=k).loc[paired] for k in
         ["solved", "hits", "budget"]}
    S, H = W["solved"].astype(bool), W["hits"].astype(float)

    rows = []
    for x in arms:
        g = d[d.arm == x].set_index("instance_id").loc[paired]
        s = g.solved.astype(bool).values
        rs = g.robust_solved if "robust_solved" in g and g.robust_solved.notna().all() else None
        steps = g.sim_steps_verify if "sim_steps_verify" in g and g.sim_steps_verify.notna().all() else None
        extra = g.sim_steps_extra.fillna(0) if "sim_steps_extra" in g else 0 * g.hits
        sg = g[g.solved.astype(bool)]
        lo, hi = boot_ci(s.astype(float))
        rows.append({
            "arm": x, "src": g.src.iloc[0], "n": n, "solved": s.mean(), "solved_ci": f"[{lo:.3f}, {hi:.3f}]",
            "robust_solved": None if rs is None else float(rs.astype(bool).mean()),
            "hits_per_sim": float(g.hits.sum() / g.budget.sum()),
            "sim_steps_verify": None if steps is None else float(steps.mean()),
            "sim_steps_extra": float(extra.mean()),
            "hits_per_10k_steps": None if steps is None else float(1e4 * g.hits.sum() / (steps + extra).sum()),
            "succ_cells": float(sg.succ_cells.mean()) if len(sg) else np.nan,
            "succ_jerk": float(sg.succ_jerk.mean()) if len(sg) else np.nan,
            "succ_manifold": float(sg.succ_manifold.mean()) if len(sg) else np.nan,
            "succ_pairwise": float(pd.to_numeric(sg.succ_pairwise, errors="coerce").mean()) if len(sg) else np.nan,
        })
    T = pd.DataFrame(rows)
    T.to_csv(out / "per_arm.csv", index=False)

    tests = []
    for ref in ["random", "wm_diverse"]:
        for x in arms:
            if x == ref:
                continue
            n01, n10, p = mcnemar(S[ref].values, S[x].values)
            diff = S[x].values.astype(float) - S[ref].values.astype(float)
            lo, hi = boot_ci(diff)
            tests.append({"arm": x, "vs": ref, "d_solved": diff.mean(), "ci": f"[{lo:+.3f}, {hi:+.3f}]",
                          "only_arm": n01, "only_ref": n10, "mcnemar_p": p,
                          "d_hits": (H[x] - H[ref]).mean(), "signflip_p_hits": signflip((H[x] - H[ref]).values)})
    P = pd.DataFrame(tests)
    P.to_csv(out / "paired_tests.csv", index=False)

    lib_any = S[[x for x in LIBRARY if x in S]].any(axis=1)
    others_any = S[[x for x in arms if x != "wm_plan"]].any(axis=1)
    only_plan_lib = sorted(S.index[S["wm_plan"] & ~lib_any])
    only_plan_all = sorted(S.index[S["wm_plan"] & ~others_any])
    only_noprim_lib = sorted(S.index[S["wm_no_prim"] & ~lib_any])
    lib_not_plan = sorted(S.index[lib_any & ~S["wm_plan"]])
    union_lib, union_all = int(lib_any.sum()), int((lib_any | S["wm_plan"] | S["wm_no_prim"]).sum())
    # strategy cells of wm_plan successes that no library arm reached on the same instance
    cl = d.pivot(index="instance_id", columns="arm", values="succ_cells_list").loc[paired]
    new_cells = []
    for i in paired:
        lib_cells = set().union(*[set(cl.at[i, x] or []) for x in LIBRARY if x in cl])
        pc = set(cl.at[i, "wm_plan"] or [])
        if pc:
            new_cells.append(len(pc - lib_cells) / len(pc))
    json.dump({"n_paired": n, "m11_instances_per_shard": per_shard, "only_wm_plan_vs_library": only_plan_lib,
               "only_wm_plan_vs_all": only_plan_all, "only_wm_no_prim_vs_library": only_noprim_lib,
               "library_solved_not_wm_plan": lib_not_plan}, open(out / "only_sets.json", "w"), indent=1)

    s3 = per_shard.get(3, 0)
    partial = not (a.s3_total and s3 >= a.s3_total)
    fmt = lambda v, k=3: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{k}f}"  # noqa: E731
    L = [f"# M11 arm A analysis{' (PARTIAL: shard 3 unfinished)' if partial else ''}", "",
         f"Generated by `scripts/analyze_m11.py`. Paired failure instances: **{n}** "
         f"(M11 wm_plan instances per shard: {per_shard}"
         + (f"; s3 {s3}/{a.s3_total}" if a.s3_total else "") + "). Shards 0-3 of the M5b selection (select_seed 2026), "
         "same pi0 failures, 16 verified sims per arm, paired continuation seeds. M5b has no robust re-verification "
         "and no sim-step logging (—).", "",
         "## Per arm", "",
         "| arm | src | solved [95% CI] | robust solved | hits/sim | sim steps verify | hits / 10k steps | succ cells | succ jerk | succ manifold | succ pairwise |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in T.itertuples():
        L.append(f"| {r.arm} | {r.src} | {r.solved:.3f} {r.solved_ci} | {fmt(r.robust_solved)} | {r.hits_per_sim:.3f} | "
                 f"{fmt(r.sim_steps_verify, 0)}{' (+' + format(r.sim_steps_extra, '.0f') + ' prefix)' if r.sim_steps_extra else ''} | "
                 f"{fmt(r.hits_per_10k_steps, 2)} | {fmt(r.succ_cells, 2)} | {fmt(r.succ_jerk)} | {fmt(r.succ_manifold)} | "
                 f"{fmt(r.succ_pairwise)} |")
    L += ["", "## Paired tests (unit = failure instance)", "",
          "| arm | vs | Δ solved [boot 95% CI] | only arm / only ref | McNemar p | Δ hits/inst | sign-flip p |",
          "|---|---|---|---|---|---|---|"]
    for r in P.itertuples():
        L.append(f"| {r.arm} | {r.vs} | {r.d_solved:+.3f} {r.ci} | {r.only_arm} / {r.only_ref} | {r.mcnemar_p:.3g} | "
                 f"{r.d_hits:+.2f} | {r.signflip_p_hits:.3g} |")
    L += ["", "## Outside the proposal library", "",
          f"- Library arms ({', '.join(LIBRARY)}) solve {union_lib}/{n} instances jointly (union); "
          f"adding wm_plan + wm_no_prim: {union_all}/{n}.",
          f"- Solved ONLY by wm_plan (no library arm solves it): **{len(only_plan_lib)}**; "
          f"by wm_plan and no other arm at all (incl. wm_noise, wm_no_prim): **{len(only_plan_all)}**.",
          f"- Solved by wm_no_prim but by no library arm: {len(only_noprim_lib)}.",
          f"- Solved by some library arm but not wm_plan: {len(lib_not_plan)}.",
          f"- Mean fraction of wm_plan success strategy cells not reached by any library arm on the same instance: "
          f"{np.mean(new_cells):.2f} (over {len(new_cells)} instances wm_plan solves)." if new_cells else "",
          "", "Instance lists: `only_sets.json`; tables: `per_arm.csv`, `paired_tests.csv`."]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
