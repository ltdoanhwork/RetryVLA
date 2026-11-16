#!/usr/bin/env python
"""T2Mem PoC 0 (diagnostic, 0 new sim): does the policy REPEAT the same failed grasp within an episode?

python scripts/poc_mem_repeat.py [--models pi1_s0 m7base_s0] [--tol 0.02]

Reads the per-step telemetry of the re-simulated failed LIBERO-Recover test episodes
(runs/_analysis/failures/resim/<model>/<iid>.npz, docs/FAILURE_ANALYSIS.md) and their failure-mode labels
(runs/_analysis/failures/modes.json). A grasp attempt = a gripper close command (a6 crosses 0 upwards) that is held
>= 5 steps. The attempt pose is the EEF position relative to the bowl at the close step. An attempt "lifts" if the bowl
rises > 3 cm above its pre-attempt height while the gripper is still closed. An attempt is a REPEAT if its relative pose
is within --tol (m, 3D) of an earlier attempt of the same episode that did not lift.

Memory (T2Mem-style "remember what failed") can only help on episodes with >= 2 attempts where later attempts repeat
an earlier failed pose. Decision rule written before running: if fewer than 20 % of the (feasible) failed episodes
contain a repeat, approach A (memory policy trained on fail -> recovery episodes) is dropped.
Output: runs/_analysis/t2mem/poc0_repeat.{md,json}. Diagnostic only (dev-set episodes, nothing is tuned on it).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESIM = REPO / "runs/_analysis/failures/resim"
OUT = REPO / "runs/_analysis/t2mem"


def attempts(t: np.ndarray, cols: list[str], min_hold: int = 5, merge_gap: int = 0, near: float = 1e9) -> list[dict]:
    c = {k: i for i, k in enumerate(cols)}
    a6 = t[:, c["a6"]]
    closed = a6 > 0
    n = len(t)
    segs, i = [], 1
    while i < n:                                  # closed segments [i, j)
        if closed[i] and not closed[i - 1]:
            j = i
            while j < n and closed[j]:
                j += 1
            if segs and i - segs[-1][1] < merge_gap:  # short re-open = chatter -> same attempt
                segs[-1][1] = j
            else:
                segs.append([i, j])
            i = j
        else:
            i += 1
    out = []
    for i, j in segs:
        if j - i < min_hold:
            continue
        eef = t[i, [c["eef_x"], c["eef_y"], c["eef_z"]]]
        bowl = t[i, [c["bowl_x"], c["bowl_y"], c["bowl_z"]]]
        if np.linalg.norm(eef - bowl) > near:
            continue
        z0 = t[max(0, i - 1), c["bowl_z"]]
        lift = float(t[i:j, c["bowl_z"]].max() - z0)
        touch = bool((t[i:j, c["c_left"]] > 0).any() or (t[i:j, c["c_right"]] > 0).any())
        out.append({"t": int(i), "hold": int(j - i), "rel": (eef - bowl).tolist(), "abs": eef.tolist(),
                    "lift": lift, "lifted": lift > 0.03, "touch": touch})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["pi1_s0", "m7base_s0"])
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--min_hold", type=int, default=5)
    ap.add_argument("--merge_gap", type=int, default=3, help="re-open shorter than this = same attempt (chatter)")
    ap.add_argument("--near", type=float, default=0.10, help="only closes with EEF within this of the bowl (m)")
    args = ap.parse_args()
    modes = {(e["model"], e["iid"]): e.get("mode") for e in json.load(open(RESIM.parent / "modes.json"))["episodes"]}
    res, lines = {}, ["# T2Mem PoC 0: repeated failed grasps within an episode", "",
                      f"tol = {args.tol} m (EEF pose relative to bowl at close), attempt = close held >= "
                      f"{args.min_hold} steps (re-open < {args.merge_gap} merged) with EEF within {args.near} m of the "
                      "bowl. Failed episodes only; `f_infeasible` excluded. Within = attempt within tol of an EARLIER "
                      "failed attempt of the same episode; cross (null) = same number of failed attempts drawn from "
                      "OTHER failure episodes of the same task.", ""]
    lines += ["| model | failed eps | attempts / ep | eps >= 2 attempts | eps with repeat | within repeat rate "
              "| cross-episode rate (null) | a2 eps with repeat |", "|---|---|---|---|---|---|---|---|"]
    for m in args.models:
        eps = []
        for f in sorted((RESIM / m).glob("*.npz")):
            iid = f.stem
            mode = modes.get((m, iid))
            d = np.load(f)
            t, cols = d["tele"], [str(x) for x in d["cols"]]
            if t[:, cols.index("success")].max() > 0 or mode in (None, "f_infeasible"):
                continue
            at = attempts(t, cols, args.min_hold, args.merge_gap, args.near)
            n_rep = 0
            for k, a in enumerate(at):
                prev = [b for b in at[:k] if not b["lifted"]]
                a["repeat"] = any(np.linalg.norm(np.subtract(a["rel"], b["rel"])) < args.tol for b in prev)
                n_rep += a["repeat"]
            eps.append({"iid": iid, "task": iid.split("__")[0], "mode": mode, "n_att": len(at), "n_rep": n_rep,
                        "attempts": at})
        # null: does attempt k (k >= 1) sit within tol of a failed attempt of ANOTHER failed episode of the same task?
        # within >> cross => episodic loop (what memory can break); within ~ cross => a fixed policy prior.
        cross_hit, cross_n = 0, 0
        rng = np.random.default_rng(0)
        for e in eps:
            others = [b for o in eps if o["task"] == e["task"] and o["iid"].split("__")[1] != e["iid"].split("__")[1]
                      for b in o["attempts"] if not b["lifted"]]
            if not others:
                continue
            for k, a in enumerate(e["attempts"]):
                p = sum(not b["lifted"] for b in e["attempts"][:k])   # matched number of comparisons
                if p == 0:
                    continue
                pick = rng.choice(len(others), size=min(p, len(others)), replace=False)
                cross_n += 1
                cross_hit += any(np.linalg.norm(np.subtract(a["rel"], others[q]["rel"])) < args.tol for q in pick)
        within_n = sum(sum(1 for k, a in enumerate(e["attempts"]) if any(not b["lifted"] for b in e["attempts"][:k]))
                       for e in eps)
        n = len(eps)
        multi = sum(e["n_att"] >= 2 for e in eps)
        rep = sum(e["n_rep"] > 0 for e in eps)
        later = sum(max(0, e["n_att"] - 1) for e in eps)
        reps = sum(e["n_rep"] for e in eps)
        a2 = [e for e in eps if e["mode"] == "a2_failed_grasp"]
        a2r = sum(e["n_rep"] > 0 for e in a2)
        res[m] = {"n": n, "multi": multi, "with_repeat": rep, "later_attempts": later, "repeats": reps,
                  "within_rate": reps / max(within_n, 1), "within_n": within_n,
                  "cross_rate": cross_hit / max(cross_n, 1), "cross_n": cross_n,
                  "a2": len(a2), "a2_with_repeat": a2r,
                  "by_mode": {md: {"n": sum(e["mode"] == md for e in eps),
                                   "with_repeat": sum(e["mode"] == md and e["n_rep"] > 0 for e in eps)}
                              for md in sorted({e["mode"] for e in eps})},
                  "episodes": eps}
        mean_att = np.mean([e["n_att"] for e in eps]) if eps else float("nan")
        lines.append(f"| {m} | {n} | {mean_att:.2f} | {multi} ({multi / max(n, 1):.0%}) | {rep} ({rep / max(n, 1):.0%}) "
                     f"| {reps}/{within_n} ({reps / max(within_n, 1):.0%}) "
                     f"| {cross_hit}/{cross_n} ({cross_hit / max(cross_n, 1):.0%}) | {a2r}/{len(a2)} |")
    lines += ["", "Per mode (episodes with >= 1 repeat / episodes):", ""]
    for m in args.models:
        lines.append(f"- {m}: " + ", ".join(f"{k} {v['with_repeat']}/{v['n']}" for k, v in res[m]["by_mode"].items()))
    tot = sum(r["n"] for r in res.values())
    tot_rep = sum(r["with_repeat"] for r in res.values())
    frac = tot_rep / max(tot, 1)
    w = sum(r["repeats"] for r in res.values()) / max(sum(r["within_n"] for r in res.values()), 1)
    x = sum(r["cross_rate"] * r["cross_n"] for r in res.values()) / max(sum(r["cross_n"] for r in res.values()), 1)
    verdict = "KEEP approach A (>= 20 %)" if frac >= 0.2 else "DROP approach A (< 20 %)"
    lines += ["", f"Pooled: {tot_rep}/{tot} = {frac:.0%} failed episodes contain a repeated failed grasp -> **{verdict}**.",
              f"Within-episode repeat rate {w:.0%} vs cross-episode null {x:.0%} (ratio {w / max(x, 1e-9):.2f}): "
              "ratio ~1 means the repeats reflect a fixed task prior, not an episodic loop."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"poc0_repeat_tol{args.tol:g}.md").write_text("\n".join(lines) + "\n")
    json.dump({"tol": args.tol, "pooled_frac": frac, "verdict": verdict, "models": res},
              open(OUT / f"poc0_repeat_tol{args.tol:g}.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
