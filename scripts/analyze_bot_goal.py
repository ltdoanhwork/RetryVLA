"""Main table for the Back on Track / ForkCast-style test-time arms on the 2610.05151 LIBERO-Goal protocol.

python scripts/analyze_bot_goal.py [--root runs/bot_goal]  ->  <root>/main.{json,md}

Per arm: SR / 100 (Wilson 95 %), paired vs its base policy (rescued / broken, exact McNemar p), trigger rate on
base-failed vs base-succeeded scenes, inference cost (per-episode GPU wall time VLA / monitor / planner, ratio vs
plain VLA, #triggers, #WM rows, batch-1 bench). Multi-seed reference policies: mean over seeds.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
G = REPO / "runs/goal_rsi"
BASES = {"m0": G / "m0_s7/ckpt_04000/eval_val", "orig7": G / "orig_s7/ckpt_02000/eval_val",
         "recwm7": G / "rec_wm_s7/ckpt_02000/eval_val"}
REFS = [("M0 (pi0 + LoRA 4k, selected on val)", ["m0_s7/ckpt_04000"]),
        ("Original (M0 + 2k on D0), 3 seeds", [f"orig_s{s}/ckpt_02000" for s in (7, 8, 9)]),
        ("F0 RetryVLA `rec_wm` (47 % anchors), 3 seeds", [f"rec_wm_s{s}/ckpt_02000" for s in (7, 8, 9)]),
        ("F0 RetryVLA `rec_rand`, 3 seeds", [f"rec_rand_s{s}/ckpt_02000" for s in (7, 8, 9)]),
        ("F1 RetryVLA `wm` (share-matched 4 %), 3 seeds", [f"f1_wm_s{s}/ckpt_02000" for s in (7, 8, 9)]),
        ("F1 RetryVLA `rand` (share-matched 4 %), 3 seeds", [f"f1_rand_s{s}/ckpt_02000" for s in (7, 8, 9)])]
ARMS = [("m0_bot_cpi", "M0 + BoT (C^pi)", "m0"), ("m0_bot_liv", "M0 + BoT (liveness)", "m0"),
        ("m0_fc_cpi", "M0 + ForkCast-style (C^pi)", "m0"), ("m0_fc_prog", "M0 + ForkCast-style (progress)", "m0"),
        ("orig7_bot_cpi", "Original s7 + BoT (C^pi)", "orig7"), ("orig7_bot_liv", "Original s7 + BoT (liveness)", "orig7"),
        ("recwm7_bot_cpi", "F0 rec_wm s7 + BoT (C^pi)", "recwm7"), ("recwm7_bot_liv", "F0 rec_wm s7 + BoT (liveness)", "recwm7")]


def eps(d: Path) -> dict | None:
    f = d / "episodes.jsonl"
    if not f.exists():
        return None
    return {r["scene"]: r for r in map(json.loads, open(f))}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/bot_goal")
    a = ap.parse_args()
    root = REPO / a.root
    out = {"refs": [], "arms": []}
    md = ["| arm | val SR / 100 [Wilson 95 %] | vs base: rescued / broken, McNemar p | trigger rate (base-fail / base-succ) "
          "| per-episode GPU s: VLA / monitor / planner | cost ratio vs plain | triggers / ep | WM rows / ep | batch-1 plan ms |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, dirs in REFS:
        srs = []
        for d in dirs:
            s = G / d / "eval_val/summary.json"
            if s.exists():
                srs.append(json.loads(s.read_text())["successes"])
        if not srs:
            md.append(f"| {name} | pending | | | | 1x (plain) | | | |")
            continue
        out["refs"].append({"name": name, "successes": srs})
        lo, hi = wilson(sum(srs), 100 * len(srs))
        md.append(f"| {name} | {np.mean(srs):.1f} ({' / '.join(map(str, srs))}) [{100*lo:.0f}, {100*hi:.0f}] | - | - | "
                  f"- | 1x (plain) | 0 | 0 | - |")
    for key, name, base in ARMS:
        e = eps(root / key)
        if e is None:
            md.append(f"| {name} | pending | | | | | | | |")
            continue
        b = eps(BASES[base])
        summ = json.loads((root / key / "summary.json").read_text())
        common = sorted(set(e) & set(b))
        y = np.array([e[s]["success"] for s in common])
        yb = np.array([b[s]["success"] for s in common])
        resc, brok = int((y & ~yb).sum()), int((~y & yb).sum())
        trig = np.array([e[s].get("bot_n_trig", 0) > 0 for s in common])
        tf = float(trig[~yb].mean()) if (~yb).any() else float("nan")
        ts = float(trig[yb].mean()) if yb.any() else float("nan")
        v = np.mean([e[s].get("bot_vla_s", 0) for s in common])
        m = np.mean([e[s].get("bot_mon_s", 0) for s in common])
        p = np.mean([e[s].get("bot_plan_s", 0) for s in common])
        nt = np.mean([e[s].get("bot_n_trig", 0) for s in common])
        wr = np.mean([e[s].get("bot_wm_rows", 0) for s in common])
        ratio = (v + m + p) / max(v, 1e-9)
        bench = summ.get("bot_bench") or {}
        k = int(y.sum())
        lo, hi = wilson(k, len(y))
        row = {"key": key, "name": name, "base": base, "successes": k, "n": len(y), "rescued": resc, "broken": brok,
               "mcnemar_p": mcnemar(resc, brok), "trig_base_fail": tf, "trig_base_succ": ts, "vla_s": v, "mon_s": m,
               "plan_s": p, "cost_ratio": ratio, "triggers_per_ep": nt, "wm_rows_per_ep": wr, "bench": bench,
               "stats": summ.get("bot_stats"), "wall_s": summ.get("wall_s")}
        out["arms"].append(row)
        md.append(f"| {name} | {k} [{100*lo:.0f}, {100*hi:.0f}] | {resc} / {brok}, p {row['mcnemar_p']:.2g} | "
                  f"{tf:.2f} / {ts:.2f} | {v:.2f} / {m:.2f} / {p:.2f} | {ratio:.1f}x | {nt:.1f} | {wr:.0f} | "
                  f"{1000 * bench.get('plan_s', float('nan')):.0f} |")
    (root / "main.json").write_text(json.dumps(out, indent=1))
    (root / "main.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
