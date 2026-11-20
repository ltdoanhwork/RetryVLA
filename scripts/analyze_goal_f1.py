"""2610.05151 follow-up F1 (share-matched distillation) vs the pre-registered rules (RESULTS "main comparison").

python scripts/analyze_goal_f1.py -> runs/goal_rsi/f1.{json,md}
Rules (written 2026-11-20 before running):
  "matches theirs"            F1 wm  - Original >= +5 with bootstrap CI lower bound > 0
  "mix volume caused the drop" F1 wm - F0 wm (= main-comparison rec_wm) > +8 with CI lower bound > 0
  "WM matters"                 F1 wm - F1 random >= +3 with CI lower bound > 0
Bootstrap = scene x seed (2000), paired by seed and scene, as analyze_goal_rsi.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_goal_rsi import R, arm, boot, mcnemar  # noqa: E402


def main():
    A = {"Original": arm("orig"), "F0_wm": arm("rec_wm"), "F0_random": arm("rec_rand"),
         "F1_wm": arm("f1_wm"), "F1_random": arm("f1_rand")}
    res = {k: {int(s): int(sum(v.values())) for s, v in x.items()} for k, x in A.items()}
    res["means"] = {k: (float(np.mean(list(v.values()))) if v else None) for k, v in list(res.items())}
    res["F1_wm_vs_orig"] = boot(A["F1_wm"], A["Original"])
    res["F1_rand_vs_orig"] = boot(A["F1_random"], A["Original"])
    res["F1_wm_vs_F0_wm"] = boot(A["F1_wm"], A["F0_wm"])
    res["F1_rand_vs_F0_rand"] = boot(A["F1_random"], A["F0_random"])
    res["F1_wm_vs_rand"] = boot(A["F1_wm"], A["F1_random"])
    pooled = {}
    for k in ("F1_wm", "F1_random"):
        n01 = n10 = 0
        for s in set(A[k]) & set(A["Original"]):
            a, b = A["Original"][s], A[k][s]
            x = mcnemar(a, b)
            n01 += x[0]; n10 += x[1]
        pooled[k] = {"fixed": n01, "broken": n10}
    res["pooled_vs_orig"] = pooled
    d1, dm, dw = res["F1_wm_vs_orig"], res["F1_wm_vs_F0_wm"], res["F1_wm_vs_rand"]
    v = {}
    if d1:
        v["matches_theirs"] = bool(d1["delta"] >= 5 and d1["ci"][0] > 0)
        v["beats_theirs"] = bool(d1["delta"] >= 10 and d1["ci"][0] > 5)
    if dm:
        v["mix_volume_caused_drop"] = bool(dm["delta"] > 8 and dm["ci"][0] > 0)
    if dw:
        v["wm_matters"] = bool(dw["delta"] >= 3 and dw["ci"][0] > 0)
    res["rules"] = v
    sh = [json.loads((R / f"f1_{a}_s{s}/ckpt_02000/meta.json").read_text())["anchors_per_source"]
          for a in ("wm", "rand") for s in (7, 8, 9) if (R / f"f1_{a}_s{s}/ckpt_02000/meta.json").exists()]
    res["recovery_anchor_share"] = [round(sum(x[1:]) / sum(x), 4) for x in sh]
    (R / "f1.json").write_text(json.dumps(res, indent=1, default=str))
    f = lambda d: "n/a" if not d else f"{d['delta']:+.1f} [{d['ci'][0]:+.1f}, {d['ci'][1]:+.1f}]"
    md = [
        "| arm | val successes s7 / s8 / s9 | mean | Δ vs Original (95 % CI) |",
        "|---|---|---|---|",
        *[f"| {k} | {' / '.join(str(res[k].get(s, '-')) for s in (7, 8, 9))} | "
          f"{res['means'][k] if res['means'][k] is None else round(res['means'][k], 1)} | "
          f"{f(boot(A[k], A['Original'])) if k != 'Original' else '-'} |" for k in A],
        "",
        f"F1 wm - F0 wm {f(dm)}; F1 random - F0 random {f(res['F1_rand_vs_F0_rand'])}; F1 wm - F1 random {f(dw)}",
        f"pooled fixed / broken vs Original: {pooled}; recovery anchor share {res['recovery_anchor_share']}",
        f"rules: {v}",
    ]
    (R / "f1.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
