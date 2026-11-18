"""2610.05151 reverse direction: main comparison table (pre-registered decision rules, PLAN "2610.05151 comparison").

python scripts/analyze_goal_rsi.py  -> runs/goal_rsi/main.{json,md}
Arms on the 100 val scenes: M0, Original (orig_s{7,8,9}), RetryVLA wm (rec_wm_s*), RetryVLA random (rec_rand_s*).
Delta vs Original: mean over seeds, scene x seed bootstrap (2000), paired by seed. Search: solved entries per arm, McNemar.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
R = REPO / "runs/goal_rsi"


def ep(path):
    return {json.loads(l)["scene"]: bool(json.loads(l)["success"]) for l in open(path)}


def arm(prefix):
    out = {}
    for s in (7, 8, 9):
        f = R / f"{prefix}_s{s}/ckpt_02000/eval_val/episodes.jsonl"
        if f.exists():
            out[s] = ep(f)
    return out


def boot(a, b, n=2000, seed=0):
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    scenes = sorted(a[seeds[0]])
    A = np.array([[a[s][k] for k in scenes] for s in seeds], float)
    B = np.array([[b[s][k] for k in scenes] for s in seeds], float)
    d = (A - B).mean()
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        si = rng.integers(len(seeds), size=len(seeds))
        ci = rng.integers(len(scenes), size=len(scenes))
        bs.append((A[si][:, ci] - B[si][:, ci]).mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"seeds": seeds, "delta": float(d) * 100, "ci": [float(lo) * 100, float(hi) * 100],
            "per_seed": {int(s): float((A[i] - B[i]).sum()) for i, s in enumerate(seeds)}}


def mcnemar(x, y):
    from scipy.stats import binomtest
    n01 = sum(1 for k in x if not x[k] and y[k]); n10 = sum(1 for k in x if x[k] and not y[k])
    return n01, n10, float(binomtest(min(n01, n10), n01 + n10).pvalue) if n01 + n10 else 1.0


def main():
    res = {"M0": sum(ep(R / "m0_s7/ckpt_04000/eval_val/episodes.jsonl").values())}
    arms = {"Original": arm("orig"), "RetryVLA_wm": arm("rec_wm"), "RetryVLA_random": arm("rec_rand")}
    for k, v in arms.items():
        res[k] = {int(s): int(sum(x.values())) for s, x in v.items()}
        res[k + "_mean"] = float(np.mean(list(res[k].values()))) if v else None
    res["delta_wm_vs_orig"] = boot(arms["RetryVLA_wm"], arms["Original"])
    res["delta_rand_vs_orig"] = boot(arms["RetryVLA_random"], arms["Original"])
    res["delta_wm_vs_rand"] = boot(arms["RetryVLA_wm"], arms["RetryVLA_random"])
    d = res["delta_wm_vs_orig"]
    if d:
        res["verdict"] = ("beats theirs" if d["delta"] >= 10 and d["ci"][0] > 5 else
                          "matches theirs" if d["delta"] >= 5 and d["ci"][0] > 0 else "below")
    rows = []
    for f in glob.glob(str(REPO / "data/recovery/goal_search/s*/p*/../p*")):
        pass
    import pandas as pd
    pi = [pd.read_json(f, lines=True) for f in glob.glob(str(REPO / "runs/*GOAL_goal_search_s*/per_instance.jsonl"))]
    if pi:
        df = pd.concat(pi).drop_duplicates(["instance_id", "arm"])
        g = df.groupby("arm")
        res["search"] = {a: {"entries": int(len(x)), "solved": float(x.solved.mean()),
                             "hits_per_verif": float(x.hits.sum() / x.budget.sum()),
                             "hits_per_1k_sim": float(1000 * x.hits.sum() / max(1, x.sim_steps_verify.sum())),
                             "trajs": int(x.n_written.sum())} for a, x in g}
        p = df.pivot(index="instance_id", columns="arm", values="solved").dropna()
        if {"wm_diverse", "random"} <= set(p.columns):
            res["search"]["mcnemar_wm_vs_random"] = mcnemar(p.random.to_dict(), p.wm_diverse.to_dict())
    bank = R / "bank.json"
    if bank.exists():
        res["bank"] = json.loads(bank.read_text())
    res["paper"] = {"M0": 77, "Original": 78, "Recovery": 88}
    (R / "main.json").write_text(json.dumps(res, indent=1, default=str))
    md = ["| arm | val successes per seed (7/8/9) | mean |", "|---|---|---|"]
    for k in ("Original", "RetryVLA_wm", "RetryVLA_random"):
        md.append(f"| {k} | {res[k]} | {res[k + '_mean']} |")
    md.append(f"\nM0 {res['M0']}; delta wm vs Original {res['delta_wm_vs_orig']}; verdict {res.get('verdict')}")
    md.append(f"search {res.get('search')}")
    (R / "main.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
