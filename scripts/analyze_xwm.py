#!/usr/bin/env python
"""Cross-WM disagreement sweep (configs/exp/xwm_sweep.yaml): paired comparison on the SAME train instances.
Arms (in-run, identical pools / CEM seeds / continuation seeds per slot): wm_unc_l0 (WM v1 alone), vj_unc_l0
(V-JEPA alone), xwm_l{0,0.5,1} (J = mean - lam |C_v1 - C_vj| - mu R). Every pick is scored by both verifiers.
Also cross-run reproducibility of the controls vs the earlier rq3 / vj sweeps (same instances, same seeds).
python scripts/analyze_xwm.py -> runs/_analysis/xwm/xwm.md + xwm.json"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_verifier import ece, mcnemar, signflip  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ARMS = ["wm_unc_l0", "vj_unc_l0", "xwm_l0", "xwm_l0.5", "xwm_l1"]
NAME = {"wm_unc_l0": "WM v1 alone", "vj_unc_l0": "V-JEPA alone", "xwm_l0": "x-WM mean (lam 0)",
        "xwm_l0.5": "x-WM min = agree (lam 0.5)", "xwm_l1": "x-WM mean - |d| (lam 1)"}


def load(pat):
    fs = sorted(glob.glob(str(REPO / pat)))
    if not fs:
        return pd.DataFrame()
    d = pd.concat([pd.read_json(f, lines=True) for f in fs])
    return d.drop_duplicates(["instance_id", "arm"], keep="last")


def boot_ci(x, n=10000, seed=0):
    r = np.random.default_rng(seed)
    m = r.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def main() -> None:
    d = load("runs/*_RQ3_xwm_sweep_s*/per_instance.jsonl")
    d = d[d.arm.isin(ARMS)]
    cnt = d.groupby("instance_id").arm.nunique()
    common = sorted(cnt[cnt == len(ARMS)].index)
    g = {a: d[d.arm == a].set_index("instance_id").loc[common] for a in ARMS}
    md = [f"# Cross-WM disagreement search: WM v1 x V-JEPA 2-AC, {len(common)} paired train instances", "",
          "8 verified picks per arm and instance (diverse top-8, in J order); same pool, CEM seeds and continuation "
          "seeds per slot in every arm. FP = P(fail | pred > 0.5) with pred = the arm's own objective value "
          "(x-WM: mean of the two C^pi); p@k = real success of the first k picks.", "",
          "| arm | solved | hits / sim | p@1 | p@4 | mean pred | FP (own pred) | ECE | FP C_v1 | FP C_vj | FP both > 0.5 | n both > 0.5 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    js = {"n": len(common), "arms": {}}
    for a in ARMS:
        x = g[a]
        y = np.concatenate(x.order_success.to_list()).astype(bool)
        p = np.concatenate(x.order_pred.to_list())
        v1 = np.concatenate(x.order_v1.to_list())
        v2 = np.concatenate(x.order_v2.to_list())
        S = np.stack([np.asarray(s, bool) for s in x.order_success])
        fp = lambda m: float((~y[m]).mean()) if m.any() else float("nan")  # noqa: E731
        both = (v1 > 0.5) & (v2 > 0.5)
        r = {"solved": float(x.solved.mean()), "hits_per_sim": float(y.mean()), "p1": float(S[:, :1].mean()),
             "p4": float(S[:, :4].mean()), "pred": float(p.mean()), "fp": fp(p > 0.5), "ece": float(ece(p, y)),
             "fp_v1": fp(v1 > 0.5), "fp_v2": fp(v2 > 0.5), "fp_both": fp(both), "n_both": int(both.sum())}
        js["arms"][a] = r
        md.append(f"| {NAME[a]} | {r['solved']:.3f} | {r['hits_per_sim']:.3f} | {r['p1']:.3f} | {r['p4']:.3f} | "
                  f"{r['pred']:.3f} | {r['fp']:.3f} | {r['ece']:.3f} | {r['fp_v1']:.3f} | {r['fp_v2']:.3f} | "
                  f"{r['fp_both']:.3f} | {r['n_both']} |")
    md += ["", "## Paired tests (x-WM arm vs each single-WM arm)", "",
           "| arm | vs | solved a / b | only a / only b | McNemar p | d hits / inst [95% CI] | sign-flip p |",
           "|---|---|---|---|---|---|---|"]
    js["tests"] = []
    for a in ["xwm_l0", "xwm_l0.5", "xwm_l1", "vj_unc_l0"]:
        for b in ["wm_unc_l0", "vj_unc_l0"]:
            if a == b:
                continue
            oa, ob, p = mcnemar(g[a].solved.to_numpy(), g[b].solved.to_numpy())
            dh = (g[a].hits - g[b].hits).to_numpy().astype(float)
            lo, hi = boot_ci(dh)
            sp = signflip(dh)
            js["tests"].append({"a": a, "b": b, "only_a": int(oa), "only_b": int(ob), "p_mcnemar": float(p),
                                "d_hits": float(dh.mean()), "ci": [float(lo), float(hi)], "p_signflip": float(sp)})
            md.append(f"| {NAME[a]} | {NAME[b]} | {g[a].solved.mean():.3f} / {g[b].solved.mean():.3f} | {oa} / {ob} | "
                      f"{p:.3g} | {dh.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] | {sp:.3g} |")
    # cross-run reproducibility of the in-run controls (same instances / seeds as the earlier sweeps)
    md += ["", "## Control reproducibility vs the earlier sweeps (same instances, pools, seeds)", ""]
    for a, pat, arm in (("wm_unc_l0", "runs/*_RQ3_verif_sweep_s*/per_instance.jsonl", "wm_unc_l0"),
                        ("vj_unc_l0", "runs/*_RQ3_vj_sweep_s*/per_instance.jsonl", "wm_unc_l0")):
        o = load(pat)
        o = o[o.arm == arm].set_index("instance_id")
        c = [i for i in common if i in o.index]
        if not c:
            continue
        same = np.mean([list(o.loc[i, "order_success"]) == list(g[a].loc[i, "order_success"]) for i in c])
        md.append(f"- {NAME[a]}: {len(c)} instances in both; solved now {g[a].loc[c].solved.mean():.3f} vs before "
                  f"{o.loc[c].solved.mean():.3f}; identical per-slot outcomes on {same:.0%} of instances")
    out = REPO / "runs/_analysis/xwm"
    out.mkdir(parents=True, exist_ok=True)
    (out / "xwm.md").write_text("\n".join(md) + "\n")
    (out / "xwm.json").write_text(json.dumps(js, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
