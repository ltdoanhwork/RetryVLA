#!/usr/bin/env python
"""Part 4 (RQ2) action-free / hybrid arms vs pi1 (action-grounded) -> runs/_analysis/af/summary.md

python scripts/analyze_af.py
Units as in scripts/stats_cluster.py: scene bootstrap (x seed) of the paired difference + sign-flip p over scenes,
exact McNemar on the 560 paired episodes (seed 0 vs pi1 seed 0). Nominal margin: >= pi1-nominal 0.796 - 2 pt
(pre-registered eps), and vs pi1 0.808. Futures-dataset stats from data/futures/af_m5b/futures.pt.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from stats_cluster import compare, eps, mcnemar  # noqa: E402

OUT = REPO / "runs/_analysis/af"
ARMS = {  # name -> (recover tags per seed, nominal tag)
    "1 demo-only": (["M9_sft_demo_only_recover_test"], "M9_sft_demo_only_nominal"),
    "2 action-grounded (pi1)": (["M5_m5c_sft_wm_diverse_rr_recover_test", "M8_seed_pi1_s1_recover_test",
                                 "M8_seed_pi1_s2_recover_test"], "M5_m5c_sft_wm_diverse_rr_nominal"),
    "3 action-free, C_pi-filtered": (["P4_af_arm3_free_filtered_s0_recover_test"], "P4_af_arm3_free_filtered_s0_nominal"),
    "4 action-free, unfiltered": (["P4_af_arm4_free_unfiltered_s0_recover_test"], "P4_af_arm4_free_unfiltered_s0_nominal"),
    "5 hybrid (2 + 3)": (["P4_af_arm5_hybrid_s0_recover_test"], "P4_af_arm5_hybrid_s0_nominal"),
}
PI1_NOM = 0.796
EPS = 0.02


def summ(tag):
    f = sorted(glob.glob(str(REPO / "runs" / f"*_{tag}")))
    if not f or not Path(f[-1], "summary.json").exists():
        return None
    s = json.load(open(Path(f[-1], "summary.json")))
    return s if s.get("status") == "done" else None


def futures_stats(lines):
    import torch

    p = REPO / "data/futures/af_m5b/futures.pt"
    if not p.exists():
        return
    d = torch.load(p, map_location="cpu", weights_only=False)
    f = d["futures"]
    var = np.asarray(f["variant"])
    lines += ["## Imagined-future dataset (data/futures/af_m5b)", "",
              f"{len(d['obs']['task'])} train failure states (M5b), {len(var)} imagined futures; WM `{d['wm_ckpt']}`, "
              f"critic `{d['critic_ckpt']}`.", "",
              "| variant | futures | mean C_pi | sim-checked | sim success (checked) |", "|---|---|---|---|---|"]
    for v in sorted(set(var)):
        m = var == v
        chk, ok = f["sim_checked"].numpy()[m], f["sim_success"].numpy()[m]
        lines.append(f"| {v} | {int(m.sum())} | {f['cpi'].numpy()[m].mean():.3f} | {int(chk.sum())} | "
                     f"{(ok[chk].mean() if chk.any() else float('nan')):.3f} |")
    lines.append("")


def main() -> None:
    rng = np.random.default_rng(0)
    lines = ["# Part 4 (RQ2): action-free imagined futures vs action-grounded recovery SFT", ""]
    futures_stats(lines)
    lines += ["## Policies (test RSR, 56 scenes x 10; nominal Spatial 500)", "",
              "| arm | seeds | RSR | close / open | nominal | nominal >= pi1-nom - 2pt | vs pi1 seed 0: diff, McNemar n01/n10 p "
              "| vs pi1 (all seeds): scene-bootstrap 95% CI, sign-flip p |", "|---|---|---|---|---|---|---|---|"]
    pi1_e = [eps(t) for t in ARMS["2 action-grounded (pi1)"][0] if eps(t)]
    res = {}
    for name, (rtags, ntag) in ARMS.items():
        es = [eps(t) for t in rtags if summ(t) and eps(t)]
        if not es:
            lines.append(f"| {name} | 0 | pending | | | | | |")
            continue
        sr = [summ(t) for t in rtags if summ(t)]
        rsr = np.mean([s["results"]["rsr"] for s in sr])
        cl = np.mean([s["results"].get("rsr_close", np.nan) for s in sr])
        op = np.mean([s["results"].get("rsr_open", np.nan) for s in sr])
        ns = summ(ntag)
        nom = ns["results"]["success_rate"] if ns else None
        keys = sorted(pi1_e[0])
        a = np.array([es[0].get(k, 0) > 0.5 for k in keys])
        b = np.array([pi1_e[0][k] > 0.5 for k in keys])
        n01, n10, p = mcnemar(a, b)
        diff, ci, pf, *_ = compare(es, pi1_e, rng)
        res[name] = {"rsr": rsr, "nominal": nom, "n_seeds": len(es), "diff_vs_pi1": diff, "ci": list(ci), "p_flip": pf,
                     "mcnemar": [n01, n10, p]}
        nm = "pending" if nom is None else f"{nom:.3f}"
        ok = "" if nom is None else ("pass" if nom >= PI1_NOM - EPS else "**FAIL**")
        lines.append(f"| {name} | {len(es)} | {rsr:.3f} | {cl:.3f} / {op:.3f} | {nm} | {ok} | "
                     f"{float(a.mean() - b.mean()):+.3f}, {n01}/{n10} p = {p:.2g} | {diff:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}], "
                     f"p = {pf:.3g} |")
    pr = sorted(glob.glob(str(REPO / "runs/_analysis/af_probe_s*/summary.md")))
    if pr:
        lines += ["", "## C_pi embedding probe (risk: does L_future reach the action expert?)", "",
                  "Linear probe (5-fold GroupKFold by scene) on the action-expert hidden state (L_future hook, t = 1) of the "
                  "256 train failure states; label = median split of C_pi(z0) (v0) or of max C_pi over the state's "
                  "filtered futures (best); null95 = 95th pct of 20 label permutations.", "",
                  open(pr[-1]).read()]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
