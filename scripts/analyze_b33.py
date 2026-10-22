#!/usr/bin/env python
"""B3.3 analysis: residual recovery RL baseline vs pi1 / pi0 (scene-level paired bootstrap) + learning curve.

python scripts/analyze_b33.py [--tag B33_b33_rl_sparse] [--out runs/_analysis/b33]

- test RSR: runs/*_<tag>_recover_test vs pi1 (M5c wm_diverse rr; + seeds 1-2 if present) and vs pi0
  (M0_recover_bddl restricted to the test scenes); unit = test scene (56), 10 trials averaged
  (scripts/stats_cluster.compare: scene x seed bootstrap CI, sign-flip p).
- nominal: runs/*_<tag>_nominal summary.
- learning curve: data/rl/<tag-without-milestone>/curve.jsonl (SR on train instances, held-out pi0 seeds) vs
  transitions, with the PoC budget marks (runs/_analysis/b33/budget.json).
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
from stats_cluster import compare, eps  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="B33_b33_rl_sparse")
    ap.add_argument("--rl_dir", default=None)
    ap.add_argument("--out", default="runs/_analysis/b33")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    L = [f"# B3.3 residual recovery RL baseline ({a.tag})", ""]
    rl = eps(f"{a.tag}_recover_test")
    pi1 = [e for e in (eps(t) for t in ["M5_m5c_sft_wm_diverse_rr_recover_test", "M8_seed_pi1_s1_recover_test",
                                        "M8_seed_pi1_s2_recover_test"]) if e]
    pi0 = eps("M0_recover_bddl")
    if rl:
        L += [f"test RSR = {np.mean(list(rl.values())):.3f} over {len(rl)} episodes", "",
              "| comparison | diff | 95% CI (scene bootstrap) | p (sign-flip) |", "|---|---|---|---|"]
        for name, Bm in [("RL vs pi1 (seeds)", pi1), ("RL vs pi0", [{k: v for k, v in pi0.items() if k in rl}])]:
            d, ci, p, ns, sa, sb = compare([rl], Bm, rng)
            L.append(f"| {name} ({sa}/{sb} seeds, {ns} scenes) | {d:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {p:.2g} |")
    else:
        L.append("test RSR: missing")
    nom = sorted(glob.glob(str(REPO / "runs" / f"*_{a.tag}_nominal" / "summary.json")))
    if nom:
        L += ["", f"nominal SR = {json.load(open(nom[-1]))['results'].get('success_rate')}"]
    rl_dir = REPO / (a.rl_dir or f"data/rl/{a.tag.split('_', 1)[1]}")
    cf = rl_dir / "curve.jsonl"
    bud = REPO / "runs/_analysis/b33/budget.json"
    if cf.exists():
        b = json.load(open(bud)) if bud.exists() else {}
        L += ["", "## learning curve (train instances, held-out pi0 seeds, deterministic residual)", "",
              f"PoC budget marks: verify-only {b.get('verify', {}).get('transitions')} | full {b.get('poc_total_matched')}",
              "", "| transitions | iteration | SR | bridge SR | n |", "|---|---|---|---|---|"]
        rows = [json.loads(l) for l in open(cf)]
        for r in rows:
            L.append(f"| {r['transitions']} | {r['it']} | {r['eval_sr']:.3f} | {r['eval_bridge_sr']:.3f} | {r['n']} |")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.plot([r["transitions"] / 1e6 for r in rows], [r["eval_sr"] for r in rows], "o-", label="residual RL")
            for k, ls in (("verify", ":"), ("full", "--")):
                x = b.get("verify", {}).get("transitions") if k == "verify" else b.get("poc_total_matched")
                if x:
                    ax.axvline(x / 1e6, ls=ls, c="gray", label=f"PoC budget ({k})")
            ax.set_xlabel("sim transitions (M)"); ax.set_ylabel("SR (train inst., held-out seeds)")
            ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out / "curve.png", dpi=150)
        except Exception as e:  # plotting optional
            L.append(f"(plot failed: {e})")
    (out / f"summary_{a.tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
