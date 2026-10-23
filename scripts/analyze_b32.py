#!/usr/bin/env python
"""B3.2 RISE-inspired baseline: analysis.  python scripts/analyze_b32.py -> runs/_analysis/b32/summary.md

1. Policy comparisons on the Spatial test (dev) split, unit = test scene (stats_cluster.compare: scene x seed
   bootstrap CI + paired sign-flip p): each B3.2 arm vs pi1 (1 seed, and 3 seeds), vs demo-only SFT, vs pi1-nominal.
2. Nominal SR of every arm.
3. Imagined-label quality (analysis only, never used in training): candidates that coincide with chunks the PoC
   verified in M5b carry the real outcome -> success rate of imagined positives vs negatives, AUC of A.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from stats_cluster import compare  # noqa: E402
from stats_cluster import eps as _eps  # noqa: E402


def eps(tag):
    """Episodes of the newest run of `tag` if it has finished (a running eval has no episodes.jsonl yet), else None."""
    runs = sorted(glob.glob(str(REPO / "runs" / f"*_{tag}")))
    return _eps(tag) if runs and Path(runs[-1], "episodes.jsonl").exists() else None

OUT = REPO / "runs/_analysis/b32"
ARMS = {"B32 cond (pool)": "B32_rise_cond_pool", "B32 AWR (pool)": "B32_rise_awr_pool",
        "B32 cond (VLA samples)": "B32_rise_cond_vla"}
PI1 = ["M5_m5c_sft_wm_diverse_rr_recover_test", "M8_seed_pi1_s1_recover_test", "M8_seed_pi1_s2_recover_test"]


def nominal(tag: str):
    f = sorted(glob.glob(str(REPO / "runs" / f"*_{tag}" / "summary.json")))
    if not f:
        return None
    return json.load(open(f[-1]))["results"].get("success_rate")


EPS_NOM = 0.02


def fmt_p(p: float) -> str:
    return "<1e-4" if p == 0 else f"{p:.2g}"   # 10k sign flips


def nom_tasks(tags):
    """[seeds, 10] per-task nominal SR (newest run of each tag) or None."""
    out = []
    for t in tags:
        f = sorted(glob.glob(str(REPO / "runs" / f"*_{t}" / "episodes.jsonl")))
        if not f:
            continue
        d = {}
        for r in map(json.loads, open(f[-1])):
            d.setdefault(r["task_id"], []).append(float(r["success"]))
        out.append([np.mean(d[k]) for k in sorted(d)])
    return np.array(out) if out else None


def nom_delta(a, b, rng, n=10000):
    if b is None:
        return (float("nan"),) * 3
    k = a.shape[1]
    bs = [a[rng.integers(0, len(a), len(a))][:, i].mean() - b[rng.integers(0, len(b), len(b))][:, i].mean()
          for i in (rng.integers(0, k, k) for _ in range(n))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return a.mean() - b.mean(), lo, hi


def _grp(s: str) -> str:
    return "pi0 samples" if s.startswith("vla") else "primitives" if s.startswith("prim") else \
        "latent CEM" if s == "wm_cem" else "perturbations" if s == "perturb" else s


def diagnosis(prop: str = "pool") -> list:
    """Why imagination-only hurts: what the cond / AWR objectives upweight, how often the critic is right about it
    (real outcomes of the SAME chunks verified in sim during M5b: zero new sim cost), and how the chunks look vs pi0."""
    import torch
    from collections import defaultdict

    R, C = defaultdict(list), defaultdict(list)
    for f in sorted(glob.glob(str(REPO / f"data/recovery/b32_rise/{prop}/inst/*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        adv = np.asarray(d["adv"], float)
        src = np.array([_grp(s) for s in d["source"]])
        lab = np.asarray(d["poc_verified_analysis_only"])
        pos = (adv >= np.quantile(adv, 0.7)) & (adv > 0)                   # cond: positive bucket
        neg = adv <= np.quantile(adv, 0.5)
        ah = (adv - adv.mean()) / (adv.std() + 1e-8)                        # AWR sampling probabilities
        w = np.exp(np.clip(ah, -2.5, 2.5) / 0.5)
        p = w / w.sum()
        a = np.asarray(d["actions"], float)
        R["frac_adv_pos"].append((adv > 0).mean()); R["V0"].append(float(d["V0"])); R["ess"].append(1 / (p ** 2).sum())
        R["n_pos"].append(pos.sum())
        for g in sorted(set(src)):
            m = src == g
            R[f"posshare|{g}"].append(pos[m].sum() / max(pos.sum(), 1)); R[f"awrmass|{g}"].append(p[m].sum())
            R[f"advpos|{g}"].append((adv[m] > 0).mean())
        stats = {"mean abs(a) xyz+rot": np.abs(a[..., :6]).mean((1, 2)),
                 "jerk mean abs(da)": np.abs(np.diff(a[..., :6], axis=1)).mean((1, 2)),
                 "gripper sign flips / chunk": (np.diff(np.sign(a[..., 6]), axis=1) != 0).sum(1),
                 "frac abs(a)>1 pre-clip": (np.abs(a) > 1).mean((1, 2))}
        for key, m in [("imagined positive (cond)", pos), ("imagined negative (cond)", neg), ("pi0 samples", src == "pi0 samples")]:
            if m.any():
                for k, v in stats.items():
                    R[f"act|{key}|{k}"].append(v[m].mean())
        top = int(np.argmax(adv))
        R["top_verified"].append(lab[top] >= 0)
        if lab[top] >= 0:
            R["top_succ"].append(lab[top] == 1); R["top_V"].append(d["value"][top])
        R["any_real_succ"].append((lab == 1).any())
        m = lab >= 0
        C["y"] += list(lab[m] == 1); C["V"] += list(d["value"][m]); C["A"] += list(adv[m]); C["pos"] += list(pos[m])
        C["w"] += list(p[m] * len(p)); C["src"] += list(src[m])
    if not R:
        return []
    y, V, A, P, W, S = (np.asarray(C[k]) for k in ("y", "V", "A", "pos", "w", "src"))
    P = P.astype(bool)
    mu = lambda k: float(np.mean(R[k]))
    L = ["", f"## Diagnosis ({prop} candidates, {len(R['V0'])} failure states x 696 imagined chunks)", "",
         f"- V(s_0) mean {mu('V0'):.3f}; fraction of chunks with A > 0: {mu('frac_adv_pos'):.3f}; "
         f"cond positives / state {mu('n_pos'):.1f} (of 696); AWR effective sample size / state {mu('ess'):.0f}.",
         "", "| source | share of cond positives | AWR sampling mass | frac A>0 within source |", "|---|---|---|---|"]
    for g in sorted({k.split("|")[1] for k in R if k.startswith("posshare|")}):
        L.append(f"| {g} | {mu('posshare|' + g):.3f} | {mu('awrmass|' + g):.3f} | {mu('advpos|' + g):.3f} |")
    L += ["", f"Critic vs reality on the {len(y)} imagined chunks that M5b also executed in sim (chunk + pi0 continuation;"
          " reused labels, 0 new sim transitions):", "",
          "| subset | n | mean imagined V(s_hat_50) | real success |", "|---|---|---|---|",
          f"| all verified | {len(y)} | {V.mean():.3f} | {y.mean():.3f} |",
          f"| cond positive (top 30%, A>0) | {P.sum()} | {V[P].mean():.3f} | {y[P].mean():.3f} |",
          f"| cond not-positive | {(~P).sum()} | {V[~P].mean():.3f} | {y[~P].mean():.3f} |",
          f"| imagined V > 0.9 | {(V > 0.9).sum()} | {V[V > 0.9].mean():.3f} | {y[V > 0.9].mean():.3f} |",
          f"| argmax-A chunk per state (verified for {np.mean(R['top_verified']):.0%}) | {len(R['top_succ'])} | "
          f"{np.mean(R['top_V']):.3f} | {np.mean(R['top_succ']):.3f} |",
          f"| AWR-weighted (w-normalised) | {len(y)} | {(W * V).sum() / W.sum():.3f} | {(W * y).sum() / W.sum():.3f} |",
          "", f"AUC(A -> real success) = {auc(A, y):.3f}; precision of the positive label = {y[P].mean():.3f} "
          f"(PoC distils only verified successes: precision 1.0). States with >= 1 real success among verified chunks: "
          f"{np.mean(R['any_real_succ']):.3f}.", "",
          "Action statistics of the chunks (raw, before the [-1,1] training clip):", "",
          "| chunks | " + " | ".join(k.split("|")[2] for k in R if k.startswith("act|imagined positive")) + " |",
          "|---|" + "---|" * len([k for k in R if k.startswith("act|imagined positive")])]
    for key in ("imagined positive (cond)", "imagined negative (cond)", "pi0 samples"):
        ks = [k for k in R if k.startswith(f"act|{key}|")]
        L.append(f"| {key} | " + " | ".join(f"{mu(k):.3f}" for k in ks) + " |")
    return L


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    r = s.argsort().argsort() + 1.0
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def main() -> None:
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    L = ["# B3.2 RISE-inspired imagination-only baseline", "",
         "Unit = test scene (56 x 10 trials); CI = scene (x seed) bootstrap; p = paired sign-flip on scenes.", "",
         "| comparison | seeds A / B | diff | 95% CI | p |", "|---|---|---|---|---|"]
    refs = [("pi1-nominal (same init)", ["M1_pi1nom_recover_test"]), ("pi1 (seed 0)", PI1[:1]), ("pi1 (3 seeds)", PI1),
            ("pi0", ["M0_recover_bddl"]), ("demo-only SFT", ["M9_sft_demo_only_recover_test"])]
    test_keys = set(eps(PI1[0]))
    for rn, rt in refs:
        Bm = [{k: v for k, v in e.items() if k in test_keys} for e in (eps(t) for t in rt) if e]
        if Bm:
            L.append(f"| reference {rn}: RSR {np.mean([np.mean(list(b.values())) for b in Bm]):.3f} | {len(Bm)} | | | |")
    for name, tag in ARMS.items():
        A = eps(f"{tag}_recover_test")
        if A is None:
            L.append(f"| {name} | missing | | | |")
            continue
        L.append(f"| {name}: RSR {np.mean(list(A.values())):.3f} (n={len(A)}) | | | | |")
        for rn, rt in refs:
            Bm = [e for e in (eps(t) for t in rt) if e]
            if not Bm:
                continue
            Bm = [{k: v for k, v in b.items() if k in A} for b in Bm]  # pi0 eval covers pilot + test: test keys only
            d, ci, p, _, sa, sb = compare([A], Bm, rng)
            L.append(f"| {name} vs {rn} | {sa} / {sb} | {d:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {fmt_p(p)} |")
    # nominal: 10 Spatial tasks x 50 episodes; CI = task-cluster bootstrap; pre-registered non-inferiority eps = 2 pt
    L += ["", f"Nominal (10 tasks x 50): delta vs reference, CI = task bootstrap. Pre-registered constraint (PLAN.md): "
          f"NSR >= NSR(pi1-nominal) - {EPS_NOM:.2f} (point estimate, >= 3 seeds; B3.2 has 1 seed).",
          "", "| policy | nominal SR | vs pi1-nominal [95% CI] | vs pi1 (3 seeds) [95% CI] | within eps (point) |",
          "|---|---|---|---|---|"]
    ref_nom = nom_tasks(["M1_pi1nom_nominal"])
    ref_pi1 = nom_tasks(["M5_m5c_sft_wm_diverse_rr_nominal", "M8_seed_pi1_s1_nominal", "M8_seed_pi1_s2_nominal"])
    for name, tag in {**ARMS, "pi1 (3 seeds)": None, "pi1-nominal": "M1_pi1nom", "demo-only": "M9_sft_demo_only",
                      "pi0": "M0_baseline_spatial"}.items():
        a = ref_pi1 if tag is None else nom_tasks([tag + "_nominal"])
        if a is None:
            L.append(f"| {name} | missing | | | |")
            continue
        c1, c2 = nom_delta(a, ref_nom, rng), nom_delta(a, ref_pi1, rng)
        L.append(f"| {name} | {a.mean():.3f} | {c1[0]:+.3f} [{c1[1]:+.3f}, {c1[2]:+.3f}] | "
                 f"{c2[0]:+.3f} [{c2[1]:+.3f}, {c2[2]:+.3f}] | {'yes' if c1[0] >= -EPS_NOM - 1e-9 else 'NO'} |")
    L += diagnosis()

    L += ["", "## Imagined labels vs real outcomes (PoC-verified chunks; analysis only)", "",
          "| data | matched chunks | real SR all | real SR imagined-positive | real SR imagined-negative | AUC(A) |",
          "|---|---|---|---|---|---|"]
    for prop in ("pool", "vla"):
        A_, Y, P_, N_ = [], [], [], []
        for f in sorted(glob.glob(str(REPO / f"data/recovery/b32_rise/{prop}/inst/*.pt"))):
            d = torch.load(f, map_location="cpu", weights_only=False)
            lab, adv = np.asarray(d["poc_verified_analysis_only"]), np.asarray(d["adv"])
            m = lab >= 0
            if not m.any():
                continue
            hi = np.quantile(adv, 0.7)
            pos = (adv >= hi) & (adv > 0)
            A_ += list(adv[m]); Y += list(lab[m] == 1)
            P_ += list(lab[m & pos] == 1); N_ += list(lab[m & ~pos] == 1)
        if Y:
            L.append(f"| {prop} | {len(Y)} | {np.mean(Y):.3f} | {np.mean(P_) if P_ else float('nan'):.3f} (n={len(P_)}) | "
                     f"{np.mean(N_) if N_ else float('nan'):.3f} (n={len(N_)}) | {auc(A_, Y):.3f} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
