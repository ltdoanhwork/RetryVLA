"""CRIT-V offline evaluation: metrics + paired instance-cluster bootstrap vs the retrained control (base).

python -m retryvla.critic.eval_variants runs/<ts>_CRITV_goal runs/<ts>_CRITV_spatial [--reps 1000]
-> runs/_analysis/critv/{critv.md, critv.json}

Per (suite, kind, view in {true, imag}), seed-mean of:
  AUC, ECE (10 bins) on all verified candidates; p@1 / p@4 = per instance, rank ALL its verified candidates by the
  critic, success rate of the top-k (mean over instances; search-pressure ranking); FP@top4 = P(fail | per-instance
  top-4 and pred >= 0.5); opt = mean pred - SR on the search-selected picks (wm_* arms); AUC on the random-arm
  (unselected) and on the search-pick subsets.
Paired difference vs base: bootstrap over instances (same resample for both critics, seed-mean inside).
Decision (PLAN CRIT-V): clearly wins iff on goal / imag dAUC >= 0.03 (CI > 0), dFP@top4 <= -0.05, dp@4 >= 0 and
spatial / imag dAUC >= -0.01.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from retryvla import REPO_ROOT
from retryvla.critic.value_model import auc, expected_calibration_error


def inst_groups(inst):
    by = {}
    for i, s in enumerate(inst):
        by.setdefault(s, []).append(i)
    return [np.asarray(v) for v in by.values()]


def metrics(p, y, groups, picks, rand):
    out = {"auc": auc(p, y), "ece": expected_calibration_error(p, y)}
    p1, p4, fp_n, fp_d = [], [], 0, 0
    for g in groups:
        o = g[np.argsort(-p[g], kind="stable")]
        p1.append(y[o[0]]); p4.append(y[o[:4]].mean())
        t4 = o[:4][p[o[:4]] >= 0.5]
        fp_n += int((y[t4] == 0).sum()); fp_d += len(t4)
    out["p@1"], out["p@4"] = float(np.mean(p1)), float(np.mean(p4))
    out["fp@top4"] = fp_n / fp_d if fp_d else float("nan")
    out["opt_picks"] = float(p[picks].mean() - y[picks].mean()) if picks.any() else float("nan")
    out["auc_picks"] = auc(p[picks], y[picks]) if picks.any() else float("nan")
    out["auc_random"] = auc(p[rand], y[rand]) if rand.any() else float("nan")
    return out


KEYS = ["auc", "ece", "p@1", "p@4", "fp@top4", "opt_picks", "auc_picks", "auc_random"]


def seed_mean(P, kind, view, y, groups, picks, rand):
    ms = [metrics(P[(k, s, v)], y, groups, picks, rand) for (k, s, v) in P if k == kind and v == view]
    return {key: float(np.nanmean([m[key] for m in ms])) for key in KEYS}, len(ms)


def bootstrap(P, kind, view, y, groups, picks, rand, reps, seed=0):
    rng = np.random.default_rng(seed)
    sk = sorted(s for (k, s, v) in P if k == kind and v == view)
    sb = sorted(s for (k, s, v) in P if k == "base" and v == view)
    diffs = {key: [] for key in KEYS}
    for _ in range(reps):
        gi = rng.integers(len(groups), size=len(groups))
        idx = np.concatenate([groups[g] for g in gi])
        # re-index groups for the resample
        off, ng = 0, []
        for g in gi:
            ng.append(np.arange(off, off + len(groups[g]))); off += len(groups[g])
        yy, pk, rd = y[idx], picks[idx], rand[idx]
        mk = [metrics(P[(kind, s, view)][idx], yy, ng, pk, rd) for s in sk]
        mb = [metrics(P[("base", s, view)][idx], yy, ng, pk, rd) for s in sb]
        for key in KEYS:
            diffs[key].append(np.nanmean([m[key] for m in mk]) - np.nanmean([m[key] for m in mb]))
    return {key: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for key, v in diffs.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--out", default="runs/_analysis/critv")
    a = ap.parse_args()
    res, lines = {}, []
    for r in a.runs:
        d = torch.load(REPO_ROOT / r / "preds.pt", weights_only=False)
        suite = "goal" if "goal" in r else "spatial"
        y = d["y"].astype(float)
        groups = inst_groups(d["inst"])
        src = np.asarray(d["src"])
        picks = np.char.startswith(src.astype(str), "wm_")
        rand = src == "random"
        P = d["p"]
        kinds = list(dict.fromkeys(k for (k, _, _) in P))
        lines += [f"\n### {suite} ({r}; eval n = {len(y)} candidates, {len(groups)} instances, SR {y.mean():.3f}; "
                  f"picks n = {int(picks.sum())} SR {y[picks].mean():.3f}; train n = {d['train_n']}, pos {d['train_pos']:.3f})\n",
                  "| kind | view | seeds | AUC | ECE | p@1 | p@4 | FP@top4 | opt (pred-SR, picks) | AUC picks | AUC random | dAUC [95% CI] | dFP@top4 [CI] | dp@4 [CI] |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for view in ("imag", "true"):
            for kind in kinds:
                m, ns = seed_mean(P, kind, view, y, groups, picks, rand)
                row = {"metrics": m, "n_seeds": ns}
                ci = ""
                if kind not in ("base", "cpi_shipped", "success_generic") and any(k == "base" for (k, _, _) in P):
                    mb, _ = seed_mean(P, "base", view, y, groups, picks, rand)
                    bs = bootstrap(P, kind, view, y, groups, picks, rand, a.reps if suite == "goal" else max(200, a.reps // 4))
                    row["delta"] = {key: m[key] - mb[key] for key in KEYS}
                    row["ci"] = bs
                    ci = " | ".join(f"{row['delta'][k]:+.3f} [{bs[k][0]:+.3f}, {bs[k][1]:+.3f}]" for k in ("auc", "fp@top4", "p@4"))
                else:
                    ci = "- | - | -"
                res[f"{suite}/{kind}/{view}"] = row
                lines.append(f"| {kind} | {view} | {ns} | " + " | ".join(f"{m[k]:.3f}" for k in KEYS) + f" | {ci} |")
        # chain diagnostics (cot): AUC of each binary chain entry on true / imagined end states
        if d.get("chain"):
            from retryvla.critic.variants import CHAIN_FLAT, chain_targets
            tgt, msk = chain_targets(torch.as_tensor(d["sym"]))
            tgt, msk = tgt.numpy(), msk.numpy()
            lines.append(f"\nChain read-out ({suite}, kind cot, seed-mean AUC of binary chain entries vs sim labels):\n")
            lines.append("| entry | rate | AUC true | AUC imag |")
            lines.append("|---|---|---|---|")
            for i, (k, t) in enumerate(CHAIN_FLAT):
                if t == "c":
                    continue
                m = msk[:, i]
                lab = tgt[m, i] > 0.5
                vals = {}
                for view in ("true", "imag"):
                    vals[view] = np.nanmean([auc(c[m, i], lab) for (kk, s, v), c in d["chain"].items()
                                             if kk == "cot" and v == view])
                lines.append(f"| {k} | {lab.mean():.3f} | {vals['true']:.3f} | {vals['imag']:.3f} |")
    g = res.get("goal/{}/imag")
    dec = {}
    for key, row in res.items():
        suite, kind, view = key.split("/")
        if suite != "goal" or view != "imag" or "delta" not in row:
            continue
        sp = res.get(f"spatial/{kind}/imag", {}).get("delta", {}).get("auc", float("nan"))
        d_, c_ = row["delta"], row["ci"]
        wins = d_["auc"] >= 0.03 and c_["auc"][0] > 0 and d_["fp@top4"] <= -0.05 and d_["p@4"] >= 0 and (
            not np.isfinite(sp) or sp >= -0.01)
        prom = (d_["auc"] >= 0.015 and c_["auc"][0] > 0) or (d_["fp@top4"] <= -0.05 and c_["fp@top4"][1] < 0)
        dec[kind] = "clearly wins" if wins else "promising" if prom else "no gain"
    del g
    lines.append("\nPre-registered decision (goal / imag vs base): " + ", ".join(f"{k}: {v}" for k, v in dec.items()))
    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "critv.md").write_text("\n".join(lines) + "\n")
    json.dump({"results": res, "decision": dec, "runs": a.runs}, open(out / "critv.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
