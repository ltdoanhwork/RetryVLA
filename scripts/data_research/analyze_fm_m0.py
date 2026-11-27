"""Analyse runs/_analysis/data_research/fm_m0/anchors.jsonl (M0 FM loss per anchor, CPU, fm_loss_m0.py).

python -m scripts.data_research.analyze_fm_m0 -> runs/_analysis/data_research/fm_m0/summary.{md,json}

Approximations (logged in the doc): an arm's recovery loss = anchor-length-weighted mean over its ELIGIBLE pool
(keep list applied, round-robin selection not replayed), segment mix from trajectories.jsonl lengths; per-task anchor
share of an arm = pool share x the arm's recovery anchor count (meta.anchors_per_source).
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from retryvla import REPO_ROOT

D = REPO_ROOT / "runs/_analysis/data_research/fm_m0"
rows = [json.loads(l) for l in open(D / "anchors.jsonl")]
for r in rows:
    if r["source"] == "onset_rand_c":
        r["source"] = "onset_rand"
    if r["source"] == "gs_rand" and r["segment"] == 1:
        pass


def boot(x, n=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


out = {"groups": {}}
L_d0 = np.mean([r["loss"] for r in rows if r["source"] == "d0"])
grp = defaultdict(list)
for r in rows:
    grp[(r["source"], r["segment"])].append(r)
lines = ["| source | segment | n | M0 FM loss (mean [95 % CI]) | x D0 | chunk d3 | grip flips / chunk |", "|---|---|---|---|---|---|---|"]
for (s, seg), rs in sorted(grp.items()):
    m, lo, hi = boot([r["loss"] for r in rs])
    out["groups"][f"{s}|{seg}"] = {"n": len(rs), "loss": m, "ci": [lo, hi], "ratio": m / L_d0,
                                   "d3": float(np.mean([r["d3"] for r in rs])), "flips": float(np.mean([r["flips"] for r in rs]))}
    lines.append(f"| {s} | {seg} | {len(rs)} | {m:.3f} [{lo:.3f}, {hi:.3f}] | {m / L_d0:.1f}x | "
                 f"{np.mean([r['d3'] for r in rs]):.3f} | {np.mean([r['flips'] for r in rs]):.2f} |")

# ---- bridge family (heuristic on the bridge actions, as scripts/wm_optimism/cache_stats.action_family)
import pyarrow.parquet as pq


def ep_actions(root):
    tabs = [pq.read_table(f, columns=["episode_index", "action", "segment"]) for f in sorted(glob.glob(f"{REPO_ROOT / root}/data/*/*.parquet"))]
    ep = np.concatenate([t["episode_index"].to_numpy() for t in tabs])
    seg = np.concatenate([t["segment"].to_numpy() for t in tabs])
    act = np.concatenate([np.stack(t["action"].to_numpy(zero_copy_only=False)) for t in tabs])
    return {int(e): act[(ep == e) & (seg == 0)] for e in np.unique(ep)}


def family(a):
    if len(a) < 4:
        return "short(<4)"
    rot0 = np.abs(a[:, 3:6]).max() == 0
    gex = np.isin(a[:, 6], [-1.0, 1.0]).all()
    rough = np.abs(np.diff(a[:, :6], 2, axis=0)).mean()
    if rot0 and gex:
        return "prim"
    if gex:
        return "prim+cem"
    return "pi0_smooth" if rough < 0.11 else ("pi0_rough" if rough < 0.205 else "perturb-like")


fam_rows = defaultdict(list)
cache = {}
for r in rows:
    if r["segment"] != 0 or not r["source"].startswith(("onset", "gs")):
        continue
    if r["root"] not in cache:
        cache[r["root"]] = ep_actions(r["root"])
    r["family"] = family(cache[r["root"]].get(r["episode"], np.zeros((0, 7))))
    fam_rows[r["family"]].append(r["loss"])
lines += ["", "| bridge family (heuristic, onset + old banks) | n | M0 FM loss | x D0 |", "|---|---|---|---|"]
out["family"] = {}
for f, v in sorted(fam_rows.items()):
    m, lo, hi = boot(v)
    out["family"][f] = {"n": len(v), "loss": m, "ci": [lo, hi]}
    lines.append(f"| {f} | {len(v)} | {m:.3f} [{lo:.3f}, {hi:.3f}] | {m / L_d0:.1f}x |")

# ---- phys filter keep vs reject (onset banks)
keep = {}
for f in ["phys_wm_diverse", "phys_random", "phys_noseam_wm_diverse", "phys_noseam_random"]:
    for l in open(REPO_ROOT / f"data/recovery/goal_search_onset_cur/{f}.jsonl"):
        q = json.loads(l)
        keep[(f.replace("phys_noseam_", "ns:").replace("phys_", "st:").split(":")[0] if False else f,
              str(Path(q["root"]).relative_to(REPO_ROOT)), int(q["episode_index"]))] = q["keep"]
lines += ["", "| onset banks, phys filter | segment | keep n / loss | reject n / loss |", "|---|---|---|---|"]
out["phys"] = {}
for f in ["phys_wm_diverse", "phys_noseam_wm_diverse"]:
    for seg in (0, 1):
        k = [r["loss"] for r in rows if r["source"] == "onset_wm" and r["segment"] == seg and keep.get((f, r["root"], r["episode"])) is True]
        j = [r["loss"] for r in rows if r["source"] == "onset_wm" and r["segment"] == seg and keep.get((f, r["root"], r["episode"])) is False]
        p = stats.mannwhitneyu(k, j).pvalue if len(k) > 2 and len(j) > 2 else float("nan")
        out["phys"][f"{f}|{seg}"] = {"keep": [len(k), float(np.mean(k)) if k else None], "reject": [len(j), float(np.mean(j)) if j else None], "p": p}
        lines.append(f"| {f} | {seg} | {len(k)} / {np.mean(k) if k else float('nan'):.3f} | {len(j)} / {np.mean(j) if j else float('nan'):.3f} (MWU p {p:.2f}) |")

# ---- arm-level: data definition -> pool rows
def traj_meta(glob_):
    t = {}
    for root in sorted(glob.glob(str(REPO_ROOT / glob_))):
        rel = str(Path(root).relative_to(REPO_ROOT))
        for l in open(Path(root) / "retryvla/trajectories.jsonl"):
            q = json.loads(l)
            t[(rel, int(q["episode_index"]))] = q
    return t


def pool_loss(src, glob_, kl=None, seg0_only=False, cont_max=0):
    """anchor-weighted M0 loss of the eligible pool + per-task anchor share and per-task loss."""
    tm = traj_meta(glob_)
    if kl:
        ks = {(str(Path(json.loads(l)["root"]).relative_to(REPO_ROOT)), int(json.loads(l)["episode_index"]))
              for f in kl for l in open(REPO_ROOT / f) if json.loads(l)["keep"]}
        tm = {k: v for k, v in tm.items() if k in ks}
    seg_mean = {s: np.mean([r["loss"] for r in rows if r["source"] == src and r["segment"] == s] or [np.nan]) for s in (0, 1)}
    per = {(r["root"], r["episode"], r["segment"]): r["loss"] for r in rows if r["source"] == src}
    mass, n, task_n, task_mass = 0.0, 0, defaultdict(float), defaultdict(float)
    lens = None
    for (root, e), q in tm.items():
        if "recovery_len" not in q and "length" not in q:   # LRS windows: anchor count from the parquet
            if lens is None:
                lens = {}
                for f in sorted(glob.glob(f"{REPO_ROOT / root}/data/*/*.parquet")):
                    ee = pq.read_table(f, columns=["episode_index"])["episode_index"].to_numpy()
                    for k_, c_ in zip(*np.unique(ee, return_counts=True)):
                        lens[int(k_)] = lens.get(int(k_), 0) + int(c_)
            q = q | {"recovery_len": lens.get(e, 0), "continuation_len": 0}
        n0 = q.get("recovery_len", q.get("length", 0))
        n1 = 0 if seg0_only else q.get("continuation_len", 0)
        if cont_max:
            n1 = min(n1, cont_max)
        task = q["task"].replace(" ", "_") if "task" in q else q["instance_id"].split("__")[0]
        for s, ns in ((0, n0), (1, n1)):
            if ns:
                l_ = per.get((root, e, s), seg_mean[s])
                if np.isnan(l_):
                    l_ = seg_mean[0]
                mass += ns * l_
                n += ns
                task_n[task] += ns
                task_mass[task] += ns * l_
    return mass / max(n, 1), {t: task_n[t] / n for t in task_n}, {t: task_mass[t] / task_n[t] for t in task_n}


ONSET = "data/recovery/goal_search_onset_cur/s*/p0/"
GS = "data/recovery/goal_search/s*/p*/"
KL = "data/recovery/goal_search_onset_cur/"
ARMS = {  # arm: (run prefix, seeds, [(src, glob, keep lists, seg0_only, cont_max)])
    "F0_wm (all 282)": ("rec_wm", [7, 8, 9], [("gs_wm", GS + "wm_diverse", None, False, 0)]),
    "F0_random (all 282)": ("rec_rand", [7, 8, 9], [("gs_rand", GS + "random", None, False, 0)]),
    "F1_wm (bridge only, 36)": ("f1_wm", [7, 8, 9], [("gs_wm", GS + "wm_diverse", None, True, 0)]),
    "F1_random (bridge only, 36)": ("f1_rand", [7, 8, 9], [("gs_rand", GS + "random", None, True, 0)]),
    "D5 (old bank 36)": ("d5_wm", [8, 9], [("gs_wm", GS + "wm_diverse", None, False, 0)]),
    "G2_wm@36": ("g2_wm", [7, 8, 9], [("onset_wm", ONSET + "wm_diverse", None, False, 0)]),
    "G2_wm@18": ("g2_wm18", [7, 8, 9], [("onset_wm", ONSET + "wm_diverse", None, False, 0)]),
    "G2_wm@5": ("g2_wm5", [7, 8, 9], [("onset_wm", ONSET + "wm_diverse", None, False, 0)]),
    "G2_random@36": ("g2_rand", [7, 8, 9], [("onset_rand", ONSET + "random", None, False, 0)]),
    "G2-phys (18)": ("g2phys_wm", [7, 8, 9, 10, 11], [("onset_wm", ONSET + "wm_diverse", [KL + "phys_wm_diverse.jsonl"], False, 0)]),
    "G2-physns (36)": ("g2physns_wm", [7, 8, 9], [("onset_wm", ONSET + "wm_diverse", [KL + "phys_noseam_wm_diverse.jsonl"], False, 0)]),
    "strict-phys (46)": ("g3physr", [7, 8, 9, 10, 11], [("onset_wm", ONSET + "wm_diverse", [KL + "phys_wm_diverse.jsonl"], False, 0),
                                                        ("onset_rand", ONSET + "random", [KL + "phys_random.jsonl"], False, 0)]),
    "P2-3 A (72)": ("p23_A", [7, 8, 9], [("onset_wm", ONSET + "wm_diverse", None, False, 0)]),
    "G3-rep (72, s10-12)": ("g3rep_A", [10, 11, 12], [("onset_wm", ONSET + "wm_diverse", None, False, 0)]),
    "G3-c20 (72, cont<=20)": ("g3c20_A", [7, 8, 9], [("onset_wm", ONSET + "wm_diverse", None, False, 20)]),
    "LRS v1": ("lrs", [7, 8, 9], [("lrs_v1", "data/recovery/lrs_goal_ds", None, True, 0)]),
    "LRS v2": ("lrs2", [7, 8, 9], [("lrs_v2", "data/recovery/lrs_goal_v2_ds", None, True, 0)]),
    "LRS v2-GT": ("lrs2gt", [7], [("lrs_v2gt", "data/recovery/lrs_goal_v2gt_ds", None, True, 0)]),
}


def val(run):
    p = REPO_ROOT / f"runs/goal_rsi/{run}/ckpt_02000/eval_val/summary.json"
    return json.loads(p.read_text()) if p.exists() else None


orig = {s: val(f"orig_s{s}") for s in range(7, 12)}
orig_mean = np.mean([o["sr"] for o in orig.values()]) * 100
orig_task = {t: np.mean([o["by_task"][t] for o in orig.values()]) for t in orig[7]["by_task"]}
arm_tab, task_pts = [], []
for name, (pre, seeds, parts) in ARMS.items():
    vs = {s: val(f"{pre}_s{s}") for s in seeds}
    vs = {s: v for s, v in vs.items() if v}
    if not vs:
        continue
    meta = json.loads((REPO_ROOT / f"runs/goal_rsi/{pre}_s{min(vs)}/ckpt_02000/meta.json").read_text())
    n_rec = sum(meta["anchors_per_source"][1:])
    n_d0 = meta["anchors_per_source"][0]
    losses, shares, tl = [], defaultdict(float), defaultdict(float)
    for src, g, kl, s0, cm in parts:
        Lp, sh, tlp = pool_loss(src, g, kl, s0, cm)
        losses.append(Lp)
        for t in sh:
            shares[t] += sh[t] / len(parts)
            tl[t] = tlp[t]
    L_rec = float(np.mean(losses))
    gshare = n_rec * L_rec / (n_rec * L_rec + n_d0 * L_d0)
    d = np.mean([v["sr"] for v in vs.values()]) * 100 - np.mean([orig[s]["sr"] * 100 for s in vs if s in orig] or [orig_mean])
    arm_tab.append({"arm": name, "seeds": sorted(vs), "n_rec": n_rec, "share": n_rec / (n_rec + n_d0), "L_rec": L_rec,
                    "L_ratio": L_rec / L_d0, "grad_share": gshare, "delta_nominal": float(d)})
    tasks_seen = set(shares)
    for t in orig_task:
        dt = np.mean([v["by_task"][t] for v in vs.values()]) - orig_task[t]
        exc = (n_rec * shares.get(t, 0.0) * (tl.get(t, L_d0) - L_d0)) / (n_d0 / 10)   # excess loss mass vs the task's D0 mass
        task_pts.append({"arm": name, "task": t, "delta": float(dt), "excess": float(exc), "has_data": t in tasks_seen,
                         "rec_share_task": float(n_rec * shares.get(t, 0.0) / (n_d0 / 10))})

lines += ["", "| arm | seeds | rec anchors | share | M0 loss rec (x D0) | gradient share (loss-weighted) | Δ nominal vs Original |",
          "|---|---|---|---|---|---|---|"]
for a in arm_tab:
    lines.append(f"| {a['arm']} | {','.join(map(str, a['seeds']))} | {a['n_rec']} | {100 * a['share']:.1f} % | {a['L_rec']:.3f} ({a['L_ratio']:.1f}x) | "
                 f"{100 * a['grad_share']:.1f} % | {a['delta_nominal']:+.1f} |")
dn = np.array([a["delta_nominal"] for a in arm_tab])
corr = {}
for k in ("share", "L_rec", "grad_share"):
    x = np.array([a[k] for a in arm_tab])
    r_, p_ = stats.spearmanr(x, dn)
    corr[k] = {"spearman": float(r_), "p": float(p_)}
# without the LRS arms (the collapse arms dominate the ranks)
sub = [a for a in arm_tab if not a["arm"].startswith("LRS")]
for k in ("share", "L_rec", "grad_share"):
    r_, p_ = stats.spearmanr([a[k] for a in sub], [a["delta_nominal"] for a in sub])
    corr[k + "_noLRS"] = {"spearman": float(r_), "p": float(p_)}
# per-task, arm-demeaned
ex = np.array([p["excess"] for p in task_pts])
dl = np.array([p["delta"] for p in task_pts])
arm_mean = {a: np.mean([p["delta"] for p in task_pts if p["arm"] == a]) for a in {p["arm"] for p in task_pts}}
dl_dm = np.array([p["delta"] - arm_mean[p["arm"]] for p in task_pts])
r1 = stats.spearmanr(ex, dl)
r2 = stats.spearmanr(ex, dl_dm)
sh_t = np.array([p["rec_share_task"] for p in task_pts])
r3 = stats.spearmanr(sh_t, dl_dm)
corr["task_excess_vs_delta"] = {"spearman": float(r1.statistic), "p": float(r1.pvalue), "n": len(task_pts)}
corr["task_excess_vs_delta_arm_demeaned"] = {"spearman": float(r2.statistic), "p": float(r2.pvalue)}
corr["task_share_vs_delta_arm_demeaned"] = {"spearman": float(r3.statistic), "p": float(r3.pvalue)}
nod = [p["delta"] for p in task_pts if not p["has_data"]]
corr["tasks_without_data_mean_delta"] = {"n": len(nod), "mean": float(np.mean(nod)) if nod else None}
lines += ["", "Correlations (Spearman):", "```", json.dumps(corr, indent=1), "```", f"D0 loss {L_d0:.3f}; Original 5-seed {orig_mean:.1f}"]
out |= {"L_d0": float(L_d0), "arms": arm_tab, "corr": corr, "task_points": task_pts}
(D / "summary.json").write_text(json.dumps(out, indent=1))
(D / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
