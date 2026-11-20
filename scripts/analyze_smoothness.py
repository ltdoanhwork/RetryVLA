"""Action / EE smoothness of verified recovery data vs nominal demos vs pi0's own rollouts (CPU only, offline).

Sources (all train / pilot data; no test-scene demos):
  Spatial  demo      lerobot/libero episodes of the libero_spatial tasks (the SFT nominal data)
           bank seg0 searched 50-step open-loop bridge of the M5b banks (wm_diverse = the pi1 m5c_rr bank; random,
                     wm_noise, wm_qd for reference), segment 0
           bank seg1 pi0 (SmolVLA, replan every 10) closed-loop continuation inside the same episodes, segment 1
           pool      data/wm/collect_v1 candidates at the failure states, by proposal source, verified-success only
  Goal     demo      data/goal_rsi/d0_train (replayed LIBERO-Goal train demos, = 2610.05151's D0)
           bank      data/recovery/goal_search/s*/p0/{wm_diverse,random} seg0 / seg1 (pi0 3B, replan every 5)
           pi0 roll  data/goal_rsi/audit_m0_s7 full M0 rollouts (success / failure)

Metrics per segment (arm dims 0:6 of the 7-d normalised delta-EE action; actions are velocity-like commands, so
Delta a ~ acceleration, Delta^2 a ~ jerk; the realised EE position p = state[:3] gives jerk = Delta^3 p):
  d1      mean_t ||a_{t+1} - a_t||_2
  d2      mean_t ||Delta^2 a_t||_2
  d3      mean_t ||Delta^3 a_t||_2              (requested |Delta^3 a|)
  sat     fraction of arm entries with |a| >= 0.98 (CEM clipping)
  grip    gripper sign changes per 100 steps
  bnd     mean ||Delta a|| at replan boundaries / mean ||Delta a|| inside chunks (pi0 rollouts; 1 = no seam)
  hand    ||a_start(seg1) - a_end(seg0)|| (bridge -> pi0 handoff jump), banks only
  ee_j    mean ||Delta^3 p|| (mm / step^3) of the realised EE position
  ee_d1   mean ||Delta^2 p|| (mm / step^2)
  path    path length / straight-line distance of p (median; segments with chord < 1 cm dropped)

usage: python scripts/analyze_smoothness.py [--out runs/_analysis/smoothness] [--max_eps 400]
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
LEROBOT_LIBERO = Path("/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.cache/huggingface/hub/datasets--lerobot--libero")


def seg_metrics(a: np.ndarray, p: np.ndarray | None = None, k: int | None = None, start: int = 0) -> dict:
    a = np.asarray(a, np.float64)
    arm = a[:, :6]
    T = len(a)
    out = {"T": T}
    if T < 5:
        return out
    da = np.diff(arm, axis=0)
    n1 = np.linalg.norm(da, axis=1)
    out["d1"] = n1.mean()
    out["d2"] = np.linalg.norm(np.diff(arm, 2, axis=0), axis=1).mean()
    out["d3"] = np.linalg.norm(np.diff(arm, 3, axis=0), axis=1).mean()
    out["sat"] = (np.abs(arm) >= 0.98).mean()
    g = np.sign(a[:, 6])
    out["grip"] = 100.0 * (np.diff(g) != 0).sum() / T
    if k:
        # Delta index i = a[i+1] - a[i]; a replan happens at step t with (t - start) % k == 0, t > start
        t = np.arange(1, T)
        is_b = ((t - start) % k == 0) & (t > start)
        if is_b.any() and (~is_b).any():
            out["bnd"] = n1[is_b].mean() / max(n1[~is_b].mean(), 1e-9)
    if p is not None and len(p) >= 5:
        pp = np.asarray(p, np.float64)[:, :3] * 1000.0  # mm
        out["ee_d1"] = np.linalg.norm(np.diff(pp, 2, axis=0), axis=1).mean()
        out["ee_j"] = np.linalg.norm(np.diff(pp, 3, axis=0), axis=1).mean()
        chord = np.linalg.norm(pp[-1] - pp[0])
        if chord >= 10.0:
            out["path"] = np.linalg.norm(np.diff(pp, axis=0), axis=1).sum() / chord
    return out


def read_lerobot(root: Path, keep_tasks: set[int] | None, max_eps: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    files = sorted((root / "data").rglob("*.parquet"))
    cols = ["episode_index", "frame_index", "action", "observation.state"]
    names = pq.read_schema(files[0]).names
    has_seg = "segment" in names
    if has_seg:
        cols.append("segment")
    if keep_tasks is not None:
        cols.append("task_index")
    eps = []
    for f in files:
        df = pq.read_table(f, columns=cols).to_pandas()
        if keep_tasks is not None:
            df = df[df.task_index.isin(keep_tasks)]
        for _, e in df.groupby("episode_index"):
            e = e.sort_values("frame_index")
            seg = np.array([np.ravel(v)[0] for v in e.segment.values]).astype(int) if has_seg else np.zeros(len(e), int)
            eps.append((np.stack(e.action.values), np.stack(e["observation.state"].values), seg))
            if len(eps) >= max_eps:
                return eps
    return eps


def bank_rows(pattern: str, label: str, k: int, max_eps: int) -> list[dict]:
    rows = []
    for d in sorted(glob.glob(str(REPO / pattern))):
        if not (Path(d).parent / "CLOSED").exists():
            continue
        for a, s, seg in read_lerobot(Path(d), None, max_eps):
            i0 = np.nonzero(seg == 0)[0]
            i1 = np.nonzero(seg == 1)[0]
            if len(i0):
                rows.append({"src": f"{label} seg0 (searched bridge)", **seg_metrics(a[i0], s[i0])})
            if len(i1) >= 5:
                r = {"src": f"{label} seg1 (pi0 continuation)", **seg_metrics(a[i1], s[i1], k=k, start=0)}
                if len(i0):
                    r["hand"] = float(np.linalg.norm(a[i1[0], :6] - a[i0[-1], :6]))
                rows.append(r)
            if len(i0) and len(i1):  # whole episode, replan seams only in seg1
                rows.append({"src": f"{label} full episode", **seg_metrics(a, s)})
    return rows


def spatial_task_ids(root: Path) -> set[int]:
    from libero.libero import benchmark

    langs = {t.language for t in benchmark.get_benchmark_dict()["libero_spatial"]().tasks}
    tasks = pd.read_parquet(root / "meta" / "tasks.parquet")
    return {int(r.task_index) for lang, r in tasks.iterrows() if lang in langs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/_analysis/smoothness")
    ap.add_argument("--max_eps", type=int, default=400)
    args = ap.parse_args()
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    # ---------------- Spatial
    snap = sorted((LEROBOT_LIBERO / "snapshots").iterdir())[0]
    for a, s, _ in read_lerobot(snap, spatial_task_ids(snap), args.max_eps):
        rows.append({"src": "S demo (lerobot/libero spatial)", **seg_metrics(a, s, k=10)})
    for arm in ("wm_diverse", "random", "wm_noise", "wm_qd"):
        lab = "S bank " + arm + (" [pi1 bank]" if arm == "wm_diverse" else "")
        rows += bank_rows(f"data/recovery/m5b/s*/p*/{arm}", lab, k=10, max_eps=args.max_eps)
    # pool candidates by proposal source (actions only), verified-success vs all
    import torch

    pool = {}
    for f in sorted(glob.glob(str(REPO / "data/wm/collect_v1/*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        A, ok = d["actions"].numpy(), d["success"].numpy()
        for i, srcname in enumerate(d["source"]):
            fam = "prim" if srcname.startswith("prim_") else srcname
            pool.setdefault(fam, []).append((seg_metrics(A[i]), bool(ok[i])))
    for fam, lst in pool.items():
        for m, ok in lst:
            rows.append({"src": f"S pool {fam} (all)", **m})
            if ok:
                rows.append({"src": f"S pool {fam} (verified success)", **m})

    # ---------------- Goal
    for a, s, _ in read_lerobot(REPO / "data/goal_rsi/d0_train", None, args.max_eps):
        rows.append({"src": "G demo (D0 train)", **seg_metrics(a, s, k=5)})
    for arm in ("wm_diverse", "random"):
        rows += bank_rows(f"data/recovery/goal_search/s*/p0/{arm}", f"G bank {arm}", k=5, max_eps=args.max_eps)
    acts = np.load(REPO / "data/goal_rsi/audit_m0_s7/actions_s7.npz")
    for l in open(REPO / "data/goal_rsi/audit_m0_s7/rollouts.jsonl"):
        r = json.loads(l)
        tag = "success" if r["success"] else "failure"
        rows.append({"src": f"G pi0 rollout ({tag}, audit s7)", **seg_metrics(acts[r["scene"]], None, k=5)})

    df = pd.DataFrame(rows)
    df.to_csv(out / "segments.csv", index=False)
    cols = ["d1", "d2", "d3", "sat", "grip", "bnd", "hand", "ee_d1", "ee_j", "path"]
    agg = df.groupby("src", sort=False).agg(n=("T", "size"), T=("T", "mean"), **{c: (c, "mean") for c in cols if c != "path"},
                                            path=("path", "median"))
    agg.to_csv(out / "summary.csv")
    with open(out / "summary.md", "w") as fh:
        fh.write("| source | n | T | d1 | d2 | d3 | sat | grip/100 | bnd | hand | EE d2p mm | EE jerk mm | path ratio (med) |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for src, r in agg.iterrows():
            f = lambda v, p=3: "-" if pd.isna(v) else f"{v:.{p}f}"
            fh.write(f"| {src} | {int(r.n)} | {r['T']:.0f} | {f(r.d1)} | {f(r.d2)} | {f(r.d3)} | {f(r.sat, 3)} | "
                     f"{f(r.grip, 2)} | {f(r.bnd, 2)} | {f(r.hand)} | {f(r.ee_d1, 3)} | {f(r.ee_j, 3)} | {f(r.path, 2)} |\n")
    print(open(out / "summary.md").read())
    whatif(out)
    plots(df, out)


def whatif(out: Path, max_eps: int = 400):
    """Offline what-if for a post-hoc smoothing step: low-pass the searched bridges (seg0 of the pi1 bank) and report
    the jerk reduction vs the open-loop drift it causes (integrated commanded displacement, converted with the
    empirical gain of the bank data: realised EE step ~ 10.3 mm per action unit, r = 0.89, so
    drift_mm = 10.3 * ||sum_t (a_s - a)_{0:3}||). Large drift = smoothing changes the
    bridge enough that it must be re-verified in sim (it is not a free cosmetic fix)."""
    from scipy.signal import savgol_filter

    rows = []
    for d in sorted(glob.glob(str(REPO / "data/recovery/m5b/s*/p*/wm_diverse"))):
        if not (Path(d).parent / "CLOSED").exists():
            continue
        for a, s, seg in read_lerobot(Path(d), None, max_eps):
            i0 = np.nonzero(seg == 0)[0]
            if len(i0) < 12:
                continue
            A = a[i0].astype(np.float64)
            base = seg_metrics(A)
            for name, fn in (("movavg5", lambda x: np.apply_along_axis(lambda c: np.convolve(np.pad(c, 2, mode="edge"), np.ones(5) / 5, "valid"), 0, x)),
                             ("savgol11_3", lambda x: savgol_filter(x, 11, 3, axis=0, mode="interp"))):
                As = A.copy()
                As[:, :6] = fn(A[:, :6])
                m = seg_metrics(As)
                cum = np.cumsum(As[:, :3] - A[:, :3], axis=0) * 10.3
                rows.append({"filter": name, "d3_before": base["d3"], "d3_after": m["d3"],
                             "drift_end_mm": float(np.linalg.norm(cum[-1])), "drift_max_mm": float(np.linalg.norm(cum, axis=1).max())})
    df = pd.DataFrame(rows)
    g = df.groupby("filter").agg(n=("d3_before", "size"), d3_before=("d3_before", "mean"), d3_after=("d3_after", "mean"),
                                 drift_end_med=("drift_end_mm", "median"), drift_max_med=("drift_max_mm", "median"),
                                 drift_max_p90=("drift_max_mm", lambda x: np.percentile(x, 90)))
    txt = "| filter | n | d3 before | d3 after | open-loop drift end (median mm) | max drift (median mm) | max drift p90 (mm) |\n|---|---|---|---|---|---|---|\n"
    for f, r in g.iterrows():
        txt += f"| {f} | {int(r.n)} | {r.d3_before:.3f} | {r.d3_after:.3f} | {r.drift_end_med:.1f} | {r.drift_max_med:.1f} | {r.drift_max_p90:.1f} |\n"
    (out / "smoothing_whatif.md").write_text(txt)
    print(txt)


def plots(df: pd.DataFrame, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["S demo (lerobot/libero spatial)", "S bank wm_diverse [pi1 bank] seg0 (searched bridge)",
             "S bank wm_diverse [pi1 bank] seg1 (pi0 continuation)", "S bank random seg0 (searched bridge)",
             "S pool vla_s1.0 (verified success)", "S pool perturb (verified success)", "S pool prim (verified success)",
             "G demo (D0 train)", "G bank wm_diverse seg0 (searched bridge)", "G bank wm_diverse seg1 (pi0 continuation)",
             "G pi0 rollout (success, audit s7)", "G pi0 rollout (failure, audit s7)"]
    order = [o for o in order if o in set(df.src)]
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.5))
    for ax, (c, name) in zip(axs, [("d1", "mean ||Δa|| (arm, normalised units)"), ("d3", "mean ||Δ³a||"),
                                    ("ee_j", "realised EE jerk mean ||Δ³p|| (mm)")]):
        data = [df.loc[df.src == o, c].dropna().values for o in order]
        keep = [i for i, d in enumerate(data) if len(d)]
        ax.boxplot([data[i] for i in keep], positions=[i + 1 for i in keep], orientation="horizontal", showfliers=False)
        ax.set_yticks(range(1, len(order) + 1))
        ax.set_yticklabels(order if ax is axs[0] else [""] * len(order), fontsize=8)
        ax.set_ylim(0.5, len(order) + 0.5)
        ax.set_title(name, fontsize=10)
        ax.invert_yaxis()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_smoothness_box.png", dpi=130)
    plt.close(fig)

    # example traces: one Spatial demo vs one pi1-bank episode (bridge -> pi0 continuation), position dims
    snap = sorted((LEROBOT_LIBERO / "snapshots").iterdir())[0]
    demo = read_lerobot(snap, spatial_task_ids(snap), 1)[0]
    d = sorted(glob.glob(str(REPO / "data/recovery/m5b/s*/p*/wm_diverse")))[0]
    eps = read_lerobot(Path(d), None, 12)
    bank = max(eps, key=lambda e: min((e[2] == 1).sum(), 80))
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.5), sharey=True)
    for ax, (a, s, seg), title in ((axs[0], demo, "Spatial nominal demo (lerobot/libero)"),
                                   (axs[1], bank, "pi1 bank episode: searched bridge (seg0) -> pi0 continuation (seg1, replan every 10)")):
        for j, c in zip(range(3), ("tab:red", "tab:green", "tab:blue")):
            ax.plot(a[:, j], color=c, lw=1.2, label="xyz"[j])
        ax.plot(a[:, 6] * 0.9, color="k", lw=0.8, ls=":", label="gripper x0.9")
        i1 = np.nonzero(seg == 1)[0]
        if len(i1):
            ax.axvline(i1[0], color="purple", lw=2, alpha=0.6)
            for t in range(i1[0] + 10, len(a), 10):
                ax.axvline(t, color="grey", lw=0.5, alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.3)
    axs[0].legend(ncol=4, fontsize=8)
    axs[1].set_xlabel("env step")
    fig.tight_layout()
    fig.savefig(out / "fig_example_traces.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
