#!/usr/bin/env python
"""Log-based failure analysis of the best RetryVLA policies on LIBERO-Recover Spatial test (56 scenes x 10 trials).

python scripts/analyze_failures.py  ->  docs/failure_figs/*.png + runs/_analysis/failures/stats.json + tables.md

CPU only: reads existing runs/<id>/episodes.jsonl (no simulation). Items: per-task / per-scene RSR, gripper-init /
capture-step / trial breakdown, seed consistency, pi0 -> best transitions, end-state proxy (final bowl-plate xy
distance = -progress), training-data coverage per task (m5b wm_diverse round-robin 208 subset).
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
FIG = REPO / "docs/failure_figs"
OUT = REPO / "runs/_analysis/failures"

# name -> list of run globs (seeds)
MODELS = {
    "pi0": ["*_M0_recover_bddl"],
    "pi1-nominal": ["*_M1_pi1nom_recover_test"],
    "pi1 (RetryVLA)": ["*_M5_m5c_sft_wm_diverse_rr_recover_test", "*_M8_seed_pi1_s1_recover_test",
                       "*_M8_seed_pi1_s2_recover_test"],
    "M7 base (best)": ["*_M7_sft_base_recover_test", "*_M8_seed_base_s1_recover_test",
                       "*_M8_seed_base_s2_recover_test"],
}
EXTRA = {  # only for the weak-task / data section
    "M7 weak": ["*_M7_sft_weak_recover_test", "*_M8_seed_weak_s1_recover_test", "*_M8_seed_weak_s2_recover_test"],
    "M8 aug": ["*_M8_sft_aug_recover_test"],
}
COL = {"pi0": "#8c8c8c", "pi1-nominal": "#6a7fa0", "pi1 (RetryVLA)": "#e08a1e", "M7 base (best)": "#2e8b57",
       "M7 weak": "#9467bd", "M8 aug": "#d62728"}
SHORT = {
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate": "between plate & ramekin",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate": "in top drawer",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate": "next to cookie box",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate": "next to plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate": "next to ramekin",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate": "on cookie box",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": "on ramekin",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": "on stove",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate": "on wooden cabinet",
}
D_BINS = [(0.0, 0.05, "<= 5 cm (at the plate)"), (0.05, 0.15, "5-15 cm"), (0.15, 0.5, "15-50 cm"),
          (0.5, 9.0, "> 50 cm (knocked away)")]


def atomic_write(path: Path, data: bytes | str, tries: int = 20) -> None:
    """/pfss quota sometimes fills up (EDQUOT): write tmp + rename, retry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    b = data.encode() if isinstance(data, str) else data
    for k in range(tries):
        tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
        try:
            tmp.write_bytes(b)
            os.replace(tmp, path)
            return
        except OSError as e:
            print(f"write {path} failed ({e}); retry {k + 1}/{tries}", flush=True)
            try:
                tmp.unlink()
            except OSError:
                pass
            time.sleep(30)
    raise RuntimeError(f"could not write {path}")


def save_fig(fig, name: str) -> None:
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    atomic_write(FIG / name, buf.getvalue())
    print("saved", FIG / name, flush=True)


def load_runs(globs_: list[str]) -> list[dict]:
    """One dict per seed: key (scene, gripper_init, trial) -> row (test part only)."""
    out = []
    for g in globs_:
        hits = sorted(glob.glob(str(REPO / "runs" / g)))
        assert len(hits) == 1, (g, hits)
        rows = [json.loads(l) for l in open(Path(hits[0]) / "episodes.jsonl")]
        d = {(r["scene"], r["gripper_init"], r["trial"]): r for r in rows if r.get("part", "test") == "test"}
        assert len(d) == 560, (g, len(d))
        out.append(d)
    return out


def rate(seeds: list[dict], keys) -> float:
    return float(np.mean([s[k]["success"] for s in seeds for k in keys])) if keys else float("nan")


def analysis_logs() -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = {m: load_runs(g) for m, g in {**MODELS, **EXTRA}.items()}
    keys = sorted(M["pi0"][0])
    tasks = sorted({k[0].split("__")[0] for k in keys})
    by_task = {t: [k for k in keys if k[0].startswith(t + "__")] for t in tasks}
    scenes = sorted({k[0] for k in keys})
    st: dict = {"overall": {}, "task": {}, "grip": {}, "capture": {}, "trial": {}}
    for m, seeds in M.items():
        per_seed = [rate([s], keys) for s in seeds]
        st["overall"][m] = {"mean": float(np.mean(per_seed)), "sd": float(np.std(per_seed, ddof=1)) if len(seeds) > 1 else 0.0,
                            "per_seed": per_seed}
        st["task"][m] = {t: {"mean": rate(seeds, ks), "per_seed": [rate([s], ks) for s in seeds], "n": len(ks)}
                         for t, ks in by_task.items()}
        st["grip"][m] = {g: rate(seeds, [k for k in keys if k[1] == g]) for g in ("close", "open")}
        st["capture"][m] = {c: rate(seeds, [k for k in keys if k[0].endswith(c)]) for c in ("step_0180", "step_0200")}
        st["trial"][m] = {f"{g}{t}": rate(seeds, [k for k in keys if k[1] == g and k[2] % 5 == t % 5])
                          for g in ("close", "open") for t in range(5)}

    main = list(MODELS)
    # ---- Fig 1: per-task RSR ----
    order = sorted(tasks, key=lambda t: -st["task"]["M7 base (best)"][t]["mean"])
    fig, ax = plt.subplots(figsize=(12, 4.6))
    w = 0.2
    for i, m in enumerate(main):
        mu = [st["task"][m][t]["mean"] for t in order]
        ps = [st["task"][m][t]["per_seed"] for t in order]
        err = [np.std(p, ddof=1) if len(p) > 1 else 0 for p in ps]
        x = np.arange(len(order)) + (i - 1.5) * w
        ax.bar(x, mu, w, yerr=err, color=COL[m], label=f"{m}  (overall {st['overall'][m]['mean']:.3f})", capsize=2)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f"{SHORT[t]}\n({len(by_task[t]) // 10} scenes)" for t in order], fontsize=8, rotation=12)
    ax.set_ylabel("test RSR (mean over seeds; bar = sd)")
    ax.set_ylim(0, 1)
    ax.axhline(st["overall"]["M7 base (best)"]["mean"], color=COL["M7 base (best)"], ls=":", lw=1)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Fig 1. RSR per task (LIBERO-Recover Spatial test, 56 scenes x 10 trials)")
    save_fig(fig, "fig1_task_rsr.png")
    plt.close(fig)

    # ---- Fig 2: per-scene heatmap ----
    cols = [("pi0", 0), ("pi1-nominal", 0)] + [("pi1 (RetryVLA)", i) for i in range(3)] + \
           [("M7 base (best)", i) for i in range(3)]
    sc_order = sorted(scenes, key=lambda s: (order.index(s.split("__")[0]),
                                             -rate(M["M7 base (best)"], [k for k in keys if k[0] == s])))
    H = np.array([[rate([M[m][i]], [k for k in keys if k[0] == s]) for m, i in cols] for s in sc_order])
    st["scene"] = {s: {f"{m}#{i}": float(H[r, c]) for c, (m, i) in enumerate(cols)} for r, s in enumerate(sc_order)}
    fig, ax = plt.subplots(figsize=(8.5, 15))
    im = ax.imshow(H, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["pi0", "pi1-nom", "pi1 s0", "pi1 s1", "pi1 s2", "M7 s0", "M7 s1", "M7 s2"], rotation=30)
    ax.set_yticks(range(len(sc_order)))
    ax.set_yticklabels([f"{SHORT[s.split('__')[0]]} | {s.split('__')[1][-3:]}/{s.split('__')[2][-3:]}"
                        for s in sc_order], fontsize=7)
    prev = None
    for r, s in enumerate(sc_order):
        t = s.split("__")[0]
        if prev is not None and t != prev:
            ax.axhline(r - 0.5, color="k", lw=1.2)
        prev = t
    for r in range(H.shape[0]):
        for c in range(H.shape[1]):
            ax.text(c, r, f"{H[r, c] * 10:.0f}", ha="center", va="center", fontsize=6)
    ax.axvline(1.5, color="k", lw=1)
    ax.axvline(4.5, color="k", lw=1)
    plt.colorbar(im, ax=ax, fraction=0.03, label="RSR of the scene (10 trials; number = successes / 10)")
    ax.set_title("Fig 2. Per-scene RSR (rows = test scenes: task | episode/capture step)")
    save_fig(fig, "fig2_scene_heatmap.png")
    plt.close(fig)
    return {"M": M, "keys": keys, "tasks": tasks, "by_task": by_task, "order": order, "st": st, "plt": plt}


def analysis_more(A: dict) -> None:
    M, keys, by_task, order, st, plt = A["M"], A["keys"], A["by_task"], A["order"], A["st"], A["plt"]
    main = list(MODELS)
    # ---- Fig 3: gripper init / capture step / trial noise ----
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [1, 1, 2]})
    for ax, key, labs, title in ((axs[0], "grip", ("close", "open"), "gripper init"),
                                 (axs[1], "capture", ("step_0180", "step_0200"), "failure captured at step")):
        for i, m in enumerate(main):
            ax.bar(np.arange(2) + (i - 1.5) * 0.2, [st[key][m][l] for l in labs], 0.2, color=COL[m], label=m)
        ax.set_xticks(range(2))
        ax.set_xticklabels(labs)
        ax.set_ylim(0, 0.8)
        ax.set_title(title)
    axs[0].legend(fontsize=7)
    tl = [f"{g}{t}" for g in ("close", "open") for t in range(5)]
    for m in main:
        axs[2].plot(range(10), [st["trial"][m][x] for x in tl], "o-", color=COL[m], label=m)
    axs[2].set_xticks(range(10))
    axs[2].set_xticklabels([f"{x[:-1]}\nk={x[-1]}" for x in tl], fontsize=8)
    axs[2].set_ylim(0, 0.8)
    axs[2].set_title("per trial (k = object-noise seed, same for every model)")
    fig.suptitle("Fig 3. RSR by gripper init, capture step (no L1-L4 labels released) and trial noise")
    save_fig(fig, "fig3_breakdown.png")
    plt.close(fig)

    # ---- consistency over 3 seeds ----
    cons = {}
    for m in ("pi1 (RetryVLA)", "M7 base (best)", "M7 weak"):
        cnt = {k: sum(s[k]["success"] for s in M[m]) for k in keys}
        cons[m] = {"all": collections.Counter(cnt.values()),
                   "task": {t: collections.Counter(cnt[k] for k in ks) for t, ks in by_task.items()},
                   "cnt": cnt}
        sc_any = collections.defaultdict(int)
        for k, c in cnt.items():
            sc_any[k[0]] += c
        cons[m]["scenes_never"] = sorted(s for s, v in sc_any.items() if v == 0)
    st["consistency"] = {m: {"all": {str(i): v["all"].get(i, 0) for i in range(4)},
                             "task": {t: {str(i): c.get(i, 0) for i in range(4)} for t, c in v["task"].items()},
                             "scenes_never_solved": v["scenes_never"]} for m, v in cons.items()}
    # union of everything (any seed of any model)
    allm = [s for m in M for s in M[m]]
    st["never_any_model"] = int(sum(1 for k in keys if not any(s[k]["success"] for s in allm)))
    st["never_any_model_scenes"] = sorted({k[0] for k in keys} - {k[0] for k in keys if any(s[k]["success"] for s in allm)})
    # episodes no run ever solves AND whose final bowl-plate distance is identical in every run (object never moved):
    # d > 0.55 m = bowl / plate already off the table after the +-2 cm load noise (infeasible start)
    inf, untouched = [], []
    for k in keys:
        rs = [s[k] for s in allm]
        d = np.array([-r["progress"] for r in rs])
        if any(r["success"] for r in rs):
            continue
        if d.min() > 0.55:  # in EVERY run (0.55: a bowl on the wooden cabinet starts ~0.52 m from the plate)
            inf.append(k)
        elif d.std() < 0.005:
            untouched.append(k)
    st["infeasible_eps"] = {"n": len(inf), "runs": len(allm),
                            "per_task": dict(collections.Counter(SHORT[k[0].split("__")[0]] for k in inf)),
                            "ids": [f"{k[0]}__{k[1]}{k[2]}" for k in inf]}
    st["untouched_by_all_runs"] = {"n": len(untouched),
                                   "per_task": dict(collections.Counter(SHORT[k[0].split("__")[0]] for k in untouched))}

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 2.6]})
    cc = ["#c0392b", "#f39c12", "#9acd32", "#2e8b57"]
    for j, m in enumerate(("pi1 (RetryVLA)", "M7 base (best)")):
        v = cons[m]["all"]
        bot = 0
        for i in range(4):
            axs[0].bar(j, v.get(i, 0) / 560, bottom=bot, color=cc[i], label=f"{i}/3 seeds" if j == 0 else None)
            axs[0].text(j, bot + v.get(i, 0) / 1120, f"{v.get(i, 0) / 5.6:.0f}%", ha="center", fontsize=8)
            bot += v.get(i, 0) / 560
    axs[0].set_xticks([0, 1])
    axs[0].set_xticklabels(["pi1 (3 seeds)", "M7 base (3 seeds)"])
    axs[0].set_ylabel("fraction of the 560 test episodes")
    axs[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axs[0].set_title("episodes solved by k of 3 seeds")
    m = "M7 base (best)"
    for ti, t in enumerate(order):
        v = cons[m]["task"][t]
        n = len(by_task[t])
        bot = 0
        for i in range(4):
            axs[1].bar(ti, v.get(i, 0) / n, bottom=bot, color=cc[i])
            bot += v.get(i, 0) / n
    axs[1].set_xticks(range(len(order)))
    axs[1].set_xticklabels([SHORT[t] for t in order], fontsize=8, rotation=20, ha="right")
    axs[1].set_title("M7 base, per task: red = never solved (0/3), green = always (3/3)")
    fig.suptitle("Fig 4a. Consistency over training seeds (same episode = same failure state + noise + policy seed)")
    save_fig(fig, "fig4a_seed_consistency.png")
    plt.close(fig)

    # ---- transition pi0 -> best ----
    p0 = M["pi0"][0]
    trans = {}
    for m in ("pi1 (RetryVLA)", "M7 base (best)"):
        T = np.zeros((2, 2))
        for s in M[m]:
            for k in keys:
                T[int(p0[k]["success"]), int(s[k]["success"])] += 1 / len(M[m])
        trans[m] = T
        st.setdefault("transition", {})[m] = {"pi0_fail->fail": T[0, 0], "pi0_fail->succ": T[0, 1],
                                              "pi0_succ->fail": T[1, 0], "pi0_succ->succ": T[1, 1]}
    tt = {}
    for t, ks in by_task.items():
        T = np.zeros(4)
        for s in M["M7 base (best)"]:
            for k in ks:
                a, b = p0[k]["success"], s[k]["success"]
                T[(0 if (not a and b) else 1 if (not a and not b) else 2 if (a and not b) else 3)] += 1 / 3
        tt[t] = T / len(ks)
    st["transition_task_m7"] = {t: dict(zip(["fixed", "still_failing", "broken", "kept"], map(float, v)))
                                for t, v in tt.items()}
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw={"width_ratios": [1, 1, 2.4]})
    for ax, m in zip(axs[:2], ("pi1 (RetryVLA)", "M7 base (best)")):
        T = trans[m]
        ax.imshow(T, cmap="Blues", vmin=0, vmax=560)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{T[i, j]:.0f}\n({T[i, j] / 5.6:.0f}%)", ha="center", va="center", fontsize=11,
                        color="white" if T[i, j] > 280 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{m.split(' ')[0]} fail", f"{m.split(' ')[0]} success"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["pi0 fail", "pi0 success"])
        ax.set_title(f"pi0 -> {m} (mean of 3 seeds)", fontsize=10)
    lab = [("fixed (pi0 fail -> success)", "#2e8b57"), ("still failing", "#c0392b"),
           ("broken (pi0 success -> fail)", "#f39c12"), ("kept (both succeed)", "#9acd32")]
    for ti, t in enumerate(order):
        bot = 0
        for i, (l, c) in enumerate(lab):
            axs[2].bar(ti, tt[t][i], bottom=bot, color=c, label=l if ti == 0 else None)
            bot += tt[t][i]
    axs[2].set_xticks(range(len(order)))
    axs[2].set_xticklabels([SHORT[t] for t in order], fontsize=8, rotation=20, ha="right")
    axs[2].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axs[2].set_title("per task, pi0 -> M7 base")
    fig.suptitle("Fig 4b. Transition matrix pi0 -> best policies (560 test episodes)")
    save_fig(fig, "fig4b_transition.png")
    plt.close(fig)

    # ---- end-state proxy: final bowl-plate xy distance of failures ----
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ends = {}
    bc = ["#3498db", "#9b59b6", "#e67e22", "#7f8c8d"]
    for ti, t in enumerate(order):
        for j, m in enumerate(("pi0", "M7 base (best)")):
            d = np.array([-s[k]["progress"] for s in M[m] for k in by_task[t] if not s[k]["success"]])
            frac = [float(np.mean((d >= lo) & (d < hi))) if len(d) else 0.0 for lo, hi, _ in D_BINS]
            ends.setdefault(m, {})[t] = {"n_fail": int(len(d)), "frac": frac,
                                         "median_d": float(np.median(d)) if len(d) else None}
            bot = 0
            for i, f in enumerate(frac):
                ax.bar(ti + (j - 0.5) * 0.38, f, 0.36, bottom=bot, color=bc[i], alpha=0.55 if j == 0 else 1.0,
                       label=D_BINS[i][2] if (ti == 0 and j == 1) else None)
                bot += f
            ax.text(ti + (j - 0.5) * 0.38, 1.01, ("pi0" if j == 0 else "M7") + f"\n{len(d)}", ha="center",
                    fontsize=6)
    for m in ("pi0", "M7 base (best)"):
        d = np.array([-s[k]["progress"] for s in M[m] for k in keys if not s[k]["success"]])
        ends.setdefault(m, {})["ALL"] = {"n_fail": int(len(d)),
                                         "frac": [float(np.mean((d >= lo) & (d < hi))) for lo, hi, _ in D_BINS]}
    succ_d = np.array([-s[k]["progress"] for s in M["M7 base (best)"] for k in keys if s[k]["success"]])
    st["success_d_percentiles"] = np.percentile(succ_d, [5, 50, 95]).round(3).tolist()
    st["end_state"] = ends
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([SHORT[t] for t in order], fontsize=8, rotation=20, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("fraction of failed episodes")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="bowl-plate xy dist. at step 520", title_fontsize=7)
    ax.set_title("Fig 6. Where is the bowl when the episode times out? (left, faded = pi0; right = M7 base 3 seeds; "
                 f"successes end at {st['success_d_percentiles'][1] * 100:.0f} cm)", fontsize=9)
    save_fig(fig, "fig6_end_state_logs.png")
    plt.close(fig)


def analysis_data(A: dict) -> None:
    from retryvla.data.mix import expand_roots, recovery_episode_subsets

    st, plt, order = A["st"], A["plt"], A["order"]
    split = json.loads((REPO / "data/splits/spatial_v1.json").read_text())
    task_of = lambda s: s.split("__")[0]  # noqa: E731
    roots = [Path(r) for r in expand_roots([REPO / "data/recovery/m5b/s*/p*/wm_diverse"])]
    subs = recovery_episode_subsets(roots, 208, 0, mode="roundrobin")
    D = {t: collections.Counter() for t in SHORT}
    for t in SHORT:
        D[t]["train_scenes"] = sum(task_of(s) == t for s in split["train"])
        D[t]["test_scenes"] = sum(task_of(s) == t for s in split["test"])
    for root, sub in zip(roots, subs):
        for l in open(root.parent / "done.jsonl"):
            D[task_of(json.loads(l)["instance_id"])]["searched_failures"] += 1
        trajs = [json.loads(l) for l in open(root / "retryvla" / "trajectories.jsonl")]
        keep = set(sub or [])
        solved = set()
        for r in trajs:
            t = task_of(r["instance_id"])
            D[t]["pool_traj"] += 1
            solved.add(r["instance_id"])
            if r["episode_index"] in keep:
                D[t]["sft_traj_208"] += 1
        for i in solved:
            D[task_of(i)]["solved_failures"] += 1
    for t in D:
        D[t]["train_failure_states"] = D[t]["train_scenes"] * 10
    st["data"] = {t: dict(v) for t, v in D.items()}
    tot = sum(v["sft_traj_208"] for v in D.values())
    st["data_total_sft"] = tot

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(order))
    for i, (k, c, l) in enumerate((("train_scenes", "#95a5a6", "train scenes"),
                                   ("searched_failures", "#3498db", "pi0 train failures searched (M5b)"),
                                   ("solved_failures", "#e67e22", "... with >= 1 verified recovery"),
                                   ("sft_traj_208", "#2e8b57", "trajectories in the SFT subset (208)"))):
        axs[0].bar(x + (i - 1.5) * 0.2, [D[t][k] for t in order], 0.2, color=c, label=l)
        for xi, t in zip(x, order):
            axs[0].text(xi + (i - 1.5) * 0.2, D[t][k] + 0.5, str(D[t][k]), ha="center", fontsize=6)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels([SHORT[t] for t in order], fontsize=8, rotation=20, ha="right")
    axs[0].legend(fontsize=7)
    axs[0].set_title("training-side coverage per task (sorted by M7 test RSR, high -> low)")
    rs = [st["task"]["M7 base (best)"][t]["mean"] for t in order]
    for t, r in zip(order, rs):
        axs[1].scatter(D[t]["sft_traj_208"], r, s=30 + 8 * D[t]["test_scenes"], color="#2e8b57")
        axs[1].annotate(SHORT[t], (D[t]["sft_traj_208"], r), fontsize=7, xytext=(4, 3), textcoords="offset points")
    axs[1].set_xlabel("recovery trajectories of the task in the 208 SFT subset")
    axs[1].set_ylabel("M7 base test RSR (3 seeds)")
    axs[1].set_title("data amount vs test RSR (size = # test scenes)")
    fig.suptitle("Fig 7. Data scarcity vs behaviour gap")
    save_fig(fig, "fig7_data_vs_rsr.png")
    plt.close(fig)


def tables_md(A: dict) -> str:
    st, order = A["st"], A["order"]
    L = ["| task | test scenes | " + " | ".join(list(MODELS) + list(EXTRA)) + " |",
         "|---|---|" + "---|" * (len(MODELS) + len(EXTRA))]
    for t in order:
        L.append(f"| {SHORT[t]} | {len(A['by_task'][t]) // 10} | " + " | ".join(
            f"{st['task'][m][t]['mean']:.3f}" for m in list(MODELS) + list(EXTRA)) + " |")
    L.append("| **all** | 56 | " + " | ".join(f"**{st['overall'][m]['mean']:.3f}**" for m in list(MODELS) + list(EXTRA)) + " |")
    L += ["", "| task | train scenes | train failure states (x10) | searched (M5b) | solved | pool traj | SFT 208 | "
          "M7 test RSR |", "|---|---|---|---|---|---|---|---|"]
    for t in order:
        d = st["data"][t]
        L.append(f"| {SHORT[t]} | {d.get('train_scenes', 0)} | {d.get('train_failure_states', 0)} | "
                 f"{d.get('searched_failures', 0)} | {d.get('solved_failures', 0)} | {d.get('pool_traj', 0)} | "
                 f"{d.get('sft_traj_208', 0)} | {st['task']['M7 base (best)'][t]['mean']:.3f} |")
    return "\n".join(L) + "\n"


def main() -> None:
    A = analysis_logs()
    analysis_more(A)
    analysis_data(A)
    st = A["st"]

    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    atomic_write(OUT / "stats.json", json.dumps(conv(st), indent=1))
    atomic_write(OUT / "tables.md", tables_md(A))
    print(tables_md(A))
    print(json.dumps(conv({k: st[k] for k in ("overall", "grip", "capture", "transition", "never_any_model",
                                               "success_d_percentiles", "data_total_sft")}), indent=1))
    print(json.dumps(conv(st["consistency"]), indent=0)[:3000])
    print(json.dumps(conv(st["end_state"]), indent=0)[:4000])
    print(json.dumps(conv(st["transition_task_m7"]), indent=0))


if __name__ == "__main__":
    main()
