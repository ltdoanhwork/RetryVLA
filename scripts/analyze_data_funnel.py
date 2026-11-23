#!/usr/bin/env python3
"""Data funnel + catastrophic-forgetting audit (offline, CPU, reads existing artefacts only).

python scripts/analyze_data_funnel.py  -> runs/_analysis/funnel/{funnel.json, tables.md, *.png}

(A) LIBERO-Recover Spatial (SmolVLA, M5b search -> M5c pi1) + Object (M10)
(B) LIBERO-Goal, 2610.05151 protocol (pi0 3B + LoRA; F0 / F1)
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
from stats_cluster import mcnemar  # noqa: E402

OUT = REPO / "runs/_analysis/funnel"
SP = REPO / "data/assets/scenes/scenes/selected_bddl_scenes_spatial"
OB = REPO / "data/assets/scenes/scenes/selected_bddl_scenes_object"
GR = REPO / "runs/goal_rsi"


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def newest(tag, n_min):
    """episodes of the newest finished run dir *_{tag} with >= n_min rows"""
    for d in sorted(glob.glob(str(REPO / "runs" / f"*_{tag}")), reverse=True):
        try:
            if json.load(open(Path(d, "summary.json"))).get("status") != "done":
                continue
            r = jl(Path(d, "episodes.jsonl"))
        except Exception:
            continue
        if len(r) >= n_min:
            return r
    return None


def levels(root, official):
    from retryvla.envs.splits import load_levels
    return load_levels(root, official_only=official)


def scene_of(inst):  # instance id -> scene tag (strip __close2 / __open5)
    return inst.rsplit("__", 1)[0]


def search_funnel(per_instance_glob, precheck_glob, bank_glob, n_select_rr, arms, lv_all, lv_off):
    rows = {}
    for f in sorted(glob.glob(per_instance_glob)):
        for r in jl(f):
            rows[(r["instance_id"], r["arm"])] = r   # later (resumed) runs overwrite
    pc = {}
    for f in sorted(glob.glob(precheck_glob)):
        for r in csv.DictReader(open(f)):
            pc[r["instance_id"]] = r["pi0_success"] == "True"
    insts = sorted({k[0] for k in rows})
    out = {"precheck_drawn": len(pc), "precheck_pi0_success": int(sum(pc.values())),
           "failure_states": len(insts), "arms": {}, "unique_scenes": len({scene_of(i) for i in insts}),
           "_scenes": sorted({scene_of(i) for i in insts})}
    out["failures_by_level_all"] = dict(Counter(lv_all.get(scene_of(i), "unk") for i in insts))
    out["failures_by_level_official"] = dict(Counter(lv_off.get(scene_of(i), "unofficial") for i in insts))
    out["failures_by_task"] = dict(Counter(i.split("__")[0] for i in insts))
    from retryvla.data.mix import expand_roots, recovery_episode_subsets
    for a in arms:
        x = [rows[(i, a)] for i in insts if (i, a) in rows]
        roots = expand_roots([bank_glob.format(arm=a)])
        bank = [t for r in roots for t in jl(Path(r) / "retryvla/trajectories.jsonl")]
        d = {"states": len(x), "candidates_scored_per_state": float(np.mean([r["n_scored"] for r in x])),
             "candidates_scored_total": int(sum(r["n_scored"] for r in x)),
             "sim_verifications": int(sum(r["budget"] for r in x)), "verified_successes": int(sum(r["hits"] for r in x)),
             "states_solved": int(sum(bool(r["solved"]) for r in x)), "bank_trajectories": len(bank),
             "sim_steps_verify": int(sum(r.get("sim_steps_verify", 0) for r in x))}
        d["hit_rate"] = d["verified_successes"] / max(1, d["sim_verifications"])
        d["solve_rate"] = d["states_solved"] / max(1, d["states"])
        d["solved_by_level_all"] = dict(Counter(lv_all.get(scene_of(r["instance_id"]), "unk") for r in x if r["solved"]))
        d["solved_by_task"] = dict(Counter(r["instance_id"].split("__")[0] for r in x if r["solved"]))
        if n_select_rr:
            subs = recovery_episode_subsets(roots, n_select_rr, 0, mode="roundrobin")
            sel = []
            for r, keep in zip(roots, subs):
                ks = set(keep)
                sel += [t for t in jl(Path(r) / "retryvla/trajectories.jsonl") if t["episode_index"] in ks]
            d["selected"] = len(sel)
            d["selected_unique_states"] = len({t["instance_id"] for t in sel})
            d["selected_frames"] = int(sum(t["length"] for t in sel))
            d["selected_bridge_frames"] = int(sum(t.get("recovery_len", 0) for t in sel))
            d["selected_continuation_frames"] = int(sum(t.get("continuation_len", 0) for t in sel))
            d["selected_by_level_all"] = dict(Counter(lv_all.get(scene_of(t["instance_id"]), "unk") for t in sel))
            d["selected_by_task"] = dict(Counter(t["instance_id"].split("__")[0] for t in sel))
        out["arms"][a] = d
    return out


def rec_eps(tag, keys=None):
    if tag == "@pi0":
        r = newest("M0_recover_bddl", 1)
    else:
        r = newest(tag, 560 if keys is None else len(keys))
    if r is None:
        return None
    e = {(x["scene"], x["gripper_init"], x["trial"]): bool(x["success"]) for x in r}
    if keys is not None:
        e = {k: v for k, v in e.items() if k in keys}
    return e


def rsr_block(eps_list, lv_all, lv_off):
    if not eps_list:
        return None
    per = [np.mean(list(e.values())) for e in eps_list]
    o = {"seeds": len(eps_list), "rsr": float(np.mean(per)), "per_seed": [round(float(p), 3) for p in per],
         "sd": float(np.std(per, ddof=1)) if len(per) > 1 else 0.0}
    for view, lv in (("all", lv_all), ("official", lv_off)):
        acc = defaultdict(list)
        for e in eps_list:
            for k, v in e.items():
                L = lv.get(k[0])
                if L:
                    acc[L].append(v)
        o[f"by_level_{view}"] = {L: {"rate": float(np.mean(v)), "n": len(v)} for L, v in sorted(acc.items())}
    return o


def nom_eps(tag):
    r = newest(tag, 500) if tag else None
    if r is None:
        return None
    return {(x["task"], x["episode"]): bool(x["success"]) for x in r}


def nom_block(eps_list):
    if not eps_list:
        return None
    tasks = sorted({k[0] for k in eps_list[0]})
    per_task = {t: float(np.mean([np.mean([v for k, v in e.items() if k[0] == t]) for e in eps_list])) for t in tasks}
    per = [np.mean(list(e.values())) for e in eps_list]
    return {"seeds": len(eps_list), "sr": float(np.mean(per)), "per_seed": [round(float(p), 3) for p in per],
            "per_task": per_task}


def paired_nominal(A, B):
    """A, B: lists of nominal episode dicts (same init states). Pool McNemar over seed pairs (A[i] vs B[i % len(B)])."""
    n01 = n10 = 0
    for i, a in enumerate(A):
        b = B[i % len(B)]
        ks = sorted(set(a) & set(b))
        x = np.array([a[k] for k in ks]); y = np.array([b[k] for k in ks])
        m = mcnemar(x, y)       # n01 = A ok & B fail ("A fixed"), n10 = B ok & A fail ("A broke")
        n01 += m[0]; n10 += m[1]
    from math import comb
    n = n01 + n10
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(n01, n10) + 1)) / 2 ** n) if n else 1.0
    return {"A_only": n01, "B_only": n10, "p": p}


def task_deltas(new, ref, has_data):
    d = {t: new["per_task"][t] - ref["per_task"][t] for t in ref["per_task"] if t in new["per_task"]}
    drop = [t for t, v in d.items() if v < -1e-9]
    return {"delta": d, "n_tasks": len(d), "n_drop": len(drop), "n_gain": sum(v > 1e-9 for v in d.values()),
            "n_drop_ge5pt": sum(v <= -0.05 for v in d.values()),
            "drops_on_tasks_without_recovery_data": [t for t in drop if not has_data.get(t)],
            "mean_delta_with_data": float(np.mean([v for t, v in d.items() if has_data.get(t)] or [np.nan])),
            "mean_delta_without_data": float(np.mean([v for t, v in d.items() if not has_data.get(t)] or [np.nan]))}


def spatial():
    from analyze_ablation import ARMS
    lv_all, lv_off = levels(SP, False), levels(SP, True)
    split = json.load(open(REPO / "data/splits/spatial_v1.json"))
    S = {"split": {p: len(split[p]) for p in ("train", "pilot", "test")}}
    for p in ("train", "test"):
        S["split"][f"{p}_by_level_all"] = dict(Counter(lv_all.get(t, "unk") for t in split[p]))
        S["split"][f"{p}_by_level_official"] = dict(Counter(lv_off.get(t, "unofficial") for t in split[p]))
    S["split"]["train_trial_slots"] = len(split["train"]) * 10
    S["split"]["train_trial_slots_unseen_by_wm"] = len(split["train"]) * 8   # trials 2..9 (WM saw 0, 1)
    S["search"] = search_funnel(str(REPO / "runs/*_M5_m5b_compare_arms_s*of8/per_instance.jsonl"),
                                str(REPO / "runs/*_M5_m5b_compare_arms_s*of8/precheck.csv"),
                                str(REPO / "data/recovery/m5b/s*/p*/{arm}"), 208,
                                ["random", "wm_diverse", "wm_noise", "wm_qd"], lv_all, lv_off)
    sc = S["search"].pop("_scenes")
    S["search"]["failure_scenes_in_train_split"] = f"{sum(s in set(split['train']) for s in sc)}/{len(sc)}"
    sm = json.load(open(REPO / "runs/20261016-042447_M5_m5c_sft_wm_diverse_rr/summary.json"))["results"]
    S["sft_mix"] = {"nominal_frames": sm["n_original_frames"], "recovery_frames": sm["n_recovery_frames"],
                    "recovery_episodes": sm["n_recovery_episodes"], "recovery_sampling_ratio": 0.5,
                    "recovery_frame_share": sm["n_recovery_frames"] / (sm["n_original_frames"] + sm["n_recovery_frames"]),
                    "steps": 6000, "batch": 32, "samples_seen": 6000 * 32}
    keys = set(rec_eps("M5_m5c_sft_wm_diverse_rr_recover_test"))
    policies = {
        "pi0": (["@pi0"], ["M0_baseline_spatial_nominal"]),
        "pi1-nominal": (["M1_pi1nom_recover_test"], ["M1_pi1nom_nominal"]),
    }
    for arm, cells in ARMS.items():
        policies[arm] = ([c[0] for c in cells], [c[1] for c in cells])
    R, N, NE = {}, {}, {}
    for k, (rt, nt) in policies.items():
        R[k] = rsr_block([e for e in (rec_eps(t, keys) for t in rt) if e], lv_all, lv_off)
        NE[k] = [e for e in (nom_eps(t) for t in nt) if e]
        N[k] = nom_block(NE[k])
    S["rsr"], S["nominal"] = R, N
    sel_tasks = S["search"]["arms"]["wm_diverse"]["selected_by_task"]
    has = {t: sel_tasks.get(t, 0) > 0 for t in N["pi0"]["per_task"]}
    S["recovery_traj_per_nominal_task"] = {t: sel_tasks.get(t, 0) for t in N["pi0"]["per_task"]}
    pi1 = "2 action-grounded (pi1)"
    S["forgetting"] = {}
    for new in N:
        if new in ("pi1-nominal",):
            continue
        for ref in ("pi1-nominal", "pi0"):
            if new == ref or N[new] is None:
                continue
            S["forgetting"][f"{new} vs {ref}"] = {"dNSR": N[new]["sr"] - N[ref]["sr"],
                                                  **task_deltas(N[new], N[ref], has),
                                                  "mcnemar": paired_nominal(NE[new], NE[ref])}
    S["forgetting"]["pi1 vs demo-only (same init/steps)"] = {
        "dNSR": N[pi1]["sr"] - N["1 demo-only"]["sr"], **task_deltas(N[pi1], N["1 demo-only"], has),
        "mcnemar": paired_nominal(NE[pi1], NE["1 demo-only"])}
    return S


def objct():
    lv_all, lv_off = levels(OB, False), levels(OB, True)
    split = json.load(open(REPO / "data/splits/object_v1.json"))
    O = {"split": {p: len(split[p]) for p in ("train", "pilot", "test")}}
    O["split"]["train_by_level_all"] = dict(Counter(lv_all.get(t, "unk") for t in split["train"]))
    O["search"] = search_funnel(str(REPO / "runs/*_M10_obj_search_s*of4/per_instance.jsonl"),
                                str(REPO / "runs/*_M10_obj_search_s*of4/precheck.csv"),
                                str(REPO / "data/recovery/obj/s*/p*/{arm}"), 123,
                                ["random", "wm_diverse", "wm_goal"], lv_all, lv_off)
    sm = json.load(open(REPO / "runs/20261021-063617_M10_sft_comp_obj/summary.json"))["results"]
    O["sft_mix"] = {"nominal_frames": sm["n_original_frames"], "recovery_frames": sm["n_recovery_frames"],
                    "recovery_episodes": sm["n_recovery_episodes"],
                    "recovery_frame_share": sm["n_recovery_frames"] / (sm["n_original_frames"] + sm["n_recovery_frames"])}
    pol = {k: (f"M10_obj_{k}_recover_test", f"M10_obj_{k}_nominal")
           for k in ("pi0", "pi1nom", "demo_only", "random", "comp", "goal")}
    R, N, NE = {}, {}, {}
    for k, (rt, nt) in pol.items():
        r = newest(rt, 260)
        e = {(x["scene"], x["gripper_init"], x["trial"]): bool(x["success"]) for x in r} if r else None
        R[k] = rsr_block([e] if e else [], lv_all, lv_off)
        NE[k] = [e for e in [nom_eps(nt)] if e]
        N[k] = nom_block(NE[k])
    O["rsr"], O["nominal"] = R, N
    sel = O["search"]["arms"]["wm_diverse"]["selected_by_task"]
    has = {t: sel.get(t, 0) > 0 for t in N["pi0"]["per_task"]}
    O["recovery_traj_per_nominal_task"] = {t: sel.get(t, 0) for t in N["pi0"]["per_task"]}
    O["forgetting"] = {}
    for new in ("comp", "random", "goal", "demo_only"):
        for ref in ("pi1nom", "pi0"):
            O["forgetting"][f"{new} vs {ref}"] = {"dNSR": N[new]["sr"] - N[ref]["sr"], **task_deltas(N[new], N[ref], has),
                                                  "mcnemar": paired_nominal(NE[new], NE[ref])}
    return O


def goal_select(roots, max_traj, seed):
    """replicates retryvla/train/pi0_sft.Pool round-robin trajectory selection"""
    per = {}
    for k, r in enumerate(roots, 1):
        for t in jl(Path(r) / "retryvla/trajectories.jsonl"):
            per.setdefault(t["instance_id"], []).append(t)
    insts = sorted(per)
    np.random.default_rng(seed).shuffle(insts)
    sel, rnd = [], 0
    while len(sel) < max_traj and any(len(v) > rnd for v in per.values()):
        for i in insts:
            if len(per[i]) > rnd and len(sel) < max_traj:
                sel.append(per[i][rnd])
        rnd += 1
    return sel


def goal_arm(prefix):
    out = {}
    for s in (7, 8, 9):
        f = GR / f"{prefix}_s{s}/ckpt_02000/eval_val/episodes.jsonl"
        if f.exists():
            out[s] = {r["scene"]: bool(r["success"]) for r in jl(f)}
    return out


def goal():
    from retryvla.data.mix import expand_roots
    G = {"audit": {}}
    for s in (7, 8):
        a = json.load(open(REPO / f"data/goal_rsi/audit_m0_s{s}/summary.json"))["seeds"][str(s)]
        G["audit"][s] = {k: a[k] for k in ("n", "successes", "failures", "entries", "by_task_fail")}
    info = json.load(open(REPO / "data/goal_rsi/d0_train/meta/info.json"))
    G["d0"] = {"episodes": info["total_episodes"], "frames": info["total_frames"]}
    G["search"] = search_funnel(str(REPO / "runs/*_GOAL_goal_search_s*of3/per_instance.jsonl"), "/nonexistent",
                                str(REPO / "data/recovery/goal_search/s*/p*/{arm}"), 0, ["random", "wm_diverse"], {}, {})
    G["search"].pop("_scenes")
    rows = {}
    for f in sorted(glob.glob(str(REPO / "runs/*_GOAL_goal_search_s*of3/per_instance.jsonl"))):
        for r in jl(f):
            rows[(r["instance_id"], r["arm"])] = r
    for a in ("random", "wm_diverse"):
        cut = defaultdict(lambda: [0, 0, 0])
        for (i, aa), r in rows.items():
            if aa == a:
                c = i.rsplit("__", 1)[1]
                cut[c][0] += 1; cut[c][1] += int(r["solved"]); cut[c][2] += int(r["hits"])
        G["search"]["arms"][a]["by_cut"] = {c: {"entries": v[0], "solved": v[1], "hits": v[2]} for c, v in sorted(cut.items())}
    G["wm_critic_training"] = {"entries_seed8": 168, "candidates_per_entry": 32, "sim_rollouts": 168 * 32}
    # distilled mixes (actual anchors from checkpoint meta; selection replicated for unique-entry coverage)
    mixes = {}
    for prefix, arm, segs in (("rec_wm", "wm_diverse", None), ("rec_rand", "random", None),
                              ("f1_wm", "wm_diverse", [0]), ("f1_rand", "random", [0])):
        for s in (7, 8, 9):
            m = GR / f"{prefix}_s{s}/ckpt_02000/meta.json"
            if not m.exists():
                continue
            meta = json.loads(m.read_text())
            aps = meta["anchors_per_source"]
            roots = expand_roots([str(REPO / f"data/recovery/goal_search/s*/p*/{arm}")])
            sel = goal_select(roots, meta["args"]["max_traj"], meta["args"]["seed"])
            mixes[f"{prefix}_s{s}"] = {"max_traj": meta["args"]["max_traj"], "segments": segs,
                                       "d0_anchors": aps[0], "recovery_anchors": sum(aps[1:]),
                                       "recovery_share": sum(aps[1:]) / sum(aps),
                                       "selected_unique_entries": len({t["instance_id"] for t in sel}),
                                       "selected_by_task": dict(Counter(t["instance_id"].split("__")[0] for t in sel)),
                                       "selected_by_cut": dict(Counter(t["instance_id"].rsplit("__", 1)[1] for t in sel))}
    G["mixes"] = mixes
    # results
    m0 = {r["scene"]: bool(r["success"]) for r in jl(GR / "m0_s7/ckpt_04000/eval_val/episodes.jsonl")}
    A = {"Original": goal_arm("orig"), "F0_wm": goal_arm("rec_wm"), "F0_random": goal_arm("rec_rand"),
         "F1_wm": goal_arm("f1_wm"), "F1_random": goal_arm("f1_rand")}
    task = lambda sc: sc.split("__")[0]
    tasks = sorted({task(k) for k in m0})
    pt = lambda e: {t: sum(v for k, v in e.items() if task(k) == t) for t in tasks}
    res = {"M0": {"successes": int(sum(m0.values())), "per_task": pt(m0)}}
    for k, x in A.items():
        if not x:
            continue
        res[k] = {"per_seed": {s: int(sum(e.values())) for s, e in x.items()},
                  "mean": float(np.mean([sum(e.values()) for e in x.values()])),
                  "per_task": {t: float(np.mean([pt(e)[t] for e in x.values()])) for t in tasks}}
    G["results"] = res
    has = {t: mixes.get("rec_wm_s7", {}).get("selected_by_task", {}).get(t, 0) > 0 for t in tasks}
    G["tasks_with_recovery_data_F0_wm"] = has
    solved_by_m0 = [t for t in tasks if res["M0"]["per_task"][t] >= 9]
    G["tasks_solved_by_M0"] = solved_by_m0
    from scipy.stats import binomtest
    fg = {}
    for k in ("F0_wm", "F0_random", "F1_wm", "F1_random"):
        if k not in res:
            continue
        for ref in ("Original", "M0"):
            n01 = n10 = 0
            for s, e in A[k].items():
                b = m0 if ref == "M0" else A["Original"].get(s)
                if b is None:
                    continue
                n01 += sum(1 for sc in e if e[sc] and not b[sc]); n10 += sum(1 for sc in e if b[sc] and not e[sc])
            p = float(binomtest(min(n01, n10), n01 + n10).pvalue) if n01 + n10 else 1.0
            refpt = res[ref]["per_task"]
            seeds = sorted(A[k]) if ref == "M0" else sorted(set(A[k]) & set(A["Original"]))
            # per-task delta on the SAME seeds as the reference
            kpt = {t: float(np.mean([pt(A[k][s])[t] for s in seeds])) for t in tasks}
            rpt = refpt if ref == "M0" else {t: float(np.mean([pt(A["Original"][s])[t] for s in seeds])) for t in tasks}
            d = {t: kpt[t] - rpt[t] for t in tasks}
            fg[f"{k} vs {ref}"] = {
                "seeds": seeds, "delta_successes": float(np.mean([sum(A[k][s].values()) for s in seeds]) -
                                                         (res["M0"]["successes"] if ref == "M0" else
                                                          np.mean([sum(A["Original"][s].values()) for s in seeds]))),
                "fixed": n01, "broken": n10, "mcnemar_p": p, "per_task_delta": d,
                "n_drop": sum(v < 0 for v in d.values()), "n_gain": sum(v > 0 for v in d.values()),
                "delta_on_M0_solved_tasks": float(sum(d[t] for t in solved_by_m0)),
                "n_drop_on_M0_solved_tasks": sum(d[t] < 0 for t in solved_by_m0),
                "drops_on_tasks_without_data": [t for t in tasks if d[t] < 0 and not has[t]],
                "delta_tasks_without_data": float(sum(d[t] for t in tasks if not has[t])),
                "delta_tasks_with_data": float(sum(d[t] for t in tasks if has[t]))}
    G["forgetting"] = fg
    G["paper"] = {"M0": 77, "Original": 78, "Recovery": 88, "recovery_anchor_share": 0.041}
    G["f1_status"] = {f"f1_{a}_s{s}": (GR / f"f1_{a}_s{s}/ckpt_02000/eval_val/summary.json").exists()
                      for a in ("wm", "rand") for s in (7, 8, 9)}
    return G


def plots(S, G):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = S["search"]["arms"]["wm_diverse"]
    st = [("train trial slots\n(unseen by WM)", S["split"]["train_trial_slots_unseen_by_wm"]),
          ("drawn for\npi0 pre-check", S["search"]["precheck_drawn"]), ("pi0 failure\nstates", S["search"]["failure_states"]),
          ("WM-imagined +\ncritic-scored", w["candidates_scored_total"]), ("sim\nverifications", w["sim_verifications"]),
          ("verified\nsuccesses", w["verified_successes"]), ("bank traj\n(<=4/state)", w["bank_trajectories"]),
          ("states with\n>=1 recovery", w["states_solved"]), ("distilled traj\n(round-robin)", w["selected"]),
          ("unique states\nin SFT mix", w["selected_unique_states"])]
    gw = G["search"]["arms"]["wm_diverse"]
    mx = G["mixes"]
    gt = [("audit rollouts\n(seed 7)", G["audit"][7]["n"]), ("failed\nrollouts", G["audit"][7]["failures"]),
          ("entries\n(3 cuts)", G["audit"][7]["entries"]), ("WM-imagined +\ncritic-scored", gw["candidates_scored_total"]),
          ("sim\nverifications", gw["sim_verifications"]), ("verified\nsuccesses", gw["verified_successes"]),
          ("entries with\n>=1 recovery", gw["states_solved"]), ("bank traj", gw["bank_trajectories"]),
          ("F0 distilled\n(matched)", mx["rec_wm_s7"]["max_traj"]), ("F1 distilled\n(bridge-only)", mx["f1_wm_s7"]["max_traj"])]
    for name, stages, title in (("funnel_spatial", st, "Spatial (SmolVLA, wm_diverse arm, M5b -> pi1)"),
                                ("funnel_goal", gt, "LIBERO-Goal (pi0 3B, wm_diverse arm, 2610.05151 protocol)")):
        fig, ax = plt.subplots(figsize=(12, 4.2))
        v = [x[1] for x in stages]
        ax.bar(range(len(v)), v, color=["#9aa5b1"] * 3 + ["#3b6fd8"] * 3 + ["#2f9e6b"] * (len(v) - 6))
        ax.set_yscale("log")
        for i, x in enumerate(v):
            ax.text(i, x * 1.1, f"{x:,}", ha="center", fontsize=9)
        ax.set_xticks(range(len(v)), [x[0] for x in stages], fontsize=8)
        ax.set_title(title); ax.set_ylabel("count (log)")
        fig.tight_layout(); fig.savefig(OUT / f"{name}.png", dpi=130); plt.close(fig)
    # per-task forgetting
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    f = S["forgetting"]["2 action-grounded (pi1) vs pi1-nominal"]
    ts = list(f["delta"]); has = S["recovery_traj_per_nominal_task"]
    short = lambda t: t.replace("pick_up_the_black_bowl_", "").replace("_and_place_it_on_the_plate", "")[:28]
    axs[0].bar(range(len(ts)), [100 * f["delta"][t] for t in ts], color=["#3b6fd8" if has[t] else "#d64545" for t in ts])
    f0 = S["forgetting"]["2 action-grounded (pi1) vs pi0"]
    axs[0].scatter(range(len(ts)), [100 * f0["delta"][t] for t in ts], color="k", marker="x", label="vs pi0")
    axs[0].axhline(0, color="k", lw=.5); axs[0].legend()
    axs[0].set_xticks(range(len(ts)), [f"{short(t)} ({has[t]})" for t in ts], rotation=60, ha="right", fontsize=7)
    axs[0].set_title("Spatial nominal: pi1 (3 seeds) - pi1-nominal, pt per task\n(red = 0 recovery traj in mix; x = vs pi0)")
    g = G["forgetting"]
    tk = list(g["F0_wm vs Original"]["per_task_delta"]); hg = G["tasks_with_recovery_data_F0_wm"]
    wdt = 0.4
    axs[1].bar(np.arange(len(tk)) - wdt / 2, [10 * g["F0_wm vs Original"]["per_task_delta"][t] for t in tk], wdt,
               color=["#3b6fd8" if hg[t] else "#d64545" for t in tk], label="F0 wm (47 % share)")
    if "F1_wm vs Original" in g:
        axs[1].bar(np.arange(len(tk)) + wdt / 2, [10 * g["F1_wm vs Original"]["per_task_delta"][t] for t in tk], wdt,
                   color="#f0a030", label=f"F1 wm (4 % share, seeds {g['F1_wm vs Original']['seeds']})")
    axs[1].axhline(0, color="k", lw=.5); axs[1].legend(fontsize=8)
    axs[1].set_xticks(range(len(tk)), [t[:30] for t in tk], rotation=60, ha="right", fontsize=7)
    axs[1].set_title("Goal val: arm - Original, pt per task (mean of seeds)\n(red = no recovery data for the task in F0)")
    fig.tight_layout(); fig.savefig(OUT / "forgetting_per_task.png", dpi=130); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S, O, G = spatial(), objct(), goal()
    O["search"].pop("_scenes", None)
    res = {"spatial": S, "object": O, "goal": G}
    (OUT / "funnel.json").write_text(json.dumps(res, indent=1, default=str))
    plots(S, G)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
