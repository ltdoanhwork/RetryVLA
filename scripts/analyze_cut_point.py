"""Cut-point study (docs/research/CUT_POINT_STUDY.md), offline, CPU only, no sim.

LIBERO-Goal (2610.05151 protocol): break the existing search / bank / SFT artefacts down by the cut fraction of the
audit entry (t in {T/4, T/2, 3T/4} = {75, 150, 225} of failed M0 rollouts with T = 300).
LIBERO-Recover (official LIBERO-Recovery v1.0.0 annotations): VLM stage / level by snapshot step.

usage: python scripts/analyze_cut_point.py [--out runs/_analysis/cut_point]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
FR = {75: "T/4", 150: "T/2", 225: "3T/4"}
ARMS = ("random", "wm_diverse")


def frac_of(iid: str) -> int:
    return int(iid.rsplit("__t", 1)[1])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load_search() -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(REPO / "runs/*GOAL_goal_search_s*of3/per_instance.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    return rows


def load_verif() -> dict:
    """data/wm/goal_verif/<entry>.pt: 32 verified candidates (16 random + 16 wm_diverse) per seed-7 entry."""
    import torch

    out = {}
    for f in sorted(glob.glob(str(REPO / "data/wm/goal_verif/*.pt"))):
        d = torch.load(f, weights_only=False, map_location="cpu")
        out[d["instance_id"]] = {"success": d["success"].numpy().astype(bool), "source": list(d["source"]),
                                 "prefix_success": d["prefix_success"].numpy().astype(bool),
                                 "success_step": d["success_step"].numpy(), "actions": d["actions"].numpy()}
    return out


def load_collect() -> dict:
    """data/wm/goal_collect/<entry>.pt: seed-8 entries, 32 candidates by proposal source (WM / critic training)."""
    import torch

    out = {}
    for f in sorted(glob.glob(str(REPO / "data/wm/goal_collect/*.pt"))):
        d = torch.load(f, weights_only=False, map_location="cpu")
        out[d["instance_id"]] = {"success": d["success"].numpy().astype(bool), "source": list(d["source"]),
                                 "prefix_success": d["prefix_success"].numpy().astype(bool)}
    return out


def bank_episodes(arm: str) -> list[dict]:
    """Per bank trajectory: entry, cut, seg0 / seg1 lengths, arm jerk (|Delta^3 a| of the bridge)."""
    eps = []
    for d in sorted(glob.glob(str(REPO / f"data/recovery/goal_search/s*/p0/{arm}"))):
        meta = [json.loads(l) for l in open(Path(d) / "retryvla/trajectories.jsonl")]
        by_ep = {m["episode_index"]: m for m in meta}
        cols = ["episode_index", "frame_index", "action", "segment"]
        acts = collections.defaultdict(list)
        for f in sorted((Path(d) / "data").rglob("*.parquet")):
            t = pq.read_table(f, columns=cols).to_pandas()
            for e, g in t.groupby("episode_index"):
                g = g.sort_values("frame_index")
                acts[int(e)].append((np.stack(g.action.values),
                                     np.array([np.ravel(v)[0] for v in g.segment.values]).astype(int)))
        for e, m in by_ep.items():
            a, s = acts[e][0]
            a0 = a[s == 0]
            d3 = float(np.linalg.norm(np.diff(a0[:, :6], 3, axis=0), axis=1).mean()) if len(a0) >= 5 else float("nan")
            eps.append({"arm": arm, "instance_id": m["instance_id"], "t": frac_of(m["instance_id"]),
                        "task": m["task"], "seg0": int((s == 0).sum()), "seg1": int((s == 1).sum()),
                        "len": int(len(s)), "bridge_d3": d3})
    return eps


def matched_subset(eps: list[dict], max_traj: int, seed: int) -> list[dict]:
    """Same round-robin selection as retryvla/train/pi0_sft.py Pool(max_traj, seed)."""
    per = collections.defaultdict(list)
    for e in eps:
        per[e["instance_id"]].append(e)
    insts = sorted(per)
    np.random.default_rng(seed).shuffle(insts)
    sel, rnd = [], 0
    while len(sel) < max_traj and any(len(v) > rnd for v in per.values()):
        for i in insts:
            if len(per[i]) > rnd and len(sel) < max_traj:
                sel.append(per[i][rnd])
        rnd += 1
    return sel


def entry_state_nominal_dist(verif_q: dict) -> dict:
    """Distance (mm) from the entry EEF position to the closest D0 (nominal demo) state of the same task."""
    import pandas as pd

    tasks = pd.read_parquet(REPO / "data/goal_rsi/d0_train/meta/tasks.parquet")
    lang2idx = {lang: int(r.task_index) for lang, r in tasks.iterrows()}
    st = collections.defaultdict(list)
    for f in sorted((REPO / "data/goal_rsi/d0_train/data").rglob("*.parquet")):
        t = pq.read_table(f, columns=["observation.state", "task_index"]).to_pandas()
        for ti, g in t.groupby("task_index"):
            st[int(ti)].append(np.stack(g["observation.state"].values))
    st = {k: np.concatenate(v) for k, v in st.items()}
    out = {}
    for iid, (task, q0) in verif_q.items():
        ref = st[lang2idx[task]]
        dpos = np.linalg.norm(ref[:, :3] - q0[None, :3], axis=1) * 1000.0
        dfull = np.linalg.norm(ref - q0[None], axis=1)
        out[iid] = {"nn_pos_mm": float(dpos.min()), "nn_state": float(dfull.min()),
                    "grip": float(q0[6] - q0[7])}
    return out


def per_task_eval() -> dict:
    """Mean val successes per task over seeds 7-9 for Original and the F0 banks (wm / rand)."""
    res = {}
    for arm in ("orig", "rec_wm", "rec_rand"):
        acc = collections.defaultdict(list)
        for s in (7, 8, 9):
            f = glob.glob(str(REPO / f"runs/goal_rsi/{arm}_s{s}/ckpt_02000/eval_val/episodes.jsonl"))
            if not f:
                continue
            for l in open(f[0]):
                r = json.loads(l)
                acc[r["task"]].append(float(r["success"]))
        res[arm] = {k: 10.0 * float(np.mean(v)) for k, v in acc.items()}
    return res


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def liberorecover_annotations() -> dict:
    A = [json.load(open(f)) for f in glob.glob(str(REPO / "data/assets/levels/annotations/*/*.json"))]
    out = {}
    for suite in ("libero_spatial", "libero_object"):
        nat = [d for d in A if d["source"] == suite and "injection" not in d]
        inj = [d for d in A if d["source"] == suite and "injection" in d]
        steps = sorted({d["step"] for d in nat})
        rows = {}
        for s in steps:
            S = [d for d in nat if d["step"] == s]
            rows[s] = {"n": len(S), "stage": dict(collections.Counter(d["stage"] for d in S)),
                       "level": dict(collections.Counter(d["level"] for d in S))}
        eps = collections.defaultdict(list)
        for d in nat:
            eps[(d["task"], d["episode_idx_in_task"])].append(d)
        lvl_change = sum(len({x["level"] for x in v}) > 1 for v in eps.values())
        fc = [d for d in nat if d.get("first_contact_step") is not None and d["fail_steps"]]
        out[suite] = {"models": dict(collections.Counter(d["model"] for d in nat)),
                      "episode_length": dict(collections.Counter(d["episode_length"] for d in nat)),
                      "by_step": rows, "episodes": len(eps), "level_changes_within_episode": lvl_change,
                      "fail_step_median": float(np.median([d["fail_steps"][0] for d in nat if d["fail_steps"]])),
                      "fail_step_before_first_contact": [sum(d["fail_steps"][0] < d["first_contact_step"] for d in fc),
                                                         len(fc)],
                      "first_contact_median": float(np.median([d["first_contact_step"] for d in fc])),
                      "l4_injected": {"n": len(inj),
                                      "strategy": dict(collections.Counter(d["injection"]["perturbed"]["strategy"]
                                                                           for d in inj)),
                                      "param_m": dict(collections.Counter(d["injection"]["perturbed"]["param_m"]
                                                                          for d in inj)),
                                      "settle_steps": sorted({d["injection"]["settle_steps"] for d in inj}),
                                      "steps": dict(collections.Counter(d["step"] for d in inj))}}
    return out


def rsr_by_step() -> dict:
    """RSR of existing LIBERO-Recover test runs split by the snapshot step of the scene (step_0180 vs step_0200 ...)."""
    runs = {"pi0 (Spatial test)": (["M0_recover_bddl"], "test"),
            "pi1-nominal (Spatial)": (["M1_pi1nom_recover_test"], None),
            "pi1 RetryVLA (Spatial, 3 seeds)": (["M5_m5c_sft_wm_diverse_rr_recover_test", "M8_seed_pi1_s1_recover_test",
                                                 "M8_seed_pi1_s2_recover_test"], None),
            "[Object] pi0": (["M10_obj_pi0_recover_test"], None),
            "[Object] competence SFT (pi1)": (["M10_obj_comp_recover_test"], None)}
    out = {}
    for name, (tags, part) in runs.items():
        acc = collections.defaultdict(list)
        for tag in tags:
            f = sorted(glob.glob(str(REPO / "runs" / f"*_{tag}")))
            if not f or not Path(f[-1], "episodes.jsonl").exists():
                continue
            for l in open(Path(f[-1], "episodes.jsonl")):
                r = json.loads(l)
                if part and r.get("part") != part:
                    continue
                acc[int(r["scene"].rsplit("step_", 1)[1])].append(float(r["success"]))
        out[name] = {s: {"n": len(v), "rsr": float(np.mean(v))} for s, v in sorted(acc.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/_analysis/cut_point")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    R: dict = {}

    # ---------- audit entries
    ents = [json.loads(l) for l in open(REPO / "data/goal_rsi/audit_m0_s7/entries_s7.jsonl")]
    R["entries"] = {"n": len(ents), "T": dict(collections.Counter(e["T"] for e in ents)),
                    "t": dict(collections.Counter(e["t"] for e in ents))}

    # ---------- search (seed-7 entries, 16 verifications / entry / arm)
    S = load_search()
    R["search"] = {}
    for arm in ARMS:
        R["search"][arm] = {}
        for t in FR:
            X = [r for r in S if r["arm"] == arm and frac_of(r["instance_id"]) == t]
            k = sum(r["solved"] for r in X)
            hits = sum(r["hits"] for r in X)
            sim = sum(r["sim_steps_verify"] for r in X)
            R["search"][arm][FR[t]] = {
                "n": len(X), "solved": k / len(X), "solved_ci": wilson(k, len(X)),
                "hits_per_verif": hits / (16 * len(X)), "hits_per_1k_sim": 1000 * hits / sim,
                "sim_per_verif": sim / (16 * len(X)),
                "jerk_all": float(np.nanmean([r["all_jerk"] for r in X])),
                "jerk_succ": float(np.nanmean([r["succ_jerk"] for r in X if r["succ_jerk"] is not None])),
                "manifold_all": float(np.nanmean([r["all_manifold"] for r in X])),
                "manifold_succ": float(np.nanmean([r["succ_manifold"] for r in X if r["succ_manifold"] is not None]))}
    # paired pattern over the 3 cuts of the same failed rollout (either arm solved = recoverable by the pool)
    pat = collections.Counter()
    by_ep = collections.defaultdict(dict)
    for r in S:
        ep = r["instance_id"].rsplit("__t", 1)[0]
        by_ep[ep][frac_of(r["instance_id"])] = by_ep[ep].get(frac_of(r["instance_id"]), False) or r["solved"]
    for ep, d in by_ep.items():
        pat["".join("1" if d.get(t) else "0" for t in FR)] += 1
    R["search"]["pattern_union_T4_T2_3T4"] = dict(sorted(pat.items()))
    mono = sum(n for p, n in pat.items() if list(p) == sorted(p, reverse=True))
    R["search"]["monotone_nonincreasing_share"] = mono / len(by_ep)

    # ---------- verified candidates: success inside the bridge (cut already near done) and continuation length
    V = load_verif()
    R["verif"] = {}
    for t in FR:
        X = [v for k, v in V.items() if frac_of(k) == t]
        succ = np.concatenate([v["success"] for v in X])
        pre = np.concatenate([v["prefix_success"] for v in X])
        sst = np.concatenate([v["success_step"][v["success"]] for v in X])
        R["verif"][FR[t]] = {"n_cand": int(len(succ)), "succ_rate": float(succ.mean()),
                             "succ_within_bridge_share": float(pre[succ].mean()) if succ.any() else None,
                             "success_step_median": float(np.median(sst)) if len(sst) else None}

    # ---------- seed-8 collect: M0 self-branch proxy (vla_s1.0 = M0's own chunk, then M0 continuation)
    C = load_collect()
    R["collect_s8"] = {}
    for t in FR:
        X = [v for k, v in C.items() if frac_of(k) == t]
        fam = collections.defaultdict(list)
        anyk = 0
        for v in X:
            anyk += bool(v["success"].any())
            for s, ok in zip(v["source"], v["success"]):
                fam["prim" if s.startswith("prim") else s].append(bool(ok))
        self_any = sum(bool(np.array([ok for s, ok in zip(v["source"], v["success"]) if s == "vla_s1.0"]).any())
                       for v in X)
        R["collect_s8"][FR[t]] = {"n_entries": len(X), "any_of_32": anyk / len(X),
                                  "m0_self_branch_any_of_6": self_any / len(X),
                                  "by_source": {f: float(np.mean(v)) for f, v in sorted(fam.items())}}

    # ---------- entry state vs nominal manifold (seed-7 entries; q0 = entry proprio of the WM verification data)
    import torch

    q0 = {}
    for f in sorted(glob.glob(str(REPO / "data/wm/goal_verif/*.pt"))):
        d = torch.load(f, weights_only=False, map_location="cpu")
        q0[d["instance_id"]] = (d["task"], d["q"][0, 0].numpy().astype(np.float64))
    ND = entry_state_nominal_dist(q0)
    solved_any = {}
    for r in S:
        solved_any[r["instance_id"]] = solved_any.get(r["instance_id"], False) or r["solved"]
    R["entry_state"] = {}
    for t in FR:
        ks = [k for k in ND if frac_of(k) == t]
        R["entry_state"][FR[t]] = {"nn_pos_mm_median": float(np.median([ND[k]["nn_pos_mm"] for k in ks])),
                                   "nn_pos_mm_q75": float(np.quantile([ND[k]["nn_pos_mm"] for k in ks], 0.75)),
                                   "grip_open_mm_median": float(1000 * np.median([ND[k]["grip"] for k in ks]))}
    xs = [ND[k]["nn_pos_mm"] for k in ND]
    ys = [float(solved_any[k]) for k in ND]
    R["entry_state"]["spearman_nn_pos_vs_solved"] = spearman(xs, ys)

    # ---------- banks, SFT mixes (F0 matched 282 all segments; F1 36 bridge-only)
    R["bank"], R["mix"] = {}, {}
    D0_ANCH = 38240
    for arm in ARMS:
        E = bank_episodes(arm)
        R["bank"][arm] = {}
        for t in FR:
            X = [e for e in E if e["t"] == t]
            R["bank"][arm][FR[t]] = {"trajs": len(X), "share_trajs": len(X) / len(E),
                                     "frames": int(sum(e["len"] for e in X)),
                                     "share_frames": sum(e["len"] for e in X) / sum(e["len"] for e in E),
                                     "seg0_mean": float(np.mean([e["seg0"] for e in X])),
                                     "seg1_mean": float(np.mean([e["seg1"] for e in X])),
                                     "bridge_only_success_share": float(np.mean([e["seg1"] == 0 for e in X])),
                                     "bridge_d3": float(np.nanmean([e["bridge_d3"] for e in X]))}
        mix = {}
        for name, mt, segs in (("F0_matched282_all", 282, (0, 1)), ("F1_36_bridge_only", 36, (0,))):
            per = {}
            for seed in (7, 8, 9):
                sel = matched_subset(E, mt, seed)
                an = collections.Counter()
                for e in sel:
                    an[e["t"]] += (e["seg0"] if 0 in segs else 0) + (e["seg1"] if 1 in segs else 0)
                tot = sum(an.values())
                per[seed] = {FR[t]: {"anchors": an[t], "share_of_recovery": an[t] / tot,
                                     "share_of_mix": an[t] / (tot + D0_ANCH)} for t in FR}
            mix[name] = {FR[t]: {k: float(np.mean([per[s][FR[t]][k] for s in per])) for k in
                                 ("anchors", "share_of_recovery", "share_of_mix")} for t in FR}
        R["mix"][arm] = mix
        # per task: bank anchors by cut (F0 matched, mean of 3 seeds) for the per-task effect
        if arm == "wm_diverse":
            pt = collections.defaultdict(lambda: collections.Counter())
            for seed in (7, 8, 9):
                for e in matched_subset(E, 282, seed):
                    pt[e["task"]][e["t"]] += e["len"] / 3
            R["per_task_anchors_wm"] = {k: {FR[t]: float(v[t]) for t in FR} for k, v in pt.items()}

    # ---------- per-task SFT effect vs cut composition
    PE = per_task_eval()
    rows = []
    anch = {t.replace(" ", "_"): v for t, v in R["per_task_anchors_wm"].items()}
    for key in sorted(PE.get("orig", {})):
        v = anch.get(key, {f: 0.0 for f in FR.values()})
        tot = sum(v.values())
        if key in PE.get("rec_wm", {}):
            rows.append({"task": key, "orig": PE["orig"][key], "wm": PE["rec_wm"][key],
                         "rand": PE.get("rec_rand", {}).get(key),
                         "delta": PE["rec_wm"][key] - PE["orig"][key], "anchors": tot,
                         "share_T4": v["T/4"] / tot if tot else None, "share_3T4": v["3T/4"] / tot if tot else None})
    fails = json.load(open(REPO / "data/goal_rsi/audit_m0_s7/summary.json"))["seeds"]["7"]["by_task_fail"]
    for r in rows:
        r["audit_fail"] = fails.get(r["task"], 0)
    R["per_task"] = sorted(rows, key=lambda r: r["delta"])
    rr = [r for r in rows if r["anchors"] > 0]
    if len(rr) >= 5:
        R["per_task_spearman"] = {"n_tasks": len(rr),
                                  "delta_vs_anchors": spearman([r["delta"] for r in rr], [r["anchors"] for r in rr]),
                                  "delta_vs_share_T4": spearman([r["delta"] for r in rr], [r["share_T4"] for r in rr]),
                                  "delta_vs_audit_fail": spearman([r["delta"] for r in rr],
                                                                  [r["audit_fail"] for r in rr])}

    # ---------- LIBERO-Recover
    R["liberorecover_annotations"] = liberorecover_annotations()
    R["liberorecover_rsr_by_step"] = rsr_by_step()

    (out / "summary.json").write_text(json.dumps(R, indent=1, default=float))
    print(json.dumps(R, indent=1, default=float))


if __name__ == "__main__":
    main()
