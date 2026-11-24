"""Shared helpers for the LIBERO-Goal recovery demos (docs/research/RECOVERY_DEMOS.md).

Entries = seed-7 M0 audit failure entries (2610.05151 reverse direction, retryvla.search.goal_instances).
For every entry we use, all from logged data (no policy inference):
  (a) M0 alone: the audit rollout's own actions[t:T] (M0 continuing from the entry state; the audit outcome is FAIL);
  (b) WM bridge (verified): a wm_diverse trajectory of data/recovery/goal_search (bridge + M0 continuation, full action
      sequence stored in the LeRobot dataset, success verified in sim);
  (c) WM false positive: the wm_diverse candidate of data/wm/goal_verif with the highest C^pi value among those whose
      sim verification FAILED (bridge actions logged; the M0 continuation is re-run with the logged policy seed).
"""

from __future__ import annotations

import json
import os
import zlib
from pathlib import Path

import h5py
import numpy as np

from retryvla import REPO_ROOT

AUDIT_DIR = "data/goal_rsi/audit_m0_s7"
AUDIT_SEED = 7
SPLIT = "goal_rsi_v1"
SEARCH_ROOT = REPO_ROOT / "data/recovery/goal_search"
VERIF = REPO_ROOT / "data/wm/goal_verif"
M0_CKPT = "runs/goal_rsi/m0_s7/ckpt_04000"
CONT = 300            # search.continuation_max_steps (configs/exp/goal_search.yaml)
H = 50                # bridge length
ARMS = ["random", "wm_diverse"]   # verification order in compare_arms (slot j of an arm -> policy_seed = seed + j)

ENTRIES = [
    "put_the_cream_cheese_in_the_bowl__demo_13__s7__t075",
    "put_the_wine_bottle_on_the_rack__demo_19__s7__t075",
    "push_the_plate_to_the_front_of_the_stove__demo_32__s7__t150",
    "open_the_middle_drawer_of_the_cabinet__demo_46__s7__t075",
    "put_the_bowl_on_the_plate__demo_49__s7__t150",
    "open_the_top_drawer_and_put_the_bowl_inside__demo_32__s7__t150",
]


def entry_meta(iid: str) -> dict:
    for l in open(REPO_ROOT / AUDIT_DIR / f"entries_s{AUDIT_SEED}.jsonl"):
        e = json.loads(l)
        if e["entry_id"] == iid:
            return e
    raise KeyError(iid)


def entry_seed(iid: str) -> int:
    return zlib.crc32(iid.encode()) % (2**31)


def audit_actions(scene: str) -> np.ndarray:
    return np.load(REPO_ROOT / AUDIT_DIR / f"actions_s{AUDIT_SEED}.npz")[scene].astype(np.float32)


def build_entry(sim, iid: str) -> dict:
    """Exactly retryvla.search.goal_instances.build_goal_instances, in-process: load_init_state(10 settle steps) ->
    replay the audit actions[:t] -> snapshot. Returns bddl, language, snapshot, obs, the audit actions and the list of
    prefix observations (for rendering the context before the entry)."""
    from libero.libero import get_libero_path

    e = entry_meta(iid)
    sp = json.loads((REPO_ROOT / f"data/splits/{SPLIT}.json").read_text())["tasks"][e["task"]]
    with h5py.File(sp["hdf5"]) as h:
        dd = h[f"data/demo_{e['demo']}"]
        init = np.asarray(dd.attrs["init_state"]) if "init_state" in dd.attrs else np.asarray(dd["states"][0])
    bddl = os.path.join(get_libero_path("bddl_files"), "libero_goal", f"{e['task']}.bddl")
    out = sim.load_init_state(bddl, init, num_steps_wait=10, seed=0, return_snapshot=False)
    acts = audit_actions(e["scene"])
    obs_list = [out["obs"]]
    succ = False
    for a in acts[: e["t"]]:
        r = sim.step(a)
        obs_list.append(r.obs)
        succ |= r.success
    return {"entry": e, "bddl": bddl, "lang": sp["language"], "snap": sim.snapshot(), "obs": obs_list[-1],
            "prefix_obs": obs_list, "audit": acts, "prefix_success": succ}


def _parquet_episode(shard: int, arm: str, ep: int) -> dict:
    import pandas as pd

    root = SEARCH_ROOT / f"s{shard}" / "p0" / arm
    cols = ["action", "observation.state", "segment", "episode_index", "frame_index"]
    dfs = [pd.read_parquet(f, columns=cols) for f in sorted((root / "data").rglob("*.parquet"))]
    df = pd.concat(dfs)
    df = df[df.episode_index == ep].sort_values("frame_index")
    return {"action": np.stack(df["action"].to_numpy()).astype(np.float32),
            "state": np.stack(df["observation.state"].to_numpy()).astype(np.float32),
            "segment": df["segment"].to_numpy().astype(int)}


def wm_success_traj(iid: str) -> dict:
    """Best-ranked verified wm_diverse trajectory with a real M0 continuation (>= 20 steps, else the longest)."""
    rows = []
    for s in range(3):
        p = SEARCH_ROOT / f"s{s}" / "p0" / "wm_diverse" / "retryvla" / "trajectories.jsonl"
        for l in open(p):
            r = json.loads(l)
            if r["instance_id"] == iid:
                rows.append({**r, "shard": s})
    assert rows, f"no verified wm_diverse trajectory for {iid}"
    good = [r for r in rows if r["continuation_len"] >= 20] or sorted(rows, key=lambda r: -r["continuation_len"])[:1]
    r = max(good, key=lambda r: r["score"])
    ep = _parquet_episode(r["shard"], "wm_diverse", r["episode_index"])
    v = verif(iid)
    bridge = ep["action"][: r["recovery_len"]]
    k = int(np.argmin([np.abs(np.asarray(v["actions"][i][: len(bridge)]) - bridge).max() for i in range(len(v["source"]))]))
    match = float(np.abs(np.asarray(v["actions"][k][: len(bridge)]) - bridge).max())
    return {**r, **ep, "verif_k": k, "verif_match": match, "pred_value_verif": float(v["pred_value"][k]),
            "verif_success": bool(v["success"][k]), "verif_success_step": int(v["success_step"][k])}


def verif(iid: str) -> dict:
    import torch

    return torch.load(VERIF / f"{iid}.pt", weights_only=False)


def wm_false_positive(iid: str) -> dict:
    """wm_diverse candidate with the highest C^pi value whose sim verification failed."""
    v = verif(iid)
    src = list(v["source"])
    ks = [k for k, s in enumerate(src) if s == "wm_diverse" and not bool(v["success"][k])]
    assert ks, f"no wm_diverse false positive for {iid}"
    k = max(ks, key=lambda k: float(v["pred_value"][k]))
    j = k - src.index("wm_diverse")
    acts = v["actions"]
    a = np.asarray(acts[k] if not isinstance(acts, list) else acts[k], np.float32)[:H]
    return {"verif_k": k, "slot": j, "policy_seed": (entry_seed(iid) + j) % (2**31 - 1), "bridge": a,
            "pred_value": float(v["pred_value"][k]), "prefix_success": bool(v["prefix_success"][k]),
            "q": v["q"][k].numpy(), "valid": v["valid"][k].numpy().astype(bool),
            "n_wm": sum(s == "wm_diverse" for s in src),
            "n_wm_succ": int(sum(bool(v["success"][i]) for i, s in enumerate(src) if s == "wm_diverse"))}


def wm_slot_seed(iid: str, k: int) -> int:
    src = list(verif(iid)["source"])
    return (entry_seed(iid) + k - src.index(src[k])) % (2**31 - 1)
