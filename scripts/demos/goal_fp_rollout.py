"""Re-run the M0 continuation of logged WM candidates (GPU, pi0) for the demos: the false positive (highest C^pi among
failed wm_diverse picks) and, as a reproducibility check, the verified success shown in panel (b).

python -m scripts.demos.goal_fp_rollout [--entries ...] [--out runs/_analysis/demos/goal/fp]

Same protocol as compare_arms verification: restore the entry snapshot, bridge open-loop (stop on success), then pi0
M0 with execute 5 / replan, chunk noise seed = policy_seed * 100003 + n_calls, policy_seed = crc32(entry) + slot,
<= 300 continuation steps, stop at BDDL success. The original run batched many workers per policy call (bf16), so the
re-run is not guaranteed bit-identical; the check reports whether the verified success reproduces action-for-action.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from retryvla import REPO_ROOT
from scripts.demos.goal_demo_common import CONT, ENTRIES, M0_CKPT, build_entry, verif, wm_false_positive, \
    wm_slot_seed, wm_success_traj


def rollout(sim, policy, E, bridge: np.ndarray, pseed: int) -> dict:
    sim.restore_snapshot(E["bddl"], E["snap"])
    obs, acts, succ = E["obs"], [], False
    for a in bridge:
        r = sim.step(a)
        acts.append(np.asarray(a, np.float32))
        obs = r.obs
        if r.success:
            return {"actions": np.asarray(acts), "success": True, "n_bridge": len(acts)}
    nb, k = len(acts), 0
    while len(acts) < nb + CONT and not succ:
        ch = policy.predict_chunk([obs], [E["lang"]], seeds=[pseed * 100_003 + k])[0]
        k += 1
        for a in ch[: min(5, nb + CONT - len(acts))]:
            r = sim.step(a)
            acts.append(np.asarray(a, np.float32))
            obs = r.obs
            if r.success:
                succ = True
                break
    return {"actions": np.asarray(acts), "success": succ, "n_bridge": nb}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", nargs="*", default=ENTRIES)
    ap.add_argument("--out", default="runs/_analysis/demos/goal/fp")
    a = ap.parse_args()
    from retryvla.envs.libero_env import LiberoSim
    from retryvla.policy.pi0_wrapper import Pi0Wrapper

    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    policy = Pi0Wrapper(M0_CKPT, "cuda", 5)
    sim = LiberoSim(128)
    rep = {}
    for iid in a.entries:
        E = build_entry(sim, iid)
        fp = wm_false_positive(iid)
        r = rollout(sim, policy, E, fp["bridge"], fp["policy_seed"])
        np.savez_compressed(out / f"{iid}.npz", actions=r["actions"], n_bridge=r["n_bridge"], success=r["success"],
                            pred_value=fp["pred_value"], slot=fp["slot"], verif_k=fp["verif_k"])
        W = wm_success_traj(iid)
        v = verif(iid)
        br = np.asarray(v["actions"][W["verif_k"]], np.float32)[:50]
        rt = rollout(sim, policy, E, br, wm_slot_seed(iid, W["verif_k"]))
        n = min(len(rt["actions"]), len(W["action"]))
        rep[iid] = {"fp_slot": fp["slot"], "fp_cpi": fp["pred_value"], "fp_rerun_success": r["success"],
                    "fp_logged_success": False, "tp_rerun_success": rt["success"], "tp_rerun_len": len(rt["actions"]),
                    "tp_logged_len": len(W["action"]),
                    "tp_action_maxdiff": float(np.abs(rt["actions"][:n] - W["action"][:n]).max())}
        print(iid, json.dumps(rep[iid]), flush=True)
    (out / "rerun_report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
