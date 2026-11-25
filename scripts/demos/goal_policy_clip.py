"""Closed-loop clip of a policy from a LIBERO-Goal failure entry, ON CPU (no GPU used), same protocol as
retryvla.eval.eval_goal_entries: init state + 10 settle steps, M0 audit actions[:t] open loop, then the policy alone
for <= 300 steps (execute 5 / replan, chunk seed = crc32('<entry>|0|eval') * 100003 + n_calls), stop at BDDL success.
Policy obs at 128 px (as in eval); the logged actions are re-rendered at 256 px by scripts/demos/render_full_episode.py.
CPU fp32 instead of batched GPU bf16, so the rollout is not guaranteed to reproduce the logged eval outcome; the npz
records both.

python -m scripts.demos.goal_policy_clip --entry <entry_id> --ckpt <ckpt> --tag g2phys --audit_dir data/goal_rsi/audit_m0_s8
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zlib

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from retryvla import REPO_ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--audit_dir", default="data/goal_rsi/audit_m0_s8")
    ap.add_argument("--cont", type=int, default=300)
    ap.add_argument("--out", default="runs/_analysis/visual/fullfps/policy_clips")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    from libero.libero import get_libero_path

    from retryvla.envs.libero_env import LiberoSim
    from retryvla.policy.pi0_wrapper import Pi0Wrapper

    e = next(json.loads(l) for l in open(REPO_ROOT / a.audit_dir / f"entries_s{a.audit_dir[-1]}.jsonl")
             if json.loads(l)["entry_id"] == a.entry)
    sp = json.loads((REPO_ROOT / "data/splits/goal_rsi_v1.json").read_text())["tasks"][e["task"]]
    with h5py.File(sp["hdf5"]) as h:
        dd = h[f"data/demo_{e['demo']}"]
        init = np.asarray(dd.attrs["init_state"]) if "init_state" in dd.attrs else np.asarray(dd["states"][0])
    pre = np.load(REPO_ROOT / a.audit_dir / f"actions_s{e['seed']}.npz")[e["scene"]][: e["t"]].astype(np.float32)
    bddl = os.path.join(get_libero_path("bddl_files"), "libero_goal", f"{e['task']}.bddl")
    t0 = time.time()
    policy = Pi0Wrapper(a.ckpt, device="cpu", n_action_steps=5)
    print(f"loaded {a.ckpt} in {time.time() - t0:.0f}s", flush=True)
    sim = LiberoSim(128)
    obs = sim.load_init_state(bddl, init, num_steps_wait=10, seed=0, return_snapshot=False)["obs"]
    for x in pre:
        obs = sim.step(x).obs
    pseed = zlib.crc32(f"{a.entry}|0|eval".encode()) % (2**31 - 1)
    acts, k, succ, tc = [], 0, None, []
    while len(acts) < a.cont and succ is None:
        t1 = time.time()
        ch = policy.predict_chunk([obs], [sp["language"]], seeds=[pseed * 100_003 + k])[0]
        tc.append(time.time() - t1)
        k += 1
        for x in ch[: min(5, a.cont - len(acts))]:
            r = sim.step(x)
            acts.append(np.asarray(x, np.float32))
            obs = r.obs
            if r.success:
                succ = len(acts)
                break
        if k % 10 == 0:
            print(f"step {len(acts)} call {k} {np.mean(tc):.1f}s/call", flush=True)
    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{a.entry}__{a.tag}.npz"
    np.savez_compressed(f, actions=np.asarray(acts), success_step=-1 if succ is None else succ, t=e["t"],
                        policy_seed=pseed, ckpt=a.ckpt)
    print(json.dumps({"entry": a.entry, "tag": a.tag, "success_step": succ, "n": len(acts), "calls": k,
                      "s_per_call": float(np.mean(tc)), "file": str(f)}), flush=True)


if __name__ == "__main__":
    main()
