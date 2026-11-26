#!/usr/bin/env python
"""E1 step R: TRUE end state of every pick of one WM (bridge-only replay from the exact failure snapshot, no
policy, NVIDIA EGL like the stored latents; PLAN E1 D7).

python -m scripts.e1.replay --tag A2_s1 [--sets bud32 bud128 bud512 unc] [--workers 12]

Writes data/wm/e1/true/<tag>.pt: {instance: {idx [M] (into the picks), z_end [M, 72, 384] fp16 (normalised with WM
v1's normaliser), q_end [M, 8], c_true [M] (C^pi v2), c_true_base [M] (CRIT-V base_s0), prefix_success [M],
sym_end [M, 25] (privileged end state), steps [M]}}. Early end (task finished inside the bridge) -> the terminal
observation (as encode_record).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from retryvla import REPO_ROOT
from retryvla.envs.vec import SimPool
from retryvla.eval.rollout import Job, run_jobs_resilient
from retryvla.utils import setup_logging
from scripts.e1.search import BASE_CRITIC, POOLS, critic_v
from retryvla.search.wm_search import WMCriticRanker

log = logging.getLogger("e1_replay")
PICKS = REPO_ROOT / "data/wm/e1/picks"
OUT = REPO_ROOT / "data/wm/e1/true"
RICH = {"every": 50, "pc_at": [], "res": 64}


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sets", nargs="*", default=["bud32", "bud128", "bud512", "unc"])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.tag}.pt"
    if out.exists():
        log.info("exists %s", out); return
    picks = torch.load(PICKS / f"{a.tag}.pt", weights_only=False)
    dev = torch.device("cuda")
    rk = WMCriticRanker((REPO_ROOT / "data/wm/latest_wm.txt").read_text().strip(),
                        (REPO_ROOT / "data/wm/latest_critic_v2.txt").read_text().strip(),
                        "facebook/dinov2-small", 6, dev, 0.0, 1.0)
    base = WMCriticRanker((REPO_ROOT / "data/wm/latest_wm.txt").read_text().strip(), str(BASE_CRITIC),
                          "facebook/dinov2-small", 6, dev, 0.0, 1.0)
    jobs, owner = [], []
    for iid, p in picks.items():
        pool = torch.load(POOLS / f"{iid}.pt", weights_only=False)
        idx = [k for k, s in enumerate(p["set"]) if s in a.sets]
        for k in idx:
            jobs.append(Job("restore_snapshot", dict(bddl=pool["bddl"], snap=pool["snapshot"]), task=p["task"],
                            max_steps=50, prefix=p["actions"][k, :50].numpy().astype(np.float32), record="all",
                            record_cfg={"stride": 50, "img_until": 1, "rich": RICH}))
            owner.append((iid, k))
    log.info("%s: %d bridges over %d instances", a.tag, len(jobs), len(picks))
    t0 = time.time()
    res_by: dict[str, dict] = {}
    CH = 400
    with SimPool(a.workers, 256) as pool:
        for c0 in range(0, len(jobs), CH):
            res = run_jobs_resilient(pool, None, jobs[c0: c0 + CH], retries=2, desc="replay")
            imgs_a = np.stack([r.record["final_image"] for r in res])
            imgs_w = np.stack([r.record["final_image2"] for r in res])
            st = np.stack([r.record["final_state"] for r in res]).astype(np.float32)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                za = rk.enc.encode_images(torch.as_tensor(imgs_a, device=dev))
                zw = rk.enc.encode_images(torch.as_tensor(imgs_w, device=dev))
            z = rk.norm.z(torch.cat([za, zw], 1).float())
            q = rk.norm.q(torch.as_tensor(st, device=dev))
            for j, r in enumerate(res):
                iid, k = owner[c0 + j]
                d = res_by.setdefault(iid, {"idx": [], "z": [], "q": [], "ps": [], "sym": [], "steps": []})
                d["idx"].append(k); d["z"].append(z[j]); d["q"].append(q[j]); d["ps"].append(bool(r.prefix_success))
                d["sym"].append(np.asarray(r.record["rich"]["sym"][-1], np.float32)); d["steps"].append(int(r.steps))
                r.record = None
            log.info("%d / %d bridges (%.0f s)", min(c0 + CH, len(jobs)), len(jobs), time.time() - t0)
    o = {}
    for iid, d in res_by.items():
        z, q = torch.stack(d["z"]), torch.stack(d["q"])
        task = picks[iid]["task"]
        o[iid] = {"idx": np.asarray(d["idx"]), "z_end": z.half().cpu(), "q_end": q.cpu(),
                  "c_true": critic_v(rk.critic, rk.vocab, task, z, q, dev),
                  "c_true_base": critic_v(base.critic, base.vocab, task, z, q, dev),
                  "prefix_success": np.asarray(d["ps"]), "sym_end": np.stack(d["sym"]), "steps": np.asarray(d["steps"])}
    tmp = Path(str(out) + ".tmp")
    torch.save(o, tmp)
    tmp.replace(out)
    log.info("wrote %s (%.0f s)", out, time.time() - t0)


if __name__ == "__main__":
    main()
