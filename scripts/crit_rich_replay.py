#!/usr/bin/env python
"""CRIT-V: rich records for already-verified candidates (bridge-only replay, no policy).

python -m scripts.crit_rich_replay --src data/wm/goal_verif --kind goal --audit_dir data/goal_rsi/audit_m0_s7 \
    --audit_seed 7 --cfg configs/exp/goal_search.yaml --out data/critic3d/goal_verif [--workers 24]

Every logged 50-step bridge is re-executed open-loop from its exact failure snapshot (restore_snapshot is bit-exact,
the bridge does not depend on the policy / continuation seed) with the opt-in rich recorder
(retryvla.critic.rich via record_cfg["rich"]). Output per source file (compact, fp16 / uint8):
  instance_id, task, sym [K, 11, S] (t = 0, 5, ..., 50; frames after an early end repeat the end state),
  pc [K, 6, PC_N, 3] fp16 + pcl [K, 6, PC_N] uint8 (t = 0, 10, ..., 50, same early-end rule), n_steps [K],
  eef_check [K] (max |eef xy replay - logged q xy| over valid frames, m), success / success_step / prefix_success /
  source copied from the source log (the verified OUTCOME labels).
kinds: goal (LIBERO-Goal audit entries), m5b (compare_arms instances), collect (wm.collect instances; --no_exclude for
verif_r1 = pilot instances).
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

from retryvla import REPO_ROOT
from retryvla.config import load_config
from retryvla.envs.vec import SimPool
from retryvla.eval.rollout import Job, run_jobs_resilient
from retryvla.utils import setup_logging

log = logging.getLogger("crit_rich_replay")
H, STRIDE, PC_STRIDE = 50, 5, 10
RICH = {"every": STRIDE, "pc_at": list(range(0, H + 1, PC_STRIDE)), "res": 96}


def goal_snaps(pool, audit_dir: str, seed: int, ids: set[str]) -> dict:
    from retryvla.search.goal_instances import build_goal_instances, load_entries

    ents = [e for e in load_entries(audit_dir, seed) if e["entry_id"] in ids]
    insts = build_goal_instances(pool, ents, audit_dir)
    return {i.instance_id: (i.bddl, i.snapshot) for i in insts}


def align(rich: dict, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rich record -> sym [11, S], pc [6, N, 3], pcl [6, N] on the fixed grids (early end: repeat the end state)."""
    t, sym = rich["t"].astype(int), rich["sym"]
    pt, pc, pcl = rich["pc_t"].astype(int), rich["pc"], rich["pcl"]

    def pick(ts, grid):
        return [int(np.nonzero(ts == g)[0][0]) if g in set(ts.tolist()) and g <= n else len(ts) - 1 for g in grid]

    gs = list(range(0, H + 1, STRIDE))
    gp = list(range(0, H + 1, PC_STRIDE))
    return sym[pick(t, gs)], pc[pick(pt, gp)], pcl[pick(pt, gp)]


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--kind", choices=["goal", "m5b", "collect"], required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit_dir", default="")
    ap.add_argument("--audit_seed", type=int, default=7)
    ap.add_argument("--no_exclude", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument("--max_cands", type=int, default=0)
    a = ap.parse_args()
    cfg = load_config(a.cfg)
    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(REPO_ROOT / a.src / "*.pt")))[a.shard:: a.num_shards][: a.max_files or None]
    files = [f for f in files if not (out / Path(f).name).exists()]
    log.info("%d files to replay (%s)", len(files), a.src)
    if not files:
        return
    ids = {torch.load(f, weights_only=False, mmap=True)["instance_id"] for f in files}
    t0 = time.time()
    with SimPool(a.workers, int(cfg.env.resolution)) as pool:
        if a.kind == "goal":
            snaps = goal_snaps(pool, a.audit_dir, a.audit_seed, ids)
        elif a.kind == "m5b":
            from scripts.replay_bridges import m5b_instances
            snaps = m5b_instances(pool, cfg, ids)
        else:
            from scripts.replay_bridges import collect_instances
            snaps = collect_instances(pool, cfg, ids, exclude=not a.no_exclude)
        log.info("restored %d / %d instances (%.0fs)", len(snaps), len(ids), time.time() - t0)
        for n, f in enumerate(files):
            d = torch.load(f, weights_only=False)
            i = d["instance_id"]
            if i not in snaps:
                log.warning("no snapshot for %s", i)
                continue
            bddl, snap = snaps[i]
            acts = d["actions"]
            acts = acts if not isinstance(acts, list) else torch.stack([x[:H] for x in acts])
            K = min(len(acts), a.max_cands or len(acts))
            jobs = [Job("restore_snapshot", dict(bddl=bddl, snap=snap), task=d["task"], max_steps=H,
                        prefix=acts[k, :H].numpy().astype(np.float32), record="all",
                        record_cfg={"stride": 10 ** 6, "img_until": 0, "rich": RICH}) for k in range(K)]
            res = run_jobs_resilient(pool, None, jobs, retries=2, desc="rich")
            SY, PC, PL, NS, CHK = [], [], [], [], []
            for k, r in enumerate(res):
                sy, pc, pl = align(r.record["rich"], int(r.steps))
                SY.append(sy); PC.append(pc); PL.append(pl); NS.append(int(r.steps))
                v = d["valid"][k].numpy().astype(bool) if "valid" in d else np.zeros(11, bool)
                q = d["q"][k].numpy() if "q" in d else None
                m = int(v.sum())
                # sym[:, 7:9] = eef xy (world), q[:, :2] = logged eef xy
                CHK.append(float(np.abs(sy[:m, 7:9].astype(np.float32) - q[:m, :2]).max()) if (m and q is not None)
                           else np.nan)
                r.record = None
            o = {"instance_id": i, "task": d["task"], "sym": torch.as_tensor(np.stack(SY)),
                 "pc": torch.as_tensor(np.stack(PC)), "pcl": torch.as_tensor(np.stack(PL)),
                 "n_steps": torch.as_tensor(NS), "eef_check": torch.as_tensor(np.asarray(CHK, np.float32)),
                 "success": d["success"][:K].clone(), "success_step": d["success_step"][:K].clone(),
                 "prefix_success": d["prefix_success"][:K].clone(), "source": list(d["source"])[:K],
                 "prefix_success_replay": torch.as_tensor([r.prefix_success for r in res])}
            tmp = out / (Path(f).name + ".tmp")
            torch.save(o, tmp)
            os.replace(tmp, out / Path(f).name)
            log.info("[%d/%d] %s: %d bridges | eef check max %.4f m | prefix agree %.3f | %.0fs", n + 1, len(files),
                     i[-40:], K, float(np.nanmax(CHK)) if np.isfinite(CHK).any() else -1,
                     float((o["prefix_success_replay"] == o["prefix_success"]).float().mean()), time.time() - t0)


if __name__ == "__main__":
    main()
