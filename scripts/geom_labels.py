#!/usr/bin/env python
"""RGR: privileged geometry labels for the WM training bridges (collect_v1, TRAIN split only).

python -m scripts.geom_labels --src data/wm/collect_v1 --cfg configs/exp/m2_collect.yaml \
    --out data/geom/labels_collect_v1 [--workers 8] [--shard i --num_shards n]

Re-simulates every logged 50-step bridge from its exact failure snapshot (same restore path as retryvla.wm.collect /
scripts/replay_bridges --kind collect; physics is bit-exact) and records retryvla.geom.features.sim_geom at
t = 0, 5, ..., 50, aligned with the stored latent frames z [K, 11, P, D]. Output per source file:
{instance_id, task, geom [K, 11, G] float32 (frames after an early end repeat the terminal state), n_steps [K],
 eef_check [K] (max |eef_replay - q_logged[:3]| over valid frames, metres: alignment / determinism check)}.
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
from retryvla.geom.exec import run_bridges
from retryvla.utils import setup_logging

log = logging.getLogger("geom_labels")


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/wm/collect_v1")
    ap.add_argument("--cfg", default="configs/exp/m2_collect.yaml")
    ap.add_argument("--out", default="data/geom/labels_collect_v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_files", type=int, default=0)
    a = ap.parse_args()
    from scripts.replay_bridges import collect_instances

    cfg = load_config(a.cfg)
    H, stride, F = 50, 5, 11
    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(REPO_ROOT / a.src / "*.pt")))[a.shard:: a.num_shards][: a.max_files or None]
    files = [f for f in files if not (out / Path(f).name).exists()]
    log.info("%d files to label", len(files))
    if not files:
        return
    ids = {torch.load(f, weights_only=False, mmap=True)["instance_id"] for f in files}
    t0 = time.time()
    with SimPool(a.workers, int(cfg.env.resolution)) as pool:
        snaps = collect_instances(pool, cfg, ids)
        log.info("restored %d / %d instances (%.0fs)", len(snaps), len(ids), time.time() - t0)
        for n, f in enumerate(files):
            d = torch.load(f, weights_only=False)
            i = d["instance_id"]
            if i not in snaps:
                log.warning("no snapshot for %s", i)
                continue
            bddl, snap = snaps[i]
            acts = d["actions"].numpy().astype(np.float32)
            res = run_bridges(pool, [(bddl, snap, acts[k, :H]) for k in range(len(acts))], every=stride,
                              snapshot=False)
            geom = np.zeros((len(acts), F, res[0]["geom"].shape[-1]), np.float32)
            chk = np.zeros(len(acts), np.float32)
            for k, r in enumerate(res):
                g = r["geom"][:F]
                geom[k, : len(g)] = g
                geom[k, len(g):] = g[-1]
                v = d["valid"][k].numpy().astype(bool)
                q = d["q"][k].numpy()
                m = min(len(g), int(v.sum()))
                # sim_geom stores eef z relative to the table top; compare xy only (z offset = table height)
                chk[k] = float(np.abs(g[:m, 7:9] - q[:m, :2]).max()) if m else np.nan
            o = {"instance_id": i, "task": d["task"], "geom": torch.as_tensor(geom),
                 "n_steps": torch.as_tensor([int(r["n"]) for r in res]), "eef_check": torch.as_tensor(chk)}
            tmp = out / (Path(f).name + ".tmp")
            torch.save(o, tmp)
            os.replace(tmp, out / Path(f).name)
            log.info("[%d/%d] %s | eef check max %.4f m | %.0fs", n + 1, len(files), i[-40:], float(np.nanmax(chk)),
                     time.time() - t0)


if __name__ == "__main__":
    main()
