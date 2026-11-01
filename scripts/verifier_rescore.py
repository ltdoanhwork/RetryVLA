#!/usr/bin/env python
"""RQ3 offline (0 new sim): re-score every sim-verified candidate in the M5b-instance verification logs with the
SAME verifier (WM v1 + C^pi v2) and its ensemble disagreement U_ens (std over WM members of C^pi(z_H^e)).

python -m scripts.verifier_rescore [--out runs/_analysis/verifier/rescored.parquet]   (GPU; ~2 GB)

The failure-state latent z0 / q0 of each instance comes from data/wm/verif_m5b (frame 0 of the recorded
verification rollouts; the later M9 / M11 / B2 logs are slimmed but use the same instances). One row per verified
candidate: dir, instance_id, arm, rank (pick order inside the arm = descending arm score), logged pred_value (the
arm's own critic), success, and the re-scored value / u_ens / u_wm / rough / v_members.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from retryvla import REPO_ROOT
from retryvla.search.scorers import ensemble_values
from retryvla.search.wm_search import WMCriticRanker
from retryvla.utils import setup_logging

log = logging.getLogger("verifier_rescore")
DIRS = ["verif_m5b", "verif_m9_mech", "verif_m9_b32", "verif_b2_value", "verif_b2x_value", "verif_m11_a",
        "verif_m11_b"]


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/_analysis/verifier/rescored.parquet")
    ap.add_argument("--dirs", nargs="*", default=DIRS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boot", default="data/wm/boot_heads_20261014-103001_M2_train_wm_v1.pt")
    ap.add_argument("--mc_dropout", type=float, default=0.1)
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument("--extra_glob", nargs="*", default=[], help="extra verif dirs (globs) with z0/q0 or z inside")
    a = ap.parse_args()
    dev = torch.device(a.device)
    rk = WMCriticRanker((REPO_ROOT / "data/wm/latest_wm.txt").read_text().strip(),
                        (REPO_ROOT / "data/wm/latest_critic_v2.txt").read_text().strip(),
                        "facebook/dinov2-small", 6, dev, 0.5, 1.0)
    from retryvla.wm.boot_heads import load_boot

    rk.boot, rk.mc_dropout = load_boot(REPO_ROOT / a.boot, dev), a.mc_dropout
    z0s = {}
    for f in sorted(glob.glob(str(REPO_ROOT / "data/wm/verif_m5b/*.pt"))):
        d = torch.load(f, weights_only=False, mmap=True)
        z0s[d["instance_id"]] = (d["z"][0, 0].float().clone(), d["q"][0, 0].float().clone())
    log.info("z0 for %d instances", len(z0s))
    rows = []
    dirs = [str(REPO_ROOT / "data/wm" / x) for x in a.dirs] + [p for g in a.extra_glob for p in glob.glob(str(REPO_ROOT / g))]
    for dd in dirs:
        n_f = 0
        for f in sorted(glob.glob(dd + "/*.pt"))[: a.max_files or None]:
            d = torch.load(f, weights_only=False, mmap=True)
            i = d["instance_id"]
            if "z0" in d:  # slim RQ3 logs: z0 / q0 already normalized (ranker.encode at search time)
                z0, q0 = d["z0"].float().to(dev), d["q0"].float().to(dev)
            elif i in z0s:
                z0, q0 = rk.norm.z(z0s[i][0][None].to(dev)), rk.norm.q(z0s[i][1][None].to(dev))
            else:
                continue
            acts = d["actions"]
            if isinstance(acts, list) or acts.shape[1] != 50:
                continue
            ev = ensemble_values(rk, z0, q0, d["task"], acts.numpy().astype(np.float32))
            src = list(d["source"])
            rank, seen = [], {}
            for s in src:
                rank.append(seen.get(s, 0)); seen[s] = seen.get(s, 0) + 1
            pu = d.get("pred_uens")
            for j in range(len(src)):
                rows.append({"dir": Path(dd).name, "instance_id": i, "task": d["task"], "arm": src[j], "rank": rank[j],
                             "pred_logged": float(d["pred_value"][j]), "success": bool(d["success"][j]),
                             "value": float(ev["value"][j]), "u_ens": float(ev["u_ens"][j]), "u_wmb": float(ev["u_wmb"][j]),
                             "u_crit": float(ev["u_crit"][j]), "v_wmb_mean": float(ev["v_wmb_mean"][j]),
                             "u_wm": float(ev["u_wm"][j]),
                             "rough": float(ev["rough"][j]), "v_members": ev["v_members"][j].tolist(),
                             "pick_budget": int(d["pick_budget"][j]) if "pick_budget" in d else -1,
                             "uens_logged": float(pu[j]) if pu is not None else np.nan})
            n_f += 1
        log.info("%s: %d instances", Path(dd).name, n_f)
    out = REPO_ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    pd.DataFrame(rows).to_parquet(tmp)
    os.replace(tmp, out)
    log.info("wrote %d rows -> %s", len(rows), out)


if __name__ == "__main__":
    main()
