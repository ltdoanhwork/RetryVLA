"""ACR magnitude calibration (pre-registered: PLAN "ACR"; ONLY the 3 Tier B calibration states, never eval states).

python -m retryvla.causal.calibrate configs/exp/acr_poc.yaml
For every calibration state x hidden type x magnitude on a wide grid: every scripted family's prefix (3 variants,
NO policy) is executed and its physical sub-goal measured: grasp families -> object lifted >= 3 cm at the end;
push_goal -> object-goal xy distance reduced >= 1 cm (or prefix success). Output: runs/_analysis/acr/calib_grid.csv
+ a printed table. Range choice rule (fixed before looking): the range of a type = [weakest, strongest] magnitude of the
grid at which the type's own target sub-goal rate is <= 50 % of nominal while at least one OTHER sub-goal stays
>= 50 % of nominal (the cause is decision-relevant and not "nothing works").
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from retryvla import REPO_ROOT
from retryvla.causal import families as FAM
from retryvla.causal.exec import run_seqs
from retryvla.config import cli_config
from retryvla.envs.vec import SimPool
from retryvla.geom.features import IDX
from retryvla.utils import setup_logging

GRID = {"slippery": [0.3, 0.1, 0.03, 0.01, 0.003, 0.001], "anchored": [1, 3, 10, 30, 100],
        "heavy": [0.3, 1, 3, 10, 30], "weak_grip": [0.3, 0.1, 0.03, 0.01, 0.003]}
FAMS = ["grasp_slow", "grasp_firm", "push_goal"]


def main() -> None:
    setup_logging()
    cfg = cli_config()
    from retryvla.envs.splits import load_split, scenes_for
    from retryvla.search.acr_search import restore_instances
    from retryvla.search.compare_arms import make_safe_run
    from retryvla.policy import SmolVLAWrapper

    sel = json.loads((REPO_ROOT / cfg.acr.instances).read_text())
    policy = SmolVLAWrapper(cfg.policy.path, cfg.policy.device, cfg.policy.n_action_steps)
    rows = []
    with SimPool(int(cfg.eval.n_workers), int(cfg.env.resolution)) as pool:
        srun = make_safe_run(pool, 2)
        insts = restore_instances(pool, policy, cfg, sel["tier_b_calib"], srun)
        for inst in insts:
            g0 = np.asarray(run_seqs(pool, [[("restore_hidden", (inst.bddl, inst.snapshot, None), {}),
                                             ("state_info", (), {})]])[0]["geom"])
            hs = [None] + [{"type": t, "mag": m} for t, ms in GRID.items() for m in ms]
            items, own = [], []
            for h in hs:
                for f in FAMS:
                    for j in range(3):
                        wps = FAM.build(f, g0, np.random.default_rng([7, j]), float(cfg.acr.perception_sigma))
                        items.append([("restore_hidden", (inst.bddl, inst.snapshot, h), {}),
                                      ("run_family", (wps,), {"snapshot": False})])
                        own.append((h, f, j))
            res = run_seqs(pool, items)
            d0 = np.linalg.norm(g0[[IDX["obj_x"], IDX["obj_y"]]] - g0[[IDX["goal_x"], IDX["goal_y"]]])
            for (h, f, j), r in zip(own, res):
                g1 = r["geom1"]
                d1 = np.linalg.norm(g1[[IDX["obj_x"], IDX["obj_y"]]] - g1[[IDX["goal_x"], IDX["goal_y"]]])
                lifted = g1[IDX["obj_z"]] - g0[IDX["obj_z"]] >= 0.03
                pushed = bool(r["success"]) or (d0 - d1 >= 0.01)
                rows.append({"instance_id": inst.instance_id, "type": (h or {}).get("type", "nominal"),
                             "mag": (h or {}).get("mag", 1.0), "family": f, "j": j,
                             "ok": bool(lifted if f.startswith("grasp") else pushed), "success": bool(r["success"])})
    d = pd.DataFrame(rows)
    out = REPO_ROOT / "runs/_analysis/acr"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "calib_grid.csv", index=False)
    d["fam2"] = np.where(d.family.str.startswith("grasp"), "grasp", "push")
    t = d.groupby(["type", "mag", "fam2"]).ok.mean().unstack()
    print(t.to_string())
    print(d.groupby(["instance_id", "type", "mag", "fam2"]).ok.mean().unstack().to_string())


if __name__ == "__main__":
    main()
