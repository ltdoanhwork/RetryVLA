"""LRS-G diagnosis: is pi0_sft --masked_recovery's loss identical to the standard pi0 loss on D0 anchors, and how large
is the per-target loss on LRS teacher windows at the M0 init / after training?

python -m scripts.lrs.check_masked_loss [--ckpts runs/goal_rsi/m0_s7/ckpt_04000 runs/goal_rsi/lrs_s7/ckpt_02000 ...]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from retryvla import REPO_ROOT
from retryvla.policy import pi0_lora as P
from retryvla.train.pi0_sft import D0, MaskedLoss, Pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=["runs/goal_rsi/m0_s7/ckpt_04000", "runs/goal_rsi/orig_s7/ckpt_02000",
                                                   "runs/goal_rsi/lrs_s7/ckpt_02000"])
    ap.add_argument("--rec", default="data/recovery/lrs_goal_ds")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors

    dev = torch.device("cuda")
    out = {}
    for ck in a.ckpts:
        policy, cfg, _ = P.load_policy(ck, device="cuda", for_training=True)
        policy.eval()
        stats = LeRobotDatasetMetadata("local/d0_train", root=REPO_ROOT / D0).stats
        pre, _ = make_pre_post_processors(cfg, dataset_stats=stats, preprocessor_overrides={"device_processor": {"device": "cuda"}})
        data = Pool([REPO_ROOT / D0, REPO_ROOT / a.rec], cfg.chunk_size)
        data.masked = True
        data.lmin_filter(5)
        ml = MaskedLoss(policy)
        rng = np.random.default_rng(0)
        d0 = [i for i, (k, _) in enumerate(data.idx) if k == 0]
        rc = [i for i, (k, _) in enumerate(data.idx) if k == 1]
        r = {"std_d0": [], "masked_d0": [], "masked_rec": [], "std_rec_unmasked": []}
        for b_i in range(a.n):
            for name, pool_ in (("d0", d0), ("rec", rc)):
                items = [data[int(j)] for j in rng.choice(pool_, 8, replace=False)]
                b = torch.utils.data.default_collate(items)
                b.pop("src")
                valid, w = b.pop("valid"), b.pop("w")
                b = pre(b)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    torch.manual_seed(1000 + b_i)
                    l_std, _ = policy(b)
                    torch.manual_seed(1000 + b_i)
                    l_m = ml(b, valid.to(dev), w.to(dev))
                if name == "d0":
                    r["std_d0"].append(float(l_std)); r["masked_d0"].append(float(l_m))
                else:
                    r["std_rec_unmasked"].append(float(l_std)); r["masked_rec"].append(float(l_m))
        out[ck] = {k: float(np.mean(v)) for k, v in r.items()} | {"pairs_d0": list(zip(r["std_d0"], r["masked_d0"]))[:3]}
        print(ck, json.dumps(out[ck]), flush=True)
        del policy, ml
        torch.cuda.empty_cache()
    (REPO_ROOT / "runs/_analysis/lrs_goal_masked_loss_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
