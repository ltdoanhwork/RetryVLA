"""M2: train the latent world model on collected failure-state trajectories.

python -m retryvla.wm.train_wm configs/exp/m2_train_wm.yaml

Reports open-loop latent error at horizons 1..10 model steps (5..50 env steps) on held-out failure
episodes, relative to the copy-last-frame baseline (ratio < 1 means the WM actually predicts motion).
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import torch

from retryvla import REPO_ROOT, tracking
from retryvla.config import cli_config, to_dict
from retryvla.utils import setup_logging
from retryvla.wm.data import Normalizer, group_split, load_latents, save_ckpt
from retryvla.wm.dynamics import LatentWM, horizon_errors, wm_loss

log = logging.getLogger("train_wm")


def batches(n: int, bs: int, rng: np.random.Generator):
    if n == 0:
        raise ValueError("empty training set (val_frac too high for the number of failure episodes?)")
    bs = min(bs, n)
    while True:
        p = rng.permutation(n)
        for i in range(0, n - bs + 1, bs):
            yield p[i: i + bs]


def build_model(c, P: int, D: int, k: int) -> LatentWM:
    return LatentWM(dim=D, n_patch=P, k=k, width=int(c.width), depth=int(c.depth), heads=int(c.heads),
                    n_ensemble=int(c.n_ensemble), dropout=float(c.get("dropout", 0.0)))


@torch.no_grad()
def evaluate(model, ls, norm, idx, dev, bs=256):
    out = {}
    agg: dict[int, list] = {}
    for i in range(0, len(idx), bs):
        j = torch.as_tensor(idx[i: i + bs])
        z, q = norm.z(ls.z[j].to(dev)), norm.q(ls.q[j].to(dev))
        he = horizon_errors(model, z, q, ls.actions[j].to(dev), ls.valid[j].to(dev))
        for h, v in he.items():
            agg.setdefault(h, []).append(v)
    for h, vs in sorted(agg.items()):
        n = sum(v["n"] for v in vs)
        out[h] = {k: sum(v[k] * v["n"] for v in vs) / n for k in ("z_mse", "copy_baseline", "q_mse")}
        out[h]["ratio"] = out[h]["z_mse"] / max(out[h]["copy_baseline"], 1e-9)
        out[h]["n"] = n
    return out


def main() -> None:
    setup_logging()
    cfg = cli_config()
    c = cfg.wm
    torch.manual_seed(int(c.seed))
    dev = torch.device(cfg.policy.device)
    ls = load_latents(list(c.data_roots))
    tr, va = group_split(ls, float(c.val_frac), int(c.seed))
    if len(va) == 0:  # tiny smoke runs: evaluate on train
        va = tr
    norm = Normalizer.fit(ls.subset(tr))
    N, F, P, D = ls.z.shape
    k = ls.actions.shape[1] // (F - 1)
    log.info("latents: %d traj, F=%d, P=%d, D=%d, k=%d | train %d / val %d trajectories", N, F, P, D, k, len(tr), len(va))
    model = build_model(c, P, D, k).to(dev)
    log.info("WM params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)
    opt = torch.optim.AdamW(model.parameters(), lr=float(c.lr), weight_decay=float(c.weight_decay))
    steps = int(c.steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(c.lr), total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(int(c.seed))
    it = batches(len(tr), int(c.batch_size), rng)
    with tracking.start(cfg) as run:
        t0 = time.time()
        for step in range(1, steps + 1):
            j = torch.as_tensor(tr[next(it)])
            z, q = norm.z(ls.z[j].to(dev, non_blocking=True)), norm.q(ls.q[j].to(dev))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                L = wm_loss(model, z, q, ls.actions[j].to(dev), ls.valid[j].to(dev), lam_open=float(c.lam_open))
            opt.zero_grad(set_to_none=True)
            L["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % int(c.log_every) == 0 or step == 1:
                run.log({f"wm/{k_}": float(v) for k_, v in L.items()}, step=step)
                log.info("step %d/%d loss %.4f z1 %.4f q1 %.4f open %.4f | %.1f it/s", step, steps, float(L["loss"]),
                         float(L["z1"]), float(L["q1"]), float(L["open"]), step / (time.time() - t0))
            if step % int(c.eval_every) == 0 or step == steps:
                model.eval()
                he = evaluate(model, ls, norm, va, dev)
                model.train()
                for h, v in he.items():
                    run.log({f"val/ratio_h{h}": v["ratio"], f"val/zmse_h{h}": v["z_mse"], f"val/qmse_h{h}": v["q_mse"]},
                            step=step)
                log.info("val open-loop (WM mse / copy-baseline): %s",
                         {h: round(v["ratio"], 3) for h, v in he.items()})
        ck = run.path("wm.pt")
        save_ckpt(ck, model=model.state_dict(), norm=norm.state_dict(), P=P, D=D, k=k, cfg=to_dict(c),
                  val_instances=sorted({ls.inst[i] for i in va}))
        run.summary.update({"horizon_errors": he, "checkpoint": str(ck), "n_train": len(tr), "n_val": len(va)})
        (REPO_ROOT / c.out_link).parent.mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / c.out_link).write_text(str(ck) + "\n")
        log.info("saved %s", ck)


if __name__ == "__main__":
    main()
