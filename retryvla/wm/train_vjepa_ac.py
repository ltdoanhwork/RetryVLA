"""Fine-tune the V-JEPA 2-AC predictor (retryvla.wm.vjepa2_ac.VJWorldModel) on RetryVLA failure-state rollouts.

python -m retryvla.wm.train_vjepa_ac configs/exp/vj_train_wm.yaml

Data = the SAME bridges WM v1 was trained on (collect_v1: 144 train failure instances x 32 candidate bridges, incl.
failed / recovering ones), re-simulated for frames (scripts/replay_bridges.py) and encoded by the frozen ViT-g
(scripts/vjepa_encode.py); same episode-level val split as WM v1 (group_split, val_frac 0.15, seed 0).
Loss (V-JEPA 2-AC recipe, L1 on layer-normed tokens): teacher-forced next-frame loss on all 10 transitions +
open-loop rollout loss over `ar_steps` model steps from frame 0 (search uses 10-step open-loop imagination),
+ MSE on normalized proprio (q_head). Step 0 eval = zero-shot (pretrained predictor + physical action map).
Reports the same horizon metric as train_wm (open-loop MSE / copy-last-frame MSE).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from retryvla import REPO_ROOT, tracking
from retryvla.config import cli_config, to_dict
from retryvla.utils import setup_logging
from retryvla.wm.data import group_split, load_latents, save_ckpt
from retryvla.wm.train_wm import batches
from retryvla.wm.vjepa2_ac import K, VJWorldModel

log = logging.getLogger("train_vjepa_ac")


def losses(model: VJWorldModel, z, q, a, v, ar_steps: int, lam_q: float):
    """z [B, F, P, D] (fp32), q [B, F, 8] normalized, a [B, H, 7], v [B, F] bool."""
    B, F = z.shape[:2]
    ac = a[:, : (F - 1) * K].reshape(B, F - 1, K * 7)
    m = (v[:, 1:] & v[:, :-1]).float()                                   # [B, F-1] valid transitions
    zp = model.predict_next(z[:, :-1], q[:, :-1], ac)                   # teacher forcing
    l_tf = ((zp.float() - z[:, 1:]).abs().mean((2, 3)) * m).sum() / m.sum().clamp_min(1)
    qp = model.next_q(zp, q[:, :-1], ac)
    l_q = (((qp.float() - q[:, 1:]) ** 2).mean(-1) * m).sum() / m.sum().clamp_min(1)
    Z, Q, l_ar, n_ar = [z[:, 0]], [q[:, 0]], z.new_zeros(()), 0
    mm = v[:, 0].float()
    for t in range(min(ar_steps, F - 1)):
        zt = model.predict_next(torch.stack(Z, 1), torch.stack(Q, 1), ac[:, : t + 1])[:, -1]
        qt = model.next_q(zt, Q[-1], ac[:, t])
        mm = mm * v[:, t + 1].float()
        l_ar = l_ar + ((zt.float() - z[:, t + 1]).abs().mean((1, 2)) * mm).sum() / mm.sum().clamp_min(1)
        l_q = l_q + (((qt.float() - q[:, t + 1]) ** 2).mean(-1) * mm).sum() / mm.sum().clamp_min(1) / max(1, ar_steps)
        Z.append(zt); Q.append(qt); n_ar += 1
    l_ar = l_ar / max(1, n_ar)
    return {"loss": l_tf + l_ar + lam_q * l_q, "tf": l_tf.detach(), "ar": l_ar.detach(), "q": l_q.detach()}


@torch.no_grad()
def evaluate(model, ls, qn, idx, dev, bs=32):
    """open-loop MSE vs copy-last-frame at horizons 1..10 (same definition as dynamics.horizon_errors) + L1."""
    agg = {}
    for i in range(0, len(idx), bs):
        j = torch.as_tensor(idx[i: i + bs])
        z, q, v = ls.z[j].to(dev).float(), qn(ls.q[j].to(dev)), ls.valid[j].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            _, _, _, zs, qs = model.rollout(z[:, 0], q[:, 0], ls.actions[j].to(dev).float())
        for h, (zh, qh) in enumerate(zip(zs, qs), 1):
            ok = v[:, h] & v[:, 0]
            if not ok.any():
                continue
            e = ((zh.float() - z[:, h]) ** 2).mean((1, 2))[ok]
            c = ((z[:, 0] - z[:, h]) ** 2).mean((1, 2))[ok]
            eq = ((qh.float() - q[:, h]) ** 2).mean(1)[ok]
            a_ = agg.setdefault(h, {"z": 0.0, "c": 0.0, "q": 0.0, "n": 0})
            a_["z"] += float(e.sum()); a_["c"] += float(c.sum()); a_["q"] += float(eq.sum()); a_["n"] += int(ok.sum())
    return {h: {"z_mse": x["z"] / x["n"], "copy_baseline": x["c"] / x["n"], "ratio": x["z"] / max(x["c"], 1e-9),
                "q_mse": x["q"] / x["n"], "n": x["n"]} for h, x in sorted(agg.items())}


def main() -> None:
    setup_logging()
    cfg = cli_config()
    c = cfg.vjwm
    torch.manual_seed(int(c.seed))
    dev = torch.device(cfg.policy.device)
    ls = load_latents(list(c.data_roots))
    tr, va = group_split(ls, float(c.val_frac), int(c.seed))           # = WM v1's split (same ids / crc32)
    if len(va) == 0:  # tiny smoke runs only
        va = tr
    vq = ls.valid[torch.as_tensor(tr)]
    qtr = ls.q[torch.as_tensor(tr)][vq]
    q_mean, q_std = qtr.mean(0), qtr.std(0).clamp_min(1e-3)
    qn = lambda q: (q.float() - q_mean.to(q.device)) / q_std.to(q.device)  # noqa: E731
    N, F, P, D = ls.z.shape
    log.info("VJ latents: %d traj, F=%d, P=%d, D=%d | train %d / val %d", N, F, P, D, len(tr), len(va))
    model = VJWorldModel(q_mean, q_std, max_frames=F, activation_checkpointing=bool(c.get("ckpt_act", True))).to(dev)
    log.info("VJ-AC params: %.1fM", sum(p.numel() for p in model.parameters()) / 1e6)
    fast = [p for n, p in model.named_parameters() if n.startswith(("q_head", "pred.action_encoder", "pred.state_encoder"))]
    slow = [p for n, p in model.named_parameters() if not n.startswith(("q_head", "pred.action_encoder", "pred.state_encoder"))]
    opt = torch.optim.AdamW([{"params": slow, "lr": float(c.lr)}, {"params": fast, "lr": float(c.lr_fast)}],
                            weight_decay=float(c.weight_decay))
    steps = int(c.steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[float(c.lr), float(c.lr_fast)], total_steps=steps,
                                                pct_start=0.05)
    rng = np.random.default_rng(int(c.seed))
    it = batches(len(tr), int(c.batch_size), rng)
    va_eval = va[:: max(1, len(va) // int(c.get("n_val_eval", 256)))]
    with tracking.start(cfg) as run:
        model.eval()
        he0 = evaluate(model, ls, qn, va_eval, dev)
        model.train()
        log.info("zero-shot val open-loop ratio: %s", {h: round(v["ratio"], 3) for h, v in he0.items()})
        run.summary["zero_shot_horizon_errors"] = he0
        t0 = time.time()
        he = he0
        for step in range(1, steps + 1):
            j = torch.as_tensor(tr[next(it)])
            z, q = ls.z[j].to(dev, non_blocking=True).float(), qn(ls.q[j].to(dev))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                L = losses(model, z, q, ls.actions[j].to(dev).float(), ls.valid[j].to(dev), int(c.ar_steps),
                           float(c.lam_q))
            opt.zero_grad(set_to_none=True)
            L["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % int(c.log_every) == 0 or step == 1:
                run.log({f"vjwm/{k_}": float(v) for k_, v in L.items()}, step=step)
                log.info("step %d/%d loss %.4f tf %.4f ar %.4f q %.4f | %.2f it/s | %.1f GB", step, steps,
                         float(L["loss"]), float(L["tf"]), float(L["ar"]), float(L["q"]), step / (time.time() - t0),
                         torch.cuda.max_memory_allocated() / 2**30)
            if step % int(c.eval_every) == 0 or step == steps:
                model.eval()
                he = evaluate(model, ls, qn, va_eval, dev)
                model.train()
                for h, v in he.items():
                    run.log({f"val/ratio_h{h}": v["ratio"], f"val/qmse_h{h}": v["q_mse"]}, step=step)
                log.info("val open-loop ratio: %s", {h: round(v["ratio"], 3) for h, v in he.items()})
                ck = run.path("vj_wm.pt")
                save_ckpt(str(ck) + ".tmp", model=model.state_dict(), q_mean=q_mean, q_std=q_std, cfg=to_dict(c),
                          val_instances=sorted({ls.inst[i] for i in va}), step=step)
                __import__("os").replace(str(ck) + ".tmp", ck)
        run.summary.update({"horizon_errors": he, "checkpoint": str(ck), "n_train": len(tr), "n_val": len(va)})
        (REPO_ROOT / c.out_link).parent.mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / c.out_link).write_text(str(ck) + "\n")
        log.info("saved %s", ck)


if __name__ == "__main__":
    main()
