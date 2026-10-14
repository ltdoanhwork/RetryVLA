"""M2: train the recovery critic on (end-of-segment latent, outcome) pairs.

python -m retryvla.critic.train_critic configs/exp/m2_train_critic.yaml

Inputs per candidate: the latent state after the 50-step recovery segment (or the terminal state if
the segment itself succeeded), label = did frozen pi0 then complete the task.
Training uses BOTH real end latents and WM-imagined end latents (open-loop rollout of the same
actions from the failure state), so the critic is calibrated for what it will see during search.

Key metric for search: per held-out instance, rank its candidates by the critic on IMAGINED end
states; success rate of the top-k vs. the instance's mean success rate ("ranking lift").
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from retryvla import REPO_ROOT, tracking
from retryvla.config import cli_config, to_dict
from retryvla.critic.value_model import RecoveryValue, auc, expected_calibration_error
from retryvla.utils import setup_logging
from retryvla.wm.data import Normalizer, group_split, load_latents, save_ckpt, task_vocab
from retryvla.wm.dynamics import LatentWM

log = logging.getLogger("train_critic")


def load_wm(path: str | Path, dev) -> tuple[LatentWM, Normalizer, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    wm = LatentWM(dim=ck["D"], n_patch=ck["P"], k=ck["k"], width=int(c["width"]), depth=int(c["depth"]),
                  heads=int(c["heads"]), n_ensemble=int(c["n_ensemble"]))
    wm.load_state_dict(ck["model"])
    return wm.to(dev).eval(), Normalizer.from_state(ck["norm"]), ck


@torch.no_grad()
def end_latents(ls, norm, wm, idx, dev, bs=256):
    """Real end latents (last valid frame) and WM-imagined ones for the same candidates."""
    zr, qr, zi, qi, unc = [], [], [], [], []
    last = ls.last_valid()
    for i in range(0, len(idx), bs):
        j = torch.as_tensor(idx[i: i + bs])
        lv = last[j]
        z = ls.z[j].to(dev)
        q = ls.q[j].to(dev)
        ar = torch.arange(len(j), device=dev)
        zr.append(norm.z(z[ar, lv.to(dev)]).half().cpu()); qr.append(norm.q(q[ar, lv.to(dev)]).cpu())
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            z0, q0 = norm.z(z[:, 0]), norm.q(q[:, 0])
            zH, qH, u, zs, qs = wm.rollout(z0, q0, ls.actions[j].to(dev))
        # imagined state at the same horizon as the real end latent (terminal successes end early)
        Zs, Qs = torch.stack(zs, 1), torch.stack(qs, 1)          # [B, F-1, P, D]
        hi = (lv - 1).clamp_min(0).to(dev)
        zi.append(Zs[ar, hi].float().half().cpu()); qi.append(Qs[ar, hi].float().cpu()); unc.append(u.float().cpu())
    return torch.cat(zr), torch.cat(qr), torch.cat(zi), torch.cat(qi), torch.cat(unc)


@torch.no_grad()
def predict(model, z, q, t, dev, bs=512):
    out = []
    for i in range(0, len(z), bs):
        out.append(torch.sigmoid(model.logits(z[i:i + bs].to(dev).float(), q[i:i + bs].to(dev), t[i:i + bs].to(dev))).cpu())
    return torch.cat(out).numpy()


def ranking_lift(p: np.ndarray, y: np.ndarray, inst: list[str], k: int = 4) -> dict:
    """Per instance: success rate of top-k by p vs. mean success; also 'any success in top-k'."""
    by: dict[str, list[int]] = {}
    for i, s in enumerate(inst):
        by.setdefault(s, []).append(i)
    top, mean, hit, hit_rand = [], [], [], []
    for ids in by.values():
        ids = np.asarray(ids)
        yy = y[ids]
        if len(ids) < k:
            continue
        o = ids[np.argsort(-p[ids])[:k]]
        top.append(y[o].mean()); mean.append(yy.mean()); hit.append(y[o].any())
        # P(at least one success among k random picks without replacement)
        n, s = len(ids), int(yy.sum())
        hit_rand.append(1 - (np.prod([(n - s - i) / (n - i) for i in range(k)]) if n - s >= k else 0.0))
    return {"topk_success": float(np.mean(top)), "mean_success": float(np.mean(mean)),
            "topk_hit": float(np.mean(hit)), "random_k_hit": float(np.mean(hit_rand)), "n_instances": len(top)}


def main() -> None:
    setup_logging()
    cfg = cli_config()
    c = cfg.critic
    torch.manual_seed(int(c.seed))
    dev = torch.device(cfg.policy.device)
    wm_path = c.wm_ckpt if c.get("wm_ckpt") else (REPO_ROOT / c.wm_link).read_text().strip()
    wm, norm, wck = load_wm(wm_path, dev)
    ls = load_latents(list(c.data_roots))
    # use the SAME held-out failure episodes as the WM (critic val = WM val)
    val_set = set(wck.get("val_instances", []))
    va = np.array([i for i, s in enumerate(ls.inst) if s in val_set]) if val_set else group_split(ls, 0.15)[1]
    tr = np.setdiff1d(np.arange(len(ls)), va)
    if len(tr) == 0:  # tiny smoke runs only
        tr = va
    vocab = task_vocab(ls.task)
    T = torch.as_tensor([vocab[t] for t in ls.task])
    y = ls.success.float()
    zr, qr, zi, qi, unc = end_latents(ls, norm, wm, np.arange(len(ls)), dev)
    log.info("critic data: %d train / %d val candidates, pos rate %.3f", len(tr), len(va), float(y.mean()))
    model = RecoveryValue(dim=ls.z.shape[-1], n_patch=ls.z.shape[2], n_tasks=max(16, len(vocab)), width=int(c.width),
                          depth=int(c.depth), dropout=float(c.dropout)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(c.lr), weight_decay=float(c.weight_decay))
    pos_w = torch.tensor((1 - y[tr].mean()) / y[tr].mean().clamp_min(1e-3)).clamp(1, 20).to(dev)
    rng = np.random.default_rng(int(c.seed))
    steps, bs = int(c.steps), int(c.batch_size)
    with tracking.start(cfg) as run:
        t0 = time.time()
        for step in range(1, steps + 1):
            b = torch.as_tensor(rng.choice(tr, bs))
            use_imag = torch.rand(bs) < float(c.imagined_frac)
            z = torch.where(use_imag[:, None, None], zi[b].float(), zr[b].float()).to(dev)
            q = torch.where(use_imag[:, None], qi[b], qr[b]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                lg = model.logits(z, q, T[b].to(dev))
            loss = Fn.binary_cross_entropy_with_logits(lg.float(), y[b].to(dev), pos_weight=pos_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % int(c.log_every) == 0:
                run.log({"critic/loss": float(loss)}, step=step)
            if step % int(c.eval_every) == 0 or step == steps:
                model.eval()
                pr, pi_ = predict(model, zr[va], qr[va], T[va], dev), predict(model, zi[va], qi[va], T[va], dev)
                model.train()
                yv = y[va].numpy()
                inst_v = [ls.inst[i] for i in va]
                m = {"auc_real": auc(pr, yv), "auc_imagined": auc(pi_, yv),
                     "ece_imagined": expected_calibration_error(pi_, yv)}
                rl = ranking_lift(pi_, yv, inst_v, k=int(c.topk))
                m.update({f"rank_{k_}": v for k_, v in rl.items()})
                run.log({f"val/{k_}": v for k_, v in m.items()}, step=step)
                log.info("step %d loss %.3f | val AUC real %.3f imagined %.3f | ECE %.3f | top-%d success %.3f vs mean %.3f "
                         "| P(hit in top-%d) %.3f vs random %.3f | %.0fs", step, float(loss), m["auc_real"],
                         m["auc_imagined"], m["ece_imagined"], int(c.topk), rl["topk_success"], rl["mean_success"],
                         int(c.topk), rl["topk_hit"], rl["random_k_hit"], time.time() - t0)
        ck = run.path("critic.pt")
        save_ckpt(ck, model=model.state_dict(), vocab=vocab, cfg=to_dict(c), wm_ckpt=str(wm_path),
                  n_patch=ls.z.shape[2], dim=ls.z.shape[-1])
        run.summary.update(m | {"checkpoint": str(ck)})
        (REPO_ROOT / c.out_link).write_text(str(ck) + "\n")


if __name__ == "__main__":
    main()
