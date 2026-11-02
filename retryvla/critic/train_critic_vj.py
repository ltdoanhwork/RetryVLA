"""C^pi on V-JEPA 2 latents: the same recovery critic (RecoveryValue) and recipe as C^pi v2 (m5_train_critic_v2:
collect_v1 + verif_r1 minus the M5 round-2 held-out instances, real + imagined end latents 50/50, 8000 steps),
but on pooled V-JEPA tokens (64 x 1408, agent view) and imagined ends from the fine-tuned V-JEPA 2-AC WM.

python -m retryvla.critic.train_critic_vj configs/exp/vj_train_critic.yaml
"""

from __future__ import annotations

import glob
import logging
import time

import numpy as np
import torch
import torch.nn.functional as Fn

from retryvla import REPO_ROOT, tracking
from retryvla.config import cli_config, to_dict
from retryvla.critic.train_critic import predict, ranking_lift
from retryvla.critic.value_model import RecoveryValue, auc, expected_calibration_error
from retryvla.utils import setup_logging
from retryvla.wm.data import save_ckpt, task_vocab
from retryvla.wm.vjepa2_ac import load_vj_wm

log = logging.getLogger("train_critic_vj")


@torch.no_grad()
def end_sets(roots, exclude: set[str], ex_roots: set[str], wm, qn, dev, bs=64):
    """per candidate: real end latent (last valid frame), WM-imagined end at the same horizon, labels."""
    ZR, QR, ZI, QI, Y, INST, TASK = [], [], [], [], [], [], []
    for r in roots:
        for f in sorted(glob.glob(str(REPO_ROOT / r / "*.pt"))):
            d = torch.load(f, weights_only=False)
            if exclude and (not ex_roots or r in ex_roots) and d["instance_id"] in exclude:
                continue
            z, q, v = d["z"], d["q"].float(), d["valid"].bool()
            K = len(z)
            lv_store = v.long().cumsum(1).argmax(1)
            lv = d["last_valid"] if "last_valid" in d else lv_store
            ar = torch.arange(K)
            ZR.append(z[ar, lv_store]); QR.append(qn(q[ar, lv_store]))
            for i in range(0, K, bs):
                j = slice(i, min(K, i + bs))
                z0, q0 = z[j, 0].to(dev).float(), qn(q[j, 0]).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    _, _, _, zs, qs = wm.rollout(z0, q0, d["actions"][j].to(dev).float())
                Zs, Qs = torch.stack(zs, 1), torch.stack(qs, 1)
                hi = (lv[j] - 1).clamp_min(0).to(dev)
                a_ = torch.arange(len(hi), device=dev)
                ZI.append(Zs[a_, hi].float().half().cpu()); QI.append(Qs[a_, hi].float().cpu())
            Y.append(d["success"].float()); INST += [d["instance_id"]] * K; TASK += [d["task"]] * K
    return (torch.cat(ZR), torch.cat(QR), torch.cat(ZI), torch.cat(QI), torch.cat(Y), INST, TASK)


def main() -> None:
    setup_logging()
    cfg = cli_config()
    c = cfg.critic
    torch.manual_seed(int(c.seed))
    dev = torch.device(cfg.policy.device)
    wm_path = (REPO_ROOT / c.wm_link).read_text().strip()
    wm, wck = load_vj_wm(wm_path, dev)
    qn = lambda q: (q.float() - wck["q_mean"]) / wck["q_std"]  # noqa: E731
    ex = {l.strip() for l in open(REPO_ROOT / c.exclude_instances_file) if l.strip()} if c.get("exclude_instances_file") else set()
    zr, qr, zi, qi, y, inst, tasks = end_sets(list(c.data_roots), ex, set(c.get("exclude_only_roots") or []), wm, qn, dev)
    val_set = set(wck.get("val_instances", []))
    va = np.array([i for i, s in enumerate(inst) if s in val_set])
    tr = np.setdiff1d(np.arange(len(y)), va)
    vocab = task_vocab(tasks)
    T = torch.as_tensor([vocab[t] for t in tasks])
    log.info("VJ critic data: %d train / %d val candidates, pos rate %.3f", len(tr), len(va), float(y.mean()))
    model = RecoveryValue(dim=zr.shape[-1], n_patch=zr.shape[1], n_tasks=max(16, len(vocab)), width=int(c.width),
                          depth=int(c.depth), dropout=float(c.dropout)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(c.lr), weight_decay=float(c.weight_decay))
    pos_w = torch.tensor((1 - y[tr].mean()) / y[tr].mean().clamp_min(1e-3)).clamp(1, 20).to(dev)
    rng = np.random.default_rng(int(c.seed))
    m = {}
    with tracking.start(cfg) as run:
        t0 = time.time()
        for step in range(1, int(c.steps) + 1):
            b = torch.as_tensor(rng.choice(tr, int(c.batch_size)))
            use_imag = torch.rand(len(b)) < float(c.imagined_frac)
            z = torch.where(use_imag[:, None, None], zi[b].float(), zr[b].float()).to(dev)
            q = torch.where(use_imag[:, None], qi[b], qr[b]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                lg = model.logits(z, q, T[b].to(dev))
            loss = Fn.binary_cross_entropy_with_logits(lg.float(), y[b].to(dev), pos_weight=pos_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % int(c.eval_every) == 0 or step == int(c.steps):
                model.eval()
                pr, pi_ = predict(model, zr[va], qr[va], T[va], dev), predict(model, zi[va], qi[va], T[va], dev)
                model.train()
                yv = y[va].numpy()
                rl = ranking_lift(pi_, yv, [inst[i] for i in va], k=int(c.topk))
                m = {"auc_real": auc(pr, yv), "auc_imagined": auc(pi_, yv), "ece_imagined": expected_calibration_error(pi_, yv),
                     "ece_real": expected_calibration_error(pr, yv), **{f"rank_{k}": v for k, v in rl.items()}}
                run.log({f"val/{k}": v for k, v in m.items()}, step=step)
                log.info("step %d loss %.3f | val AUC real %.3f imagined %.3f | ECE imag %.3f | top-%d %.3f vs mean %.3f | %.0fs",
                         step, float(loss), m["auc_real"], m["auc_imagined"], m["ece_imagined"], int(c.topk),
                         rl["topk_success"], rl["mean_success"], time.time() - t0)
        ck = run.path("critic.pt")
        save_ckpt(ck, model=model.state_dict(), vocab=vocab, cfg=to_dict(c), wm_ckpt=str(wm_path),
                  n_patch=zr.shape[1], dim=zr.shape[-1])
        run.summary.update(m | {"checkpoint": str(ck), "n_train": int(len(tr)), "n_val": int(len(va))})
        (REPO_ROOT / c.out_link).write_text(str(ck) + "\n")


if __name__ == "__main__":
    main()
