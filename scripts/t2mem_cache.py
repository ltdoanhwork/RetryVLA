#!/usr/bin/env python
"""T2Mem PoC 1, stage 1 (GPU, 0 new sim): cache per-candidate critic embeddings of imagined vs TRUE end states.

python -m scripts.t2mem_cache [--dirs verif_m5b] [--out runs/_analysis/t2mem/cache_verif_m5b.pt]

For every sim-verified candidate in the full verif logs (z stored = true latents every 5 env steps), roll the WM v1 out
open-loop from the failure-state latent (exactly like scripts/diag_wm_true.py), take the imagined end state at the same
horizon as the true one, and store C^pi v2's CLS embedding (pre-head, width 256) of
  h_wm  = critic trunk on the IMAGINED end state,    h_true = critic trunk on the TRUE end state,
plus logits, success, arm, rank, actions and pooled latents (start / imagined end / true end, per camera mean token).
Small (~100 MB) on purpose (/pfss quota). Stage 2 = scripts/t2mem_poc1.py (instance-level test-time memory replay).
"""

from __future__ import annotations

import argparse
import glob
import logging
import os

import torch

from retryvla import REPO_ROOT
from retryvla.search.wm_search import WMCriticRanker
from retryvla.utils import setup_logging

log = logging.getLogger("t2mem_cache")


def cls_embed(critic, z, q, tid):
    n = z.shape[0]
    x = torch.cat([critic.cls.expand(n, -1, -1), critic.task(tid)[:, None], critic.q_proj(q)[:, None],
                   critic.in_proj(z)], 1) + critic.pos
    return critic.trunk(x)[:, 0]


def pool2(z):  # [B, P=72, D] -> [B, 2*D] (agent | wrist mean token)
    h = z.shape[1] // 2
    return torch.cat([z[:, :h].mean(1), z[:, h:].mean(1)], -1)


@torch.no_grad()
def file_rows(rk: WMCriticRanker, d: dict, bs: int = 64) -> dict | None:
    dev = rk.dev
    z, q, valid = d["z"], d["q"].float(), d["valid"].bool()
    acts = d["actions"]
    if isinstance(acts, list) or acts.shape[1] != 50 or z.shape[1] != 11:
        return None
    K = len(z)
    tid0 = rk.vocab.get(d["task"], 0)
    lv = valid.long().cumsum(1).argmax(1)
    src = list(d["source"])
    rank, seen = [], {}
    for s in src:
        rank.append(seen.get(s, 0)); seen[s] = seen.get(s, 0) + 1
    H_wm, H_true, L_wm, L_true, P0, PW, PT, QW, QT = [], [], [], [], [], [], [], [], []
    for i in range(0, K, bs):
        j = slice(i, min(K, i + bs))
        zn = rk.norm.z(z[j].to(dev).float())
        qn = rk.norm.q(q[j].to(dev))
        n = zn.shape[0]
        ar = torch.arange(n, device=dev)
        lvj = lv[j].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            _, _, _, zs, qs = rk.wm.rollout(zn[:, 0], qn[:, 0], acts[j].to(dev).float())
        Zs, Qs = torch.stack(zs, 1).float(), torch.stack(qs, 1).float()
        hi = (lvj - 1).clamp_min(0)
        zi, qi = Zs[ar, hi], Qs[ar, hi]
        zt, qt = zn[ar, lvj], qn[ar, lvj]
        ti = torch.full((n,), tid0, device=dev, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            hw = cls_embed(rk.critic, zi, qi, ti).float()
            ht = cls_embed(rk.critic, zt, qt, ti).float()
            lw = rk.critic.head(hw).squeeze(-1).float()
            lt = rk.critic.head(ht).squeeze(-1).float()
        H_wm.append(hw.half().cpu()); H_true.append(ht.half().cpu())
        L_wm.append(lw.cpu()); L_true.append(lt.cpu())
        P0.append(pool2(zn[:, 0]).half().cpu()); PW.append(pool2(zi).half().cpu()); PT.append(pool2(zt).half().cpu())
        QW.append(qi.half().cpu()); QT.append(qt.half().cpu())
    return {"instance_id": d["instance_id"], "task": d["task"], "tid": tid0, "arm": src, "rank": rank,
            "success": torch.as_tensor(d["success"]).bool(), "actions": acts.half(),
            "h_wm": torch.cat(H_wm), "h_true": torch.cat(H_true), "l_wm": torch.cat(L_wm), "l_true": torch.cat(L_true),
            "p0": torch.cat(P0), "p_wm": torch.cat(PW), "p_true": torch.cat(PT), "q_wm": torch.cat(QW),
            "q_true": torch.cat(QT), "q0": rk.norm.q(q[:, 0].to(dev)).half().cpu()}


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=["verif_m5b"])
    ap.add_argument("--out", default="runs/_analysis/t2mem/cache_verif_m5b.pt")
    ap.add_argument("--max_files", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rk = WMCriticRanker((REPO_ROOT / "data/wm/latest_wm.txt").read_text().strip(),
                        (REPO_ROOT / "data/wm/latest_critic_v2.txt").read_text().strip(),
                        "facebook/dinov2-small", 6, dev, 0.5, 1.0)
    head = {k: v.detach().cpu() for k, v in rk.critic.head.state_dict().items()}
    out, n_files = [], 0
    for name in a.dirs:
        fs = sorted(glob.glob(str(REPO_ROOT / "data/wm" / name / "*.pt")))[: a.max_files or None]
        for n, f in enumerate(fs):
            d = torch.load(f, weights_only=False)
            if d.get("slimmed"):
                continue
            r = file_rows(rk, d)
            if r is not None:
                r["dir"] = name
                out.append(r)
            n_files += 1
            if n % 16 == 0:
                log.info("%s %d/%d files, %d instances", name, n + 1, len(fs), len(out))
    p = REPO_ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    torch.save({"instances": out, "critic_head": head, "vocab": rk.vocab}, tmp)
    os.replace(tmp, p)
    log.info("wrote %d instances -> %s", len(out), p)


if __name__ == "__main__":
    main()
