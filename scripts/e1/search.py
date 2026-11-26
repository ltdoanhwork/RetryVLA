#!/usr/bin/env python
"""E1 step S: imagination-only search with one WM on every held-out pool (PLAN E1 D2), no simulator.

python -m scripts.e1.search --wm runs/e1/wm/A2_s1/wm.pt --tag A2_s1     (or --wm shipped --tag WMv1)

Per held-out instance, with the arm's WM + C^pi v2 (fixed critic):
  bud<N>  arm_wm_budget logic (lambda 0): first N/2 of a fixed permutation of the shared pool + ONE elite-local CEM
          round of N/2 around the top-max(2, min(16, N/8)), top-8 by J = C^pi - R, N in {8, 32, 128, 512}
  unc     wm_unc_l0 pipeline: pool + 2 elite-local CEM rounds (768), diverse top-8 on imagined end features
CEM / diversity seeds depend only on the instance (same for every WM -> paired). Writes data/wm/e1/picks/<tag>.pt:
{instance: {actions [32, 50, 7], set [32] (bud8 ... unc), c_imag [32] (C^pi v2), c_imag_base [32] (CRIT-V base_s0),
z_end [32, 72, 384] fp16 + q_end [32, 8] (imagined, normalised), rough [32]}}.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import zlib
from pathlib import Path

import numpy as np
import torch

from retryvla import REPO_ROOT
from retryvla.critic.train_critic import load_wm
from retryvla.search.cem import CEM, CEMConfig
from retryvla.search.wm_search import WMCriticRanker, diverse_ranking
from retryvla.utils import setup_logging

log = logging.getLogger("e1_search")
POOLS = REPO_ROOT / "data/wm/e1/pools"
OUT = REPO_ROOT / "data/wm/e1/picks"
BASE_CRITIC = REPO_ROOT / "runs/20261124_CRITV_spatial/ckpt/base_s0.pt"
BUDGETS = (8, 32, 128, 512)
K = 8


def load_ranker(wm_path: str, dev) -> WMCriticRanker:
    rk = WMCriticRanker((REPO_ROOT / "data/wm/latest_wm.txt").read_text().strip(),
                        (REPO_ROOT / "data/wm/latest_critic_v2.txt").read_text().strip(),
                        "facebook/dinov2-small", 6, dev, 0.0, 1.0)
    if wm_path != "shipped":
        rk.wm, rk.norm, _ = load_wm(wm_path, dev)
    return rk


@torch.no_grad()
def end_states(rk, z0, q0, cands, bs=256):
    Z, Q = [], []
    for i in range(0, len(cands), bs):
        a = torch.as_tensor(cands[i: i + bs], device=rk.dev)
        n = len(a)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=rk.dev.type == "cuda"):
            zH, qH, _, _, _ = rk.wm.rollout(z0.expand(n, -1, -1), q0.expand(n, -1), a)
        Z.append(zH.float()); Q.append(qH.float())
    return torch.cat(Z), torch.cat(Q)


@torch.no_grad()
def critic_v(critic, vocab, task, z, q, dev, bs=512):
    tid = vocab.get(task, 0)
    out = []
    for i in range(0, len(z), bs):
        n = len(z[i: i + bs])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out.append(torch.sigmoid(critic.logits(z[i: i + bs], q[i: i + bs],
                                                   torch.full((n,), tid, device=dev, dtype=torch.long))).float())
    return torch.cat(out).cpu().numpy()


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.tag}.pt"
    if out.exists():
        log.info("exists %s", out); return
    dev = torch.device("cuda")
    rk = load_ranker(a.wm, dev)
    base = WMCriticRanker((REPO_ROOT / "data/wm/latest_wm.txt").read_text().strip(), str(BASE_CRITIC),
                          "facebook/dinov2-small", 6, dev, 0.0, 1.0)
    held = json.loads((REPO_ROOT / "configs/exp/e1_split.json").read_text())["heldout"]
    res = {}
    for n, iid in enumerate(held):
        f = POOLS / f"{iid}.pt"
        if not f.exists():
            log.warning("no pool %s", iid); continue
        p = torch.load(f, weights_only=False)
        z0, q0 = p["z0"].float().to(dev), p["q0"].float().to(dev)
        pool_c, task = p["pool_c"], p["task"]
        seed = zlib.crc32(iid.encode()) % (2**31)
        base_cs = rk.score(z0, q0, task, pool_c)
        acts, sets = [], []
        # (i) exploitation-curve budgets
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(pool_c))
        for N in BUDGETS:
            n0 = min(N // 2, len(pool_c))
            idx = perm[:n0]
            c0 = pool_c[idx]
            s0 = base_cs["value"][idx] - base_cs["rough"][idx]
            ne = max(2, min(16, N // 8, n0))
            el = np.argsort(-s0)[:ne]
            cem = CEM(pool_c.shape[1], 7, CEMConfig(n_samples=N - n0, init_std=0.25, smooth_kernel=5), seed=seed + N)
            new = cem.sample_local(c0[el], s0[el], 1, n=N - n0, temperature=0.05, decay=0.6)
            ns = rk.score(z0, q0, task, new)
            c = np.concatenate([c0, new])
            s = np.concatenate([s0, ns["value"] - ns["rough"]])
            top = np.argsort(-s)[:K]
            acts.append(c[top]); sets += [f"bud{N}"] * len(top)
        # (ii) wm_unc_l0 pipeline (lambda 0): pool + 2 elite-local CEM rounds, diverse top-8
        cand, cs = pool_c, base_cs
        cem = CEM(pool_c.shape[1], 7, CEMConfig(n_samples=len(pool_c), init_std=0.25, smooth_kernel=5), seed=seed)
        for it in range(1, 3):
            s = cs["value"] - cs["rough"]
            el = np.argsort(-s)[:16]
            new = cem.sample_local(cand[el], s[el], it, temperature=0.05, decay=0.6)
            ns = rk.score(z0, q0, task, new)
            cand = np.concatenate([cand, new])
            cs = {k: np.concatenate([cs[k], ns[k]]) for k in ("value", "rough", "zfeat")}
        s = cs["value"] - cs["rough"]
        pk = diverse_ranking(s, cs["zfeat"], K, seed=seed)
        acts.append(cand[pk]); sets += ["unc"] * len(pk)
        A = np.concatenate(acts).astype(np.float32)
        zE, qE = end_states(rk, z0, q0, A)
        c_im = critic_v(rk.critic, rk.vocab, task, zE, qE, dev)
        c_ib = critic_v(base.critic, base.vocab, task, zE, qE, dev)
        from retryvla.search.cem import roughness
        keep = np.isin(np.array(sets), ["bud512", "unc"])
        # fixed candidates: the 64 verified verif_m5b candidates of this held-out instance (16 random + 48 picked
        # by WM v1's search): imagined C under THIS WM from the stored true frame 0 vs C on the stored TRUE end
        v = torch.load(REPO_ROOT / "data/wm/verif_m5b" / f"{iid}.pt", weights_only=False)
        vz = rk.norm.z(v["z"].float().to(dev)); vq = rk.norm.q(v["q"].float().to(dev))
        lv = v["valid"].long().cumsum(1).argmax(1).to(dev)
        ar = torch.arange(len(vz), device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            _, _, _, zs_, qs_ = rk.wm.rollout(vz[:, 0], vq[:, 0], v["actions"].float().to(dev))
        Zs, Qs = torch.stack(zs_, 1).float(), torch.stack(qs_, 1).float()
        hi = (lv - 1).clamp_min(0)
        fz, fq = Zs[ar, hi], Qs[ar, hi]
        fixed = {"source": list(v["source"]), "success": v["success"].numpy(),
                 "c_imag": critic_v(rk.critic, rk.vocab, task, fz, fq, dev),
                 "c_true": critic_v(rk.critic, rk.vocab, task, vz[ar, lv], vq[ar, lv], dev),
                 "c_imag_base": critic_v(base.critic, base.vocab, task, fz, fq, dev),
                 "c_true_base": critic_v(base.critic, base.vocab, task, vz[ar, lv], vq[ar, lv], dev),
                 "tok_err": ((fz - vz[ar, lv]) ** 2).mean(-1).cpu().numpy().astype(np.float16)}
        res[iid] = {"actions": torch.as_tensor(A), "set": sets, "c_imag": c_im, "c_imag_base": c_ib,
                    "z_end": zE[torch.as_tensor(keep, device=dev)].half().cpu(), "z_end_mask": keep,
                    "q_end": qE.cpu(), "rough": roughness(A), "task": task, "fixed": fixed}
        if n % 10 == 0:
            log.info("[%d/%d] %s: bud512 C_imag %.3f unc %.3f", n + 1, len(held), iid[-40:],
                     float(c_im[np.array(sets) == "bud512"].mean()), float(c_im[np.array(sets) == "unc"].mean()))
    tmp = Path(str(out) + ".tmp")
    torch.save(res, tmp)
    tmp.replace(out)
    log.info("wrote %s (%d instances)", out, len(res))


if __name__ == "__main__":
    main()
