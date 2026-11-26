#!/usr/bin/env python
"""WM over-optimism, visual study, step 1: per-candidate statistics on sim-verified candidates (CPU ok).

python -m scripts.wm_optimism.cache_stats [--dirs verif_m5b true_end_rq3 verif_r1 collect_v1] [--device cpu]

For every candidate of the listed latent logs (data/wm/<dir>/*.pt, TRUE latents z stored) roll WM v1 open loop from
the true frame-0 latent (same as scripts.diag_wm_true) and store
  c_wm / c_true            C^pi v2 on the imagined / TRUE end latent (same horizon = last valid frame)
  cw{h} / ct{h}            C^pi v2 on the imagined / true latent at every 5-step horizon h = 1..10 (NaN after an
                           early end; the critic was trained on END states, intermediate values are diagnostic only)
  ez{h}                    normalized-token MSE imagined vs true at horizon h
  eef_err                  |eef xyz imagined - true| at the end (m, de-normalized robot state)
  tok_err [72]             per-token normalized MSE at the end (6x6 agent-view grid + 6x6 wrist grid) -> tok_err.npy
  fam                      proposal family: logged (collect_v1 / verif_r1 sources) or action heuristic (prim = exact
                           primitive structure; pi0_s1.0 / pi0_s1.5 / perturb by 2nd-difference roughness, calibrated
                           on collect_v1 labels); CEM / pCN children of the search arms keep their arm as `arm`
  flips, rough             gripper sign flips, mean |2nd difference| of the arm actions
Outputs: runs/_analysis/wm_optimism/cache/{rows.parquet, tok_err.npy}
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
from retryvla.search.wm_search import WMCriticRanker
from retryvla.utils import setup_logging

log = logging.getLogger("wm_opt_cache")
OUT = REPO_ROOT / "runs/_analysis/wm_optimism/cache"


def action_family(a: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """a [K, 50, 7] -> heuristic family, gripper flips, roughness, exact-gripper flag."""
    rot0 = np.abs(a[:, :, 3:6]).max((1, 2)) == 0
    gex = np.isin(a[:, :, 6], [-1.0, 1.0]).all(1)
    flips = (np.diff(np.sign(a[:, :, 6]), axis=1) != 0).sum(1)
    rough = np.abs(np.diff(a[:, :, :6], 2, axis=1)).mean((1, 2))
    fam = []
    for k in range(len(a)):
        if rot0[k] and gex[k]:
            fam.append("prim")
        elif gex[k]:
            fam.append("prim+cem")          # search-arm child of a primitive (gripper schedule kept, arm mutated)
        elif rough[k] < 0.11:
            fam.append("pi0_s1.0")
        elif rough[k] < 0.205:
            fam.append("pi0_s1.5")
        else:
            fam.append("perturb")
    return fam, flips, rough, gex


def logged_family(s: str) -> str | None:
    if s.startswith("prim"):
        return "prim"
    if s.startswith("vla_s1.0"):
        return "pi0_s1.0"
    if s.startswith("vla_s1.5"):
        return "pi0_s1.5"
    if s == "perturb":
        return "perturb"
    return None


@torch.no_grad()
def file_rows(rk: WMCriticRanker, d: dict, dname: str, stem: str, bs: int = 64):
    dev = rk.dev
    z, q, valid = d["z"], d["q"].float(), d["valid"].bool()
    acts = d["actions"]
    if isinstance(acts, list) or acts.shape[1] != 50 or z.shape[1] != 11:
        return [], []
    K = len(z)
    tid = rk.vocab.get(d["task"], 0)
    lv = valid.long().cumsum(1).argmax(1)
    src = list(d["source"])
    rank, seen = [], {}
    for s in src:
        rank.append(seen.get(s, 0)); seen[s] = seen.get(s, 0) + 1
    an = acts.numpy().astype(np.float32)
    hfam, flips, rough, gex = action_family(an)
    qm, qs_ = rk.norm.q_mean.float().to(dev), rk.norm.q_std.float().to(dev)
    rows, toks = [], []
    for i in range(0, K, bs):
        j = slice(i, min(K, i + bs))
        zn = rk.norm.z(z[j].to(dev).float())
        qn = rk.norm.q(q[j].to(dev))
        n = zn.shape[0]
        ar = torch.arange(n, device=dev)
        lvj = lv[j].to(dev)
        _, _, _, zs, qs = rk.wm.rollout(zn[:, 0], qn[:, 0], acts[j].to(dev).float())
        Zs, Qs = torch.stack(zs, 1).float(), torch.stack(qs, 1).float()          # [n, 10, P, D]
        hi = (lvj - 1).clamp_min(0)
        zi, qi, zt, qt = Zs[ar, hi], Qs[ar, hi], zn[ar, lvj], qn[ar, lvj]
        ti = torch.full((n,), tid, device=dev, dtype=torch.long)
        c_wm = torch.sigmoid(rk.critic.logits(zi, qi, ti)).float()
        c_true = torch.sigmoid(rk.critic.logits(zt, qt, ti)).float()
        # per-horizon critic (imagined vs true)
        cw = torch.sigmoid(rk.critic.logits(Zs.flatten(0, 1), Qs.flatten(0, 1), ti.repeat_interleave(10))).view(n, 10)
        ct = torch.sigmoid(rk.critic.logits(zn[:, 1:].flatten(0, 1), qn[:, 1:].flatten(0, 1),
                                            ti.repeat_interleave(10))).view(n, 10)
        e_z = ((Zs - zn[:, 1:]) ** 2).mean((2, 3))
        tok = ((zi - zt) ** 2).mean(-1)                                            # [n, 72]
        eef_i = qi[:, :3] * qs_[:3] + qm[:3]
        eef_t = qt[:, :3] * qs_[:3] + qm[:3]
        eef_err = (eef_i - eef_t).norm(dim=-1)
        vf = valid[j]
        for b in range(n):
            k = i + b
            lf = logged_family(src[k])
            r = {"dir": dname, "file": stem, "k": k, "instance_id": d["instance_id"], "task": d["task"],
                 "arm": src[k], "rank": rank[k], "success": bool(d["success"][k]),
                 "prefix_success": bool(d["prefix_success"][k]), "last_valid": int(lvj[b]),
                 "pick_budget": int(d["pick_budget"][k]) if "pick_budget" in d else -1,
                 "pred_logged": float(d["pred_value"][k]) if "pred_value" in d else np.nan,
                 "fam": lf or hfam[k], "fam_logged": lf is not None, "flips": int(flips[k]),
                 "rough": float(rough[k]), "grip_exact": bool(gex[k]),
                 "c_wm": float(c_wm[b]), "c_true": float(c_true[b]), "eef_err": float(eef_err[b]),
                 "mse_end": float(tok[b].mean())}
            for h in range(10):
                ok = bool(vf[b, h + 1])
                r[f"ez{h + 1}"] = float(e_z[b, h]) if ok else np.nan
                r[f"cw{h + 1}"] = float(cw[b, h]) if ok else np.nan
                r[f"ct{h + 1}"] = float(ct[b, h]) if ok else np.nan
            rows.append(r)
        toks.append(tok.cpu().numpy().astype(np.float16))
    return rows, toks


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=["verif_m5b", "true_end_rq3", "verif_r1", "collect_v1"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument("--stride", nargs="*", type=int, default=[], help="per-dir file stride (subsample), default 1")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    rk = WMCriticRanker((REPO_ROOT / "data/wm/latest_wm.txt").read_text().strip(),
                        (REPO_ROOT / "data/wm/latest_critic_v2.txt").read_text().strip(),
                        "facebook/dinov2-small", 6, dev, 0.5, 1.0)
    rows, toks = [], []
    for di, name in enumerate(a.dirs):
        st = a.stride[di] if di < len(a.stride) else 1
        fs = sorted(glob.glob(str(REPO_ROOT / "data/wm" / name / "*.pt")))[::st][: a.max_files or None]
        for n, f in enumerate(fs):
            d = torch.load(f, weights_only=False)
            if d.get("slimmed") or "z" not in d:
                continue
            r, t = file_rows(rk, d, name, Path(f).stem)
            rows += r; toks += t
            if n % 16 == 0:
                log.info("%s %d/%d files, %d rows", name, n + 1, len(fs), len(rows))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "rows.parquet")
    np.save(out / "tok_err.npy", np.concatenate(toks))
    log.info("wrote %d rows -> %s", len(rows), out)


if __name__ == "__main__":
    main()
