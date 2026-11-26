"""Shared helpers for the WM over-optimism visual study (scripts/wm_optimism/*)."""

from __future__ import annotations

import numpy as np

from scripts.wm_optimism.cache_stats import action_family

FAM_VI = {"prim": "primitive", "prim+cem": "primitive + CEM", "pi0_s1.0": "pi0 sample (noise 1.0)",
          "pi0_s1.5": "pi0 sample (noise 1.5)", "perturb": "perturbation"}


def select_cases(cache: dict, n_fp: int = 10, n_fpc: int = 2, n_tp: int = 3) -> list[dict]:
    """Deterministic gallery picks from gallery.pt: WM-caused false positives (C(imag) > 0.8, C(true) < 0.2, real
    fail; one per instance first, families diversified), critic-caused FPs (C(imag) > 0.8, C(true) > 0.7, fail) and
    true positives (C(imag) > 0.8, C(true) > 0.8, success), each from distinct instances where possible."""
    rows = []
    for stem, d in cache.items():
        fam, flips, rough, _ = action_family(d["actions"].numpy())
        for k in range(len(d["success"])):
            rows.append({"stem": stem, "k": k, "arm": d["source"][k], "fam": fam[k], "flips": int(flips[k]),
                         "c_wm": float(d["c_wm"][k]), "c_true": float(d["c_true"][k]),
                         "success": bool(d["success"][k])})
    out = []

    def take(pool, n, kind, key):
        used_inst = {r["stem"] for r in out if r["kind"] == kind}
        used_fam: dict[str, int] = {}
        pool = sorted(pool, key=key)
        for _ in range(n):
            best = None
            for r in pool:
                if any(r["stem"] == o["stem"] and r["k"] == o["k"] for o in out):
                    continue
                score = (r["stem"] in used_inst, used_fam.get(r["fam"], 0))
                if best is None or score < best[0]:
                    best = (score, r)
            if best is None:
                break
            r = dict(best[1], kind=kind)
            out.append(r)
            used_inst.add(r["stem"]); used_fam[r["fam"]] = used_fam.get(r["fam"], 0) + 1

    take([r for r in rows if r["c_wm"] > 0.8 and r["c_true"] < 0.2 and not r["success"]], n_fp, "FP_wm",
         lambda r: -(r["c_wm"] - r["c_true"]))
    take([r for r in rows if r["c_wm"] > 0.8 and r["c_true"] > 0.7 and not r["success"]], n_fpc, "FP_critic",
         lambda r: -r["c_true"])
    take([r for r in rows if r["c_wm"] > 0.8 and r["c_true"] > 0.8 and r["success"]], n_tp, "TP",
         lambda r: -r["c_wm"])
    return out


def nn_retrieve(cache: dict, stem: str, k: int, z_query, exclude_own: bool = True):
    """Nearest real frame (same scene bank: all candidates x all valid frames of this instance) to an imagined
    latent, by mean squared distance over the 72 normalized DINOv2 tokens. Returns (k', f', dist, own_end_dist,
    rank of the own true end frame)."""
    import torch

    d = cache[stem]
    zt = d["zt"].float()                                       # [K, 11, 72, 384]
    K, F = zt.shape[:2]
    valid = torch.zeros(K, F, dtype=torch.bool)
    for kk in range(K):
        valid[kk, : int(d["lv"][kk]) + 1] = True
    dist = ((zt - z_query.float()[None, None]) ** 2).mean((2, 3))   # [K, F]
    own = float(dist[k, int(d["lv"][k])])
    m = valid.clone()
    if exclude_own:
        m[k] = False
    dm = torch.where(m, dist, torch.full_like(dist, float("inf")))
    idx = int(torch.argmin(dm))
    kk, ff = divmod(idx, F)
    rank = int((dist[valid] < own).sum())
    return kk, ff, float(dm[kk, ff]), own, rank


def grid_err(d: dict, k: int) -> np.ndarray:
    """per-token squared error (normalized) of the imagined vs true END latent -> [2, 6, 6] (agent, wrist)."""
    lv = int(d["lv"][k])
    zi = d["zi"][k, max(lv - 1, 0)].float()
    zt = d["zt"][k, lv].float()
    e = ((zi - zt) ** 2).mean(-1).numpy()
    return e.reshape(2, 6, 6)
