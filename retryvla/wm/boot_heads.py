"""RQ3: a NON-degenerate ensemble for an existing latent WM (bootstrap last-layer heads on the frozen trunk).

python -m retryvla.wm.boot_heads [--wm data/wm/latest_wm.txt] [--data data/wm/collect_v1] [--n 4] [--out ...]

Why: every LatentWM checkpoint so far has 4 bit-identical ensemble heads (zero init + identical targets ->
identical gradients for ever), so U_WM == 0 and the `- lambda U` term of every WM search was a no-op
(found 2026-11-01, experiments/NOTES.md "RQ3"). Retraining the WM would also change the imagined states the
critic C^pi v2 was trained on, so instead we keep the WM (trunk + its own heads = the mean prediction used by
C^pi, unchanged) and fit E extra linear z / q heads on the frozen trunk features of the WM's OWN training
transitions (same data, same train / val group split), each on a Poisson(1) bootstrap of the training
trajectories, by closed-form ridge regression (exact; the trunk is frozen so each head is a linear model).
A member's rollout = trunk + boot head e fed back on its own trajectory (retryvla.search.scorers.member_rollout),
so the disagreement compounds over the horizon; U_ens = std_e C^pi(z_H^e).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch

from retryvla import REPO_ROOT
from retryvla.critic.train_critic import load_wm
from retryvla.utils import setup_logging
from retryvla.wm.data import group_split, load_latents

log = logging.getLogger("boot_heads")


def trunk_features(wm, z, q, a_chunk):
    """The LatentWM.step trunk output y [N, P+2, W] (heads read y[:, :P] for z and y[:, P] for q)."""
    x = torch.cat([wm.in_proj(z), wm.q_in(q)[:, None], wm.a_in(a_chunk.flatten(1))[:, None]], 1)
    return wm.trunk(x + wm.pos)


def load_boot(path: str | Path, dev) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return {k: (v.to(dev).float() if torch.is_tensor(v) else v) for k, v in ck.items()}


@torch.no_grad()
def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", default="data/wm/latest_wm.txt", help="WM checkpoint or a *.txt link to one")
    ap.add_argument("--data", nargs="*", default=None, help="latent roots (default: the WM's own cfg.data_roots)")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--ridge", type=float, default=1e-2, help="ridge = this x mean diag(X^T X)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm_path = Path(a.wm)
    wm_path = Path((REPO_ROOT / wm_path).read_text().strip()) if wm_path.suffix == ".txt" else REPO_ROOT / wm_path
    wm, norm, ck = load_wm(wm_path, dev)
    roots = a.data or list(ck["cfg"]["data_roots"])
    ls = load_latents(roots)
    tr, va = group_split(ls, float(ck["cfg"]["val_frac"]), int(ck["cfg"]["seed"]))
    k, P, E = wm.k, wm.P, a.n
    W = wm.in_proj.out_features
    g = torch.Generator().manual_seed(a.seed)
    wts = torch.poisson(torch.ones(E, len(ls)), generator=g)       # bootstrap multiplicity per trajectory
    wts[:, va] = 0.0                                                 # val failures stay unseen (as for the WM)
    log.info("WM %s | %d traj (train %d / val %d), %d members, P=%d, W=%d", wm_path, len(ls), len(tr), len(va), E, P, W)
    # sufficient statistics of E weighted ridge regressions (bias via an appended 1)
    Az = torch.zeros(E, W + 1, W + 1, device=dev, dtype=torch.float64)
    Bz = torch.zeros(E, W + 1, wm.dim, device=dev, dtype=torch.float64)
    Aq = torch.zeros(E, W + 1, W + 1, device=dev, dtype=torch.float64)
    Bq = torch.zeros(E, W + 1, 8, device=dev, dtype=torch.float64)
    idx = np.concatenate([tr, va])
    n_tr = 0
    for i in range(0, len(idx), a.bs):
        j = torch.as_tensor(idx[i: i + a.bs])
        z, q = norm.z(ls.z[j].to(dev)), norm.q(ls.q[j].to(dev))
        m, act = ls.valid[j].to(dev), ls.actions[j].to(dev)
        w_tr = wts[:, j].to(dev)
        if float(w_tr.sum()) == 0:
            continue
        for f in range(z.shape[1] - 1):
            tgt = (m[:, f + 1] & m[:, f])
            if not tgt.any():
                break
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                y = trunk_features(wm, z[:, f], q[:, f], act[:, f * k:(f + 1) * k]).float()
            yz = torch.cat([y[:, :P], torch.ones_like(y[:, :P, :1])], -1).double()      # [b, P, W+1]
            yq = torch.cat([y[:, P], torch.ones_like(y[:, P, :1])], -1).double()        # [b, W+1]
            dz = (z[:, f + 1] - z[:, f]).double()                                        # [b, P, D]
            dq = (q[:, f + 1] - q[:, f]).double()
            w = (w_tr * tgt[None].float()).double()                                      # [E, b]
            Az += torch.einsum("eb,bpi,bpj->eij", w, yz, yz)
            Bz += torch.einsum("eb,bpi,bpd->eid", w, yz, dz)
            Aq += torch.einsum("eb,bi,bj->eij", w, yq, yq)
            Bq += torch.einsum("eb,bi,bd->eid", w, yq, dq)
            n_tr += int(tgt.sum())
    heads = {}
    for name, A_, B_ in (("z", Az, Bz), ("q", Aq, Bq)):
        lam = a.ridge * A_.diagonal(dim1=1, dim2=2)[:, :-1].mean(1)                      # [E]
        reg = torch.eye(W + 1, device=dev, dtype=torch.float64)[None] * lam[:, None, None]
        reg[:, -1, -1] = 0.0                                                              # no penalty on the bias
        sol = torch.linalg.solve(A_ + reg, B_)                                           # [E, W+1, out]
        heads[f"{name}_W"], heads[f"{name}_b"] = sol[:, :-1].float().cpu(), sol[:, -1].float().cpu()
    # sanity: 1-step val error of each member vs the WM's own (identical) heads
    errs = {"wm": 0.0, **{e: 0.0 for e in range(E)}}
    nv = 0
    for i in range(0, min(len(va), 512), a.bs):
        j = torch.as_tensor(va[i: i + a.bs])
        z, q = norm.z(ls.z[j].to(dev)), norm.q(ls.q[j].to(dev))
        m, act = ls.valid[j].to(dev), ls.actions[j].to(dev)
        tgt = m[:, 1] & m[:, 0]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            y = trunk_features(wm, z[:, 0], q[:, 0], act[:, :k]).float()
            z1, _, _, _ = wm.step(z[:, 0], q[:, 0], act[:, :k])
        errs["wm"] += float(((z1.float() - z[:, 1]) ** 2).mean((1, 2))[tgt].sum())
        for e in range(E):
            ze = z[:, 0] + y[:, :P] @ heads["z_W"][e].to(dev) + heads["z_b"][e].to(dev)
            errs[e] += float(((ze - z[:, 1]) ** 2).mean((1, 2))[tgt].sum())
        nv += int(tgt.sum())
    errs = {str(kk): v / max(nv, 1) for kk, v in errs.items()}
    log.info("val 1-step z mse (n=%d): %s", nv, {kk: round(v, 4) for kk, v in errs.items()})
    out = Path(a.out) if a.out else REPO_ROOT / "data/wm" / f"boot_heads_{wm_path.parent.name}.pt"
    tmp = out.with_suffix(".tmp")
    torch.save({**heads, "wm": str(wm_path), "n": E, "ridge": a.ridge, "seed": a.seed, "roots": roots,
                "n_train_transitions": n_tr, "val_1step_mse": errs}, tmp)
    os.replace(tmp, out)
    log.info("saved %s", out)


if __name__ == "__main__":
    main()
