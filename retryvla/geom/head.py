"""RGR observation-only geometry head: WM v1 latent (normalized DINOv2 tokens of both cams + proprio) -> geometry
vector g [G] (retryvla.geom.features.GEOM_KEYS), so E_geom can be evaluated on WM-IMAGINED states.

Train (TRAIN split only: collect_v1 bridges + privileged labels from scripts/geom_labels.py):
python -m retryvla.geom.head --labels data/geom/labels_collect_v1 [--steps 6000] [--imag 1]
  --imag 1: also train on WM-imagined latents of the same bridges with the TRUE labels (the head is applied to
            imagined states at search time). Validation = the WM's own held-out failure episodes (group_split
            val_frac 0.15 seed 0), reported separately on real and on imagined latents.
Output: runs/<ts>_RGR_geom_head/{geom_head.pt, metrics.json}; data/geom/latest_geom_head.txt -> checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from retryvla import REPO_ROOT
from retryvla.geom.features import BINARY, G, GEOM_KEYS, IDX, geom_energy

log = logging.getLogger("geom_head")
BIN_IDX = [IDX[k] for k in BINARY]
CONT_IDX = [i for i in range(G) if i not in BIN_IDX]


class GeomHead(nn.Module):
    def __init__(self, dim: int = 384, n_patch: int = 72, width: int = 256, depth: int = 2, heads: int = 4,
                 state_dim: int = 8):
        super().__init__()
        self.inp = nn.Linear(dim, width)
        self.q_proj = nn.Linear(state_dim, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.pos = nn.Parameter(torch.zeros(1, n_patch + 2, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.1, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, depth)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, G))
        self.register_buffer("y_mean", torch.zeros(G))
        self.register_buffer("y_std", torch.ones(G))

    def forward(self, z, q):
        """z [N,P,D], q [N,8] normalized -> raw outputs [N,G] (continuous standardized, binary logits)."""
        n = z.shape[0]
        x = torch.cat([self.cls.expand(n, -1, -1), self.q_proj(q)[:, None], self.inp(z)], 1) + self.pos
        return self.out(self.enc(x)[:, 0])

    @torch.no_grad()
    def predict(self, z, q) -> torch.Tensor:
        """-> g [N,G] in feature units (binary entries as probabilities)."""
        o = self.forward(z.float(), q.float()).float()
        g = o * self.y_std + self.y_mean
        g[:, BIN_IDX] = torch.sigmoid(o[:, BIN_IDX])
        return g


def load_head(path, dev) -> GeomHead:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    h = GeomHead(**ck["arch"])
    h.load_state_dict(ck["model"])
    return h.to(dev).eval()


def loss_fn(h: GeomHead, o, y):
    yc = (y - h.y_mean) / h.y_std
    lc = Fn.smooth_l1_loss(o[:, CONT_IDX], yc[:, CONT_IDX], beta=0.5)
    lb = Fn.binary_cross_entropy_with_logits(o[:, BIN_IDX], y[:, BIN_IDX].clamp(0, 1))
    return lc + lb


def auc(score, lab) -> float:
    lab = np.asarray(lab, bool)
    if lab.all() or (~lab).all():
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(score)
    n1 = lab.sum()
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * (~lab).sum()))


def metrics(gp: np.ndarray, gt: np.ndarray, g0p: np.ndarray, g0t: np.ndarray) -> dict:
    """gp/gt predicted/true geometry [N,G] of frames; g0p/g0t predicted/true of the bridge's frame 0 [N,G]."""
    out = {}
    for k in GEOM_KEYS:
        i = IDX[k]
        if k in BINARY:
            out[f"auc_{k}"] = auc(gp[:, i], gt[:, i] > 0.5)
            out[f"rate_{k}"] = float((gt[:, i] > 0.5).mean())
        else:
            out[f"mae_{k}"] = float(np.abs(gp[:, i] - gt[:, i]).mean())
    po = np.linalg.norm(gp[:, :3] - gt[:, :3], axis=1)
    out["obj_pos_err_cm_median"], out["obj_pos_err_cm_p90"] = float(np.median(po) * 100), float(np.quantile(po, .9) * 100)
    pg = np.linalg.norm(gp[:, 3:6] - gt[:, 3:6], axis=1)
    out["goal_pos_err_cm_median"] = float(np.median(pg) * 100)
    Ep, vp, _ = geom_energy(g0p, gp)
    Et, vt, _ = geom_energy(g0t, gt)
    out["E_corr"] = float(np.corrcoef(Ep, Et)[0, 1]) if Et.std() > 0 else float("nan")
    out["invalid_rate_true"] = float((~vt).mean())
    out["invalid_auc"] = auc(Ep, ~vt)
    out["valid_agree"] = float((vp == vt).mean())
    out["n"] = int(len(gt))
    return out


def load_data(labels_dir: str, wm_ckpt: str, dev):
    from retryvla.critic.train_critic import load_wm
    from retryvla.wm.data import LatentSet, concat_latents, group_split

    lab_files = sorted(glob.glob(str(REPO_ROOT / labels_dir / "*.pt")))
    have = {Path(f).name for f in lab_files}
    src = sorted(f for f in glob.glob(str(REPO_ROOT / "data/wm/collect_v1/*.pt")) if Path(f).name in have)

    sets, geoms = [], []
    for f in src:
        d = torch.load(f, weights_only=False)
        k = len(d["success"])
        sets.append(LatentSet(d["z"], d["q"].float(), d["valid"], d["actions"].float(), d["success"],
                              d["prefix_success"], [d["instance_id"]] * k, [d["task"]] * k, list(d["source"])))
        geoms.append(torch.load(REPO_ROOT / labels_dir / Path(f).name, weights_only=False)["geom"])
    ls = concat_latents(sets)
    geom = torch.cat(geoms).float()                     # [N, F, G]
    wm, norm, _ = load_wm(wm_ckpt, dev)
    tr, va = group_split(ls, 0.15, 0)
    return ls, geom, wm, norm, tr, va


@torch.no_grad()
def imagine(wm, norm, ls, idx, dev, bs=256):
    """WM-imagined normalized latents for frames 1..F-1 of bridges idx -> z [n, F-1, P, D] fp16, q [n, F-1, 8]."""
    Z, Q = [], []
    for i in range(0, len(idx), bs):
        j = torch.as_tensor(idx[i: i + bs])
        z0, q0 = norm.z(ls.z[j, 0].to(dev)), norm.q(ls.q[j, 0].to(dev))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            _, _, _, zs, qs = wm.rollout(z0, q0, ls.actions[j].to(dev))
        Z.append(torch.stack(zs, 1).half().cpu()); Q.append(torch.stack(qs, 1).float().cpu())
    return torch.cat(Z), torch.cat(Q)


def main() -> None:
    from retryvla.utils import setup_logging

    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/geom/labels_collect_v1")
    ap.add_argument("--wm_link", default="data/wm/latest_wm.txt")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--imag", type=int, default=1)
    ap.add_argument("--out_link", default="data/geom/latest_geom_head.txt")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    wm_ck = (REPO_ROOT / a.wm_link).read_text().strip()
    ls, geom, wm, norm, tr, va = load_data(a.labels, wm_ck, dev)
    F = ls.z.shape[1]
    log.info("bridges %d (train %d / val %d), frames %d", len(ls), len(tr), len(va), F)
    run = REPO_ROOT / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}_RGR_geom_head"
    run.mkdir(parents=True)
    # real frames: (bridge, frame) with valid latent; labels geom[b, f]
    V = ls.valid.bool()

    def frames(idx):
        b, f = torch.nonzero(V[torch.as_tensor(idx)], as_tuple=True)
        return torch.as_tensor(idx)[b], f

    trb, trf = frames(tr)
    vab, vaf = frames(va)
    zi_tr, qi_tr = imagine(wm, norm, ls, tr, dev) if a.imag else (None, None)
    zi_va, qi_va = imagine(wm, norm, ls, va, dev)
    head = GeomHead().to(dev)
    y_tr = geom[trb, trf]
    head.y_mean.copy_(y_tr.mean(0).to(dev)); head.y_std.copy_(y_tr.std(0).clamp_min(1e-3).to(dev))
    head.y_mean[BIN_IDX], head.y_std[BIN_IDX] = 0.0, 1.0
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    rng = np.random.default_rng(0)
    tr_pos = {int(b): k for k, b in enumerate(tr)}
    # imagined train frames: bridge k of tr, frame f in 1..F-1, valid
    if a.imag:
        ib, if_ = torch.nonzero(V[torch.as_tensor(tr)][:, 1:], as_tuple=True)
    t0 = time.time()
    for step in range(a.steps):
        head.train()
        n_r = a.bs // 2 if a.imag else a.bs
        s = rng.integers(len(trb), size=n_r)
        b, f = trb[s], trf[s]
        z = norm.z(ls.z[b, f].to(dev)); q = norm.q(ls.q[b, f].to(dev)); y = geom[b, f].to(dev)
        if a.imag:
            s2 = rng.integers(len(ib), size=a.bs - n_r)
            k2, f2 = ib[s2], if_[s2]
            z = torch.cat([z, zi_tr[k2, f2].to(dev).float()]); q = torch.cat([q, qi_tr[k2, f2].to(dev)])
            y = torch.cat([y, geom[torch.as_tensor(tr)[k2], f2 + 1].to(dev)])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            o = head(z, q).float()
        loss = loss_fn(head, o, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sch.step()
        if step % 500 == 0:
            log.info("step %d loss %.4f (%.0fs)", step, float(loss), time.time() - t0)
    head.eval()

    def pred(z, q, bs=1024):
        out = []
        for i in range(0, len(z), bs):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out.append(head.predict(z[i: i + bs].to(dev).float(), q[i: i + bs].to(dev).float()).cpu())
        return torch.cat(out).numpy()

    # val real frames
    zr, qr = norm.z(ls.z[vab, vaf].float()), norm.q(ls.q[vab, vaf])
    gp = pred(zr, qr)
    g0p_all = pred(norm.z(ls.z[torch.as_tensor(va), 0].float()), norm.q(ls.q[torch.as_tensor(va), 0]))
    pos = {int(b): k for k, b in enumerate(va)}
    kk = np.array([pos[int(b)] for b in vab])
    m_real = metrics(gp, geom[vab, vaf].numpy(), g0p_all[kk], geom[vab, 0].numpy())
    # val imagined frames (1..F-1, valid), and end-of-bridge (last valid frame) only
    vb, vf = torch.nonzero(V[torch.as_tensor(va)][:, 1:], as_tuple=True)
    gpi = pred(zi_va[vb, vf].float(), qi_va[vb, vf])
    gt_i = geom[torch.as_tensor(va)[vb], vf + 1].numpy()
    g0t_i = geom[torch.as_tensor(va)[vb], 0].numpy()
    m_imag = metrics(gpi, gt_i, g0p_all[vb.numpy()], g0t_i)
    last = (V[torch.as_tensor(va)].long().cumsum(1).argmax(1) - 1).clamp_min(0)
    endm = (vf == last[vb]).numpy()
    m_imag_end = metrics(gpi[endm], gt_i[endm], g0p_all[vb.numpy()][endm], g0t_i[endm])
    # copy-frame baseline for imagined: predict the frame-0 geometry (no motion)
    m_copy_end = metrics(g0p_all[vb.numpy()][endm], gt_i[endm], g0p_all[vb.numpy()][endm], g0t_i[endm])
    res = {"val_real": m_real, "val_imagined_all": m_imag, "val_imagined_end": m_imag_end,
           "val_copy_frame0_end": m_copy_end, "n_train_bridges": int(len(tr)), "n_val_bridges": int(len(va)),
           "wm_ckpt": wm_ck, "labels": a.labels, "args": vars(a),
           "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()}
    json.dump(res, open(run / "metrics.json", "w"), indent=1)
    ck = run / "geom_head.pt"
    torch.save({"model": head.state_dict(), "arch": {}, "keys": GEOM_KEYS, "metrics": res}, ck)
    (REPO_ROOT / a.out_link).parent.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / a.out_link).write_text(str(ck))
    for k, m in res.items():
        if isinstance(m, dict) and "n" in m:
            log.info("%s: obj pos err median %.2f cm | goal %.2f cm | mae d_grip_obj %.3f | mae upz %.3f | auc contact "
                     "%.3f | auc off %.3f | E corr %.3f | invalid AUC %.3f (rate %.3f) | n %d", k,
                     m["obj_pos_err_cm_median"], m["goal_pos_err_cm_median"], m["mae_d_grip_obj"], m["mae_obj_upz"],
                     m["auc_finger_obj_contact"], m["auc_off_table"], m["E_corr"], m["invalid_auc"],
                     m["invalid_rate_true"], m["n"])


if __name__ == "__main__":
    main()
