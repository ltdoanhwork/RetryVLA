"""P2 data for the policy-agnostic value critics (B3 progress, B4 generic task success).

Every sample = (latent state, label, task) in the SAME token space as the WM / competence critic C^pi
(raw DINOv2 tokens of both cams [P=72, D=384] + 8-d state; the WM normalizer is applied by the trainer), with an
optional WM-IMAGINED version of the same state (open-loop WM rollout of the actions that led there), so that every
critic is trained on real + imagined inputs like C^pi (train_critic.py, imagined_frac).

Sources
  demo      nominal LIBERO-Spatial demos (scripts/build_demo_latents.py). Frame t of an episode of length T.
            real: latent at t. imagined: WM rollout from frame t-h with the demo actions t-h..t-1 (h = 50, or the
            largest multiple of the WM step that fits).
  fail      failure states = frame 0 of every LIBERO-Recover train instance in collect_v1 / verif_r1 (benchmark
            failure states; no continuation outcome is used). real only.
  segment   end of a 50-step candidate segment from a failure state (collect / verification logs): real end latent
            (last valid frame) and the WM-imagined end latent of the same actions (as in train_critic.end_latents).
            label = task success of segment + continuation under the policy that produced the log.
"""

from __future__ import annotations

import glob
import logging
import re
import zlib
from dataclasses import dataclass, field

import numpy as np
import torch

from retryvla import REPO_ROOT

log = logging.getLogger("value_data")


@dataclass
class Samples:
    zr: list = field(default_factory=list)      # real latent [P, D] fp16 (raw)
    qr: list = field(default_factory=list)      # [8]
    zi: list = field(default_factory=list)      # imagined latent (normalized space) or None
    qi: list = field(default_factory=list)
    y: list = field(default_factory=list)       # float label
    task: list = field(default_factory=list)
    group: list = field(default_factory=list)   # leakage unit (failure episode / demo episode)
    src: list = field(default_factory=list)     # source name (for mixture weights + per-source metrics)
    inst: list = field(default_factory=list)
    extra: list = field(default_factory=list)   # e.g. prefix_success flag

    def add(self, zr, qr, zi, qi, y, task, group, src, inst, extra=0):
        self.zr.append(zr); self.qr.append(qr); self.zi.append(zi); self.qi.append(qi); self.y.append(float(y))
        self.task.append(task); self.group.append(group); self.src.append(src); self.inst.append(inst)
        self.extra.append(int(extra))

    def __len__(self):
        return len(self.y)


def trial_of(iid: str) -> int:
    return int(re.search(r"(\d+)$", iid).group(1))


def group_of(iid: str) -> str:
    p = iid.split("__")
    return f"{p[0]}__{p[1]}"


def load_segment_root(root: str, wm, norm, dev, keep_trials=None, exclude: set | None = None, H: int = 50,
                      need_imagined: bool = True, frame0_only: bool = False):
    """Yield per-file dicts with real end latents, imagined end latents (normalized), labels.
    keep_trials: None = all, else set of trial indices to keep. exclude: instance ids to drop.
    Candidates whose action length != H (e.g. M7 two-stage 2H segments) are skipped: their latents stop at H."""
    files = sorted(glob.glob(str(REPO_ROOT / root / "*.pt")))
    n_skip = 0
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        iid = d["instance_id"]
        if exclude and iid in exclude:
            n_skip += 1
            continue
        if keep_trials is not None and trial_of(iid) not in keep_trials:
            n_skip += 1
            continue
        if frame0_only:
            yield {"iid": iid, "task": d["task"], "z0": d["z"][0, 0], "q0": d["q"][0, 0].float()}
            continue
        acts = d["actions"]
        if isinstance(acts, list):
            ok = [i for i, a in enumerate(acts) if a.shape[0] == H]
            A = torch.stack([acts[i] for i in ok]).float() if ok else None
        else:
            ok = list(range(len(acts))) if acts.shape[1] == H else []
            A = acts.float() if ok else None
        if not ok:
            continue
        ok_t = torch.as_tensor(ok)
        z, q, v = d["z"][ok_t], d["q"][ok_t].float(), d["valid"][ok_t]
        lv = v.long().cumsum(1).argmax(1)
        ar = torch.arange(len(ok))
        out = {"iid": iid, "task": d["task"], "zr": z[ar, lv], "qr": q[ar, lv],
               "y": d["success"][ok_t].float(), "prefix": d["prefix_success"][ok_t].bool(), "z0": z[0, 0], "q0": q[0, 0]}
        if need_imagined:
            out["zi"], out["qi"] = imagine(wm, norm, z[:, 0], q[:, 0], A, lv, dev)
        yield out
    log.info("%s: skipped %d instances (excluded / trial filter)", root, n_skip)


@torch.no_grad()
def imagine(wm, norm, z0, q0, A, lv, dev, bs: int = 256):
    """WM-imagined latent at the same horizon as the real end latent (terminal successes end early)."""
    zi, qi = [], []
    for i in range(0, len(A), bs):
        j = slice(i, i + bs)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            _, _, _, zs, qs = wm.rollout(norm.z(z0[j].to(dev)), norm.q(q0[j].to(dev)), A[j].to(dev))
        Zs, Qs = torch.stack(zs, 1), torch.stack(qs, 1)
        hi = (lv[j] - 1).clamp_min(0).to(dev)
        ar = torch.arange(len(hi), device=dev)
        zi.append(Zs[ar, hi].float().half().cpu()); qi.append(Qs[ar, hi].float().cpu())
    return torch.cat(zi), torch.cat(qi)


@torch.no_grad()
def demo_samples(path: str, wm, norm, dev, label: str, horizon: int = 50):
    """Demo frames. label='progress' -> t / (T-1); 'success' -> 1 (the expert completes the task from every
    demo state). Imagined version: WM rollout of the demo actions over the h steps that led to frame t."""
    D = torch.load(REPO_ROOT / path, weights_only=False)
    k = wm.k
    out = []
    for ep in D["episodes"]:
        T, ts, z, q, A = ep["T"], ep["t"].tolist(), ep["z"], ep["q"].float(), ep["actions"]
        pos = {t: i for i, t in enumerate(ts)}
        z0s, q0s, aa, tgt = [], [], [], []
        for i, t in enumerate(ts):
            h = min(horizon, (t // k) * k)
            if h >= k and (t - h) in pos:
                z0s.append(z[pos[t - h]]); q0s.append(q[pos[t - h]])
                a = A[t - h: t]
                aa.append(torch.cat([a, a[-1:].expand(horizon - h, -1)]) if h < horizon else a)  # pad (cut below)
                tgt.append((i, h))
        zi = [None] * len(ts)
        qi = [None] * len(ts)
        if tgt:
            # group by horizon so every rollout uses exactly h steps
            for h in sorted({h for _, h in tgt}):
                sel = [n for n, (_, hh) in enumerate(tgt) if hh == h]
                Z0 = torch.stack([z0s[n] for n in sel]).to(dev)
                Q0 = torch.stack([q0s[n] for n in sel]).to(dev)
                AA = torch.stack([aa[n][:h] for n in sel]).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    zH, qH, _, _, _ = wm.rollout(norm.z(Z0), norm.q(Q0), AA)
                for m, n in enumerate(sel):
                    zi[tgt[n][0]] = zH[m].float().half().cpu(); qi[tgt[n][0]] = qH[m].float().cpu()
        for i, t in enumerate(ts):
            y = t / max(1, T - 1) if label == "progress" else 1.0
            out.append((z[i], q[i], zi[i], qi[i], y, ep["task"], f"demo_ep{ep['episode']}", ep["episode"]))
    return out


def demo_is_val(ep: int, frac: float, seed: int = 0) -> bool:
    return (zlib.crc32(f"{seed}:demo{ep}".encode()) % 1000) / 1000 < frac


def eval_instances() -> set[str]:
    """Instance ids of the M5b / M9 search instances (the P2 evaluation set and its siblings): never trained on."""
    import pandas as pd

    fs = glob.glob(str(REPO_ROOT / "runs" / "*_M5_m5b_compare_arms_s*" / "per_instance.jsonl")) + \
        glob.glob(str(REPO_ROOT / "runs" / "*_M9_mech_search_s*" / "per_instance.jsonl"))
    return set(pd.concat([pd.read_json(f, lines=True) for f in fs]).instance_id) if fs else set()
