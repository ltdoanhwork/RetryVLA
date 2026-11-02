#!/usr/bin/env python
"""Re-simulate recorded recovery bridges (open-loop 50-step prefixes, NO policy continuation) from their exact
failure-state snapshots, to recover what the logs do not store: the TRUE latent trajectory (slim RQ3 logs) and/or
the raw camera frames (for a foundation-WM encoder such as V-JEPA 2).

python -m scripts.replay_bridges --src data/wm/verif_rq3_sweep --cfg configs/exp/rq3_verifier_sweep.yaml \
    --kind m5b --out data/wm/true_end_rq3 [--frames_out data/foundation/frames/rq3] [--workers 24]
    [--shard i --num_shards n] [--check]

--kind m5b     instances = compare_arms.select_jobs over ALL m5b shards (train split, trials 2..9), restored with
               0 policy steps (exactly the compare_arms precheck path for known pi0 failures).
--kind collect instances = run_search.build_instances on the collect config's scenes (train split, hard groups
               excluded), precheck_rollouts 0 (exactly retryvla.wm.collect).
Per source file one output file with the source's keys (actions, success, source, ...) plus z [K, 11, 72, 384] fp16
(DINOv2-small 6x6 x 2 cams, frames every 5 env steps; `--no_dino` skips it), q [K, 11, 8], valid [K, 11].
--frames_out: agent-view uint8 frames [K, F, 256, 256, 3] at `--frame_stride` (+ frame-0 wrist) as .npz (compressed).
--check: compare re-encoded z with the source's stored z (full logs only) -> determinism / renderer check.
The bridge is open-loop and restore_snapshot is bit-exact, so the end state does not depend on continuation seeds.
Physics is identical on any node; rendering on a CPU-only shell is Mesa (software) instead of NVIDIA EGL ->
use --check on a full log to quantify the latent difference before relying on it.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch

from retryvla import REPO_ROOT
from retryvla.config import load_config
from retryvla.envs.splits import assert_allowed, load_split, scenes_for
from retryvla.envs.vec import SimPool
from retryvla.eval.rollout import Job, run_jobs_resilient
from retryvla.utils import setup_logging

log = logging.getLogger("replay_bridges")


def m5b_instances(pool, cfg, ids: set[str]) -> dict[str, tuple]:
    from retryvla.search.compare_arms import iid, select_jobs

    split = load_split(REPO_ROOT / cfg.split.file)
    scenes = scenes_for(split, "train", REPO_ROOT / cfg.recover.scene_root)
    assert_allowed([s.tag for s in scenes], split)
    mb = cfg.m5b
    jobs = []
    for sh in range(int(mb.num_shards)):
        mb.shard = sh
        jobs += select_jobs(scenes, cfg, mb)
    todo = []
    for j in jobs:
        if iid(j.meta) in ids:
            jj = Job(**{**j.__dict__})
            jj.return_snapshot, jj.max_steps = True, 0
            todo.append(jj)
    res = run_jobs_resilient(pool, None, todo, retries=2, desc="restore")
    return {iid(r.meta): (todo[k].load_kwargs["bddl"], r.load_out["snapshot"]) for k, r in enumerate(res)}


def collect_instances(pool, cfg, ids: set[str], exclude: bool = True) -> dict[str, tuple]:
    from retryvla.search.run_search import build_instances
    from retryvla.wm.collect import group_of

    wc = cfg.wm_collect
    split = load_split(REPO_ROOT / cfg.split.file)
    scenes = scenes_for(split, wc.split, REPO_ROOT / cfg.recover.scene_root)
    assert_allowed([s.tag for s in scenes], split)
    if exclude and wc.get("exclude_instances_file"):
        ex = {group_of(l.strip()) for l in open(REPO_ROOT / wc.exclude_instances_file) if l.strip()}
        scenes = [s for s in scenes if s.group not in ex]
    keep_tags = {i.rsplit("__", 1)[0] for i in ids}
    scenes = [s for s in scenes if s.tag in keep_tags]
    cfg.search_on.precheck_rollouts = 0
    insts, _ = build_instances(pool, None, scenes, cfg, wc.split)
    return {i.instance_id: (i.bddl, i.snapshot) for i in insts if i.instance_id in ids}


def jpeg(img: np.ndarray, q: int = 95) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img[..., ::-1]), [int(cv2.IMWRITE_JPEG_QUALITY), q])
    assert ok
    return buf.tobytes()


def frames_of(rec: dict, horizon: int, stride: int, key: str = "image") -> tuple[list, np.ndarray]:
    """frames of camera `key` at t = 0, stride, ..., horizon (same rule as wm.collect.encode_record: an early end
    contributes the terminal observation once, later frames are invalid copies) + valid."""
    img_t = list(rec["img_t"])
    F = horizon // stride + 1
    out, valid = [], []
    ended = False
    for f in range(F):
        t = f * stride
        if t in img_t and not ended:
            out.append(rec[key][img_t.index(t)]); valid.append(True)
        elif not ended and t >= len(rec["action"]) and f"final_{key}" in rec:
            out.append(rec[f"final_{key}"]); valid.append(True); ended = True
        else:
            out.append(out[-1]); valid.append(False); ended = True
    return out, np.asarray(valid)


def frames_state(rec: dict, horizon: int, stride: int) -> np.ndarray:
    """robot state [F, 8] aligned with frames_of (terminal state after an early end, then repeated)."""
    F = horizon // stride + 1
    n = len(rec["action"])
    out = []
    for f in range(F):
        t = f * stride
        out.append(rec["state"][t] if t < n else rec.get("final_state", rec["state"][-1]))
    return np.stack(out).astype(np.float32)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--kind", choices=["m5b", "collect"], required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--frames_out", default="")
    ap.add_argument("--frame_stride", type=int, default=5)
    ap.add_argument("--frame_cams", choices=["agent", "both"], default="both")
    ap.add_argument("--frame_ends", action="store_true", help="store only the first + last valid frame")
    ap.add_argument("--no_exclude", action="store_true", help="--kind collect: do not drop the hard groups "
                    "(verif_r1 = pilot instances)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_files", type=int, default=0)
    ap.add_argument("--max_cands", type=int, default=0, help="only the first N candidates per file (smoke/check)")
    ap.add_argument("--sources", nargs="*", default=[], help="only candidates whose logged source/arm is listed")
    ap.add_argument("--no_dino", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = load_config(a.cfg)
    H, stride = 50, 5
    files = sorted(glob.glob(str(REPO_ROOT / a.src / "*.pt")))[a.shard:: a.num_shards][: a.max_files or None]
    out = REPO_ROOT / a.out if a.out else None
    fout = REPO_ROOT / a.frames_out if a.frames_out else None
    for p in (out, fout):
        if p is not None:
            p.mkdir(parents=True, exist_ok=True)

    def done(f: str) -> bool:
        n = Path(f).stem
        return ((out is None or (out / f"{n}.pt").exists()) and (fout is None or (fout / f"{n}.pt").exists()))

    files = [f for f in files if not done(f)]
    log.info("%d source files to replay (%s, shard %d/%d)", len(files), a.src, a.shard, a.num_shards)
    if not files:
        return
    enc = None
    if out is not None and not a.no_dino:
        from retryvla.wm.encoder_dino import DinoEncoder
        from retryvla.wm.collect import encode_record

        torch.set_num_threads(max(1, min(8, (os.cpu_count() or 8) // 4)))
        enc = DinoEncoder("facebook/dinov2-small", pool_grid=6, device=a.device)
    heads = {}
    for f in files:
        d = torch.load(f, weights_only=False, mmap=True)
        heads[f] = d["instance_id"]
    t0 = time.time()
    with SimPool(int(a.workers), int(cfg.env.resolution)) as pool:
        if a.kind == "m5b":
            snaps = m5b_instances(pool, cfg, set(heads.values()))
        else:
            snaps = collect_instances(pool, cfg, set(heads.values()), exclude=not a.no_exclude)
        log.info("restored %d / %d instances (%.0fs)", len(snaps), len(set(heads.values())), time.time() - t0)
        for n, f in enumerate(files):
            d = torch.load(f, weights_only=False)
            i = d["instance_id"]
            if i not in snaps:
                log.warning("no snapshot for %s, skipped", i)
                continue
            bddl, snap = snaps[i]
            nC = len(d["success"])
            sel = [k for k in range(nC) if not a.sources or d["source"][k] in a.sources][: a.max_cands or None]
            if len(sel) < nC:  # candidate subset (--sources / --max_cands): index every per-candidate field
                ix = torch.as_tensor(sel, dtype=torch.long)
                d = {k: (v[ix] if torch.is_tensor(v) and v.ndim and len(v) == nC else
                         [v[j] for j in sel] if isinstance(v, list) and len(v) == nC else v) for k, v in d.items()}
            acts = d["actions"]
            acts = acts if not isinstance(acts, list) else torch.stack([x[:H] for x in acts])
            K = len(acts)
            jobs = [Job("restore_snapshot", dict(bddl=bddl, snap=snap), task=d["task"], max_steps=H,
                        prefix=acts[k, :H].numpy().astype(np.float32), record="all",
                        record_cfg={"stride": min(stride, a.frame_stride), "img_until": H}) for k in range(K)]
            res = run_jobs_resilient(pool, None, jobs, retries=2, desc="bridges")
            stem = Path(f).stem
            if out is not None:
                o = {k: v for k, v in d.items() if k not in ("z", "q", "valid", "z0", "q0", "slimmed")}
                o["prefix_success_replay"] = torch.as_tensor([r.prefix_success for r in res])
                if enc is not None:
                    e = [encode_record(enc, r.record, H, stride) for r in res]
                    o["z"] = torch.stack([x["z"] for x in e])
                    o["q"] = torch.as_tensor(np.stack([x["q"] for x in e]))
                    o["valid"] = torch.as_tensor(np.stack([x["valid"] for x in e]))
                    if a.check and "z" in d:
                        zs = d["z"][:K].float()
                        m = o["valid"] & d["valid"][:K].bool()
                        diff = ((o["z"].float() - zs) ** 2).mean((2, 3))[m]
                        scale = (zs ** 2).mean((2, 3))[m]
                        qd = (o["q"] - d["q"][:K].float()).abs().amax(-1)[m]
                        log.info("CHECK %s: latent rel MSE mean %.2e max %.2e | state max|dq| %.2e | prefix_success "
                                 "agree %.3f", i[-40:], float((diff / scale).mean()), float((diff / scale).max()),
                                 float(qd.max()), float((o["prefix_success_replay"] == d["prefix_success"][:K]).float().mean()))
                tmp = out / f"{stem}.pt.tmp"
                torch.save(o, tmp)
                os.replace(tmp, out / f"{stem}.pt")
            if fout is not None:
                # JPEG q95 bytes: frames[k][f] (agent view; + wrist if --frame_cams both), frames every frame_stride
                # env steps; --frame_ends keeps only the first and the last valid frame (eval sets)
                fr, va, fw = [], [], []
                for r in res:
                    x, v = frames_of(r.record, H, a.frame_stride, "image")
                    lv = int(np.nonzero(v)[0].max())
                    keep = [0, lv] if a.frame_ends else list(range(len(x)))
                    fr.append([jpeg(x[t]) for t in keep]); va.append(v)
                    if a.frame_cams == "both":
                        xw, _ = frames_of(r.record, H, a.frame_stride, "image2")
                        fw.append([jpeg(xw[t]) for t in keep])
                tmp = fout / f"{stem}.pt.tmp"
                torch.save({"instance_id": i, "task": d["task"], "frame_stride": a.frame_stride,
                            "frame_ends": bool(a.frame_ends), "agent": fr, "wrist": fw or None,
                            "valid": torch.as_tensor(np.stack(va)),
                            "state": torch.as_tensor(np.stack([frames_state(r.record, H, a.frame_stride) for r in res])),
                            "actions": acts[:K, :H].clone(), "success": d["success"][:K].clone(),
                            "prefix_success": d["prefix_success"][:K].clone(), "source": list(d["source"])[:K]}, tmp)
                os.replace(tmp, fout / f"{stem}.pt")
            for r in res:
                r.record = None
            log.info("[%d/%d] %s: %d bridges | %.0fs", n + 1, len(files), i[-45:], K, time.time() - t0)


if __name__ == "__main__":
    main()
