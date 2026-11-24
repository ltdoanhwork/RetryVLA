"""pi0 LoRA SFT for the 2610.05151 reverse-direction track (their recipe; see retryvla/policy/pi0_lora.py for D2).

torchrun --nproc_per_node 2 -m retryvla.train.pi0_sft --out runs/<id> --steps 5000 --save_every 1000 \
    [--init <ckpt>] [--recovery data/...] [--warmup 1000 --decay_steps 30000] [--seed 7] [--bs 48]

- Data: D0 = data/goal_rsi/d0_train (300 train demos replayed in our sim, 128 px). Windows = every frame as anchor,
  50-step action chunk, repeated-tail padding at the episode end (lerobot clamping = openpi / their baseline).
- --recovery: extra LeRobotDatasets pooled with D0 and shuffled uniformly (their mix: no oversampling), optional
  --recovery_segments to keep only the recovery segment (segment 0) as anchors.
- Optimiser = theirs: AdamW betas (0.9, 0.95), eps 1e-8, wd 1e-10, clip 1, peak 2.5e-5, linear warmup then cosine to
  2.5e-6 over decay_steps (openpi CosineDecaySchedule). M0: warmup 1000 / decay 30000 (openpi defaults, the paper
  gives none for M0); continuation: warmup 200 / decay 2000 (paper). No EMA. Normalisation stats from D0 only.
- Global batch --bs split over ranks; bf16 autocast; gradient checkpointing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from retryvla import REPO_ROOT
from retryvla.policy import pi0_lora as P

D0 = "data/goal_rsi/d0_train"


def lr_at(step, peak, warmup, decay_steps, decay_lr):
    if step < warmup:
        return peak * (step + 1) / (warmup + 1)
    prog = min(1.0, (step - warmup) / max(1, decay_steps - warmup))
    return decay_lr + (peak - decay_lr) * 0.5 * (1 + math.cos(math.pi * prog))


class Pool(torch.utils.data.Dataset):
    """Concatenation of LeRobotDatasets; optional anchor filter on the `segment` column for recovery sets."""

    def __init__(self, roots, chunk, seg_filter=None, max_traj: int = 0, seed: int = 0, groups=None, caps=None):
        """max_traj > 0: keep at most that many recovery trajectories over ALL recovery roots, round-robin over
        instances (= train.subset_mode roundrobin / match_trajectory_count_of of the SmolVLA recipe)."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.ds, self.idx = [], []
        keep_eps = None
        if max_traj:
            per_inst = {}
            rng = np.random.default_rng(seed)
            for k, r in enumerate(roots[1:], 1):
                tj = Path(r) / "retryvla/trajectories.jsonl"
                if tj.exists():
                    for l in open(tj):
                        t = json.loads(l)
                        per_inst.setdefault(t["instance_id"], []).append((k, int(t["episode_index"])))
                else:   # plain LeRobotDataset (diagnostic D4: D0 itself as the "recovery" root): one instance per episode
                    n_ep = json.loads((Path(r) / "meta/info.json").read_text())["total_episodes"]
                    for e in range(n_ep):
                        per_inst.setdefault(f"{k}:{e}", []).append((k, e))
            def rr(pi, cap):
                insts = sorted(pi)
                rng.shuffle(insts)
                out, rnd = [], 0
                while len(out) < cap and any(len(v) > rnd for v in pi.values()):
                    for i in insts:
                        if len(pi[i]) > rnd and len(out) < cap:
                            out.append(pi[i][rnd])
                    rnd += 1
                return out
            if caps:   # per-group caps ("glob@N" in --recovery): round-robin inside each group separately
                sel = []
                for gname, cap in caps.items():
                    ks = {k for k in range(1, len(roots)) if groups[k - 1] == gname}
                    sel += rr({i: [x for x in v if x[0] in ks] for i, v in per_inst.items()
                               if any(x[0] in ks for x in v)}, cap)
            else:
                sel = rr(per_inst, max_traj)
            keep_eps = set(sel)
            self.n_traj = len(sel)
            self.n_traj_by_group = {g: sum(1 for k, _ in sel if groups and groups[k - 1] == g) for g in (caps or {})}
        for k, r in enumerate(roots):
            meta_fps = json.loads((Path(r) / "meta/info.json").read_text())["fps"]
            d = LeRobotDataset(f"local/{Path(r).name}", root=r,
                               delta_timestamps={"action": [i / meta_fps for i in range(chunk)]})
            keep = np.arange(len(d))
            if keep_eps is not None and k > 0:
                ep = np.asarray(d.hf_dataset.with_format(None).select_columns(["episode_index"])["episode_index"]).reshape(-1)
                keep = np.nonzero(np.isin(ep, [e for kk, e in keep_eps if kk == k]))[0]
            if seg_filter is not None and k > 0 and "segment" in d.features:
                seg = np.asarray(d.hf_dataset.with_format(None).select_columns(["segment"])["segment"]).reshape(-1)
                keep = keep[np.isin(seg[keep], seg_filter)]
            self.ds.append(d)
            self.idx += [(len(self.ds) - 1, int(i)) for i in keep]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        k, j = self.idx[i]
        it = self.ds[k][j]
        out = {"observation.images.image": it["observation.images.image"],
               "observation.images.image2": it["observation.images.image2"],
               "observation.state": it["observation.state"], "action": it["action"], "task": it["task"],
               "src": k}
        if getattr(self, "masked", False):   # LRS-G (opt-in): validity mask + quality weight per anchor
            T = it["action"].shape[0]
            if k > 0 and "q" in it:
                out["valid"] = (~it["action_is_pad"].bool()).float()
                out["w"] = torch.as_tensor(float(it["q"].reshape(-1)[0]), dtype=torch.float32)
            else:   # D0: their baseline (repeated-tail padding, no mask), weight 1
                out["valid"] = torch.ones(T)
                out["w"] = torch.tensor(1.0)
        return out

    def lmin_filter(self, l_min: int) -> int:
        """LRS-G: keep recovery anchors with >= l_min valid targets (their L_min = 5). Returns #dropped."""
        keep, drop = [], 0
        cache = {}
        for k, j in self.idx:
            if k == 0 or "run_end" not in self.ds[k].features:
                keep.append((k, j))
                continue
            if k not in cache:
                h = self.ds[k].hf_dataset.with_format(None).select_columns(["frame_index", "run_end"])
                cache[k] = (np.asarray(h["frame_index"]).reshape(-1), np.asarray(h["run_end"]).reshape(-1))
            fi, re_ = cache[k]
            if re_[j] - fi[j] + 1 >= l_min:
                keep.append((k, j))
            else:
                drop += 1
        self.idx = keep
        return drop


class MaskedLoss(torch.nn.Module):
    """LRS-G quality-weighted, validity-masked flow matching (D-L7: loss mask only, no attention / noisy-token mask).
    loss = sum_b w_b sum_t m_bt l_bt / sum_b sum_t m_bt, l_bt = mean over the 7 action dims of the pi0 FM loss."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch, valid, w):
        p = self.policy
        images, img_masks = p._preprocess_images(batch)
        lang_tokens, lang_masks = batch["observation.language.tokens"], batch["observation.language.attention_mask"]
        losses = p.model.forward(images, img_masks, lang_tokens, lang_masks, p.prepare_state(batch),
                                 p.prepare_action(batch))
        losses = losses[:, :, : p.config.output_features["action"].shape[0]].mean(-1)   # [B, T]
        m = valid[:, : losses.shape[1]].to(losses)
        return (losses * m * w.to(losses)[:, None]).sum() / m.sum().clamp_min(1.0)


def save_resume(rp: Path, step: int, opt, policy) -> None:
    """Quota-safe resume state (2026-11-22): trainable params + AdamW moments in bf16 (~2.7 GB instead of 5.3 GB),
    ONE copy. The old copy is deleted BEFORE writing (the shared /pfss quota cannot hold two). A failed write is logged
    and skipped: training continues, and the next resume point retries. Moments are cast back to fp32 on load (logged
    deviation: a resumed run continues from bf16-rounded optimizer state)."""
    sd = opt.state_dict()
    st = {k: {kk: (vv.to(torch.bfloat16) if torch.is_tensor(vv) and vv.is_floating_point() and vv.ndim else vv)
              for kk, vv in v.items()} for k, v in sd["state"].items()}
    obj = {"step": step, "opt": {"state": st, "param_groups": sd["param_groups"]}, "compact": True,
           "trainable": {k: v.detach().to(torch.bfloat16).cpu() for k, v in policy.named_parameters() if v.requires_grad}}
    tmp = Path(str(rp) + ".tmp")
    try:
        rp.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
        torch.save(obj, tmp)
        tmp.replace(rp)
        print(f"resume state saved at step {step}", flush=True)
    except (OSError, RuntimeError) as e:
        tmp.unlink(missing_ok=True)
        print(f"WARNING resume save failed at step {step} ({str(e)[:100]}); continuing without a resume point", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None)
    ap.add_argument("--recovery", nargs="*", default=[])
    ap.add_argument("--recovery_segments", nargs="*", type=int, default=None)
    ap.add_argument("--max_traj", type=int, default=0)
    ap.add_argument("--stop_at", type=int, default=0,
                    help="diagnostic: exit right after the resume save at this step (forces a later resume)")
    ap.add_argument("--save_dtype", default="fp32", choices=["fp32", "bf16"],
                    help="dtype of the final trainable.pt (bf16 halves it: 0.9 GB; /pfss quota, F1 2026-11-22)")
    ap.add_argument("--resume_every", type=int, default=0,
                    help="> 0 (opt-in, single process): save trainable + AdamW state every N steps to <out>/resume.pt "
                         "and continue from it on restart (back-off kills on borrowed GPUs)")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--peak", type=float, default=2.5e-5)
    ap.add_argument("--decay_lr", type=float, default=2.5e-6)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--decay_steps", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation micro-steps (global bs unchanged)")
    ap.add_argument("--masked_recovery", action="store_true",
                    help="LRS-G (opt-in): recovery anchors get a validity mask (targets past the window end) and "
                         "their per-frame quality weight q; D0 unchanged")
    ap.add_argument("--l_min", type=int, default=5, help="LRS-G: min valid targets per recovery anchor")
    a = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(a.seed + rank)
    out = REPO_ROOT / a.out
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(a), indent=1))

    policy, cfg, info = P.load_policy(a.init, device=str(dev), for_training=True)
    policy.config.gradient_checkpointing = True
    if hasattr(policy.model, "gradient_checkpointing_enable"):
        policy.model.gradient_checkpointing_enable()
    if rank == 0:
        print("params", info, flush=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors

    stats = LeRobotDatasetMetadata("local/d0_train", root=REPO_ROOT / D0).stats
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats,
                                         preprocessor_overrides={"device_processor": {"device": str(dev)}})
    import glob as _g
    rec_roots, groups, caps = [], [], {}
    for r in a.recovery:   # "glob" or "glob@N" (per-group trajectory cap, e.g. P2-3 B = curated D_R@36 + D_P@n)
        gl, _, cap = r.partition("@")
        for p_ in sorted(_g.glob(str(REPO_ROOT / gl))):
            if p_ not in rec_roots:
                rec_roots.append(p_)
                groups.append(gl)
        if cap:
            caps[gl] = int(cap)
    data = Pool([REPO_ROOT / D0] + rec_roots, cfg.chunk_size, a.recovery_segments,
                a.max_traj or (sum(caps.values()) if caps else 0), a.seed, groups, caps or None)
    if a.masked_recovery:
        data.masked = True
        nd = data.lmin_filter(a.l_min)
        if rank == 0:
            print(f"masked_recovery: dropped {nd} recovery anchors with < {a.l_min} valid targets", flush=True)
    n_src = np.bincount([k for k, _ in data.idx])
    if rank == 0:
        print("anchors per source", n_src.tolist(), flush=True)
    g = torch.Generator().manual_seed(a.seed * 1000 + rank)
    sampler = torch.utils.data.RandomSampler(data, replacement=True, num_samples=a.steps * a.bs // world + a.bs,
                                             generator=g)
    dl = torch.utils.data.DataLoader(data, batch_size=a.bs // world // a.accum, sampler=sampler, num_workers=a.workers,
                                     drop_last=True, persistent_workers=True, prefetch_factor=4)
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.peak, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    core = MaskedLoss(policy) if a.masked_recovery else policy
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(core, device_ids=[dev.index], find_unused_parameters=True)
    else:
        model = core
    meta = {"base": P.BASE, "parent": a.init, "args": vars(a), "params": info, "anchors_per_source": n_src.tolist(),
            "recovery_roots": [str(r) for r in rec_roots], "n_recovery_traj": getattr(data, "n_traj", None),
            "n_traj_by_group": getattr(data, "n_traj_by_group", None)}
    start = 0
    rp = out / "resume.pt"
    if a.resume_every and rp.exists():
        st = torch.load(rp, map_location="cpu", weights_only=False)
        named = dict(policy.named_parameters())
        for k, v in st["trainable"].items():
            named[k].data.copy_(v.to(named[k].device, named[k].dtype))
        opt.load_state_dict(st["opt"])   # floating moments are cast to the param dtype (fp32) by torch
        for v in opt.state.values():
            for kk, vv in list(v.items()):
                if torch.is_tensor(vv) and vv.is_floating_point() and vv.ndim:
                    v[kk] = vv.float()
        start = int(st["step"])
        g.manual_seed(a.seed * 1000 + rank + 7919 * start)   # fresh sample order after a resume (logged)
        print(f"RESUMED from step {start}", flush=True)
    t0, it = time.time(), iter(dl)
    policy.train()
    for step in range(start, a.steps):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, a.peak, a.warmup, a.decay_steps, a.decay_lr)
        opt.zero_grad(set_to_none=True)
        for _ in range(a.accum):
            b = next(it)
            b.pop("src")
            valid, w = b.pop("valid", None), b.pop("w", None)
            b = pre(b)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if a.masked_recovery:
                    loss = model(b, valid.to(dev), w.to(dev))
                else:
                    loss, _ = model(b)
            (loss / a.accum).backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if rank == 0 and (step % 50 == 0 or step == a.steps - 1):
            el, nd = time.time() - t0, step - start + 1
            print(f"step {step} loss {loss.item():.4f} gn {gn.item():.2f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"{el / nd:.2f}s/it eta {el / nd * (a.steps - step - 1) / 60:.0f} min", flush=True)
            try:   # /pfss quota (EDQUOT) must not kill training
                with open(out / "metrics.jsonl", "a") as f:
                    f.write(json.dumps({"step": step, "loss": loss.item(), "gn": gn.item()}) + "\n")
            except OSError as e:
                print(f"WARNING metrics write failed ({e})", flush=True)
        if a.resume_every and rank == 0 and (step + 1) % a.resume_every == 0 and step + 1 < a.steps:
            save_resume(rp, step + 1, opt, policy)
            if a.stop_at and step + 1 == a.stop_at:
                print(f"STOP_AT {a.stop_at} (diagnostic forced restart)", flush=True)
                return
        if rank == 0 and ((step + 1) % a.save_every == 0 or step + 1 == a.steps):
            for k in range(12):   # /pfss quota is shared workspace-wide: wait it out instead of losing the run
                try:
                    P.save_ckpt(policy, out / f"ckpt_{step + 1:05d}", meta | {"step": step + 1, "save_dtype": a.save_dtype},
                                pre, post, dtype=torch.bfloat16 if a.save_dtype == "bf16" else torch.float32)
                    break
                except (OSError, RuntimeError) as e:
                    Path(out / f"ckpt_{step + 1:05d}" / "trainable.pt.tmp").unlink(missing_ok=True)
                    print(f"final ckpt save failed ({str(e)[:80]}), retry {k + 1}/12 in 5 min", flush=True)
                    time.sleep(300)
            else:
                raise SystemExit("could not save the final checkpoint (quota)")
            print("saved", step + 1, flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        (out / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        rp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
