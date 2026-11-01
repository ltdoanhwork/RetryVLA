"""M1c: recovery SFT of SmolVLA on D_original U D_recovery (standard flow-matching loss).

python -m retryvla.train.sft configs/exp/m1_sft_oracle_cem.yaml [train.steps=...]

Keeps the pretrained checkpoint's normalization stats (pre/post-processors are loaded from
``train.init_from``) so pi1 sees the same input/output scaling as pi0. Checkpoints are saved in
lerobot format under runs/<id>/checkpoints/<step>/ and can be evaluated with
``policy.path=<that dir>``.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from retryvla import REPO_ROOT, tracking
from retryvla.config import cli_config
from retryvla.data.mix import (FutureDataset, MixDataset, count_trajectories, expand_roots, recovery_episode_subsets,
                               spatial_episode_ids)
from retryvla.utils import setup_logging

log = logging.getLogger("sft")


def lr_lambda(warmup: int, total: int, final_frac: float = 0.05):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total - warmup)
        return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return f


def build_datasets(cfg, policy_cfg):
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    from retryvla.data.mix import patch_lerobot_index_mapping

    patch_lerobot_index_mapping()

    tc = cfg.train
    meta = LeRobotDatasetMetadata(tc.original_dataset)
    dts = resolve_delta_timestamps(policy_cfg, meta)
    eps = spatial_episode_ids(tc.original_dataset, tc.original_suite_filter) if tc.original_suite_filter else None
    log.info("original dataset %s: %s episodes", tc.original_dataset, "all" if eps is None else len(eps))
    orig = LeRobotDataset(tc.original_dataset, episodes=eps, delta_timestamps=dts)
    recs = []
    n_match = tc.get("max_recovery_trajectories")
    if tc.get("match_trajectory_count_of"):
        n_match = count_trajectories(REPO_ROOT / tc.match_trajectory_count_of)
        log.info("matching trajectory count of %s: %d", tc.match_trajectory_count_of, n_match)
    roots = [Path(r) for r in expand_roots([REPO_ROOT / r for r in tc.recovery_datasets])]
    log.info("recovery roots: %d", len(roots))
    subs = recovery_episode_subsets(roots, int(n_match) if n_match else None, int(tc.get("min_traj_len", 0)),
                                    mode=str(tc.get("subset_mode", "score")))
    for root, sub in zip(roots, subs):
        if sub is not None and not sub:
            continue
        ds = LeRobotDataset("retryvla/recovery", root=root, episodes=sub, delta_timestamps=dts)
        log.info("recovery dataset %s: %d episodes, %d frames", root, ds.num_episodes, ds.num_frames)
        recs.append(ds)
    return orig, recs


def build_future_datasets(tc, policy_cfg) -> list:
    """train.future_datasets: [{path, variant, n, filters}] (Part 4; default [] = OFF)."""
    out = []
    for k, spec in enumerate(tc.get("future_datasets") or []):
        fd = FutureDataset(REPO_ROOT / spec["path"], str(spec["variant"]), spec.get("n"), chunk=int(policy_cfg.chunk_size),
                           action_dim=int(policy_cfg.action_feature.shape[0]), seed=int(tc.seed) + k,
                           center=bool(tc.get("future_center", True)),
                           filters=dict(spec["filters"]) if spec.get("filters") else None)
        log.info("future dataset %s [%s]: %d imagined futures on %d failure states | cpi %.3f | sim-checked %d "
                 "(success %d)", spec["path"], spec["variant"], len(fd), int(fd.obs_idx.unique().numel()),
                 float(fd.cpi.mean()) if len(fd) else float("nan"), int(fd.sim_checked.sum()), int(fd.sim_success.sum()))
        out.append(fd)
    return out


def build_preserve(tc, policy, device):
    """Optional L_preserve (P3, default OFF: train.preserve_beta = 0). Frozen teacher = the init policy
    (train.preserve_teacher, default train.init_from). On NOMINAL demo samples of each batch the student's predicted
    flow-matching velocity is pulled toward the teacher's at the SAME (noise, time, x_t, observation):
        L = L_FM(all samples) + beta * mean_{nominal, valid steps, real action dims} || v_student - v_teacher ||^2
    (a distillation term in the native output space of the flow-matching head, not a KL on sampled actions)."""
    beta = float(tc.get("preserve_beta", 0.0) or 0.0)
    if beta <= 0:
        return None
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    path = tc.get("preserve_teacher") or tc.init_from
    teacher = SmolVLAPolicy.from_pretrained(path)
    teacher.config.device = str(device)
    teacher.to(device).eval().requires_grad_(False)
    cap: dict[str, torch.Tensor] = {}
    hs = [policy.model.action_out_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("s", o)),
          teacher.model.action_out_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("t", o))]
    on = str(tc.get("preserve_on", "nominal"))
    log.info("L_preserve on: beta %.3g, teacher %s, samples=%s", beta, path, on)
    return {"beta": beta, "teacher": teacher, "cap": cap, "hooks": hs, "on": on}


def preserve_forward(pres, policy, batch, src):
    """Student FM loss + L_preserve with shared noise / time (see build_preserve)."""
    cfg = policy.config
    B = batch["action"].shape[0]
    dev = batch["action"].device
    noise = policy.model.sample_noise((B, cfg.chunk_size, cfg.max_action_dim), dev)
    t = policy.model.sample_time(B, dev)
    loss, out = policy.forward(batch, noise=noise, time=t)
    v_s = pres["cap"].pop("s")
    with torch.no_grad():
        pres["teacher"].forward(batch, noise=noise, time=t)
    v_t = pres["cap"].pop("t").detach()
    d = cfg.action_feature.shape[0]
    se = ((v_s[..., :d].float() - v_t[..., :d].float()) ** 2).mean(-1)            # [B, chunk]
    m = torch.ones_like(se)
    if batch.get("action_is_pad") is not None:
        m = m * (~batch["action_is_pad"]).float()
    if pres["on"] == "nominal":
        m = m * (src == 0).float()[:, None]
    lp = (se * m).sum() / m.sum().clamp_min(1.0)
    return loss, out, lp


def build_future(tc, policy, device, future_dim: int):
    """Optional L_future (Part 4 / RQ2, default OFF: train.future_beta = 0). A small head on the action-expert hidden
    states (the INPUT of ``action_out_proj``, i.e. the same hook point as build_preserve), mean-pooled over the chunk,
    predicts the pooled DINOv2 latent of the sample's target future (FutureDataset). Applied only to samples with
    has_future; samples with has_action False are masked out of the flow-matching loss (no pseudo action labels)."""
    beta = float(tc.get("future_beta", 0.0) or 0.0)
    if beta <= 0 or future_dim <= 0:
        return None
    hid = int(policy.model.action_out_proj.in_features)
    w = int(tc.get("future_head_width", 512))
    head = torch.nn.Sequential(torch.nn.LayerNorm(hid), torch.nn.Linear(hid, w), torch.nn.GELU(),
                               torch.nn.Linear(w, future_dim)).to(device)
    cap: dict[str, torch.Tensor] = {}
    h = policy.model.action_out_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("h", i[0]))
    kind = str(tc.get("future_loss", "cosine"))
    assert kind in ("cosine", "l2"), kind
    log.info("L_future on: beta %.3g, loss %s, head %d->%d->%d", beta, kind, hid, w, future_dim)
    return {"beta": beta, "head": head, "cap": cap, "hooks": [h], "loss": kind}


def future_forward(fut, policy, batch, has_action, has_future, target):
    """Masked FM loss (mean over has_action samples; = policy.forward's mean when all have actions) + L_future."""
    per, out = policy.forward(batch, reduction="none")                    # [B]
    m = has_action.float()
    loss = (per * m).sum() / m.sum().clamp_min(1.0)
    h = fut["cap"].pop("h").float().mean(1)                                 # [B, hidden] pooled over the chunk
    pred = fut["head"](h)
    if fut["loss"] == "cosine":
        lf_i = 1.0 - torch.nn.functional.cosine_similarity(pred, target.float(), dim=-1)
    else:
        lf_i = ((pred - target.float()) ** 2).mean(-1)
    mf = has_future.float()
    lf = (lf_i * mf).sum() / mf.sum().clamp_min(1.0)
    return loss, out, lf


def main() -> None:
    setup_logging()
    from retryvla.baselines.sft_rise import safe_log, save_checkpoint, short_tmpdir  # EDQUOT-safe (B3.2 fixes)
    short_tmpdir()  # DataLoader workers die with `AF_UNIX path too long` under the inherited ~90-char TMPDIR
    cfg = cli_config()
    tc = cfg.train
    torch.manual_seed(int(tc.seed))
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    with tracking.start(cfg) as run:
        device = torch.device(cfg.policy.device)
        policy = SmolVLAPolicy.from_pretrained(tc.init_from)
        policy.config.device = str(device)
        policy.to(device)
        pre, post = make_pre_post_processors(policy.config, pretrained_path=tc.init_from,
                                             preprocessor_overrides={"device_processor": {"device": str(device)}})
        orig, recs = build_datasets(cfg, policy.config)
        n_items = int(tc.steps) * int(tc.batch_size)
        futs = build_future_datasets(tc, policy.config)
        if futs:
            mix = MixDataset(orig, recs, float(tc.recovery_ratio), n_items, tuple(tc.recovery_segments),
                             seed=int(tc.seed), futures=futs, future_ratio=float(tc.get("future_ratio", 0.0)))
            run.summary.update({"n_future_samples": sum(len(f) for f in futs), "future_ratio": mix.future_ratio})
        else:  # default path (bit-for-bit pre-Part-4)
            mix = MixDataset(orig, recs, float(tc.recovery_ratio), n_items, tuple(tc.recovery_segments),
                             seed=int(tc.seed))
        log.info("mix: %d original frames, %d recovery frames, ratio %.2f", len(orig), len(mix.rec_index), mix.ratio)
        run.summary.update({"n_original_frames": len(orig), "n_recovery_frames": len(mix.rec_index),
                            "n_recovery_episodes": sum(d.num_episodes for d in recs)})
        dl = DataLoader(mix, batch_size=int(tc.batch_size), shuffle=False, num_workers=int(tc.num_workers),
                        pin_memory=True, drop_last=True, persistent_workers=int(tc.num_workers) > 0,
                        prefetch_factor=4 if int(tc.num_workers) > 0 else None)
        params = [p for p in policy.parameters() if p.requires_grad]
        fut = build_future(tc, policy, device, mix.future_dim if getattr(mix, "with_future", False) else 0)
        if fut is not None:
            assert float(tc.get("preserve_beta", 0.0) or 0.0) <= 0, "L_future + L_preserve together not supported"
            params = params + list(fut["head"].parameters())
        elif getattr(mix, "with_future", False):
            raise ValueError("future samples present but train.future_beta <= 0: they would only add zero-loss items")
        log.info("trainable params: %.1fM", sum(p.numel() for p in params) / 1e6)
        opt = torch.optim.AdamW(params, lr=float(tc.lr), betas=(0.9, 0.95), eps=1e-8,
                                weight_decay=float(tc.weight_decay))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(int(tc.warmup_steps), int(tc.steps)))
        amp = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(str(tc.amp_dtype))
        pres = build_preserve(tc, policy, device)
        policy.train()
        t0, loss_ema = time.time(), None
        ckpt_dir = None
        for step, batch in enumerate(dl, start=1):
            src = batch.pop("source")
            if fut is not None:
                f_ha, f_hf = batch.pop("has_action").to(device), batch.pop("has_future").to(device)
                f_tg = batch.pop("future_latent").to(device)
            lf = None
            batch = pre(batch)
            with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None):
                if fut is not None:
                    loss, out, lf = future_forward(fut, policy, batch, f_ha, f_hf, f_tg)
                    lp = None
                    loss = loss + fut["beta"] * lf
                elif pres is None:
                    loss, out = policy.forward(batch)
                    lp = None
                else:
                    loss, out, lp = preserve_forward(pres, policy, batch, src.to(device))
                    loss = loss + pres["beta"] * lp
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, float(tc.grad_clip))
            opt.step()
            sched.step()
            lv = float(loss.detach())
            loss_ema = lv if loss_ema is None else 0.98 * loss_ema + 0.02 * lv
            if step % int(tc.log_every) == 0 or step == 1:
                safe_log(run, {"train/loss": lv, "train/loss_ema": loss_ema, "train/grad_norm": float(gn),
                         "train/lr": sched.get_last_lr()[0], "train/recovery_frac": float(src.float().mean()),
                         "train/steps_per_s": step / (time.time() - t0),
                         **({"train/preserve_loss": float(lp.detach())} if lp is not None else {}),
                         **({"train/future_loss": float(lf.detach()), "train/future_frac": float((src == 2).float().mean())}
                            if lf is not None else {})}, step)
                log.info("step %d/%d loss %.4f (ema %.4f) gn %.2f lr %.2e | %.2f it/s%s", step, int(tc.steps), lv,
                         loss_ema, float(gn), sched.get_last_lr()[0], step / (time.time() - t0),
                         (f" | L_preserve {float(lp.detach()):.4f}" if lp is not None else "")
                         + (f" | L_future {float(lf.detach()):.4f}" if lf is not None else ""))
            if step % int(tc.save_every) == 0 or step == int(tc.steps):
                for k in range(8):  # save_checkpoint retries OSError/RuntimeError itself; safetensors' SafetensorError
                    try:            # (EDQUOT, 2026-10-23 13:59) is neither -> retry it here too
                        ckpt_dir = save_checkpoint(run, step, policy, pre, post, None, {"step": step})
                        break
                    except Exception as e:  # noqa: BLE001
                        if k == 7:
                            raise
                        log.warning("checkpoint %06d failed (%s), outer retry %d/8 in %ds", step, str(e)[:100],
                                    k + 1, 60 * (k + 1))
                        time.sleep(60 * (k + 1))
                log.info("saved %s", ckpt_dir)
            if step >= int(tc.steps):
                break
        last = run.path("checkpoints", "last")
        if ckpt_dir is not None:
            if last.exists() or last.is_symlink():
                last.unlink()
            last.symlink_to(Path(ckpt_dir).parent.name)
        if fut is not None:  # head + target centering, for the embedding probe (scripts/af_probe.py)
            tmp = run.path("future_head.pt.tmp")
            torch.save({"head": fut["head"].state_dict(), "target_mean": [f.target_mean for f in futs],
                        "loss": fut["loss"], "beta": fut["beta"]}, tmp)
            Path(tmp).replace(run.path("future_head.pt"))
        run.summary.update({"final_loss": loss_ema, "checkpoint": str(ckpt_dir)})


if __name__ == "__main__":
    main()
