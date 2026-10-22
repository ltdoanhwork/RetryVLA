"""B3.2: RISE-inspired imagination-only policy improvement, step 2 = SmolVLA fine-tuning.

python -m retryvla.baselines.sft_rise configs/exp/b32_rise_sft_cond.yaml [rise_train.mode=cond|awr ...]

Same recipe as the PoC distillation (retryvla.train.sft: init pi1-nominal, 6k optimizer steps, effective batch 32,
lr 5e-5 cosine, 50% nominal LIBERO-Spatial demo replay), but the "recovery" half of every batch comes from
IMAGINED data (retryvla.baselines.rise_lite): (obs_0 of a failure state, 50-step chunk, advantage A from WM + critic).
No candidate was ever executed in the simulator.

  mode=cond (main; RISE / RECAP-style advantage conditioning): chunks with high imagined advantage are labelled
      positive, low ones negative; the label is appended to the instruction ("<task> Advantage: positive.");
      nominal demos are labelled positive (expert data). At test time every instruction gets the positive suffix
      (retryvla.baselines.eval_cond reads it from <ckpt>/rise_condition.json).
  mode=awr (second): AWR, w = exp(clip(A_hat / beta)) with A_hat = A standardized within the failure state;
      implemented as importance sampling (candidate drawn with prob ~ w, unweighted flow-matching loss), which has
      the same expected gradient as the w-weighted loss but far lower variance (most of the 696 chunks have w ~ 0).
Both: failure states are drawn uniformly (coverage of every failure state, cf. the PoC round-robin subset).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from retryvla import REPO_ROOT, tracking
from retryvla.config import cli_config
from retryvla.data.mix import CORE_KEYS
from retryvla.train.sft import build_datasets, lr_lambda
from retryvla.utils import setup_logging

log = logging.getLogger("sft_rise")


class ImaginedBank:
    """All rise_lite instance files of one or more roots, with per-instance sampling tables."""

    def __init__(self, roots, rt):
        self.mode = str(rt.mode)
        self.items = []
        files = sorted(f for r in roots for f in (REPO_ROOT / r / "inst").glob("*.pt"))
        if rt.get("max_instances"):
            files = files[: int(rt.max_instances)]
        if not files:
            raise FileNotFoundError(f"no rise_lite instance files under {list(roots)}")
        self.stats = {"n_instances": len(files), "n_pos": 0, "n_neg": 0, "n_inst_no_pos": 0, "ess": []}
        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            adv = np.asarray(d["adv"], np.float64)
            it = {"image": d["image"], "image2": d["image2"], "state": np.asarray(d["state"], np.float32),
                  "actions": np.asarray(d["actions"], np.float32), "task": d["task"], "id": d["instance_id"]}
            if bool(rt.get("clip_actions", True)):  # latent-CEM samples can leave [-1, 1]; the env clips them anyway
                it["actions"] = np.clip(it["actions"], -1.0, 1.0)
            if self.mode == "cond":
                hi = np.quantile(adv, 1 - float(rt.pos_frac))
                lo = np.quantile(adv, float(rt.neg_frac))
                pos = np.nonzero((adv >= hi) & (adv > float(rt.pos_min_adv)))[0]
                neg = np.nonzero(adv <= lo)[0]
                it["pos"], it["neg"] = pos, neg
                self.stats["n_pos"] += len(pos)
                self.stats["n_neg"] += len(neg)
                self.stats["n_inst_no_pos"] += int(len(pos) == 0)
            elif self.mode == "awr":
                ah = (adv - adv.mean()) / (adv.std() + 1e-8)
                w = np.exp(np.clip(ah, -float(rt.awr_clip), float(rt.awr_clip)) / float(rt.awr_beta))
                p = w / w.sum()
                it["p"] = p
                self.stats["ess"].append(float(1.0 / (p ** 2).sum()))
            else:
                raise ValueError(self.mode)
            self.items.append(it)
        if self.stats["ess"]:
            self.stats["ess_mean"] = float(np.mean(self.stats.pop("ess")))
        else:
            self.stats.pop("ess")


def _img(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float().div_(255.0)


class RiseMix(Dataset):
    """Virtual dataset of length n: item i is an imagined (obs_0, chunk) sample with prob `ratio`, else a demo frame."""

    def __init__(self, original, bank: ImaginedBank, ratio: float, n: int, rt, seed: int = 0):
        self.orig, self.bank, self.ratio, self.n, self.rt, self.seed = original, bank, float(ratio), int(n), rt, seed
        self.cond = bank.mode == "cond"
        self.pos_sfx, self.neg_sfx = str(rt.get("pos_suffix", "")), str(rt.get("neg_suffix", ""))
        self.demo_sfx = self.pos_sfx if (self.cond and str(rt.get("demo_label", "positive")) == "positive") else ""

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng((self.seed, i))
        if rng.random() < self.ratio:
            it = self.bank.items[int(rng.integers(len(self.bank.items)))]
            if self.cond:
                use_pos = len(it["pos"]) and (rng.random() < float(self.rt.p_pos) or not len(it["neg"]))
                arr = it["pos"] if use_pos else it["neg"]
                k = int(arr[rng.integers(len(arr))])
                task = it["task"] + (self.pos_sfx if use_pos else self.neg_sfx)
            else:
                k = int(rng.choice(len(it["p"]), p=it["p"]))
                task = it["task"]
            act = torch.from_numpy(it["actions"][k])
            return {"observation.images.image": _img(it["image"]), "observation.images.image2": _img(it["image2"]),
                    "observation.state": torch.from_numpy(it["state"]), "action": act,
                    "action_is_pad": torch.zeros(len(act), dtype=torch.bool), "task": task,
                    "source": torch.tensor(1)}
        item = self.orig[int(rng.integers(len(self.orig)))]
        out = {k: item[k] for k in CORE_KEYS if k in item}
        for k in ("observation.images.image", "observation.images.image2", "observation.state"):
            v = out.get(k)
            if v is not None and v.ndim == (4 if "images" in k else 2) and v.shape[0] == 1:
                out[k] = v[0]
        out["task"] = out["task"] + self.demo_sfx
        out["source"] = torch.tensor(0)
        return out


def main() -> None:
    setup_logging()
    cfg = cli_config()
    tc, rt = cfg.train, cfg.rise_train
    torch.manual_seed(int(tc.seed))
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    accum = int(tc.get("grad_accum", 1))
    bs = int(tc.batch_size)
    assert bs % accum == 0, (bs, accum)
    mbs = bs // accum
    with tracking.start(cfg) as run:
        device = torch.device(cfg.policy.device)
        policy = SmolVLAPolicy.from_pretrained(tc.init_from)
        policy.config.device = str(device)
        policy.to(device)
        pre, post = make_pre_post_processors(policy.config, pretrained_path=tc.init_from,
                                             preprocessor_overrides={"device_processor": {"device": str(device)}})
        orig, recs = build_datasets(cfg, policy.config)
        assert not recs, "B3.2 uses imagined data only: train.recovery_datasets must be empty"
        bank = ImaginedBank(list(rt.data_roots), rt)
        log.info("imagined bank: %s", bank.stats)
        n_items = int(tc.steps) * bs
        mix = RiseMix(orig, bank, float(tc.recovery_ratio), n_items, rt, seed=int(tc.seed))
        cond = None
        if bank.mode == "cond":
            cond = {"mode": "cond", "positive_suffix": mix.pos_sfx, "negative_suffix": mix.neg_sfx,
                    "demo_suffix": mix.demo_sfx, "apply_at_eval": "positive_suffix appended to every instruction"}
        run.summary.update({"n_original_frames": len(orig), "bank": bank.stats, "condition": cond,
                            "grad_accum": accum, "micro_batch": mbs, "sim_verification_transitions": 0})
        dl = DataLoader(mix, batch_size=mbs, shuffle=False, num_workers=int(tc.num_workers), pin_memory=True,
                        drop_last=True, persistent_workers=int(tc.num_workers) > 0,
                        prefetch_factor=4 if int(tc.num_workers) > 0 else None)
        params = [p for p in policy.parameters() if p.requires_grad]
        log.info("trainable params: %.1fM | micro-batch %d x accum %d", sum(p.numel() for p in params) / 1e6, mbs, accum)
        opt = torch.optim.AdamW(params, lr=float(tc.lr), betas=(0.9, 0.95), eps=1e-8,
                                weight_decay=float(tc.weight_decay))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(int(tc.warmup_steps), int(tc.steps)))
        amp = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(str(tc.amp_dtype))
        policy.train()
        t0, loss_ema, ckpt_dir = time.time(), None, None
        step, lv, rf = 0, 0.0, 0.0
        opt.zero_grad(set_to_none=True)
        for mi, batch in enumerate(dl, start=1):
            src = batch.pop("source")
            batch = pre(batch)
            with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None):
                loss, _ = policy.forward(batch)
            (loss / accum).backward()
            lv += float(loss.detach()) / accum
            rf += float(src.float().mean()) / accum
            if mi % accum:
                continue
            step += 1
            gn = torch.nn.utils.clip_grad_norm_(params, float(tc.grad_clip))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            loss_ema = lv if loss_ema is None else 0.98 * loss_ema + 0.02 * lv
            if step % int(tc.log_every) == 0 or step == 1:
                run.log({"train/loss": lv, "train/loss_ema": loss_ema, "train/grad_norm": float(gn),
                         "train/lr": sched.get_last_lr()[0], "train/recovery_frac": rf,
                         "train/steps_per_s": step / (time.time() - t0)}, step=step)
                log.info("step %d/%d loss %.4f (ema %.4f) gn %.2f lr %.2e rec %.2f | %.2f it/s", step, int(tc.steps),
                         lv, loss_ema, float(gn), sched.get_last_lr()[0], rf, step / (time.time() - t0))
            lv, rf = 0.0, 0.0
            if step % int(tc.save_every) == 0 or step == int(tc.steps):
                ckpt_dir = run.path("checkpoints", f"{step:06d}", "pretrained_model")
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(ckpt_dir)
                pre.save_pretrained(ckpt_dir)
                post.save_pretrained(ckpt_dir)
                if cond is not None:
                    (Path(ckpt_dir) / "rise_condition.json").write_text(json.dumps(cond, indent=1))
                log.info("saved %s", ckpt_dir)
            if step >= int(tc.steps):
                break
        last = run.path("checkpoints", "last")
        if ckpt_dir is not None:
            if last.exists() or last.is_symlink():
                last.unlink()
            last.symlink_to(Path(ckpt_dir).parent.name)
        run.summary.update({"final_loss": loss_ema, "checkpoint": str(ckpt_dir), "optimizer_steps": step})


if __name__ == "__main__":
    main()
