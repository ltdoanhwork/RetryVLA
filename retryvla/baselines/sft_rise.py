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

Robustness (after the 2026-10-22 /pfss EDQUOT outage killed the first cond_pool run at step 5150 via a
TensorBoard write): metric logging failures are non-fatal (TB is disabled, jsonl skipped), checkpoints are written
to `checkpoints/<step>.tmp/` and renamed only when complete (retry with back-off ~36 min), and the latest
checkpoint also carries `training_state.pt` (optimizer, scheduler, step, EMA, RNG). With `train.resume=auto`
(default) a relaunch continues the newest unfinished run dir of the same milestone/tag/seed from its last
complete checkpoint; the sample stream is a pure function of (seed, item index), so the resumed data order is
identical to an uninterrupted run.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from retryvla import REPO_ROOT, RUNS_DIR, tracking
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
        self.offset = 0   # resume: skip the items already consumed (item i is a pure function of (seed, i))

    def __len__(self) -> int:
        return self.n - self.offset

    def __getitem__(self, i: int) -> dict:
        i = i + self.offset
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


def safe_log(run, metrics: dict, step: int) -> None:
    """run.log, but a failing write (EDQUOT on /pfss) must never kill training: drop TB, keep going."""
    try:
        run.log(metrics, step=step)
    except Exception as e:  # noqa: BLE001  (TensorBoard re-raises its async worker's OSError on every call)
        log.warning("metric logging failed at step %d (%s): TensorBoard disabled, jsonl best-effort", step,
                    str(e)[:120])
        run._tb = False   # also keeps Run.finish from re-raising in SummaryWriter.close()


def save_checkpoint(run, step: int, policy, pre, post, cond, state: dict, tries: int = 8,
                    wait_s: float = 60.0) -> Path:
    """Write checkpoints/<step>.tmp/{pretrained_model,training_state.pt}, then rename to checkpoints/<step>.

    The final directory exists only when complete (eval DAG globs on .../<step>/pretrained_model). Quota windows
    are waited out (60, 120, ... s; ~36 min total) before raising. Only the newest checkpoint keeps the ~0.8 GB
    optimizer state."""
    root = run.path("checkpoints", "x").parent
    final, tmp = root / f"{step:06d}", root / f"{step:06d}.tmp"
    for k in range(tries):
        try:
            shutil.rmtree(tmp, ignore_errors=True)
            ck = tmp / "pretrained_model"
            ck.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(ck)
            pre.save_pretrained(ck)
            post.save_pretrained(ck)
            if cond is not None:
                (ck / "rise_condition.json").write_text(json.dumps(cond, indent=1))
            torch.save(state, tmp / "training_state.pt")
            if final.exists():
                shutil.rmtree(final)
            tmp.rename(final)
            break
        except (RuntimeError, OSError) as e:
            shutil.rmtree(tmp, ignore_errors=True)
            if k == tries - 1:
                raise
            log.warning("checkpoint %06d failed (%s), retry %d/%d in %.0fs", step, str(e)[:100], k + 1, tries,
                        wait_s * (k + 1))
            time.sleep(wait_s * (k + 1))
    for old in root.glob("[0-9]*/training_state.pt"):
        if old.parent != final:
            old.unlink(missing_ok=True)
    return final / "pretrained_model"


def _same_recipe(a, b) -> bool:
    from omegaconf import OmegaConf
    ca = OmegaConf.to_container(a, resolve=True)
    cb = OmegaConf.to_container(b, resolve=True)
    for c in (ca, cb):
        c.get("train", {}).pop("resume", None)
    return all(ca.get(k) == cb.get(k) for k in ("train", "rise_train", "policy", "split"))


def find_resume(cfg) -> tuple[Path | None, Path | None]:
    """Newest unfinished run dir of this milestone/tag with an identical recipe and a complete checkpoint."""
    from omegaconf import OmegaConf
    for d in sorted(RUNS_DIR.glob(f"*_{cfg.milestone}_{cfg.tag}"), reverse=True):
        try:
            if json.loads((d / "summary.json").read_text()).get("status") == "done":
                continue
        except (OSError, ValueError):
            pass
        sts = sorted(d.glob("checkpoints/[0-9][0-9][0-9][0-9][0-9][0-9]/training_state.pt"))
        if not sts or not (d / "config.yaml").exists():
            continue
        if not _same_recipe(OmegaConf.load(d / "config.yaml"), cfg):
            log.info("resume: %s has a different recipe, skipped", d.name)
            continue
        return d, sts[-1]
    return None, None


def short_tmpdir() -> None:
    """DataLoader workers share tensors via an AF_UNIX socket under $TMPDIR (limit 107 chars). After the
    2026-10-23 container restart the tmux server inherited a ~90-char TMPDIR from another tool and every worker
    died with `OSError: AF_UNIX path too long`; fall back to a short private dir for this process only."""
    import os
    import tempfile
    if len(tempfile.gettempdir()) > 60:
        new = f"/tmp/rv_{os.getuid()}"
        os.makedirs(new, exist_ok=True)
        log.warning("TMPDIR %s too long for AF_UNIX sockets -> %s", tempfile.gettempdir(), new)
        os.environ["TMPDIR"] = new
        tempfile.tempdir = None


def main() -> None:
    setup_logging()
    short_tmpdir()
    cfg = cli_config()
    tc, rt = cfg.train, cfg.rise_train
    torch.manual_seed(int(tc.seed))
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    accum = int(tc.get("grad_accum", 1))
    bs = int(tc.batch_size)
    assert bs % accum == 0, (bs, accum)
    mbs = bs // accum
    resume_mode = str(tc.get("resume", "auto"))
    run_dir, st_file = find_resume(cfg) if resume_mode == "auto" else (None, None)
    with tracking.start(cfg, run_dir=run_dir) as run:
        if run_dir is not None:
            log.info("resume: continuing %s from %s", run_dir.name, st_file.parent.name)
            for t in run_dir.glob("checkpoints/*.tmp"):
                shutil.rmtree(t, ignore_errors=True)
        device = torch.device(cfg.policy.device)
        init = str(st_file.parent / "pretrained_model") if st_file is not None else tc.init_from
        policy = SmolVLAPolicy.from_pretrained(init)
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
        state = torch.load(st_file, map_location="cpu", weights_only=False) if st_file is not None else None
        step0 = int(state["step"]) if state is not None else 0
        mix.offset = step0 * bs   # one optimizer step consumes exactly bs items (drop_last, shuffle=False)
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
        step, lv, rf = step0, 0.0, 0.0
        if state is not None:
            opt.load_state_dict(state["optimizer"])
            sched.load_state_dict(state["scheduler"])
            loss_ema = state.get("loss_ema")
            torch.set_rng_state(state["torch_rng"])
            if state.get("cuda_rng") is not None and device.type == "cuda":
                torch.cuda.set_rng_state(state["cuda_rng"], device)
            ckpt_dir = st_file.parent / "pretrained_model"
            run.summary.setdefault("resumes", []).append({"from_step": step0,
                                                          "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            log.info("resumed at optimizer step %d (lr %.2e, item offset %d)", step0, sched.get_last_lr()[0],
                     mix.offset)
            del state
        opt.zero_grad(set_to_none=True)
        for mi, batch in enumerate(dl if step < int(tc.steps) else [], start=1):
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
            if step % int(tc.log_every) == 0 or step == step0 + 1:
                sps = (step - step0) / (time.time() - t0)
                safe_log(run, {"train/loss": lv, "train/loss_ema": loss_ema, "train/grad_norm": float(gn),
                               "train/lr": sched.get_last_lr()[0], "train/recovery_frac": rf,
                               "train/steps_per_s": sps}, step=step)
                log.info("step %d/%d loss %.4f (ema %.4f) gn %.2f lr %.2e rec %.2f | %.2f it/s", step, int(tc.steps),
                         lv, loss_ema, float(gn), sched.get_last_lr()[0], rf, sps)
            lv, rf = 0.0, 0.0
            if step % int(tc.save_every) == 0 or step == int(tc.steps):
                st = {"step": step, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                      "loss_ema": loss_ema, "torch_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}
                ckpt_dir = save_checkpoint(run, step, policy, pre, post, cond, st)
                del st
                log.info("saved %s", ckpt_dir)
            if step >= int(tc.steps):
                break
        last = run.path("checkpoints", "last")
        if ckpt_dir is not None:
            try:
                if last.exists() or last.is_symlink():
                    last.unlink()
                last.symlink_to(Path(ckpt_dir).parent.name)
            except OSError as e:
                log.warning("could not update checkpoints/last (%s)", e)
        run.summary.update({"final_loss": loss_ema, "checkpoint": str(ckpt_dir), "optimizer_steps": step})


if __name__ == "__main__":
    main()
