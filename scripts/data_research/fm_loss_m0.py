"""Learnability probe (2026-11-27, research task SOLUTIONS_DATA): per-anchor flow-matching loss of the FROZEN M0
(runs/goal_rsi/m0_s7/ckpt_04000, the init of every Goal SFT arm) on D0 demos vs our recovery banks vs LRS windows.

CPU only (fp32). The prefix (2 images + language) is encoded once per anchor and K (noise, time) draws reuse its KV
cache through `denoise_step` (same computation as the training forward: the prefix never attends to the suffix).
Draw k uses torch seed 1000 + k for every batch, so all sources see the same noise / time distribution.

python -m scripts.data_research.fm_loss_m0 [--k 4] [--bs 8] [--threads 24] [--only d0,onset_wm,...]
-> runs/_analysis/data_research/fm_m0/anchors.jsonl (one row per anchor; appended, resumable)
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from retryvla import REPO_ROOT

OUT = REPO_ROOT / "runs/_analysis/data_research/fm_m0"
CK = "runs/goal_rsi/m0_s7/ckpt_04000"
D0 = "data/goal_rsi/d0_train"

# (source name, glob, anchors per trajectory from segment 0, from segment 1, masked (LRS windows), max episodes per
# root (0 = all)). CPU budget (~15 s / anchor on the shared node): ordered by priority, appended + resumable.
SOURCES = [
    ("d0", D0, None, None, False, 0),
    ("lrs_v2", "data/recovery/lrs_goal_v2_ds", 2, 0, True, 0),
    ("lrs_v2gt", "data/recovery/lrs_goal_v2gt_ds", 1, 0, True, 0),
    ("onset_wm", "data/recovery/goal_search_onset_cur/s*/p0/wm_diverse", 1, 1, False, 0),
    ("onset_rand", "data/recovery/goal_search_onset_cur/s*/p0/random", 1, 0, False, 0),
    ("lrs_v1", "data/recovery/lrs_goal_ds", 1, 0, True, 0),
    ("gs_wm", "data/recovery/goal_search/s*/p*/wm_diverse", 1, 1, False, 20),
    ("onset_rand_c", "data/recovery/goal_search_onset_cur/s*/p0/random", 0, 1, False, 0),
    ("gs_rand", "data/recovery/goal_search/s*/p*/random", 1, 0, False, 20),
]
N_D0 = 64


def roughness(a: np.ndarray) -> dict:
    """a [T, 7] target chunk (raw action units)."""
    d3 = np.linalg.norm(np.diff(a[:, :6], 3, axis=0), axis=1).mean() if len(a) > 3 else 0.0
    g = np.sign(a[:, 6])
    return {"d3": float(d3), "sat": float((np.abs(a[:, :3]) >= 0.999).mean()),
            "flips": int((np.diff(g) != 0).sum()), "rot_std": float(a[:, 3:6].std())}


def pick_anchors(root: str, n0, n1, masked: bool, rng, max_ep: int = 0) -> list[tuple[int, int, int, int]]:
    """-> [(dataset row j, episode, frame_index, segment)]."""
    import pyarrow.parquet as pq
    cols = ["episode_index", "frame_index"] + (["segment"] if n0 is not None else []) + (["run_end"] if masked else [])
    tabs = [pq.read_table(f, columns=[c for c in cols if c in pq.read_schema(f).names])
            for f in sorted(glob.glob(f"{root}/data/*/*.parquet"))]
    ep = np.concatenate([t["episode_index"].to_numpy() for t in tabs])
    fi = np.concatenate([t["frame_index"].to_numpy() for t in tabs])
    if n0 is None:   # D0: uniform over all anchors
        js = rng.choice(len(ep), N_D0, replace=False)
        return [(int(j), int(ep[j]), int(fi[j]), -1) for j in sorted(js)]
    seg = np.concatenate([t["segment"].to_numpy() for t in tabs]) if "segment" in tabs[0].column_names else np.zeros(len(ep), int)
    ok = np.ones(len(ep), bool)
    if masked and "run_end" in tabs[0].column_names:
        re_ = np.concatenate([t["run_end"].to_numpy() for t in tabs])
        ok = (re_ - fi + 1) >= 5   # LRS L_min = 5 (as trained)
    out = []
    eps = np.unique(ep)
    if max_ep and len(eps) > max_ep:
        eps = np.sort(rng.choice(eps, max_ep, replace=False))
    for e in eps:
        for s, n in ((0, n0), (1, n1)):
            if not n:
                continue
            m = np.nonzero((ep == e) & (seg == s) & ok)[0]
            if len(m):
                for j in sorted(rng.choice(m, min(n, len(m)), replace=False)):
                    out.append((int(j), int(e), int(fi[j]), s))
    return out


@torch.no_grad()
def fm_losses(policy, batch, k: int, valid: torch.Tensor) -> np.ndarray:
    """[B, k] per-anchor FM loss (mean over valid targets x 7 dims) for k seeded (noise, time) draws."""
    from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks
    m = policy.model
    images, img_masks = policy._preprocess_images(batch)
    lt, lm = batch["observation.language.tokens"], batch["observation.language.attention_mask"]
    state, actions = policy.prepare_state(batch), policy.prepare_action(batch)
    pe, ppm, pam = m.embed_prefix(images, img_masks, lt, lm)
    a2 = make_att_2d_masks(ppm, pam)
    pos = torch.cumsum(ppm, dim=1) - 1
    m.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
    _, kv = m.paligemma_with_expert.forward(attention_mask=m._prepare_attention_masks_4d(a2), position_ids=pos,
                                            past_key_values=None, inputs_embeds=[pe, None], use_cache=True)
    B = actions.shape[0]
    out = np.zeros((B, k))
    D = policy.config.output_features["action"].shape[0]
    for d in range(k):
        g = torch.Generator().manual_seed(1000 + d)
        noise = torch.randn(actions.shape, generator=g)
        a_, b_ = m.config.time_sampling_beta_alpha, m.config.time_sampling_beta_beta
        tb = torch.from_numpy(np.random.default_rng(2000 + d).beta(a_, b_, B)).float()
        t = tb * m.config.time_sampling_scale + m.config.time_sampling_offset
        x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
        u_t = noise - actions
        v_t = m.denoise_step(state=state, prefix_pad_masks=ppm, past_key_values=kv, x_t=x_t, timestep=t)
        l = ((u_t - v_t) ** 2)[:, :, :D].mean(-1)   # [B, T]
        vm = valid[:, : l.shape[1]].to(l)
        out[:, d] = ((l * vm).sum(1) / vm.sum(1).clamp_min(1)).numpy()
    return out


def selfcheck(policy, batch):
    """denoise_step path == training forward on the same (noise, time)."""
    m = policy.model
    images, img_masks = policy._preprocess_images(batch)
    lt, lm = batch["observation.language.tokens"], batch["observation.language.attention_mask"]
    state, actions = policy.prepare_state(batch), policy.prepare_action(batch)
    g = torch.Generator().manual_seed(1000)
    noise = torch.randn(actions.shape, generator=g)
    t = torch.from_numpy(np.random.default_rng(2000).beta(m.config.time_sampling_beta_alpha,
                                                          m.config.time_sampling_beta_beta, actions.shape[0])).float()
    t = t * m.config.time_sampling_scale + m.config.time_sampling_offset
    with torch.no_grad():
        l_ref = m.forward(images, img_masks, lt, lm, state, actions, noise=noise, time=t)[:, :, :7].mean((1, 2))
    l_new = fm_losses(policy, batch, 1, torch.ones(actions.shape[0], actions.shape[1]))[:, 0]
    print("selfcheck forward vs cached", l_ref.numpy().round(5), l_new.round(5), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0, help="debug: max anchors per source")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    fo = OUT / "anchors.jsonl"
    done = set()
    if fo.exists():
        for l in open(fo):
            r = json.loads(l)
            done.add((r["root"], r["row"]))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors
    from retryvla.policy import pi0_lora as P

    t0 = time.time()
    policy, cfg, _ = P.load_policy(str(REPO_ROOT / CK), device="cpu")
    policy.eval().float()
    stats = LeRobotDatasetMetadata("local/d0_train", root=REPO_ROOT / D0).stats
    pre, _ = make_pre_post_processors(cfg, dataset_stats=stats, preprocessor_overrides={"device_processor": {"device": "cpu"}})
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    only = set(a.only.split(",")) if a.only else None
    checked = False
    for name, gl, n0, n1, masked, max_ep in SOURCES:
        if only and name not in only:
            continue
        for root in sorted(glob.glob(str(REPO_ROOT / gl))):
            if not Path(root, "meta/info.json").exists():
                continue
            rng = np.random.default_rng(7)
            anchors = [x for x in pick_anchors(root, n0, n1, masked, rng, max_ep) if (str(Path(root).relative_to(REPO_ROOT)), x[0]) not in done]
            if a.limit:
                anchors = anchors[: a.limit]
            if not anchors:
                continue
            fps = json.loads(Path(root, "meta/info.json").read_text())["fps"]
            ds = LeRobotDataset(f"local/{Path(root).name}", root=root,
                                delta_timestamps={"action": [i / fps for i in range(cfg.chunk_size)]})
            rel = str(Path(root).relative_to(REPO_ROOT))
            print(f"{name} {rel}: {len(anchors)} anchors", flush=True)
            for b0 in range(0, len(anchors), a.bs):
                chunk = anchors[b0: b0 + a.bs]
                items = [ds[j] for j, *_ in chunk]
                b = {"observation.images.image": torch.stack([it["observation.images.image"] for it in items]),
                     "observation.images.image2": torch.stack([it["observation.images.image2"] for it in items]),
                     "observation.state": torch.stack([it["observation.state"] for it in items]),
                     "action": torch.stack([it["action"] for it in items]), "task": [it["task"] for it in items]}
                pad = torch.stack([it["action_is_pad"] for it in items]).bool()
                valid = (~pad).float() if masked else torch.ones(pad.shape)
                raw = b["action"].numpy().copy()
                bp = pre(b)
                if not checked:
                    selfcheck(policy, bp)
                    checked = True
                t1 = time.time()
                L = fm_losses(policy, bp, a.k, valid)
                with open(fo, "a") as f:
                    for i, (j, e, fi, s) in enumerate(chunk):
                        nv = int(valid[i].sum())
                        r = {"source": name, "root": rel, "row": j, "episode": e, "frame": fi, "segment": s,
                             "task": items[i]["task"], "loss": float(L[i].mean()), "loss_k": L[i].round(5).tolist(),
                             "n_valid": nv, "n_pad": int(pad[i].sum())} | roughness(raw[i][: max(nv, 4)])
                        f.write(json.dumps(r) + "\n")
                print(f"  {b0 + len(chunk)}/{len(anchors)} {time.time() - t1:.1f} s/batch  mean {L.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
