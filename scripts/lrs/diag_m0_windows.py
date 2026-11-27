"""LRS-G debug 2026-11-27: per-dim / per-episode flow-matching loss of the M0 init on D0 vs LRS teacher windows, and
M0's own predicted actions on every teacher-window frame (control C3: same observations, M0 action labels).

python -m scripts.lrs.diag_m0_windows [--rec data/recovery/lrs_goal_v2_ds data/recovery/lrs_goal_v2gt_ds] [--k 4]
-> runs/_analysis/lrs_debug/m0_loss.json, runs/_analysis/lrs_debug/m0_pred_<name>.npz (first action of M0's chunk at
   every frame, fixed noise seed = global frame index, + the full chunk)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from retryvla import REPO_ROOT
from retryvla.policy import pi0_lora as P
from retryvla.train.pi0_sft import D0, Pool

M0 = "runs/goal_rsi/m0_s7/ckpt_04000"
OUT = REPO_ROOT / "runs/_analysis/lrs_debug"


def per_elem_loss(policy, b, K):
    """[B, T, A] FM loss averaged over K (noise, time) draws."""
    p = policy
    images, img_masks = p._preprocess_images(b)
    lt, lm = b["observation.language.tokens"], b["observation.language.attention_mask"]
    st, ac = p.prepare_state(b), p.prepare_action(b)
    A = p.config.output_features["action"].shape[0]
    acc = 0
    for k in range(K):
        torch.manual_seed(1234 + k)
        acc = acc + p.model.forward(images, img_masks, lt, lm, st, ac)[:, :, :A].float()
    return acc / K


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", nargs="*", default=["data/recovery/lrs_goal_v2_ds", "data/recovery/lrs_goal_v2gt_ds"])
    ap.add_argument("--n_d0", type=int, default=640)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--no_pred", action="store_true")
    ap.add_argument("--pred_only", action="store_true", help="skip the loss part (m0_loss.json kept)")
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors

    OUT.mkdir(parents=True, exist_ok=True)
    policy, cfg, _ = P.load_policy(M0, device="cuda", for_training=True)
    policy.eval()
    stats = LeRobotDatasetMetadata("local/d0_train", root=REPO_ROOT / D0).stats
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats,
                                         preprocessor_overrides={"device_processor": {"device": "cuda"}})
    res = json.loads((OUT / "m0_loss.json").read_text()) if (OUT / "m0_loss.json").exists() else {}
    for rec in a.rec:
        name = Path(rec).name
        data = Pool([REPO_ROOT / D0, REPO_ROOT / rec], cfg.chunk_size)
        data.masked = True
        data.lmin_filter(5)
        rng = np.random.default_rng(0)
        d0 = [i for i, (k, _) in enumerate(data.idx) if k == 0]
        rc = [i for i, (k, _) in enumerate(data.idx) if k == 1]
        sets = {} if a.pred_only else {"rec": rc} if "D0" in res else {"D0": list(rng.choice(d0, a.n_d0, replace=False)), "rec": rc}
        for sname, idx in sets.items():
            L, V, EP = [], [], []
            for s in range(0, len(idx), a.bs):
                items = [data[int(j)] for j in idx[s: s + a.bs]]
                EP += [int(data.ds[data.idx[int(j)][0]].hf_dataset[data.idx[int(j)][1]]["episode_index"]) for j in idx[s: s + a.bs]]
                b = torch.utils.data.default_collate(items)
                b.pop("src")
                valid, _ = b.pop("valid"), b.pop("w")
                b = pre(b)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    l = per_elem_loss(policy, b, a.k)
                L.append(l.cpu().numpy())
                V.append(valid.numpy())
            L, V = np.concatenate(L), np.concatenate(V)
            key = "D0" if sname == "D0" else name
            m = V[:, :, None]
            per_dim = (L * m).sum((0, 1)) / m.sum()
            per_anchor = (L.mean(-1) * V).sum(1) / V.sum(1)
            first10 = (L[:, :10] * m[:, :10]).sum((0, 1)) / m[:, :10].sum()
            eps = np.asarray(EP)
            res[key] = {"n": int(len(L)), "mean": float((L.mean(-1) * V).sum() / V.sum()),
                        "per_dim": per_dim.round(4).tolist(), "per_dim_first10": first10.round(4).tolist(),
                        "anchor_q50_q90_q99": np.quantile(per_anchor, [0.5, 0.9, 0.99]).round(3).tolist()}
            if key != "D0":
                res[key]["per_episode"] = {int(e): round(float(per_anchor[eps == e].mean()), 3) for e in np.unique(eps)}
                tot = (L.mean(-1) * V).sum()
                res[key]["share_of_loss_top3_eps"] = sorted(
                    ((int(e), round(float((L.mean(-1) * V)[eps == e].sum() / tot), 3)) for e in np.unique(eps)),
                    key=lambda x: -x[1])[:5]
            print(key, json.dumps({k: v for k, v in res[key].items() if k != "per_episode"}), flush=True)
        (OUT / "m0_loss.json").write_text(json.dumps(res, indent=1))
        if a.no_pred:
            continue
        # M0 predictions on EVERY frame of the recovery set (incl. the last L_min - 1 of each window, which are chunk
        # targets but not anchors): row j -> first action + chunk, noise seeded by the row index
        allp = Pool([REPO_ROOT / rec], cfg.chunk_size)
        PR = []
        for s in range(0, len(allp), a.bs):
            js = list(range(s, min(len(allp), s + a.bs)))
            b = torch.utils.data.default_collate([allp[j] for j in js])
            b.pop("src")
            b = pre(b)
            g = torch.Generator(device="cuda")
            noise = []
            for j in js:
                g.manual_seed(int(j))
                noise.append(torch.randn(cfg.chunk_size, cfg.max_action_dim, generator=g, device="cuda"))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                policy.reset()
                PR.append(post(policy.predict_action_chunk(b, noise=torch.stack(noise))).float().cpu().numpy())
        PR = np.concatenate(PR)
        lab = np.stack([np.asarray(allp.ds[0].hf_dataset[j]["action"]) for j in range(len(allp))])
        res.setdefault(name, {})
        res[name]["m0_pred_vs_label_mae_per_dim"] = np.abs(PR[:, 0] - lab).mean(0).round(4).tolist()
        res[name]["m0_pred_grip_agree"] = float((np.sign(PR[:, 0, 6]) == np.sign(lab[:, 6])).mean())
        print(name, "pred vs label", res[name]["m0_pred_vs_label_mae_per_dim"], res[name]["m0_pred_grip_agree"], flush=True)
        np.savez_compressed(OUT / f"m0_pred_{name}.npz", first=PR[:, 0], chunk=PR.astype(np.float16), label=lab)
        (OUT / "m0_loss.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
