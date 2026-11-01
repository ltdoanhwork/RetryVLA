"""Part 4 risk probe: does a policy's embedding separate high- vs low-C_pi failure states?

python scripts/af_probe.py --futures data/futures/af_m5b/futures.pt --out runs/_analysis/af_probe \
    --policy pi1=runs/<pi1>/checkpoints/last/pretrained_model --policy arm3=... [--policy ...]

Embedding = the action-expert hidden state at the L_future hook point (input of action_out_proj), mean-pooled over
the chunk, at flow time t = 1 (x_t = pure noise, fixed seed: the first denoising step at inference, independent of
any action label). States = the failure observations in futures.pt (256 M5b train failure states).
Labels (median split): v0 = critic C_pi(z0) of the failure state itself; best = max C_pi over its K C_pi-filtered
imagined futures (how recoverable the WM+critic think the state is). Probe = standardized logistic regression,
5-fold GroupKFold by scene (no scene in both train and test folds); report AUC mean over folds and a
label-permutation null (20 shuffles). Run on a GPU (one SmolVLA at a time, ~3 GB).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@torch.no_grad()
def embed(path: str, obs: dict, dev, bs: int = 32, seed: int = 0) -> np.ndarray:
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    pol = SmolVLAPolicy.from_pretrained(path)
    pol.config.device = str(dev)
    pol.to(dev).eval()
    pre, _ = make_pre_post_processors(pol.config, pretrained_path=path,
                                      preprocessor_overrides={"device_processor": {"device": str(dev)}})
    cap = {}
    h = pol.model.action_out_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("h", i[0]))
    n, out = len(obs["task"]), []
    C, A = pol.config.chunk_size, pol.config.max_action_dim
    g = torch.Generator(device=dev).manual_seed(seed)
    for s in range(0, n, bs):
        idx = list(range(s, min(n, s + bs)))
        img = lambda k: obs[k][idx].permute(0, 3, 1, 2).float() / 255.0  # noqa: E731
        b = {"observation.images.image": img("image"), "observation.images.image2": img("image2"),
             "observation.state": obs["state"][idx].float(), "task": [obs["task"][i] for i in idx],
             "action": torch.zeros(len(idx), C, pol.config.action_feature.shape[0]),
             "action_is_pad": torch.zeros(len(idx), C, dtype=torch.bool)}
        b = pre(b)
        noise = torch.randn((len(idx), C, A), generator=g, device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            pol.forward(b, noise=noise, time=torch.ones(len(idx), device=dev))
        out.append(cap.pop("h").float().mean(1).cpu().numpy())
    h.remove()
    del pol
    torch.cuda.empty_cache()
    return np.concatenate(out)


def probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_perm: int = 20, seed: int = 0) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def cv_auc(yy):
        aucs = []
        for tr, te in GroupKFold(5).split(X, yy, groups):
            if len(set(yy[te])) < 2 or len(set(yy[tr])) < 2:
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(X[tr], yy[tr])
            aucs.append(roc_auc_score(yy[te], m.predict_proba(X[te])[:, 1]))
        return float(np.mean(aucs)) if aucs else float("nan")

    rng = np.random.default_rng(seed)
    auc = cv_auc(y)
    null = [cv_auc(rng.permutation(y)) for _ in range(n_perm)]
    return {"auc": auc, "null_mean": float(np.mean(null)), "null_95": float(np.quantile(null, 0.95)),
            "p_perm": float((1 + sum(a >= auc for a in null)) / (1 + n_perm))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--futures", required=True)
    ap.add_argument("--policy", action="append", required=True, help="name=checkpoint_dir")
    ap.add_argument("--out", default="runs/_analysis/af_probe")
    a = ap.parse_args()
    d = torch.load(a.futures, map_location="cpu", weights_only=False)
    obs, fut = d["obs"], d["futures"]
    var = np.asarray(fut["variant"])
    o_f, c_f = fut["obs"].numpy()[var == "filtered"], fut["cpi"].numpy()[var == "filtered"]
    best = np.full(len(obs["task"]), np.nan)
    for o in np.unique(o_f):
        best[o] = c_f[o_f == o].max()
    v0 = obs["v0"].numpy()
    labels = {"v0": (v0 > np.median(v0)).astype(int), "best": (best > np.nanmedian(best)).astype(int)}
    groups = np.asarray([s for s in obs["scene"]])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {"n_states": int(len(v0)), "n_scenes": int(len(set(groups))),
           "label_stats": {"v0_median": float(np.median(v0)), "best_median": float(np.nanmedian(best)),
                           "spearman_v0_best": float(__import__("scipy.stats").stats.spearmanr(v0, best).statistic)}}
    # reference: the DINOv2 latent of the state itself (what the critic sees) -> upper-ish bound for a linear probe
    Xz = obs["z0_pool"].numpy()
    res["dino_z0"] = {k: probe(Xz, y, groups) for k, y in labels.items()}
    print("dino_z0", res["dino_z0"], flush=True)
    for spec in a.policy:
        name, path = spec.split("=", 1)
        X = embed(path, obs, dev)
        res[name] = {k: probe(X, y, groups) for k, y in labels.items()}
        res[name]["path"] = path
        print(name, res[name], flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "summary.json.tmp"
    tmp.write_text(json.dumps(res, indent=1))
    tmp.replace(out / "summary.json")
    lines = ["| embedding | AUC v0 (null95) | AUC best (null95) |", "|---|---|---|"]
    for k, r in res.items():
        if isinstance(r, dict) and "v0" in r:
            lines.append(f"| {k} | {r['v0']['auc']:.3f} ({r['v0']['null_95']:.3f}) | "
                         f"{r['best']['auc']:.3f} ({r['best']['null_95']:.3f}) |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
