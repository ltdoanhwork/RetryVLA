"""Offline proxy for "does distillation make the policy jerky?" (CPU, no sim).

pi1 rollouts were never logged with actions, so we compare predicted action chunks of three Spatial SmolVLA policies on
the SAME observations with the SAME flow-matching noise:
  pi0          HuggingFaceVLA/smolvla_libero (the bank's continuation policy)
  pi1-nominal  M1 demo-only SFT (same init / steps as pi1, no recovery data)  = the fair control
  pi1          M5c wm_diverse_rr (the distilled policy, RSR 0.439)
Observations: m5b wm_diverse bank (train scenes) at (a) the failure state = first frame of segment 0 and (b) a
mid-continuation state of segment 1 (pi0's own states). Metrics on the executed prefix (first 10) and the whole chunk;
plus the distance of the predicted chunk to the bank's searched bridge at (a) (imitation of the bridge).

usage: CUDA_VISIBLE_DEVICES= python scripts/smoothness_policy_proxy.py --n_obs 48 --seeds 2
"""

from __future__ import annotations

import argparse
import glob
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
POLICIES = {
    "pi0": "HuggingFaceVLA/smolvla_libero",
    "pi1-nominal": "runs/20261013-080655_M1_sft_nominal_only/checkpoints/006000/pretrained_model",
    "pi1": "runs/20261016-042447_M5_m5c_sft_wm_diverse_rr/checkpoints/006000/pretrained_model",
}


def dec(x):
    return np.array(Image.open(io.BytesIO(x["bytes"])).convert("RGB"))


def load_obs(n_obs: int, rng: np.random.RandomState):
    """Returns list of dicts {kind, obs, task, bridge}."""
    dirs = sorted(d for d in glob.glob(str(REPO / "data/recovery/m5b/s*/p*/wm_diverse"))
                  if (Path(d).parent / "CLOSED").exists())
    items = []
    for d in dirs:
        traj = {json.loads(l)["episode_index"]: json.loads(l) for l in open(Path(d) / "retryvla/trajectories.jsonl")}
        for f in sorted((Path(d) / "data").rglob("*.parquet")):
            t = pq.read_table(f, columns=["episode_index", "frame_index", "segment", "action"]).to_pandas()
            t["row"] = np.arange(len(t))
            for ep, e in t.groupby("episode_index"):
                e = e.sort_values("frame_index")
                seg = np.array([np.ravel(v)[0] for v in e.segment.values])
                i0, i1 = np.nonzero(seg == 0)[0], np.nonzero(seg == 1)[0]
                if len(i0) < 10 or len(i1) < 30:
                    continue
                items.append((f, e.row.values, np.stack(e.action.values), i0, i1, traj[int(ep)]["task"]))
    pick = sorted(rng.choice(len(items), size=min(n_obs, len(items)), replace=False), key=lambda j: str(items[j][0]))
    out = []
    cache = {}
    for j in pick:
        f, rows, A, i0, i1, task = items[j]
        if f not in cache:
            cache.clear()
            cache[f] = pq.read_table(f, columns=["observation.state", "observation.images.image",
                                                 "observation.images.image2"]).to_pandas()
        tab = cache[f]
        for kind, idx in (("failure", i0[0]), ("continuation", i1[len(i1) // 3])):
            r = tab.iloc[int(rows[idx])]
            obs = {"image": dec(r["observation.images.image"]), "image2": dec(r["observation.images.image2"]),
                   "state": np.asarray(r["observation.state"], np.float32)}
            out.append({"kind": kind, "obs": obs, "task": task,
                        "bridge": A[i0] if kind == "failure" else None,
                        "logged": A[idx: idx + 50]})
    return out


def met(c: np.ndarray) -> dict:
    arm = c[:, :6].astype(np.float64)
    return {"d1": np.linalg.norm(np.diff(arm, axis=0), axis=1).mean(),
            "d3": np.linalg.norm(np.diff(arm, 3, axis=0), axis=1).mean(),
            "grip": float((np.diff(np.sign(c[:, 6])) != 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_obs", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="runs/_analysis/smoothness")
    args = ap.parse_args()
    torch.set_num_threads(16)
    rng = np.random.RandomState(0)
    items = load_obs(args.n_obs, rng)
    print("observations", len(items), flush=True)
    from retryvla.policy import SmolVLAWrapper

    rows = []
    for name, path in POLICIES.items():
        p = str(REPO / path) if path.startswith("runs/") else path
        pol = SmolVLAWrapper(p, "cpu", 10)
        for i, it in enumerate(items):
            seeds = [1000 * i + s for s in range(args.seeds)]
            ch = pol.predict_chunk([it["obs"]] * len(seeds), [it["task"]] * len(seeds), seeds=seeds)
            for s, c in zip(seeds, ch):
                r = {"policy": name, "i": i, "kind": it["kind"], "seed": s}
                r.update({f"{k}_50": v for k, v in met(c).items()})
                r.update({f"{k}_10": v for k, v in met(c[:12]).items()})
                if it["bridge"] is not None:
                    H = min(len(it["bridge"]), 50)
                    r["dist_bridge"] = float(np.abs(c[:H, :6] - it["bridge"][:H, :6]).mean())
                rows.append(r)
            if i % 16 == 0:
                print(name, i, flush=True)
        del pol
    import pandas as pd

    df = pd.DataFrame(rows)
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "policy_proxy.csv", index=False)
    agg = df.groupby(["kind", "policy"])[["d1_10", "d3_10", "d1_50", "d3_50", "grip_50", "dist_bridge"]].mean()
    # paired deltas vs pi1-nominal (per observation, averaged over seeds) with bootstrap CI
    lines = ["| kind | policy | d1 (first 12) | d3 (first 12) | d1 (50) | d3 (50) | grip toggles / chunk | L1 dist to bridge |",
             "|---|---|---|---|---|---|---|---|"]
    for (k, pname), r in agg.iterrows():
        lines.append(f"| {k} | {pname} | {r.d1_10:.3f} | {r.d3_10:.3f} | {r.d1_50:.3f} | {r.d3_50:.3f} | {r.grip_50:.2f} | "
                     + ("-" if np.isnan(r.dist_bridge) else f"{r.dist_bridge:.3f}") + " |")
    lines.append("")
    lines.append("Paired delta pi1 - pi1-nominal (per observation, mean over noise seeds; 95 % bootstrap CI):")
    b = np.random.RandomState(1)
    for k in ("failure", "continuation"):
        for m in ("d3_10", "d3_50", "d1_50", "dist_bridge"):
            piv = df[df.kind == k].groupby(["i", "policy"])[m].mean().unstack()
            if m not in df or piv.isna().all().all() or "pi1" not in piv:
                continue
            dlt = (piv["pi1"] - piv["pi1-nominal"]).dropna().values
            if not len(dlt):
                continue
            bs = [b.choice(dlt, len(dlt)).mean() for _ in range(2000)]
            lines.append(f"- {k} {m}: {dlt.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] (n {len(dlt)})")
    txt = "\n".join(lines)
    (out / "policy_proxy.md").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
