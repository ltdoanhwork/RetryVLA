#!/usr/bin/env python
"""Step-by-step visualizations of the training data.

For each selected trajectory writes to runs/_analysis/viz/<name>/:
  strip.png   - frames at evenly spaced steps: agent cam (top) + wrist cam (bottom), annotated with
                step, segment (SEARCH recovery / SmolVLA continuation / DEMO), and the action taken
  actions.png - the 7 action dims over time, recovery segment shaded
  video.mp4   - full trajectory with the same overlay
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "runs" / "_analysis" / "viz"
COL = {0: (255, 140, 0), 1: (0, 160, 255), 2: (60, 180, 75)}
LAB = {0: "SEARCH recovery", 1: "SmolVLA continuation", 2: "LIBERO demo"}
try:
    FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
    FONT_S = ImageFont.truetype("DejaVuSans.ttf", 11)
except Exception:
    FONT = FONT_S = ImageFont.load_default()


def frames_of(ds) -> list[dict]:
    out = []
    for i in range(ds.num_frames):
        it = ds[i]
        out.append({
            "img": (it["observation.images.image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
            "wrist": (it["observation.images.image2"].permute(1, 2, 0).numpy() * 255).astype(np.uint8),
            "state": it["observation.state"].numpy().reshape(-1),
            "action": it["action"].numpy().reshape(-1)[:7] if it["action"].ndim == 1 else it["action"].numpy()[0],
            "seg": int(it["segment"]) if "segment" in it else 2,
        })
    return out


def annotate(f: dict, t: int, T: int) -> np.ndarray:
    tile = np.concatenate([f["img"], f["wrist"]], 0)  # 512 x 256
    im = Image.fromarray(tile).convert("RGB")
    canvas = Image.new("RGB", (256, 512 + 74), (255, 255, 255))
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    c = COL[f["seg"]]
    d.rectangle([0, 0, 255, 18], fill=c)
    d.text((4, 2), f"t={t}/{T - 1}  {LAB[f['seg']]}", fill=(255, 255, 255), font=FONT)
    a = f["action"]
    grip = "CLOSE" if a[6] > 0 else "OPEN"
    d.text((4, 516), f"move  dx {a[0]:+.2f} dy {a[1]:+.2f} dz {a[2]:+.2f}", fill=(0, 0, 0), font=FONT_S)
    d.text((4, 532), f"rot   {a[3]:+.2f} {a[4]:+.2f} {a[5]:+.2f}", fill=(0, 0, 0), font=FONT_S)
    d.text((4, 548), f"gripper cmd: {grip}   finger q={f['state'][6]:.3f}", fill=(0, 0, 0), font=FONT_S)
    d.text((4, 564), f"eef z={f['state'][2]:.3f}", fill=(0, 0, 0), font=FONT_S)
    return np.asarray(canvas)


def render(name: str, frames: list[dict], title: str, n_tiles: int = 8) -> None:
    import imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    od = OUT / name
    od.mkdir(parents=True, exist_ok=True)
    T = len(frames)
    # strip: always include the first frame, the last recovery frame, and the last frame
    rec_end = max([i for i, f in enumerate(frames) if f["seg"] == 0], default=None)
    idx = set(np.linspace(0, T - 1, n_tiles).astype(int).tolist())
    if rec_end is not None:
        idx |= {rec_end, min(rec_end + 1, T - 1)}
    idx = sorted(idx)
    tiles = [annotate(frames[i], i, T) for i in idx]
    gap = np.full((tiles[0].shape[0], 6, 3), 255, np.uint8)
    strip = np.concatenate(sum([[t, gap] for t in tiles], [])[:-1], 1)
    head = Image.new("RGB", (strip.shape[1], 30), (255, 255, 255))
    ImageDraw.Draw(head).text((6, 6), title, fill=(0, 0, 0), font=FONT)
    imageio.imwrite(od / "strip.png", np.concatenate([np.asarray(head), strip], 0))
    # action plot
    A = np.stack([f["action"] for f in frames])
    seg = np.array([f["seg"] for f in frames])
    fig, ax = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True)
    for k, lab in enumerate(["dx", "dy", "dz", "droll", "dpitch", "dyaw"]):
        ax[0].plot(A[:, k], label=lab, lw=1.3 if k < 3 else 0.8, ls="-" if k < 3 else ":")
    ax[1].step(range(T), A[:, 6], where="post", color="k", label="gripper cmd (+1 close / -1 open)")
    ax[1].plot([f["state"][6] * 25 for f in frames], color="tab:red", label="finger opening x25")
    for a_ in ax:
        if (seg == 0).any():
            a_.axvspan(0, int(np.nonzero(seg == 0)[0].max()) + 1, color=(1, 0.55, 0), alpha=0.15)
        a_.legend(fontsize=7, ncol=4, loc="upper right")
    ax[0].set(title=f"{title}  (shaded = searched recovery segment)", ylabel="arm action")
    ax[1].set(xlabel="env step (20 Hz)", ylabel="gripper")
    plt.tight_layout()
    plt.savefig(od / "actions.png", dpi=100)
    plt.close(fig)
    imageio.mimwrite(od / "video.mp4", [annotate(f, i, T) for i, f in enumerate(frames)], fps=15)
    print(f"{name}: T={T}, recovery={(seg == 0).sum()}, continuation={(seg == 1).sum()} -> {od}")


def pick(root: str, cond) -> dict | None:
    rows = [json.loads(l) for l in open(REPO / "data/recovery" / root / "retryvla/trajectories.jsonl")]
    rows = [r for r in rows if cond(r)]
    return sorted(rows, key=lambda r: -r.get("score", 0))[0] if rows else None


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from retryvla.data.mix import patch_lerobot_index_mapping, spatial_episode_ids

    patch_lerobot_index_mapping()
    cases = [
        ("1_trivial_open_gripper", "m1_random_spatial_s1", lambda r: r["length"] <= 12,
         "Trivial recovery: bowl already over the plate, a short searched segment just releases it"),
        ("2_primitive_then_smolvla", "m1_random_spatial_s1",
         lambda r: str(r.get("source", "")).startswith("prim") and r["continuation_len"] > 40,
         "Primitive recovery (hand-written move) -> SmolVLA takes over and finishes"),
        ("3_cem_hard_instance", "m1b_hard_cem_local_s0",
         lambda r: r.get("source") == "cem_local" and r["continuation_len"] > 30,
         "Hard instance: CEM-refined recovery segment -> SmolVLA continuation succeeds"),
        ("4_vla_sample_recovery", "m1_oracle_cem_spatial_s1",
         lambda r: str(r.get("source", "")).startswith("vla") and r["continuation_len"] > 40,
         "Recovery = one of many SmolVLA samples (different noise) -> continuation succeeds"),
    ]
    for name, root, cond, title in cases:
        r = pick(root, cond)
        if r is None:
            print("no match for", name)
            continue
        ds = LeRobotDataset("retryvla/recovery", root=REPO / "data/recovery" / root, episodes=[int(r["episode_index"])])
        task = r["instance_id"].split("__")[0].replace("pick_up_the_black_bowl_", "")
        meta = f"[{r['instance_id'].split('__', 1)[1]}  task: ...{task[:40]}  src={r.get('source')}  it={r.get('iteration')}]"
        render(name, frames_of(ds), f"{title}   {meta}")
    # nominal demo for comparison
    eps = spatial_episode_ids("lerobot/libero", "libero_spatial")
    ds = LeRobotDataset("lerobot/libero", episodes=[eps[0]])
    render("0_nominal_libero_demo", frames_of(ds), "Original LIBERO-Spatial demo (the other 50-75% of every SFT batch)")


if __name__ == "__main__":
    main()
