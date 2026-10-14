#!/usr/bin/env python
"""Side-by-side: SmolVLA alone vs WM-guided recovery + SmolVLA, from the exact same failure state."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

INST = "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate__episode_031__step_0180__open1"
OUT = REPO / "runs/_analysis/viz/compare_drawer_ep031"


def main() -> None:
    import imageio
    from PIL import Image, ImageDraw

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from retryvla.data.mix import patch_lerobot_index_mapping
    from retryvla.envs.libero_env import LiberoSim, discover_scenes
    from retryvla.eval.eval_recover import noise_seed
    from retryvla.policy import SmolVLAWrapper
    from scripts.viz_trajectories import FONT, frames_of

    patch_lerobot_index_mapping()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = INST.rsplit("__", 1)[0]
    sc = next(s for s in discover_scenes(REPO / "data/assets/scenes/scenes/selected_bddl_scenes_spatial") if s.tag == tag)
    policy = SmolVLAWrapper("HuggingFaceVLA/smolvla_libero", "cuda", 10)
    sim = LiberoSim()

    # A) SmolVLA alone, 3 different policy seeds, from the identical failure state (same protocol seed as search)
    alone = []
    for seed in range(3):
        o = sim.load_scene(sc.bddl, sc.npy, gripper_close=False, noise_xy=0.02, noise_seed=noise_seed(7, sc.tag, 1),
                           settle_steps=20, build_seed=7)["obs"]
        frames, succ, t = [o["image"]], False, 0
        while t < 350 and not succ:
            ch = policy.predict_chunk([o], [sc.language_bddl], seeds=[seed * 1000 + t])[0][:10]
            for a in ch:
                r = sim.step(a)
                o, t = r.obs, t + 1
                frames.append(o["image"])
                if r.success:
                    succ = True
                    break
        alone.append((frames, succ, t))
        print(f"SmolVLA alone seed {seed}: {'SUCCESS' if succ else 'fail'} after {t} steps")

    # B) WM-guided: the verified trajectory found by M3 (searched segment + SmolVLA continuation)
    rows = [json.loads(l) | {"root": str(Path(f).parents[1])} for f in
            glob.glob(str(REPO / "data/recovery/m3_wm_search_hard_s*/retryvla/trajectories.jsonl")) for l in open(f)]
    r = max([x for x in rows if x["instance_id"] == INST], key=lambda x: x["score"])
    ds = LeRobotDataset("retryvla/recovery", root=r["root"], episodes=[int(r["episode_index"])])
    wm = frames_of(ds)
    wm_imgs, wm_seg = [f["img"] for f in wm], [f["seg"] for f in wm]

    def label(img, txt, col):
        im = Image.fromarray(np.ascontiguousarray(img)).convert("RGB")
        c = Image.new("RGB", (256, 276), (255, 255, 255))
        c.paste(im, (0, 20))
        d = ImageDraw.Draw(c)
        d.rectangle([0, 0, 255, 19], fill=col)
        d.text((4, 3), txt, fill=(255, 255, 255), font=FONT)
        return np.asarray(c)

    fa, sa, ta = alone[0]
    T = max(len(fa), len(wm_imgs))
    video = []
    for t in range(T):
        a = fa[min(t, len(fa) - 1)]
        b = wm_imgs[min(t, len(wm_imgs) - 1)]
        ta_txt = f"SmolVLA alone  t={min(t, len(fa) - 1)}" + ("  FAILED" if t >= len(fa) - 1 and not sa else "")
        sb = wm_seg[min(t, len(wm_imgs) - 1)]
        tb_txt = (f"WM-chosen segment t={t}" if sb == 0 else f"SmolVLA after WM t={min(t, len(wm_imgs) - 1)}") + \
                 ("  SUCCESS" if t >= len(wm_imgs) - 1 else "")
        video.append(np.concatenate([label(a, ta_txt, (150, 150, 150)),
                                     label(b, tb_txt, (255, 140, 0) if sb == 0 else (0, 160, 255))], 1))
    imageio.mimwrite(OUT / "side_by_side.mp4", video, fps=20)
    idx = [0, 25, 50, 90, 130, 180, min(T - 1, 260), T - 1]
    rows_img = [np.concatenate([video[min(i, T - 1)] for i in idx], 1)]
    imageio.imwrite(OUT / "side_by_side_strip.png", rows_img[0])
    # strips for the other SmolVLA-alone seeds
    for s, (f, ok, t) in enumerate(alone):
        sel = [f[min(i, len(f) - 1)] for i in [0, 50, 100, 150, 200, 250, 300, len(f) - 1]]
        imageio.imwrite(OUT / f"alone_seed{s}_strip.png",
                        np.concatenate([label(x, f"alone seed{s} t={j}", (150, 150, 150)) for x, j in
                                        zip(sel, [0, 50, 100, 150, 200, 250, 300, len(f) - 1])], 1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
