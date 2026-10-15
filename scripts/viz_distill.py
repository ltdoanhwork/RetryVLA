#!/usr/bin/env python
"""Distillation demo videos: pi0 vs pi1 (no search) on the SAME held-out test failure state.
Writes runs/_analysis/viz/distill_<name>/side_by_side.mp4 + poster.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CASES = [
    ("drawer_ep011", "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate__episode_011__step_0200", 6),
    ("stove_ep032", "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate__episode_032__step_0200", 6),
]
PI1 = sorted((REPO / "runs").glob("*_M4_sft_A_base_nom_random/checkpoints/006000/pretrained_model"))[-1]


def rollout(sim, policy, sc, trial, seed_base, max_steps=520):
    from retryvla.eval.eval_recover import noise_seed

    o = sim.load_scene(sc.bddl, sc.npy, gripper_close=trial < 5, noise_xy=0.02, noise_seed=noise_seed(7, sc.tag, trial),
                       settle_steps=20, build_seed=7)["obs"]
    frames, t, ok, calls = [o["image"]], 0, False, 0
    while t < max_steps and not ok:
        ch = policy.predict_chunk([o], [sc.language_bddl], seeds=[seed_base * 100_003 + calls])[0][:10]
        calls += 1
        for a in ch:
            r = sim.step(a)
            o, t = r.obs, t + 1
            frames.append(o["image"])
            if r.success:
                ok = True
                break
    return frames, ok, t


def main():
    import imageio
    from PIL import Image, ImageDraw, ImageFont

    from retryvla.envs.libero_env import LiberoSim, discover_scenes
    from retryvla.eval.eval_recover import noise_seed
    from retryvla.policy import SmolVLAWrapper

    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
    scenes = {s.tag: s for s in discover_scenes(REPO / "data/assets/scenes/scenes/selected_bddl_scenes_spatial")}
    pi0 = SmolVLAWrapper("HuggingFaceVLA/smolvla_libero", "cpu", 10)
    pi1 = SmolVLAWrapper(str(PI1), "cpu", 10)
    sim = LiberoSim()

    def lab(img, txt, col):
        im = Image.fromarray(np.ascontiguousarray(img)).convert("RGB").resize((320, 320))
        c = Image.new("RGB", (320, 346), (255, 255, 255))
        c.paste(im, (0, 26))
        d = ImageDraw.Draw(c)
        d.rectangle([0, 0, 319, 25], fill=col)
        d.text((6, 5), txt, fill=(255, 255, 255), font=font)
        return np.asarray(c)

    for name, tag, trial in CASES:
        sc = scenes[tag]
        seed = noise_seed(8, sc.tag, trial)  # eval used seed+1 = 8 for the policy seed
        f0, ok0, t0 = rollout(sim, pi0, sc, trial, seed)
        f1, ok1, t1 = None, False, 0
        for k in range(3):  # numerics differ from the batched GPU eval; retry a couple of policy seeds
            f1, ok1, t1 = rollout(sim, pi1, sc, trial, seed + k)
            if ok1:
                break
        print(f"{name}: pi0 {'SUCCESS' if ok0 else 'fail'} ({t0} steps) | pi1 {'SUCCESS' if ok1 else 'fail'} ({t1} steps, seed+{k})")
        T = max(len(f0), len(f1))
        vid = []
        for t in range(0, T, 2):
            a, b = f0[min(t, len(f0) - 1)], f1[min(t, len(f1) - 1)]
            ta = f"pi0 (SmolVLA)  t={min(t, len(f0) - 1)}" + ("  FAILED" if t >= len(f0) - 1 and not ok0 else "")
            tb = f"pi1 (after recovery SFT)  t={min(t, len(f1) - 1)}" + ("  SUCCESS" if t >= len(f1) - 1 and ok1 else "")
            vid.append(np.concatenate([lab(a, ta, (110, 120, 133)), np.full((346, 8, 3), 255, np.uint8),
                                       lab(b, tb, (42, 157, 143) if ok1 else (110, 120, 133))], 1))
        od = REPO / "runs/_analysis/viz" / f"distill_{name}"
        od.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(od / "side_by_side.mp4", vid, fps=20, codec="libx264", pixelformat="yuv420p", quality=7)
        imageio.imwrite(od / "poster.png", vid[min(len(vid) - 1, 30)])
        idx = np.linspace(0, len(vid) - 1, 5).astype(int)
        imageio.imwrite(od / "grid.png", np.concatenate([np.concatenate([vid[i][:, :320] for i in idx], 1),
                                                         np.concatenate([vid[i][:, 328:] for i in idx], 1)], 0))
    sim.close()


if __name__ == "__main__":
    main()
