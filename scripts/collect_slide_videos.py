#!/usr/bin/env python
"""Collect stage demo videos into slides/assets/video/: re-encode to H.264 yuv420p (PowerPoint-safe),
cap size, write a poster frame and a JSON manifest with sizes."""

from __future__ import annotations

import glob
import json
import subprocess
from pathlib import Path

import imageio
import imageio_ffmpeg

REPO = Path(__file__).resolve().parents[1]
V = REPO / "runs/_analysis/viz"
OUT = REPO / "slides/assets/video"
FF = imageio_ffmpeg.get_ffmpeg_exe()

VIDEOS = {
    "nominal_demo": V / "0_nominal_libero_demo/video.mp4",
    "m1_trivial": V / "1_trivial_open_gripper/video.mp4",
    "m1_primitive": V / "2_primitive_then_smolvla/video.mp4",
    "m1_cem_bad": V / "3_cem_hard_instance/video.mp4",
    "m1_vla_sample": V / "4_vla_sample_recovery/video.mp4",
    "c1_success": V / "c1_between_the_plate_and_the_rame_k11/video.mp4",
    "c1_fail": V / "c1_between_the_plate_and_the_rame_k0/video.mp4",
    "m3_wm_drawer_ep031": V / "c4_wm_in_the_top_drawer_of_t_episode_031_step_0180_open1/video.mp4",
    "m3_wm_drawer_ep018": V / "c4_wm_in_the_top_drawer_of_t_episode_018_step_0180_open1/video.mp4",
    "m3_compare": V / "compare_drawer_ep031/side_by_side.mp4",
    "distill_drawer": V / "distill_drawer_ep011/side_by_side.mp4",
    "distill_stove": V / "distill_stove_ep032/side_by_side.mp4",
}
# pi0 failing on held-out test scenes (from the M0c eval videos)
m0 = sorted(glob.glob(str(REPO / "runs/*M0_recover_bddl/videos/*top_drawer*fail.mp4")))
if m0:
    VIDEOS["m0_pi0_fail_drawer"] = Path(m0[0])
m0b = sorted(glob.glob(str(REPO / "runs/*M0_recover_bddl/videos/*on_the_ramekin*fail.mp4")))
if m0b:
    VIDEOS["m0_pi0_fail_ramekin"] = Path(m0b[0])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man = {}
    for k, src in VIDEOS.items():
        if not src.exists():
            print("missing", k, src)
            continue
        dst = OUT / f"{k}.mp4"
        subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "slow", "-crf", "24", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-movflags", "+faststart", "-an", str(dst)], check=True)
        r = imageio.get_reader(dst)
        meta = r.get_meta_data()
        n = r.count_frames()
        poster = r.get_data(min(n - 1, int(n * 0.35)))
        imageio.imwrite(OUT / f"{k}_poster.png", poster)
        man[k] = {"w": poster.shape[1], "h": poster.shape[0], "sec": round(n / meta.get("fps", 20), 1),
                  "kb": dst.stat().st_size // 1024}
        print(k, man[k])
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
