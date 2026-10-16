#!/usr/bin/env python
"""Add the M5 demo videos to slides/assets/video/ (H.264 yuv420p, poster frame) and MERGE into manifest.json
(the older collect_slide_videos.py rewrites the manifest; this one only adds/updates its own keys)."""

from __future__ import annotations

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
    "m5_explain_drawer": (V / "explainer/in_the_top_drawer_of_the_wooden_cabinet__open6/explainer.mp4", 0.30),
    "m5_explain_cabinet": (V / "explainer/on_the_wooden_cabinet_and_place_it_on_th__open8/explainer.mp4", 0.30),
    "m5_explain_ramekin": (V / "explainer/on_the_ramekin_and_place_it_on_the_plate__open9/explainer.mp4", 0.30),
    "m5_e2e_test_drawer": (V / "e2e/test_in_the_top_drawer_of_the_wooden_ca__episode_040_step_0200__close1/e2e.mp4", 0.55),
    "m5_e2e_train_drawer": (V / "e2e/train_in_the_top_drawer_of_the_wooden_ca__episode_031_step_0200__open6/e2e.mp4", 0.55),
}


def main() -> None:
    man = json.loads((OUT / "manifest.json").read_text())
    for k, (src, at) in VIDEOS.items():
        dst = OUT / f"{k}.mp4"
        subprocess.run([FF, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-preset", "slow", "-crf", "23", "-vf", "scale='min(1600,iw)':-2",
                        "-movflags", "+faststart", "-an", str(dst)], check=True)
        r = imageio.get_reader(dst)
        n = r.count_frames()
        fps = r.get_meta_data().get("fps", 15)
        poster = r.get_data(min(n - 1, int(n * at)))
        imageio.imwrite(OUT / f"{k}_poster.png", poster)
        man[k] = {"w": poster.shape[1], "h": poster.shape[0], "sec": round(n / fps, 1), "kb": dst.stat().st_size // 1024}
        print(k, man[k])
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
