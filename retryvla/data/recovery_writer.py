"""Write verified recovery trajectories as a LeRobotDataset (v3) with the same core features
as ``lerobot/libero`` plus per-frame provenance:
  - ``segment``: 0 = searched recovery segment, 1 = nominal-policy continuation
  - ``instance_index``: index into ``instances.jsonl`` (failure instance metadata)
Episode-level metadata (scene, source, score, ...) goes to ``trajectories.jsonl``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

FPS = 10  # matches lerobot/libero metadata (one frame per env step)


def features(use_videos: bool = False) -> dict:
    img = {"dtype": "video" if use_videos else "image", "shape": (256, 256, 3),
           "names": ["height", "width", "channel"]}
    return {
        "observation.images.image": dict(img),
        "observation.images.image2": dict(img),
        "observation.state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        "segment": {"dtype": "int64", "shape": (1,), "names": None},
        "instance_index": {"dtype": "int64", "shape": (1,), "names": None},
    }


def _png_ok(path: Path) -> bool:
    from PIL import Image

    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


class RecoveryWriter:
    def __init__(self, root: str | Path, repo_id: str = "retryvla/recovery", overwrite: bool = False,
                 use_videos: bool = False, write_retries: int = 0, retry_wait_s: float = 60.0):
        """write_retries > 0 (opt-in; default keeps the old behaviour): before ``save_episode`` every PNG of the
        episode is checked and any unreadable one (lerobot's async image writer only LOGS write errors, e.g.
        EDQUOT 2026-10-22) is rewritten synchronously from the in-memory frame, retrying with linear backoff."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.write_retries, self.retry_wait_s = int(write_retries), float(retry_wait_s)
        self.root = Path(root)
        if self.root.exists():
            if not overwrite:
                raise FileExistsError(f"{self.root} exists (pass overwrite=True)")
            shutil.rmtree(self.root)
        self.ds = LeRobotDataset.create(repo_id=repo_id, fps=FPS, features=features(use_videos), root=self.root,
                                        robot_type="panda", use_videos=use_videos, image_writer_threads=4)
        self.side = self.root / "retryvla"
        self.side.mkdir(parents=True, exist_ok=True)
        self._traj = open(self.side / "trajectories.jsonl", "a")
        self._inst = open(self.side / "instances.jsonl", "a")
        self.instances: dict[str, int] = {}
        self.n_episodes = 0

    def add_instance(self, instance_id: str, meta: dict) -> int:
        if instance_id not in self.instances:
            self.instances[instance_id] = len(self.instances)
            self._inst.write(json.dumps({"instance_index": self.instances[instance_id],
                                         "instance_id": instance_id, **meta}) + "\n")
            self._inst.flush()
        return self.instances[instance_id]

    def add_trajectory(self, rec: dict, task: str, instance_id: str, meta: dict,
                       keep_continuation: bool = True) -> int:
        """rec = worker record dict: image, image2, state, action, segment arrays [T, ...]."""
        ii = self.instances[instance_id]
        T = len(rec["action"])
        if not keep_continuation:
            T = int((rec["segment"] == 0).sum())
        for t in range(T):
            self.ds.add_frame({
                "observation.images.image": rec["image"][t],
                "observation.images.image2": rec["image2"][t],
                "observation.state": rec["state"][t].astype(np.float32),
                "action": rec["action"][t].astype(np.float32),
                "segment": np.array([rec["segment"][t]], dtype=np.int64),
                "instance_index": np.array([ii], dtype=np.int64),
                "task": task,
            })
        if self.write_retries > 0 and not self.ds.meta.video_keys:
            self._repair_images(rec, T)
        self.ds.save_episode()
        ep = self.n_episodes
        self.n_episodes += 1
        n_rec = int((rec["segment"][:T] == 0).sum())
        self._traj.write(json.dumps({"episode_index": ep, "instance_index": ii, "instance_id": instance_id,
                                     "length": T, "recovery_len": n_rec, "continuation_len": T - n_rec,
                                     "task": task, **meta}, default=float) + "\n")
        self._traj.flush()
        return ep

    def _repair_images(self, rec: dict, T: int) -> None:
        import logging
        import time

        from lerobot.datasets.image_writer import image_array_to_pil_image

        log = logging.getLogger("recovery_writer")
        w = self.ds.writer
        w._wait_image_writer()
        buf = w.episode_buffer
        src = {"observation.images.image": rec["image"], "observation.images.image2": rec["image2"]}
        bad = [(k, t) for k in src for t in range(T) if not _png_ok(Path(buf[k][t]))]
        for attempt in range(self.write_retries + 1):
            if not bad:
                return
            log.warning("%s: %d unreadable PNGs in episode %d (e.g. %s), rewrite %d/%d", self.root.name, len(bad),
                        buf["episode_index"], Path(buf[bad[0][0]][bad[0][1]]).name, attempt + 1, self.write_retries)
            if attempt:
                time.sleep(self.retry_wait_s * attempt)
            still = []
            for k, t in bad:
                p = Path(buf[k][t])
                tmp = p.with_name(p.name + ".tmp.png")
                try:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    image_array_to_pil_image(np.asarray(src[k][t])).save(tmp, compress_level=1)
                    tmp.replace(p)
                except OSError:
                    tmp.unlink(missing_ok=True)
                if not _png_ok(p):
                    still.append((k, t))
            bad = still
        if bad:
            raise OSError(f"{self.root}: {len(bad)} PNGs still unwritable after {self.write_retries} retries")

    def close(self) -> None:
        self.ds.finalize()
        self._traj.close()
        self._inst.close()
