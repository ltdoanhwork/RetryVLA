"""Part 4 (action-free futures) must be OFF by default: the default MixDataset stream is bit-identical to the
pre-Part-4 implementation (git HEAD~ version loaded from git), FutureDataset / future_forward behave as specified.

python -m pytest -q tests/test_af_default.py   (CPU only, no sim, no checkpoints)
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
BASE_REV = "c6d3f8c"  # last commit before Part 4 touched retryvla/data/mix.py / retryvla/train/sft.py


def _old_mix():
    src = subprocess.run(["git", "show", f"{BASE_REV}:retryvla/data/mix.py"], cwd=REPO, capture_output=True,
                         text=True, check=True).stdout
    p = Path(tempfile.mkdtemp()) / "mix_old.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location("mix_old", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Seg:
    def __init__(self, seg):
        self.seg = seg

    def with_format(self, _):
        return self

    def select_columns(self, _):
        return {"segment": self.seg}


class FakeDS:
    def __init__(self, n, off, seg=None):
        self.n, self.off = n, off
        if seg is not None:
            self.hf_dataset = _Seg(seg)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.off + i)
        return {"observation.images.image": torch.rand(1, 3, 4, 4, generator=g),
                "observation.images.image2": torch.rand(3, 4, 4, generator=g),
                "observation.state": torch.rand(1, 8, generator=g), "action": torch.rand(5, 7, generator=g),
                "action_is_pad": torch.zeros(5, dtype=torch.bool), "task": f"t{self.off + i}", "extra": 1}


def _fake_futures(path):
    n_obs, k = 6, 3
    obs = {"instance_id": [f"i{j}" for j in range(n_obs)], "scene": [f"s{j // 2}" for j in range(n_obs)],
           "task": [f"task{j}" for j in range(n_obs)],
           "image": torch.randint(0, 255, (n_obs, 4, 4, 3), dtype=torch.uint8),
           "image2": torch.randint(0, 255, (n_obs, 4, 4, 3), dtype=torch.uint8),
           "state": torch.rand(n_obs, 8), "z0_pool": torch.rand(n_obs, 6), "v0": torch.rand(n_obs)}
    rows = [(o, v, r) for v in ("filtered", "random") for o in range(n_obs) for r in range(k)]
    fut = {"obs": torch.tensor([o for o, _, _ in rows]), "variant": [v for _, v, _ in rows],
           "rank": torch.tensor([r for _, _, r in rows]), "future_latent": torch.rand(len(rows), 6),
           "future_q": torch.rand(len(rows), 8), "cpi": torch.rand(len(rows)), "score": torch.rand(len(rows)),
           "unc": torch.rand(len(rows)), "sim_checked": torch.tensor([v == "filtered" for _, v, _ in rows]),
           "sim_success": torch.tensor([v == "filtered" and r == 0 for _, v, r in rows]),
           "actions": torch.zeros(len(rows), 5, 7, dtype=torch.float16)}
    torch.save({"obs": obs, "futures": fut}, path)


def _same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if isinstance(a[k], torch.Tensor):
            assert torch.equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


def test_default_mix_bit_identical():
    sys.path.insert(0, str(REPO))
    from retryvla.data.mix import MixDataset

    old = _old_mix()
    orig = FakeDS(50, 0)
    recs = [FakeDS(20, 1000, np.array([0] * 10 + [1] * 10)), FakeDS(7, 2000, np.array([0, 1, 1, 0, 2, 0, 1]))]
    for ratio in (0.0, 0.3, 0.5):
        a = old.MixDataset(orig, recs, ratio, 200, ("recovery", "continuation"), seed=3)
        b = MixDataset(orig, recs, ratio, 200, ("recovery", "continuation"), seed=3)
        c = MixDataset(orig, recs, ratio, 200, ("recovery", "continuation"), seed=3, futures=None, future_ratio=0.7)
        assert not b.with_future and not c.with_future and b.rec_index == a.rec_index
        for i in range(200):
            _same(a[i], b[i])
            _same(a[i], c[i])


def test_future_dataset_and_mix(tmp_path):
    sys.path.insert(0, str(REPO))
    from retryvla.data.mix import FutureDataset, MixDataset, select_futures

    p = tmp_path / "futures.pt"
    _fake_futures(p)
    d = torch.load(p, weights_only=False)
    rows = select_futures(d["futures"], "filtered", 8, seed=0)
    assert len(rows) == 8 and (d["futures"]["rank"][rows[:6]] == 0).all()  # round robin: all rank-0 first
    assert len(set(d["futures"]["obs"][rows[:6]].tolist())) == 6
    fd = FutureDataset(p, "filtered", 8, chunk=5, action_dim=7)
    it = fd[0]
    assert it["action_is_pad"].all() and not bool(it["has_action"]) and bool(it["has_future"])
    assert it["observation.images.image"].shape == (3, 4, 4) and it["observation.images.image"].max() <= 1
    assert torch.allclose(fd.target.mean(0), torch.zeros(6), atol=1e-6)
    orig = FakeDS(50, 0)
    recs = [FakeDS(20, 1000, np.array([0] * 20))]
    m = MixDataset(orig, recs, 1 / 3, 3000, ("recovery", "continuation"), seed=1, futures=[fd], future_ratio=0.25)
    src = np.array([int(m[i]["source"]) for i in range(3000)])
    fr = [(src == s).mean() for s in (0, 1, 2)]
    assert abs(fr[2] - 0.25) < 0.03 and abs(fr[0] - 0.5) < 0.03 and abs(fr[1] - 0.25) < 0.03, fr
    keys = {k for i in range(30) for k in m[i]}
    assert {"future_latent", "has_action", "has_future"} <= keys


def test_future_forward_masking():
    sys.path.insert(0, str(REPO))
    from retryvla.train.sft import future_forward

    class P:
        def forward(self, batch, reduction="mean"):
            per = batch["per"]
            return (per if reduction == "none" else per.mean()), {}

    B, hid = 6, 4
    head = torch.nn.Linear(hid, 3)
    per = torch.rand(B)
    fut = {"cap": {"h": torch.rand(B, 5, hid)}, "head": head, "loss": "cosine"}
    ha, hf = torch.ones(B, dtype=torch.bool), torch.zeros(B, dtype=torch.bool)
    loss, _, lf = future_forward(fut, P(), {"per": per}, ha, hf, torch.rand(B, 3))
    assert torch.allclose(loss, per.mean()) and float(lf) == 0.0       # all action samples -> = default FM mean
    ha = torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.bool)
    fut["cap"]["h"] = torch.rand(B, 5, hid)
    loss, _, lf = future_forward(fut, P(), {"per": per}, ha, ~ha, torch.rand(B, 3))
    assert torch.allclose(loss, per[ha].mean()) and 0 <= float(lf) <= 2  # unverified: no FM, only L_future
