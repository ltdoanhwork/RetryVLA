"""V2R data / training helpers (PLAN "V2R"; all opt-in, default sft path unchanged).

1. Recovery GROUPS with per-group counts, physical-plausibility keep lists and SMOOTH-V smooth-first ranking:
   train.recovery_groups = [{datasets: [glob...], n: 208, keep_list: [phys_<arm>.jsonl...] | null,
                             rank: score | d3, require_closed: false}, ...]
   Each group = round-robin over its instances (instance order shuffled with train.seed like recovery_episode_subsets
   roundrobin); inside an instance trajectories ordered by search score (rank score, = pi1) or by bridge d3 ascending
   (rank d3, smooth-first); keep_list rows with keep=false are not eligible.
2. AUX structured heads (train.aux_beta > 0): targets from the privileged replays of rec_phys (<bank>/phys/<arm>/*.npz,
   geom [T+1, G] after every step) for recovery frames; nominal frames are masked. Per frame f, k in K_AUX:
   chain-of-dynamics = grasped (finger contact and lift >= 2 cm), finger contact (BCE), object lift (cm), object-goal
   xy distance (cm); point-flow summary = 3-D displacement (cm) of the manipulated object and of the goal object from
   f to f + k, k in 10..40 (rigid bodies). Head = MLP on the action-expert hidden states (input of action_out_proj,
   mean over the chunk); loss FM + aux_beta * aux; the head is saved apart (aux_head.pt), never part of the policy.
"""

from __future__ import annotations

import glob as _glob
import json
import logging
from pathlib import Path

import numpy as np
import torch

from retryvla import REPO_ROOT
from retryvla.data.mix import CORE_KEYS, MixDataset, expand_roots
from retryvla.geom.features import IDX

log = logging.getLogger("v2r_data")
K_AUX = (0, 10, 20, 30, 40)
N_BIN = 2 * len(K_AUX)
N_AUX = 4 * len(K_AUX) + 2 * 3 * (len(K_AUX) - 1)


def group_subsets(tc, seed: int) -> list[tuple[Path, list[int]]]:
    out = []
    for gi, g in enumerate(tc.recovery_groups):
        pats = [REPO_ROOT / p for p in g["datasets"]]
        roots = [Path(r) for r in expand_roots(pats)]
        if g.get("require_closed"):
            roots = [r for r in roots if (r / "CLOSED").exists()]
        keep = None
        if g.get("keep_list"):
            keep = {}
            for f in g["keep_list"]:
                for l in open(REPO_ROOT / f):
                    r = json.loads(l)
                    keep[(str(Path(r["root"]).resolve()), int(r["episode_index"]))] = (bool(r["keep"]), float(r["d3"]))
        rank = str(g.get("rank", "score"))
        rows = []
        for k, root in enumerate(roots):
            for l in open(root / "retryvla" / "trajectories.jsonl"):
                r = json.loads(l)
                key = (str(root.resolve()), int(r["episode_index"]))
                if keep is not None:
                    if key not in keep or not keep[key][0]:
                        continue
                    d3 = keep[key][1]
                else:
                    d3 = 0.0
                sk = d3 if rank == "d3" else -float(r.get("score", 0.0))
                rows.append((sk, k, int(r["episode_index"]), r.get("instance_id", f"{k}_{r['episode_index']}")))
        by: dict[str, list] = {}
        for row in sorted(rows, key=lambda x: x[0]):
            by.setdefault(row[3], []).append(row)
        ids = sorted(by)
        np.random.default_rng(seed).shuffle(ids)
        rr = [by[i][j] for j in range(max((len(v) for v in by.values()), default=0)) for i in ids if j < len(by[i])]
        n = int(g.get("n", 0) or len(rr))
        if len(rr) < n:
            log.warning("group %d: only %d eligible trajectories (< %d)", gi, len(rr), n)
        sel = rr[:n]
        per = [[] for _ in roots]
        for _, k, e, _ in sel:
            per[k].append(e)
        log.info("group %d: %d roots, %d eligible, %d selected over %d instances (rank %s, keep_list %s)", gi,
                 len(roots), len(rr), len(sel), len({x[3] for x in sel}), rank, bool(keep))
        out += [(r, sorted(v)) for r, v in zip(roots, per) if v]
    return out


# ------------------------------------------------------------------ aux targets
def _targets(g: np.ndarray, f: int) -> tuple[np.ndarray, np.ndarray]:
    """geom [T+1, G] (state after every step; row 0 = start), frame f (state before action f = row f) ->
    targets [N_AUX], mask [N_AUX] (1 where f + k is inside the trajectory)."""
    T = len(g) - 1
    rest = float(g[:, IDX["obj_z"]].min())
    tb, tc, mb, mc = [], [], [], []
    for k in K_AUX:
        i = min(f + k, T)
        ok = float(f + k <= T)
        lift = float(g[i, IDX["obj_z"]] - rest)
        con = float(g[i, IDX["finger_obj_contact"]] > 0.5)
        tb += [float(con and lift >= 0.02), con]
        mb += [ok, ok]
        og = float(np.linalg.norm(g[i, IDX["obj_x"]: IDX["obj_y"] + 1] - g[i, IDX["goal_x"]: IDX["goal_y"] + 1]))
        tc += [lift * 100, og * 100]
        mc += [ok, ok]
    for k in K_AUX[1:]:
        i = min(f + k, T)
        ok = float(f + k <= T)
        for a, b in (("obj_x", "obj_z"), ("goal_x", "goal_z")):
            tc += list((g[i, IDX[a]: IDX[b] + 1] - g[f, IDX[a]: IDX[b] + 1]) * 100)
            mc += [ok] * 3
    return np.asarray(tb + tc, np.float32), np.asarray(mb + mc, np.float32)


class AuxTargets:
    def __init__(self, phys_dirs: list[str]):
        """phys_dirs: globs of rec_phys output dirs (<bank>/phys/<arm>); files <rid>_<ep>.npz with rid = root relative
        to the bank with '/' -> '_'. Index -> {(root_abs, ep): geom}."""
        self.geom = {}
        for pat in phys_dirs:
            for d in _glob.glob(str(REPO_ROOT / pat)):
                d = Path(d)
                bank = d.parent.parent
                for f in d.glob("*.npz"):
                    rid, ep = f.stem.rsplit("_", 1)
                    self.geom[(rid, int(ep), str(bank.resolve()))] = f
        self._cache = {}
        log.info("aux targets: %d trajectories with privileged replays", len(self.geom))

    def lookup(self, root: Path, ep: int):
        root = Path(root).resolve()
        for (rid, e, bank), f in self.geom.items():
            if e == ep and str(root).startswith(bank) and str(root)[len(bank) + 1:].replace("/", "_") == rid:
                return f
        return None

    def get(self, root: Path, ep: int, frame: int):
        key = (str(root), ep)
        if key not in self._cache:
            f = self.lookup(root, ep)
            self._cache[key] = np.load(f)["geom"] if f is not None else None
        g = self._cache[key]
        if g is None:
            return np.zeros(N_AUX, np.float32), np.zeros(N_AUX, np.float32)
        return _targets(g, int(frame))


class AuxMixDataset(MixDataset):
    """MixDataset (same random stream / items) + aux targets / mask for recovery frames (nominal: mask 0)."""

    def __init__(self, *a, aux: AuxTargets, **k):
        super().__init__(*a, **k)
        self.aux = aux

    def __getitem__(self, i: int) -> dict:
        rng = np.random.default_rng((self.seed, i))
        tg, mk = np.zeros(N_AUX, np.float32), np.zeros(N_AUX, np.float32)
        if rng.random() < self.ratio:
            di, j = self.rec_index[rng.integers(len(self.rec_index))]
            item, src = self.rec[di][j], 1
            tg, mk = self.aux.get(self.rec[di].root, int(item["episode_index"]), int(item["frame_index"]))
        else:
            item, src = self.orig[int(rng.integers(len(self.orig)))], 0
        out = {k: item[k] for k in CORE_KEYS if k in item}
        for k in ("observation.images.image", "observation.images.image2", "observation.state"):
            v = out.get(k)
            if v is not None and v.ndim == (4 if "images" in k else 2) and v.shape[0] == 1:
                out[k] = v[0]
        out["source"] = torch.tensor(src)
        out["aux"], out["aux_mask"] = torch.as_tensor(tg), torch.as_tensor(mk)
        return out


def aux_stats(ds: AuxMixDataset, n: int = 2000) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean / std of the continuous targets over recovery frames (standardisation; binary entries untouched)."""
    if not ds.rec_index:
        raise ValueError("aux_beta > 0 but the mix has no recovery frames")
    rng = np.random.default_rng(0)
    cols = [ds_.hf_dataset.with_format(None).select_columns(["episode_index", "frame_index"])[:] for ds_ in ds.rec]
    T, M = [], []
    for _ in range(n):
        di, j = ds.rec_index[rng.integers(len(ds.rec_index))]
        ep, fr = np.asarray(cols[di]["episode_index"]).reshape(-1)[j], np.asarray(cols[di]["frame_index"]).reshape(-1)[j]
        t, m = ds.aux.get(ds.rec[di].root, int(ep), int(fr))
        T.append(t); M.append(m)
    T, M = np.stack(T), np.stack(M)
    mu = np.where(M.sum(0) > 0, (T * M).sum(0) / np.maximum(M.sum(0), 1), 0)
    sd = np.sqrt(np.where(M.sum(0) > 0, (((T - mu) ** 2) * M).sum(0) / np.maximum(M.sum(0), 1), 1)) + 1e-3
    mu[:N_BIN], sd[:N_BIN] = 0.0, 1.0
    return torch.as_tensor(mu, dtype=torch.float32), torch.as_tensor(sd, dtype=torch.float32)


def build_aux_head(policy, device, width: int = 512):
    hid = int(policy.model.action_out_proj.in_features)
    head = torch.nn.Sequential(torch.nn.LayerNorm(hid), torch.nn.Linear(hid, width), torch.nn.GELU(),
                               torch.nn.Linear(width, N_AUX)).to(device)
    cap: dict[str, torch.Tensor] = {}
    h = policy.model.action_out_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("h", i[0]))
    return head, cap, h


def aux_loss(head, cap, tgt, mask, mu, sd):
    h = cap.pop("h").float().mean(1)
    pred = head(h)
    y = (tgt - mu) / sd
    lb = torch.nn.functional.binary_cross_entropy_with_logits(pred[:, :N_BIN], tgt[:, :N_BIN], reduction="none")
    lc = torch.nn.functional.smooth_l1_loss(pred[:, N_BIN:], y[:, N_BIN:], reduction="none", beta=1.0)
    l_ = torch.cat([lb, lc], 1) * mask
    return l_.sum() / mask.sum().clamp_min(1.0)
