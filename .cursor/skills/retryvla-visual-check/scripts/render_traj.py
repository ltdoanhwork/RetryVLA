#!/usr/bin/env python
"""RetryVLA visual check: render one trajectory (or two side by side) as MP4 + GIF + contact-sheet PNG, with a
caption bar (task, entry, segment, gripper, outcome) and an action-plot strip (EE xyz + gripper vs time, shaded
segments, moving cursor). CPU only by default (MuJoCo OSMesa / llvmpipe; never touches a GPU).

Sources (SPEC = kind:path#selector):
  bank:<dataset_root>#<episode>      RecoveryWriter LeRobot dataset (data/recovery/<run>/s*/p0/<arm>, D_P/D_F bank0,
                                     smoothed banks, LRS window datasets). Frames come from the stored PNGs; with
                                     --replay (or when the PNGs are missing) the actions are re-executed in sim from
                                     the exact failure snapshot and the 8-d state is checked against the stored one.
  lrs:<lrs_root>#<instance_id>       LRS-G teacher record (rows.jsonl + actions/<iid>.npz under a*/l*/): teacher path
                                     (all stages) + final hand-off student run, replayed in sim (LIBERO-Goal).
  rich:<critic3d .pt>#<k>            CRIT-V rich record (data/critic3d/<x>/<iid>.pt). It stores no actions or frames:
                                     bridge k is taken from data/wm/<x>/<iid>.pt and replayed (50 steps, open loop).
  audit:<goal entry_id>              LIBERO-Goal M0 alone: the audit rollout's own actions[t:T] from the entry.
  acts:<file.npy | file.npz:key>@<instance_id>   any [T, 7] action array replayed from that instance's snapshot.
Instance ids: LIBERO-Goal entry ids (<task>__demo_<d>__s<seed>__t075 / __on295 / __pre280; found in
data/goal_rsi/*/entries*.jsonl) or LIBERO-Recover ids (<scene_tag>__close2 / __open6; eval.seed 7, noise 0.02).

Usage (from the repo root, `source env/activate.sh`):
  python .cursor/skills/retryvla-visual-check/scripts/render_traj.py bank:data/recovery/m5b/s0/p0/random#3 --out DIR
  python ... --sample data/recovery/goal_search/s0/p0/wm_diverse --n 8 --out DIR   # stratified 8: random+worst+task
  python ... --side-by-side bank:<root>#<ep> lrs:data/recovery/lrs_goal#<iid> --out DIR
Outputs per trajectory: <name>.mp4, <name>.gif (<= 10 s, <= 384 px per panel), <name>_sheet.png, <name>.json (jerk /
gripper-flip / handoff-jump stats, replay state check).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
MAX_PX = 384
SEG_COL = {"bridge": (230, 140, 20), "continuation": (40, 90, 170), "teacher": (0, 150, 150),
           "handoff": (150, 60, 200), "M0": (90, 90, 90)}
OK, FAIL = (30, 160, 60), (200, 40, 40)


def gl_setup(gpu: bool) -> None:
    """Must run before anything imports mujoco / robosuite. CPU = OSMesa (llvmpipe), bit-exact physics either way."""
    if not gpu:
        os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "osmesa"
        os.environ.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)
        os.environ.setdefault("LP_NUM_THREADS", "8")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)


# ---------------------------------------------------------------------------------------------------------------- sim
_SIM = {}


def get_sim(res: int):
    if res not in _SIM:
        from retryvla.envs.libero_env import LiberoSim
        _SIM[res] = LiberoSim(resolution=res)
    return _SIM[res]


def is_goal(iid: str) -> bool:
    return re.search(r"__s\d+__(t|on|pre)\d+$", iid) is not None


_ENTRIES: dict | None = None


def goal_entry(iid: str) -> tuple[dict, Path]:
    global _ENTRIES
    if _ENTRIES is None:
        _ENTRIES = {}
        for f in sorted(glob.glob(str(REPO / "data/goal_rsi/*/entries*_s*.jsonl"))):
            for l in open(f):
                if l.strip():
                    e = json.loads(l)
                    _ENTRIES.setdefault(e["entry_id"], (e, Path(f).parent))
    if iid not in _ENTRIES:
        raise KeyError(f"goal entry {iid} not in data/goal_rsi/*/entries*.jsonl")
    return _ENTRIES[iid]


def audit_actions(e: dict, d: Path):
    import numpy as np
    f = d / f"actions_s{e['seed']}.npz"
    if not f.exists():
        f = REPO / f"data/goal_rsi/audit_m0_s{e['seed']}/actions_s{e['seed']}.npz"
    return np.load(f)[e["scene"]].astype(np.float32)


def build_instance(sim, iid: str, seed: int = 7) -> dict:
    """Exact failure snapshot of an instance, in-process (same code paths as goal_instances / compare_arms)."""
    import numpy as np
    if is_goal(iid):
        import h5py
        from libero.libero import get_libero_path
        e, d = goal_entry(iid)
        sp = json.loads((REPO / "data/splits/goal_rsi_v1.json").read_text())["tasks"][e["task"]]
        with h5py.File(sp["hdf5"]) as h:
            dd = h[f"data/demo_{e['demo']}"]
            init = np.asarray(dd.attrs["init_state"]) if "init_state" in dd.attrs else np.asarray(dd["states"][0])
        bddl = os.path.join(get_libero_path("bddl_files"), "libero_goal", f"{e['task']}.bddl")
        sim.load_init_state(bddl, init, num_steps_wait=10, seed=0, return_snapshot=False)
        acts = audit_actions(e, d)
        for a in acts[: e["t"]]:
            sim.step(a)
        return {"bddl": bddl, "snap": sim.snapshot(), "task": sp["language"], "suite": "LIBERO-Goal", "entry": e,
                "audit": acts}
    from retryvla.eval.eval_recover import noise_seed
    m = re.match(r"(.+)__(close|open)(\d+)$", iid)
    if not m:
        raise ValueError(f"unknown instance id format: {iid}")
    tag, g, k = m.group(1), m.group(2), int(m.group(3))
    hits = glob.glob(str(REPO / "data/assets/scenes/scenes/selected_bddl_scenes_*" / Path(*tag.split("__")))
                     + ".bddl")
    if not hits:
        raise FileNotFoundError(f"scene {tag} not under data/assets/scenes/scenes/")
    bddl = hits[0]
    out = sim.load_scene(bddl, bddl[:-5] + "_sim_state.npy", gripper_close=(g == "close"), noise_xy=0.02,
                         noise_seed=noise_seed(seed, tag, k), settle_steps=20, num_steps_wait=0, build_seed=seed)
    return {"bddl": bddl, "snap": out["snapshot"], "task": out["language"], "suite": "LIBERO-Recover", "entry": None}


def replay(inst: dict, sim, actions, stop_on_success: bool = True) -> dict:
    """Restore the snapshot, step the actions; frames/states = obs BEFORE each action (+ the final obs)."""
    import numpy as np
    o = sim.restore_snapshot(inst["bddl"], inst["snap"])["obs"]
    frames, states, succ = [o["image"]], [o["state"]], None
    for i, a in enumerate(np.asarray(actions, np.float32)):
        r = sim.step(a)
        frames.append(r.obs["image"])
        states.append(r.obs["state"])
        if r.success and succ is None:
            succ = i + 1
            if stop_on_success:
                break
    return {"frames": frames, "state": np.asarray(states, np.float32), "success_step": succ}


# ------------------------------------------------------------------------------------------------------------ loaders
def bank_rows(root: Path) -> list[dict]:
    return [json.loads(l) for l in open(root / "retryvla" / "trajectories.jsonl") if l.strip()]


def bank_table(root: Path, cols: list[str], ep: int | None = None):
    import pyarrow.parquet as pq
    files = sorted((root / "data").rglob("*.parquet"))
    if ep is not None:  # locate the episode's data file via meta/episodes (v3) when possible
        try:
            import pandas as pd
            me = pd.concat([pd.read_parquet(f, columns=["episode_index", "data/chunk_index", "data/file_index"])
                            for f in sorted((root / "meta" / "episodes").rglob("*.parquet"))])
            r = me[me.episode_index == ep].iloc[0]
            files = [root / f"data/chunk-{int(r['data/chunk_index']):03d}/file-{int(r['data/file_index']):03d}.parquet"]
        except Exception:
            pass
    tabs = [pq.read_table(f, columns=cols, filters=None if ep is None else [("episode_index", "=", ep)]) for f in files]
    import pyarrow as pa
    return pa.concat_tables(tabs).to_pandas().sort_values(["episode_index", "frame_index"])


def decode_png(cell):
    import cv2
    import numpy as np
    b = cell.get("bytes") if isinstance(cell, dict) else None
    if not b:
        return None
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def load_bank(root: str, ep: int, do_replay: bool, res: int) -> dict:
    import numpy as np
    root = Path(root)
    row = next(r for r in bank_rows(root) if int(r["episode_index"]) == ep)
    df = bank_table(root, ["observation.images.image", "observation.state", "action", "segment", "episode_index",
                           "frame_index"], ep)
    act = np.stack(df["action"].to_numpy()).astype(np.float32)
    st = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    seg = ["bridge" if int(s) == 0 else "continuation" for s in df["segment"].to_numpy()]
    frames = [decode_png(c) for c in df["observation.images.image"].to_numpy()]
    iid = row["instance_id"]
    info = {"bank": str(root), "episode": ep, "score": row.get("score"), "pred_value": row.get("pred_value"),
            "recovery_len": row.get("recovery_len"), "continuation_len": row.get("continuation_len")}
    succ_step = row.get("success_step")
    if do_replay or any(f is None for f in frames):
        sim = get_sim(res)
        inst = build_instance(sim, iid)
        rp = replay(inst, sim, act)
        n = min(len(st), len(rp["state"]))
        info["replay_state_maxdiff"] = float(np.abs(rp["state"][:n] - st[:n]).max())
        info["replay_success_step"] = rp["success_step"]
        frames = rp["frames"]
    arm = row.get("arm", root.name)
    run = "_".join(p for p in root.parts[-4:] if p not in ("p0", "data", "recovery"))
    return {"name": f"{run}_ep{ep}", "title": f"bank {arm} ep{ep}",
            "task": row.get("task", ""), "entry": iid, "frames": frames, "state": st, "action": act, "seg": seg,
            "success": succ_step is not None, "success_step": succ_step, "info": info}


def lrs_record(root: Path, iid: str):
    import numpy as np
    for p in sorted(Path(root).glob("**/rows.jsonl")):
        for l in open(p):
            if l.strip():
                r = json.loads(l)
                if r["instance_id"] == iid and (p.parent / "actions" / f"{iid}.npz").exists():
                    return r, np.load(p.parent / "actions" / f"{iid}.npz")
    raise KeyError(f"no LRS record for {iid} under {root}")


def load_lrs(root: str, iid: str, res: int) -> dict:
    import numpy as np
    r, z = lrs_record(Path(root), iid)
    path = z["path"].astype(np.float32)
    k = len(r["stages"]) - 1
    stud = z[f"student_{k}"].astype(np.float32) if f"student_{k}" in z else np.zeros((0, 7), np.float32)
    h = r["stages"][-1].get("handoff") or {}
    nt = int(h.get("n_test", 0))
    bounds = np.cumsum([s["n_actions"] for s in r["stages"]])
    acts = np.concatenate([path, stud], 0)
    seg = []
    for i in range(len(acts)):
        if i < len(path):
            seg.append(f"teacher st{int(np.searchsorted(bounds, i, side='right')) + 1}")
        elif i < len(path) + nt:
            seg.append("handoff")
        else:
            seg.append("continuation")
    sim = get_sim(res)
    inst = build_instance(sim, iid)
    rp = replay(inst, sim, acts)
    T = len(rp["frames"]) - 1
    info = {"lrs_status": r["status"], "judge_accept": h.get("accept"), "gt_success": h.get("gt_success"),
            "n_stages": len(r["stages"]), "teacher_len": len(path), "handoff_test": nt,
            "replay_success_step": rp["success_step"],
            "logged_gt_step": (len(path) + h["gt_step"]) if h.get("gt_success") else None}
    return {"name": f"lrs_{iid}", "title": f"LRS teacher {len(r['stages'])}st + handoff ({r['status']})",
            "task": inst["task"], "entry": iid, "frames": rp["frames"], "state": rp["state"], "action": acts[:T],
            "seg": seg[:T], "success": rp["success_step"] is not None, "success_step": rp["success_step"], "info": info}


def load_rich(pt: str, k: int, res: int) -> dict:
    import numpy as np
    import torch
    pt = Path(pt)
    d = torch.load(pt, weights_only=False, map_location="cpu")
    iid = d["instance_id"]
    src = REPO / "data/wm" / pt.parent.name / pt.name
    if not src.exists():
        raise FileNotFoundError(f"rich record has no actions; expected the source log {src}")
    s = torch.load(src, weights_only=False, map_location="cpu")
    acts = s["actions"][k] if not isinstance(s["actions"], list) else s["actions"][k]
    acts = np.asarray(acts, np.float32)[:50]
    sim = get_sim(res)
    inst = build_instance(sim, iid)
    rp = replay(inst, sim, acts)
    T = len(rp["frames"]) - 1
    info = {"rich": str(pt), "k": k, "source": list(d["source"])[k], "logged_success": bool(d["success"][k]),
            "logged_prefix_success": bool(d["prefix_success"][k]), "replay_prefix_success": rp["success_step"],
            "pred_value": float(s["pred_value"][k]) if "pred_value" in s else None}
    return {"name": f"rich_{pt.parent.name}_{iid}_k{k}", "title": f"bridge k{k} ({info['source']})",
            "task": d["task"], "entry": iid, "frames": rp["frames"], "state": rp["state"], "action": acts[:T],
            "seg": ["bridge"] * T, "success": bool(d["success"][k]), "success_step": None, "info": info,
            "outcome": f"logged rollout {'SUCCESS' if d['success'][k] else 'FAIL'} (bridge only shown)"}


def load_acts(spec: str, res: int, kind: str) -> dict:
    import numpy as np
    sim = get_sim(res)
    if kind == "audit":
        iid = spec
        inst = build_instance(sim, iid)
        acts, lab, title = inst["audit"][inst["entry"]["t"]:], "M0", "M0 alone (audit actions[t:])"
    else:
        f, iid = spec.rsplit("@", 1)
        if ":" in f:
            f, key = f.rsplit(":", 1)
            acts = np.load(f)[key]
        else:
            acts = np.load(f)
        inst, lab, title = build_instance(sim, iid), "bridge", f"actions {Path(f).name}"
    rp = replay(inst, sim, acts)
    T = len(rp["frames"]) - 1
    return {"name": f"{kind}_{iid}", "title": title, "task": inst["task"], "entry": iid, "frames": rp["frames"],
            "state": rp["state"], "action": np.asarray(acts, np.float32)[:T], "seg": [lab] * T,
            "success": rp["success_step"] is not None, "success_step": rp["success_step"], "info": {}}


def load(spec: str, a) -> dict:
    kind, rest = spec.split(":", 1)
    if kind == "bank":
        p, ep = rest.rsplit("#", 1)
        return load_bank(p, int(ep), a.replay, a.sim_res)
    if kind == "lrs":
        p, iid = rest.rsplit("#", 1)
        return load_lrs(p, iid, a.sim_res)
    if kind == "rich":
        p, k = rest.rsplit("#", 1)
        return load_rich(p, int(k), a.sim_res)
    if kind in ("audit", "acts"):
        return load_acts(rest, a.sim_res, kind)
    raise ValueError(f"unknown source kind {kind!r}")


# -------------------------------------------------------------------------------------------------------------- stats
def seg_color(label: str):
    for k, c in SEG_COL.items():
        if label.startswith(k):
            return c
    return (120, 120, 120)


def stats(act, seg) -> dict:
    """Action smoothness: jerk = mean ||d2 a[:6]||, max_step = max ||d1 a[:6]||, gripper sign flips, and the action
    jump at every segment boundary (bridge -> continuation, teacher -> handoff, ...)."""
    import numpy as np
    a = np.asarray(act, np.float64)
    if len(a) < 3:
        return {"T": len(a), "jerk": 0.0, "max_step": 0.0, "grip_flips": 0, "boundary_jumps": []}
    d1 = np.diff(a[:, :6], axis=0)
    d2 = np.diff(d1, axis=0)
    g = np.sign(a[:, 6])
    g = g[g != 0]
    jumps = [{"t": i, "from": seg[i - 1], "to": seg[i], "jump": float(np.linalg.norm(a[i, :6] - a[i - 1, :6])),
              "grip_change": bool(np.sign(a[i, 6]) != np.sign(a[i - 1, 6]))}
             for i in range(1, len(seg)) if seg[i] != seg[i - 1]]
    return {"T": len(a), "jerk": float(np.linalg.norm(d2, axis=1).mean()),
            "max_step": float(np.linalg.norm(d1, axis=1).max()), "grip_flips": int((np.diff(g) != 0).sum()),
            "boundary_jumps": jumps}


def stats_line(s: dict) -> str:
    j = max([b["jump"] for b in s["boundary_jumps"]], default=0.0)
    return f"T={s['T']} jerk={s['jerk']:.3f} max|da|={s['max_step']:.2f} flips={s['grip_flips']} handoff|da|={j:.2f}"


# ---------------------------------------------------------------------------------------------------------- rendering
def plot_strip(tr: dict, width: int, height: int):
    """EE xyz (cm, relative to t=0) + gripper command vs time, shaded segments -> (RGB image, x0 px, x1 px)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    act, st, seg = tr["action"], tr["state"], tr["seg"]
    T = len(act)
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes([0.1, 0.2, 0.76, 0.72])
    t0 = 0
    for i in range(1, T + 1):
        if i == T or seg[i] != seg[t0]:
            c = np.asarray(seg_color(seg[t0])) / 255
            ax.axvspan(t0, i, color=c, alpha=0.15, lw=0)
            if i < T:
                ax.axvline(i, color=c, ls="--", lw=0.8)
            t0 = i
    ee = (st[: T + 1, :3] - st[0, :3]) * 100
    for d, c in enumerate("rgb"):
        ax.plot(np.arange(len(ee)), ee[:, d], c, lw=1, label="xyz"[d])
    ax.set_xlim(0, max(T, 1))
    ax.tick_params(labelsize=6, pad=1)
    ax.set_ylabel("EE cm", fontsize=6, labelpad=1)
    ax.legend(fontsize=5, loc="upper left", ncol=3, handlelength=1, frameon=False, borderpad=0.1)
    ax2 = ax.twinx()
    ax2.step(np.arange(T), act[:, 6], "k", lw=0.9, where="post")
    ax2.set_ylim(-1.3, 1.3)
    ax2.set_yticks([-1, 1], ["open", "close"], fontsize=5)
    ax2.tick_params(pad=1)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    bb = ax.get_window_extent()
    plt.close(fig)
    return img, float(bb.x0), float(bb.x1)


def caption_bar(lines: list[tuple[str, tuple]], width: int, scale: float):
    import cv2
    import numpy as np
    lh = int(15 * scale) + 1
    bar = np.full((lh * len(lines) + 4, width, 3), 255, np.uint8)
    maxc = int(width / (6.4 * scale))
    for i, (txt, col) in enumerate(lines):
        cv2.putText(bar, txt[:maxc], (6, int(12 * scale) + lh * i), cv2.FONT_HERSHEY_SIMPLEX, 0.36 * scale, col, 1,
                    cv2.LINE_AA)
    return bar


def wrap(txt: str, n: int, max_lines: int) -> list[str]:
    import textwrap
    ls = textwrap.wrap(txt, n) or [""]
    return ls[: max_lines - 1] + [" ".join(ls[max_lines - 1:])] if len(ls) > max_lines else ls


def tail(txt: str, n: int) -> str:
    return txt if len(txt) <= n else ".." + txt[-(n - 2):]


def grip_txt(tr: dict, i: int) -> str:
    a = tr["action"][min(i, len(tr["action"]) - 1), 6]
    s = tr["state"][min(i, len(tr["state"]) - 1)]
    return f"grip {'CLOSE' if a > 0 else 'open'} w{(s[6] - s[7]) * 100:.1f}cm"


def outcome_txt(tr: dict) -> tuple[str, tuple]:
    if tr.get("outcome"):
        return tr["outcome"], (OK if tr["success"] else FAIL)
    if tr["success"]:
        return (f"SUCCESS @ {tr['success_step']}" if tr["success_step"] else "SUCCESS (logged)"), OK
    return "FAIL", FAIL


def panel(tr: dict, res: int) -> list:
    """One captioned frame per step: caption bar / camera frame / plot strip with a cursor."""
    import cv2
    import numpy as np
    sc = res / 256
    strip, x0, x1 = plot_strip(tr, res, max(96, int(res * 0.42)))
    T, n = len(tr["action"]), len(tr["frames"])
    out_txt, out_col = outcome_txt(tr)
    maxc = int(res / (6.4 * sc))
    res_frames = []
    for i, f in enumerate(tr["frames"]):
        si = min(i, T - 1)
        seg = tr["seg"][si] if i < T else "end"
        col = seg_color(tr["seg"][si])
        bar = caption_bar([(tr["title"], col)] + [(l, (20, 20, 20)) for l in wrap(tr["task"], maxc, 2)]
                          + [(tail(tr["entry"], maxc), (90, 90, 90)), (f"{seg} t={i}/{T} {grip_txt(tr, i)}", col)],
                          res, sc)
        im = cv2.resize(f, (res, res), interpolation=cv2.INTER_AREA if f.shape[0] > res else cv2.INTER_LINEAR)
        cv2.rectangle(im, (0, 0), (res - 1, res - 1), col, max(2, int(3 * sc)))
        if i == n - 1:
            cv2.rectangle(im, (0, res - int(22 * sc)), (res - 1, res - 1), out_col, -1)
            cv2.putText(im, out_txt, (6, res - int(7 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * sc, (255, 255, 255), 1,
                        cv2.LINE_AA)
        s = strip.copy()
        x = int(x0 + (x1 - x0) * min(i, T) / max(T, 1))
        cv2.line(s, (x, 0), (x, s.shape[0] - 1), (220, 0, 0), 1)
        res_frames.append(np.concatenate([bar, im, s], 0))
    return res_frames


def hstack(panels: list[list]) -> list:
    import numpy as np
    T = max(len(p) for p in panels)
    H = max(p[0].shape[0] for p in panels)
    out = []
    for t in range(T):
        row = []
        for p in panels:
            f = p[min(t, len(p) - 1)]
            if f.shape[0] < H:
                f = np.concatenate([f, np.full((H - f.shape[0], f.shape[1], 3), 255, np.uint8)], 0)
            row += [f, np.full((H, 4, 3), 255, np.uint8)]
        out.append(np.concatenate(row[:-1], 1))
    return out


def write_mp4(path: Path, frames: list, fps: int = 20) -> None:
    import imageio.v2 as iio
    h, w = frames[0].shape[:2]
    with iio.get_writer(str(path), fps=fps, codec="libx264", quality=6, macro_block_size=1,
                        ffmpeg_params=["-vf", f"scale={w - w % 2}:{h - h % 2}"]) as wr:
        for f in frames:
            wr.append_data(f)


def write_gif(path: Path, frames: list, n_panels: int = 1, max_frames: int = 56, fps: int = 8,
              panel_w: int = 224) -> None:
    """<= max_frames + 6 frames at `fps` (default <= 7.75 s), <= panel_w px per panel, 64 colours -> small files."""
    import cv2
    from PIL import Image
    every = max(1, -(-len(frames) // max_frames))
    fr = frames[::every] + [frames[-1]] * 6
    w = min(fr[0].shape[1], min(panel_w, MAX_PX) * n_panels + 4 * (n_panels - 1))
    if w < fr[0].shape[1]:
        h = int(fr[0].shape[0] * w / fr[0].shape[1])
        fr = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in fr]
    import numpy as np
    mosaic = Image.fromarray(np.concatenate(fr[:: max(1, len(fr) // 6)], 0))   # one global palette -> frame deltas
    pal = mosaic.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    ims = [Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in fr]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=True)


def sheet(tr: dict, n_tiles: int = 8, tile: int = 160, cols: int = 4):
    """Contact sheet: header (title / task / entry / outcome / stats) + N evenly spaced frames + plot strip with
    tick marks at the sampled steps."""
    import cv2
    import numpy as np
    n = len(tr["frames"])
    idx = sorted(set(np.linspace(0, n - 1, min(n_tiles, n)).round().astype(int).tolist()))
    T = len(tr["action"])
    tiles = []
    for i in idx:
        im = cv2.resize(tr["frames"][i], (tile, tile), interpolation=cv2.INTER_AREA)
        si = min(i, T - 1)
        col = seg_color(tr["seg"][si])
        cv2.rectangle(im, (0, 0), (tile - 1, tile - 1), col, 3)
        cv2.rectangle(im, (0, 0), (tile - 1, 14), (255, 255, 255), -1)
        g = "C" if tr["action"][si, 6] > 0 else "o"
        cv2.putText(im, f"t{i} {tr['seg'][si][:12]} g:{g}", (4, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1,
                    cv2.LINE_AA)
        tiles.append(im)
    while len(tiles) % cols:
        tiles.append(np.full((tile, tile, 3), 255, np.uint8))
    rows = [np.concatenate(tiles[r: r + cols], 1) for r in range(0, len(tiles), cols)]
    grid = np.concatenate(rows, 0)
    W = grid.shape[1]
    strip, x0, x1 = plot_strip(tr, W, 120)
    for i in idx:
        x = int(x0 + (x1 - x0) * min(i, T) / max(T, 1))
        cv2.line(strip, (x, 0), (x, 6), (220, 0, 0), 2)
    ot, oc = outcome_txt(tr)
    hdr = caption_bar([(tr["title"] + "  |  " + ot, oc), (tr["task"], (20, 20, 20)), (tr["entry"], (90, 90, 90)),
                       (stats_line(tr["stats"]), (20, 20, 20))], W, 1.0)
    return np.concatenate([hdr, grid, strip], 0)


def vstack(ims: list):
    import numpy as np
    W = max(i.shape[1] for i in ims)
    pad = [np.concatenate([i, np.full((i.shape[0], W - i.shape[1], 3), 255, np.uint8)], 1) for i in ims]
    sep = np.full((6, W, 3), 200, np.uint8)
    out = []
    for p in pad:
        out += [p, sep]
    return np.concatenate(out[:-1], 0)


def save_png(path: Path, im) -> None:
    from PIL import Image
    Image.fromarray(im).quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(
        path, optimize=True)


# ------------------------------------------------------------------------------------------------------------- main
def safe(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name if len(name) <= 120 else name[:80] + "_" + format(abs(hash(name)) % 10**8, "08d")


def render_one(tr: dict, out: Path, a, name: str | None = None) -> dict:
    tr["stats"] = stats(tr["action"], tr["seg"])
    name = safe(name or tr["name"])
    fr = panel(tr, a.res)
    paths = {"sheet": out / f"{name}_sheet.png", "gif": out / f"{name}.gif"}
    save_png(paths["sheet"], sheet(tr, a.tiles))
    write_gif(paths["gif"], fr)
    if not a.no_mp4:
        paths["mp4"] = out / f"{name}.mp4"
        write_mp4(paths["mp4"], fr + [fr[-1]] * 15)
    rec = {"name": name, "entry": tr["entry"], "task": tr["task"], "title": tr["title"], "success": tr["success"],
           "success_step": tr["success_step"], "stats": tr["stats"], "info": tr["info"],
           "paths": {k: str(v) for k, v in paths.items()}}
    (out / f"{name}.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def render_sbs(A: dict, B: dict, out: Path, a) -> dict:
    import cv2
    import numpy as np
    if A["entry"] != B["entry"]:
        print(f"[visual-check] WARNING: different entries {A['entry']} vs {B['entry']} (not a paired comparison)")
    f0 = [cv2.resize(t["frames"][0], (128, 128), interpolation=cv2.INTER_AREA).astype(float) for t in (A, B)]
    d0 = float(np.abs(f0[0] - f0[1]).mean())
    print(f"[visual-check] start-frame mean |diff| = {d0:.2f} / 255 (stored vs replayed frames differ slightly by "
          f"renderer; > ~15 = different start state)")
    for t in (A, B):
        t["stats"] = stats(t["action"], t["seg"])
    name = safe(f"sbs_{A['name']}__VS__{B['name']}")
    fr = hstack([panel(A, a.res), panel(B, a.res)])
    paths = {"sheet": out / f"{name}_sheet.png", "gif": out / f"{name}.gif"}
    save_png(paths["sheet"], vstack([sheet(A, a.tiles), sheet(B, a.tiles)]))
    write_gif(paths["gif"], fr, n_panels=2)
    if not a.no_mp4:
        paths["mp4"] = out / f"{name}.mp4"
        write_mp4(paths["mp4"], fr + [fr[-1]] * 15)
    rec = {"name": name, "entry": [A["entry"], B["entry"]], "start_frame_diff": d0,
           "A": {"title": A["title"], "success": A["success"], "stats": A["stats"], "info": A["info"]},
           "B": {"title": B["title"], "success": B["success"], "stats": B["stats"], "info": B["info"]},
           "paths": {k: str(v) for k, v in paths.items()}}
    (out / f"{name}.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def pick_sample(root: Path, n: int, seed: int) -> list[tuple[int, str]]:
    """Stratified sample: 2 worst-jerk, 1 worst boundary jump, 1 most gripper flips, 1 per task, random fill."""
    import numpy as np
    df = bank_table(root, ["action", "segment", "episode_index", "frame_index"])
    rows = {int(r["episode_index"]): r for r in bank_rows(root)}
    S = {}
    for ep, g in df.groupby("episode_index"):
        seg = ["bridge" if int(s) == 0 else "continuation" for s in g["segment"]]
        S[int(ep)] = stats(np.stack(g["action"].to_numpy()), seg)
    eps = sorted(S)
    picks: list[tuple[int, str]] = []

    def add(ep, why):
        if ep is not None and ep not in [p[0] for p in picks] and len(picks) < n:
            picks.append((ep, why))
    for ep in sorted(eps, key=lambda e: -S[e]["jerk"])[:2]:
        add(ep, f"worst-jerk ({S[ep]['jerk']:.3f})")
    bj = [e for e in eps if S[e]["boundary_jumps"]]
    if bj:
        e = max(bj, key=lambda e: max(b["jump"] for b in S[e]["boundary_jumps"]))
        add(e, f"worst handoff jump ({max(b['jump'] for b in S[e]['boundary_jumps']):.2f})")
    e = max(eps, key=lambda e: S[e]["grip_flips"])
    if S[e]["grip_flips"] > 1:
        add(e, f"most gripper flips ({S[e]['grip_flips']})")
    rng = np.random.default_rng(seed)
    tasks: dict[str, list[int]] = {}
    for e in eps:
        tasks.setdefault(rows.get(e, {}).get("task", "?"), []).append(e)
    for t in sorted(tasks):
        add(int(rng.choice(tasks[t])), f"per-task: {t[:40]}")
    for e in rng.permutation(eps):
        add(int(e), "random")
    return picks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="*", help="sources to render one by one (see above)")
    ap.add_argument("--side-by-side", nargs=2, metavar=("A", "B"), dest="sbs")
    ap.add_argument("--sample", default="", help="bank root: render a stratified sample of --n trajectories")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/_analysis/visual_check")
    ap.add_argument("--res", type=int, default=256, help=f"panel frame size (clamped to {MAX_PX})")
    ap.add_argument("--sim_res", type=int, default=256, help="camera resolution for sim replays")
    ap.add_argument("--tiles", type=int, default=8)
    ap.add_argument("--replay", action="store_true", help="bank: replay in sim even if frames are stored (+ check)")
    ap.add_argument("--no_mp4", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="use NVIDIA EGL (default: CPU OSMesa)")
    a = ap.parse_args()
    a.res = min(a.res, MAX_PX)
    gl_setup(a.gpu)
    out = Path(a.out)
    out = out if out.is_absolute() else REPO / out
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    if a.sample:
        root = Path(a.sample)
        for ep, why in pick_sample(root, a.n, a.seed):
            r = render_one(load(f"bank:{root}#{ep}", a), out, a)
            r["why"] = why
            recs.append(r)
            print(f"[visual-check] ep{ep} ({why}): {stats_line(r['stats'])}\n    {r['paths']['sheet']}", flush=True)
        md = [f"# visual check sample: {root}", ""]
        md += [f"- ep {r['info']['episode']} ({r['why']}): {stats_line(r['stats'])}\n  ![]({r['paths']['sheet']})"
               for r in recs]
        (out / "index.md").write_text("\n".join(md) + "\n")
    for s in a.specs:
        r = render_one(load(s, a), out, a)
        recs.append(r)
        print(f"[visual-check] {s}: {stats_line(r['stats'])} info={json.dumps(r['info'], default=float)}\n"
              f"    {r['paths']}", flush=True)
    if a.sbs:
        r = render_sbs(load(a.sbs[0], a), load(a.sbs[1], a), out, a)
        recs.append(r)
        print(f"[visual-check] side-by-side: {r['paths']}", flush=True)
    if not recs:
        ap.error("nothing to render (give SPECs, --sample or --side-by-side)")


if __name__ == "__main__":
    main()
