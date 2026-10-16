#!/usr/bin/env python
"""Visual comparison of the recovery STRATEGIES behind the M5b search arms.

python scripts/viz_strategies.py [--n_inst 3]

1) strategies.png: from all M5b verification logs (256 instances x 4 arms x 16 verified candidates), every
   candidate's 50-step segment is mapped to a readable strategy = gripper schedule x vertical motion x
   horizontal motion. Shows, per arm, how often each strategy was tried and how often it succeeded.
2) per instance (WM solved, random did not): grid video [SmolVLA alone | best random try (failed) |
   up to 3 verified WM recoveries using DIFFERENT strategies], plus a key-frame strip and an end-effector
   path plot. Re-simulated from the exact failure snapshot, so every panel starts from the same state.
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "runs/_analysis/viz/strategies"
ARMS = ["random", "wm_diverse", "wm_noise", "wm_qd"]
GRIP = {0: "open", 1: "grasp (open->close)", 2: "release (close->open)", 3: "hold closed"}
ZB = {0: "lift", 1: "level", 2: "lower"}
XY = {0: "", 1: "+x", 2: "+y", 3: "-x", 4: "-y"}


def strat_label(cell: int) -> str:
    g, rest = divmod(int(cell), 15)
    z, xy = divmod(rest, 5)
    s = f"{GRIP[g].split(' ')[0]} + {ZB[z]}"
    return s + (f" + move {XY[xy]}" if xy else "")


def grip_label(cell: int) -> str:
    return GRIP[int(cell) // 15]


def load_verif():
    import torch

    from retryvla.search.arms import descriptor

    rows = []
    for f in sorted(glob.glob(str(REPO / "data/wm/verif_m5b/*.pt"))):
        d = torch.load(f, weights_only=False, map_location="cpu")
        acts = d["actions"].numpy()
        cell = descriptor(acts)
        for k, (arm, s) in enumerate(zip(d["source"], d["success"].tolist())):
            rows.append({"inst": d["instance_id"], "task": d["scene"].split("__")[0], "arm": arm, "k": k,
                         "cell": int(cell[k]), "success": bool(s), "actions": acts[k]})
    return rows


def plot_strategies(rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    tried = {a: Counter(grip_label(r["cell"]) for r in rows if r["arm"] == a) for a in ARMS}
    won = {a: Counter(grip_label(r["cell"]) for r in rows if r["arm"] == a and r["success"]) for a in ARMS}
    gl = list(GRIP.values())
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"width_ratios": [1, 1, 1.35]})
    col = {"random": "#9AA5B1", "wm_diverse": "#E8962E", "wm_noise": "#1F8FE5", "wm_qd": "#2A9D8F"}
    w = 0.2
    x = np.arange(len(gl))
    for i, a in enumerate(ARMS):
        n = sum(tried[a].values())
        ax[0].bar(x + (i - 1.5) * w, [tried[a][g] / n for g in gl], w, color=col[a], label=a)
        ax[1].bar(x + (i - 1.5) * w, [won[a][g] / max(1, tried[a][g]) for g in gl], w, color=col[a], label=a)
    for k, t in enumerate(["What each arm spends its 16 sims on\n(fraction of verified candidates, by gripper schedule)",
                           "Which schedules actually work\n(verified success rate per schedule)"]):
        ax[k].set_xticks(x, [g.replace(" (", "\n(") for g in gl], fontsize=10)
        ax[k].set_title(t, fontsize=12)
        ax[k].grid(axis="y", alpha=0.3)
    ax[0].legend(frameon=False)
    # top full strategies among WM successes vs random successes
    succ = Counter(strat_label(r["cell"]) for r in rows if r["success"] and r["arm"] != "random")
    top = [s for s, _ in succ.most_common(12)]
    for i, a in enumerate(ARMS):
        c = Counter(strat_label(r["cell"]) for r in rows if r["arm"] == a and r["success"])
        ax[2].barh(np.arange(len(top)) + (i - 1.5) * w, [c[s] for s in top], w, color=col[a], label=a)
    ax[2].set_yticks(np.arange(len(top)), top, fontsize=10)
    ax[2].invert_yaxis()
    ax[2].set_title("Most common SUCCESSFUL recovery strategies\n(# verified successes, 50-step segment)", fontsize=12)
    ax[2].grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "strategies.png", dpi=130)
    plt.close()
    # drawer only (where WM helps most)
    dr = [r for r in rows if "top_drawer" in r["task"]]
    lines = ["arm,grip schedule,tried,succ,rate"]
    for a in ARMS:
        for g in gl:
            t = sum(1 for r in dr if r["arm"] == a and grip_label(r["cell"]) == g)
            s = sum(1 for r in dr if r["arm"] == a and grip_label(r["cell"]) == g and r["success"])
            lines.append(f"{a},{g},{t},{s},{s / max(1, t):.3f}")
    (OUT / "drawer_schedules.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def pick_instances(rows, n):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["inst"]][r["arm"]].append(r)
    cand = []
    for i, d in by.items():
        if any(r["success"] for r in d["random"]):
            continue
        wins = [r for a in ("wm_diverse", "wm_qd", "wm_noise") for r in d[a] if r["success"]]
        cells = {r["cell"] for r in wins}
        if len(cells) >= 2:
            cand.append((len(cells), i))
    cand.sort(reverse=True)
    out, tasks = [], set()
    for _, i in cand:  # one per task first
        t = i.split("__")[0]
        if t not in tasks:
            out.append(i)
            tasks.add(t)
        if len(out) == n:
            break
    return out, by


def render_instance(iid, d, policy, sim, cfg):
    import imageio
    from PIL import Image, ImageDraw

    from retryvla.envs.libero_env import discover_scenes
    from retryvla.eval.eval_recover import noise_seed
    from scripts.viz_trajectories import FONT, FONT_S

    tag, gt = iid.rsplit("__", 1)
    grip, k = gt.rstrip("0123456789"), int(gt[len(gt.rstrip("0123456789")):])
    sc = next(s for s in discover_scenes(REPO / cfg.recover.scene_root) if s.tag == tag)
    task = sc.instruction(cfg.recover.instruction)
    out = sim.load_scene(sc.bddl, sc.npy, gripper_close=(grip == "close"), noise_xy=float(cfg.recover.obj_noise_xy),
                         noise_seed=noise_seed(int(cfg.eval.seed), sc.tag, k),
                         settle_steps=int(cfg.recover.gripper_settle_steps), build_seed=int(cfg.eval.seed))
    snap = sim.snapshot()

    def run(prefix, seed, T=350):
        o = sim.restore_snapshot(sc.bddl, snap, build_seed=int(cfg.eval.seed))["obs"]
        frames, eef, segs, ok, t = [o["image"]], [o["state"][:3].copy()], [0 if prefix is not None else 1], False, 0
        for a in (prefix if prefix is not None else []):
            r = sim.step(a)
            o, t = r.obs, t + 1
            frames.append(o["image"]); eef.append(o["state"][:3].copy()); segs.append(0)
            if r.success:
                return frames, eef, segs, True
        while t < T and not ok:
            ch = policy.predict_chunk([o], [task], seeds=[seed + t])[0][:10]
            for a in ch:
                r = sim.step(a)
                o, t = r.obs, t + 1
                frames.append(o["image"]); eef.append(o["state"][:3].copy()); segs.append(1)
                if r.success:
                    ok = True
                    break
        return frames, eef, segs, ok

    panels = [("SmolVLA alone", run(None, 11))]
    rnd = sorted(d["random"], key=lambda r: -np.abs(r["actions"][..., :3]).sum())[:1]
    if rnd:
        panels.append((f"random try: {strat_label(rnd[0]['cell'])}", run(rnd[0]["actions"], 11)))
    seen = set()
    for a in ("wm_diverse", "wm_qd", "wm_noise"):
        for r in d[a]:
            if r["success"] and r["cell"] not in seen and len(seen) < 3:
                seen.add(r["cell"])
                panels.append((f"{a}: {strat_label(r['cell'])}", run(r["actions"], 11)))
    T = max(len(p[1][0]) for p in panels)

    def tile(img, title, seg, t, n, ok):
        im = Image.fromarray(np.ascontiguousarray(img)).convert("RGB")
        c = Image.new("RGB", (256, 296), (255, 255, 255))
        c.paste(im, (0, 40))
        dr = ImageDraw.Draw(c)
        done = t >= n - 1
        bar = (200, 60, 60) if done and not ok else (40, 160, 80) if done else (255, 140, 0) if seg == 0 else (0, 140, 230)
        dr.rectangle([0, 0, 255, 39], fill=bar)
        dr.text((4, 2), title[:38], fill=(255, 255, 255), font=FONT_S)
        status = ("SUCCESS" if ok else "FAILED") if done else ("searched segment" if seg == 0 else "SmolVLA continues")
        dr.text((4, 20), f"t={min(t, n - 1)}  {status}", fill=(255, 255, 255), font=FONT)
        return np.asarray(c)

    vid = []
    for t in range(0, T, 2):
        row = [tile(f[min(t, len(f) - 1)], ttl, s[min(t, len(s) - 1)], t, len(f), ok) for ttl, (f, e, s, ok) in panels]
        vid.append(np.concatenate(row, 1))
    od = OUT / tag[len("pick_up_the_black_bowl_"):][:40].strip("_") / gt
    od.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(od / "grid.mp4", vid, fps=15)
    keys = [0, 25, 50, 100, 175, 260]
    strip = [np.concatenate([tile(f[min(t, len(f) - 1)], ttl, s[min(t, len(s) - 1)], t, len(f), ok)
                             for t in keys], 1) for ttl, (f, e, s, ok) in panels]
    imageio.imwrite(od / "strip.png", np.concatenate(strip, 0))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for ttl, (f, e, s, ok) in panels:
        e = np.array(e)
        s = np.array(s)
        ls = "-" if ok else "--"
        for k2, (i, j) in enumerate([(0, 1), (0, 2)]):
            l, = ax[k2].plot(e[:, i], e[:, j], ls, lw=1.6, label=f"{ttl} ({'ok' if ok else 'fail'})")
            m = s == 0
            if m.any():
                ax[k2].plot(e[m, i], e[m, j], "-", lw=4, alpha=0.35, color=l.get_color())
            ax[k2].plot(e[0, i], e[0, j], "ko", ms=5)
    ax[0].set_xlabel("x"); ax[0].set_ylabel("y"); ax[1].set_xlabel("x"); ax[1].set_ylabel("z (height)")
    ax[0].set_title("end-effector path, top view (thick = searched segment)")
    ax[1].set_title("side view")
    ax[1].legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(od / "eef_paths.png", dpi=120)
    plt.close()
    print("saved", od, [(p[0], p[1][3]) for p in panels])
    return od


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_inst", type=int, default=3)
    a = ap.parse_args()
    rows = load_verif()
    plot_strategies(rows)
    insts, by = pick_instances(rows, a.n_inst)
    from retryvla.config import load_config
    from retryvla.envs.libero_env import LiberoSim
    from retryvla.policy import SmolVLAWrapper

    cfg = load_config(REPO / "configs/exp/m5b_compare_arms.yaml")
    policy = SmolVLAWrapper(cfg.policy.path, "cuda", 10)
    sim = LiberoSim()
    for i in insts:
        render_instance(i, by[i], policy, sim, cfg)


if __name__ == "__main__":
    main()
