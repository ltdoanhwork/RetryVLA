#!/usr/bin/env python
"""Rule-based failure-mode classification of the re-simulated failures (scripts/analyze_failures_resim.py output).

python scripts/classify_failures.py      (CPU only: reads runs/_analysis/failures/resim/*/<iid>.npz + .mp4)

-> runs/_analysis/failures/modes.json, docs/failure_figs/fig5_failure_modes.png, docs/failure_figs/strip_<mode>_*.png,
   docs/failure_videos/<mode>.mp4 (Vietnamese caption bar).
"""

from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.analyze_failures import SHORT, atomic_write, save_fig  # noqa: E402

RES = REPO / "runs/_analysis/failures/resim"
FIG = REPO / "docs/failure_figs"
VID = REPO / "docs/failure_videos"
MODES = {  # key -> (label EN, label VI, colour, one-line explanation VI)
    "a1_wrong_place": ("(a1) never touches the bowl (grasps elsewhere)", "(a1) không chạm tới bát (gắp chỗ khác)", "#8e1b10",
                       "không ngón nào chạm bát, bát không xê dịch: tay hạ xuống / kẹp vào chỗ trống trên bàn, lặp đi lặp lại"),
    "a2_failed_grasp": ("(a2) touches the bowl, grasp fails", "(a2) chạm bát nhưng gắp trượt", "#e74c3c",
                        "tay chạm / đẩy bát, đóng gripper nhiều lần nhưng không lần nào kẹp chắc và nhấc lên được"),
    "b_drop": ("(b) grasps, then drops / loses it", "(b) gắp được rồi đánh rơi", "#e67e22",
               "đã nhấc bát lên nhưng tuột / nhả xa đĩa và không gắp lại thành công"),
    "c_misplace": ("(c) carries, but misplaces", "(c) mang tới nhưng đặt sai", "#f1c40f",
                   "bát được mang tới gần đĩa nhưng nằm lệch / trên mép / nghiêng, task không được tính"),
    "d_stuck": ("(d) stuck / jammed", "(d) kẹt / đứng yên", "#7f8c8d",
                "tay gần như không di chuyển (hoặc kẹt vào vật / tủ) trong phần lớn thời gian"),
    "e_progress": ("(e) times out while still progressing", "(e) hết giờ khi đang làm", "#3498db",
                   "vẫn đang cầm bát / đang tiến về đĩa ở bước 520: thiếu thời gian, không phải sai hành vi"),
    "f_infeasible": ("(f) bowl / plate already off the table", "(f) bát hoặc đĩa đã văng khỏi bàn từ đầu", "#34495e",
                     "ngay sau khi load failure state (noise ±2 cm), bát hoặc đĩa đã nằm dưới mặt bàn (rơi xuống sàn): không thể cứu"),
}


def load_tele(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    cols = [str(c) for c in z["cols"]]
    t = z["tele"]
    return {c: t[:, i] for i, c in enumerate(cols)}


def features(T: dict) -> dict:
    n = len(T["eef_x"])
    eef = np.stack([T["eef_x"], T["eef_y"], T["eef_z"]], 1)
    bowl = np.stack([T["bowl_x"], T["bowl_y"], T["bowl_z"]], 1)
    plate = np.stack([T["plate_x"], T["plate_y"]], 1)
    d = np.linalg.norm(bowl[:, :2] - plate, axis=1)
    both = (T["c_left"] > 0) & (T["c_right"] > 0)
    z_rest = float(bowl[0, 2]) if not both[0] else float(np.min(bowl[:, 2]))
    lifted = bowl[:, 2] > z_rest + 0.025
    held = both & lifted
    # longest run of holding
    run = best = 0
    for h in held:
        run = run + 1 if h else 0
        best = max(best, run)
    carried = best >= 10
    g = T["a6"] > 0
    closes = int(np.sum(g[1:] & ~g[:-1]))
    w = min(200, n - 1)
    eef_range_last = float(np.max(np.ptp(eef[-w:], axis=0)))
    path_last = float(np.sum(np.linalg.norm(np.diff(eef[-w:], axis=0), axis=1)))
    cmd = np.linalg.norm(np.stack([T["a0"], T["a1"], T["a2"]], 1), axis=1)
    vel = np.r_[0, np.linalg.norm(np.diff(eef, axis=0), axis=1)]
    jam = float(np.mean((cmd[1:] > 0.3) & (vel[1:] < 0.0015)))
    env_contact = float(np.mean(T["c_robot_env"] > 0))
    held_end = bool(np.mean(held[-10:]) > 0.5)
    d_trend = float(d[max(0, n - 101)] - d[-1])  # > 0 = still approaching the plate in the last 100 steps
    last_held = int(np.max(np.nonzero(held)[0])) if held.any() else -1
    d_release = float(d[last_held]) if last_held >= 0 else float("nan")
    reach = float(np.min(np.linalg.norm(eef - bowl, axis=1)))
    n_touch = int(np.sum((T["c_left"] > 0) | (T["c_right"] > 0)))
    return dict(n=n, reach=reach, n_touch=n_touch, bowl_z0=float(bowl[0, 2]), plate_z0=float(T["plate_z"][0]), d0=float(d[0]), d_end=float(d[-1]), d_min=float(np.min(d)), carried=bool(carried),
                held_steps=int(held.sum()), longest_hold=int(best), closes=closes, eef_range_last=eef_range_last,
                path_last=path_last, jam=jam, env_contact=env_contact, held_end=held_end, d_trend=d_trend,
                last_held=last_held, d_release=d_release, bowl_up_end=float(T["bowl_up"][-1]),
                bowl_dz_end=float(bowl[-1, 2] - z_rest), grasped_at_start=bool(both[0]),
                bowl_moved=float(np.linalg.norm(bowl[-1] - bowl[0])))


def classify(f: dict) -> str:
    if f["bowl_z0"] < 0.6 or f["plate_z0"] < 0.6:  # bowl / plate already below table height (~0.9 m) at the start
        return "f_infeasible"
    if f["held_end"] and f["d_end"] > 0.03:
        return "e_progress"
    if not f["carried"]:
        if f["eef_range_last"] < 0.03 or (f["jam"] > 0.2 and f["env_contact"] > 0.4):
            return "d_stuck"
        never_touched = f["n_touch"] == 0 and f["bowl_moved"] < 0.02
        return "a1_wrong_place" if never_touched else "a2_failed_grasp"
    if f["d_end"] < 0.10 or f["d_release"] < 0.08:
        return "c_misplace"
    if f["d_trend"] > 0.05 and f["last_held"] > f["n"] - 120:
        return "e_progress"
    return "b_drop"


def fonts():
    from PIL import ImageFont

    try:
        return (ImageFont.truetype("DejaVuSans-Bold.ttf", 20), ImageFont.truetype("DejaVuSans-Bold.ttf", 15),
                ImageFont.truetype("DejaVuSans.ttf", 13))
    except Exception:  # pragma: no cover
        f = ImageFont.load_default()
        return f, f, f


def read_frames(mp4: Path) -> list[np.ndarray]:
    import imageio

    return [np.asarray(f) for f in imageio.mimread(mp4, memtest=False)]


def panel(frame: np.ndarray, T: dict, t: int, head: str, col, big: bool = True) -> np.ndarray:
    """agent cam (2x) with wrist inset + telemetry line (viz_e2e style). frame = [agent | wrist] 256 x 512."""
    from PIL import Image, ImageDraw

    F_BIG, F_MED, F_SM = fonts()
    W = 384 if big else 256
    ag, wr = frame[:, :256], frame[:, 256:512]
    cv = Image.new("RGB", (W, W + 30 + 44), (255, 255, 255))
    cv.paste(Image.fromarray(ag).resize((W, W), Image.BICUBIC), (0, 30))
    ws = W // 3
    cv.paste(Image.fromarray(wr).resize((ws, ws), Image.BICUBIC), (W - ws - 3, 30 + W - ws - 3))
    d = ImageDraw.Draw(cv)
    d.rectangle([W - ws - 5, 30 + W - ws - 5, W - 2, 30 + W - 2], outline=(255, 255, 255), width=2)
    d.rectangle([0, 0, W, 28], fill=col)
    d.text((6, 5), head, fill=(255, 255, 255), font=F_MED)
    t = min(t, len(T["eef_z"]) - 1)
    w = T["grip_w"][t]
    both = T["c_left"][t] > 0 and T["c_right"][t] > 0
    dd = float(np.hypot(T["bowl_x"][t] - T["plate_x"][t], T["bowl_y"][t] - T["plate_y"][t]))
    g = "KẸP bát" if both else ("MỞ" if w > 0.06 else "ĐÓNG")
    d.text((6, W + 33), (f"bước {t}  " if big else "") + f"gripper {g}, tay z={T['eef_z'][t]:.2f} m", fill=(0, 0, 0),
           font=F_SM)
    d.text((6, W + 52), f"bát -> đĩa (xy) = {dd * 100:.0f} cm", fill=(0, 0, 0), font=F_SM)
    return np.asarray(cv)


def key_steps(T: dict, f: dict, k: int = 6) -> list[int]:
    n = f["n"] - 1
    held = (T["c_left"] > 0) & (T["c_right"] > 0)
    ev = [0, n]
    if held.any():
        i0 = int(np.argmax(held))
        ev += [i0, int(f["last_held"]) if f["last_held"] > 0 else i0]
    ev = sorted(set(ev))
    for x in np.linspace(0, n, k):
        if len(ev) >= k:
            break
        if min(abs(int(x) - e) for e in ev) > n / (2 * k):
            ev.append(int(x))
    ev = sorted(set(ev))[:k]
    if ev[-1] != n:
        ev[-1] = n
    return ev


def make_strip(row: dict, out: Path) -> None:
    from PIL import Image, ImageDraw

    F_BIG, F_MED, F_SM = fonts()
    T, f = row["T"], row["f"]
    frames = read_frames(row["mp4"])
    lab = MODES[row["mode"]]
    ks = key_steps(T, f)
    pans = [panel(frames[min(len(frames) - 1, s // 2)], T, s, f"bước {s}", (60, 70, 80), big=False) for s in ks]
    strip = np.concatenate(pans, 1)
    hdr = Image.new("RGB", (strip.shape[1], 54), (35, 48, 59))
    d = ImageDraw.Draw(hdr)
    d.text((10, 5), f"{lab[1]}  |  {row['model']}  |  {SHORT[row['task_folder']]}  |  "
                    f"{row['scene'].split('__')[1]} {row['scene'].split('__')[2]} {row['gripper_init']}{row['trial']}",
           fill=(255, 215, 120), font=F_MED)
    d.text((10, 30), lab[3], fill=(255, 255, 255), font=F_SM)
    import io

    img = Image.fromarray(np.concatenate([np.asarray(hdr), strip], 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    atomic_write(out, buf.getvalue())
    print("saved", out, flush=True)


def make_video(row: dict, out: Path) -> None:
    import imageio
    from PIL import Image, ImageDraw

    F_BIG, F_MED, F_SM = fonts()
    T, f = row["T"], row["f"]
    frames = read_frames(row["mp4"])
    lab = MODES[row["mode"]]
    vid = []
    W = 384
    for i, fr in enumerate(frames):
        t = 2 * i
        p = panel(fr, T, t, f"{row['model']}  •  {SHORT[row['task_folder']]}", (46, 139, 87))
        wr = np.asarray(Image.fromarray(fr[:, 256:512]).resize((W, W), Image.BICUBIC))
        wrp = np.full_like(p, 255)
        wrp[30:30 + W] = wr
        body = np.concatenate([p, wrp], 1)
        cap = Image.new("RGB", (body.shape[1], 84), (35, 48, 59))
        d = ImageDraw.Draw(cap)
        d.text((10, 6), lab[1], fill=(255, 215, 120), font=F_BIG)
        d.text((10, 36), lab[3], fill=(255, 255, 255), font=F_SM)
        d.text((10, 58), f"Nhiệm vụ: \"{row['task_text'][:80]}\"  |  phải: camera cổ tay", fill=(190, 200, 210), font=F_SM)
        bar = Image.new("RGB", (body.shape[1], 14), (230, 230, 230))
        ImageDraw.Draw(bar).rectangle([0, 0, int(body.shape[1] * t / 520), 13], fill=(200, 50, 50))
        vid.append(np.concatenate([np.asarray(cap), body, np.asarray(bar)], 0))
    vid = [vid[0]] * 15 + vid + [vid[-1]] * 25
    tmp = out.with_name(out.stem + ".tmp.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(tmp, vid, fps=20, codec="libx264", pixelformat="yuv420p", macro_block_size=1, quality=6)
    os.replace(tmp, out)
    print("saved", out, os.path.getsize(out) // 1024, "KB", flush=True)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from retryvla.envs.libero_env import parse_bddl_language

    rows = [json.loads(l) for l in open(RES / "results.jsonl")]
    seen, data = set(), []
    for r in rows:
        if (r["model"], r["iid"]) in seen:
            continue
        seen.add((r["model"], r["iid"]))
        npz, mp4 = RES / r["model"] / f"{r['iid']}.npz", RES / r["model"] / f"{r['iid']}.mp4"
        if not npz.exists():
            continue
        T = load_tele(npz)
        f = features(T)
        r.update(T=T, f=f, mp4=mp4, mode=None if r["success"] else classify(f))
        tf, ep, st = r["scene"].split("__")
        bddl = REPO / f"data/assets/scenes/scenes/selected_bddl_scenes_spatial/{tf}/{ep}/{st}.bddl"
        r["task_text"] = parse_bddl_language(bddl)
        data.append(r)
    out = {"n": len(data), "per_model": {}, "episodes": []}
    for m in sorted({r["model"] for r in data}):
        dm = [r for r in data if r["model"] == m]
        fails = [r for r in dm if r["mode"]]
        out["per_model"][m] = {"n_resim": len(dm), "resim_failed_again": len(fails),
                               "resim_succeeded": len(dm) - len(fails),
                               "modes": dict(collections.Counter(r["mode"] for r in fails)),
                               "start_near_plate_not_carried": sum(1 for r in fails if r["mode"] in ("a1_wrong_place", "a2_failed_grasp")
                                                                   and r["f"]["d0"] < 0.06),
                               "succ_steps": [r["steps"] for r in dm if not r["mode"]]}
    for r in data:
        out["episodes"].append({k: r[k] for k in ("model", "iid", "task_folder", "gripper_init", "trial", "success",
                                                   "steps", "mode", "logged_progress", "progress")}
                               | {"f": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r["f"].items()}})
    fails = [r for r in data if r["mode"]]
    tasks = sorted({r["task_folder"] for r in data}, key=lambda t: list(SHORT).index(t))
    by = {t: collections.Counter(r["mode"] for r in fails if r["task_folder"] == t) for t in tasks}
    out["per_task"] = {SHORT[t]: dict(v) for t, v in by.items()}
    out["overall"] = dict(collections.Counter(r["mode"] for r in fails))
    atomic_write(REPO / "runs/_analysis/failures/modes.json", json.dumps(out, indent=1))

    # ---- Fig 5: stacked bars per task (+ overall, + per model) ----
    cols = tasks + ["__ALL_m7", "__ALL_pi1", "__ALL"]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for i, c in enumerate(cols):
        if c == "__ALL":
            cnt, n = collections.Counter(r["mode"] for r in fails), len(fails)
        elif c.startswith("__ALL_"):
            mm = "m7base_s0" if c.endswith("m7") else "pi1_s0"
            sub = [r for r in fails if r["model"] == mm]
            cnt, n = collections.Counter(r["mode"] for r in sub), len(sub)
        else:
            cnt, n = by[c], sum(by[c].values())
        bot = 0
        for k, (en, vi, colr, _) in MODES.items():
            v = cnt.get(k, 0) / max(n, 1)
            ax.bar(i, v, bottom=bot, color=colr, label=en if i == 0 else None, edgecolor="white", lw=0.5)
            if cnt.get(k, 0):
                ax.text(i, bot + v / 2, str(cnt[k]), ha="center", va="center", fontsize=7)
            bot += v
        ax.text(i, 1.02, f"n={n}", ha="center", fontsize=7)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([SHORT.get(c, {"__ALL_m7": "ALL M7 s0", "__ALL_pi1": "ALL pi1 s0", "__ALL": "ALL"}.get(c))
                        for c in cols], rotation=20, ha="right", fontsize=8)
    ax.axvline(len(tasks) - 0.5, color="k", lw=1)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("fraction of re-simulated failures")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Fig 5. Failure modes of the re-simulated failures (M7 base s0 + pi1 s0; numbers = episodes)")
    save_fig(fig, "fig5_failure_modes.png")
    plt.close(fig)

    # ---- strips + videos: representative per mode (prefer M7, longest-evidence episode) ----
    VID.mkdir(parents=True, exist_ok=True)
    reps = {}
    for k in MODES:
        cand = [r for r in fails if r["mode"] == k]
        if not cand:
            continue
        tcnt = collections.Counter(r["task_folder"] for r in cand)  # prefer the task where the mode is most common
        cand.sort(key=lambda r: (r["model"] != "m7base_s0", -tcnt[r["task_folder"]], {"a1_wrong_place": -r["f"]["closes"],
                                                            "a2_failed_grasp": -r["f"]["closes"],
                                                            "b_drop": -r["f"]["longest_hold"],
                                                            "c_misplace": r["f"]["d_end"],
                                                            "d_stuck": r["f"]["eef_range_last"],
                                                            "e_progress": -r["f"]["d_trend"],
                                                            "f_infeasible": 0}[k]))
        tsk = set()
        picks = []
        for r in cand:  # 2 strips from different tasks
            if r["task_folder"] not in tsk:
                picks.append(r)
                tsk.add(r["task_folder"])
            if len(picks) == 2:
                break
        reps[k] = picks
        for j, r in enumerate(picks):
            make_strip(r, FIG / f"strip_{k}_{j}.png")
        if k != "f_infeasible":
            make_video(picks[0], VID / f"{k}.mp4")
    out["representatives"] = {k: [r["model"] + ":" + r["iid"] for r in v] for k, v in reps.items()}
    atomic_write(REPO / "runs/_analysis/failures/modes.json", json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("n", "per_model", "per_task", "overall", "representatives")}, indent=1))


if __name__ == "__main__":
    main()
