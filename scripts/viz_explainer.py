#!/usr/bin/env python
"""Explainer video: what does a WM-found recovery actually DO, compared with SmolVLA alone and a random try.

python scripts/viz_explainer.py <instance_id> [<instance_id> ...]

Per instance -> runs/_analysis/viz/explainer/<short>/explainer.mp4 (+ keyframes.png):
  3 columns, all re-simulated from the SAME failure snapshot:
    A) SmolVLA alone (no recovery)          B) random 50-step try, then SmolVLA      C) WM-chosen 50-step segment, then SmolVLA
  each column: agent camera (2x) + wrist camera, live gripper state / hand height, a progress bar
  (orange = searched segment, blue = SmolVLA), and a Vietnamese caption per phase. The video pauses on the
  failure state, at the end of the searched segment, and at the outcome.
WM column: the first verified WM success (from M5b verification logs) that also succeeds on re-simulation.
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "runs/_analysis/viz/explainer"
H = 50
try:
    from PIL import ImageFont

    F_BIG = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    F_MED = ImageFont.truetype("DejaVuSans-Bold.ttf", 17)
    F_SM = ImageFont.truetype("DejaVuSans.ttf", 15)
except Exception:  # pragma: no cover
    F_BIG = F_MED = F_SM = None

COLS = {"A": (120, 120, 120), "B": (150, 110, 60), "C": (230, 130, 20)}
TITLE = {"A": "A. SmolVLA tự làm (không recovery)", "B": "B. 50 bước RANDOM → SmolVLA", "C": "C. 50 bước WM chọn → SmolVLA"}


def describe(a: np.ndarray) -> str:
    """Plain-language description of a 50-step segment."""
    g = a[:, 6] > 0
    g0, g1 = g[:5].mean() > 0.5, g[-5:].mean() > 0.5
    grip = {(False, False): "mở gripper (nhả vật)", (False, True): "mở rồi đóng lại (gắp lại)",
            (True, False): "đang kẹp rồi nhả ra", (True, True): "giữ chặt vật"}[(bool(g0), bool(g1))]
    m = a[:, :3].mean(0)
    mv = []
    if m[2] > 0.1:
        mv.append("nâng tay lên")
    elif m[2] < -0.1:
        mv.append("hạ tay xuống")
    if np.linalg.norm(m[:2]) > 0.1:
        ax = 0 if abs(m[0]) >= abs(m[1]) else 1
        mv.append("dịch ngang " + ("+" if m[ax] > 0 else "-") + "xy"[ax])
    return grip + (", " + ", ".join(mv) if mv else ", gần như đứng yên")


def main() -> None:
    import imageio
    import torch
    from PIL import Image, ImageDraw

    from retryvla.config import load_config
    from retryvla.envs.libero_env import LiberoSim, discover_scenes
    from retryvla.eval.eval_recover import noise_seed
    from retryvla.policy import SmolVLAWrapper

    cfg = load_config(REPO / "configs/exp/m5b_compare_arms.yaml")
    policy = SmolVLAWrapper(cfg.policy.path, "cuda", 10)
    sim = LiberoSim()
    scenes = {s.tag: s for s in discover_scenes(REPO / cfg.recover.scene_root)}

    for iid in sys.argv[1:]:
        tag, gt = iid.rsplit("__", 1)
        grip0 = gt.rstrip("0123456789")
        k = int(gt[len(grip0):])
        sc = scenes[tag]
        task = sc.instruction(cfg.recover.instruction)
        out = sim.load_scene(sc.bddl, sc.npy, gripper_close=(grip0 == "close"),
                             noise_xy=float(cfg.recover.obj_noise_xy), noise_seed=noise_seed(int(cfg.eval.seed), sc.tag, k),
                             settle_steps=int(cfg.recover.gripper_settle_steps), build_seed=int(cfg.eval.seed))
        snap = sim.snapshot()
        v = torch.load(REPO / "data/wm/verif_m5b" / f"{iid}.pt", weights_only=False, map_location="cpu")
        acts, src, ok = v["actions"].numpy(), list(v["source"]), v["success"].numpy().astype(bool)
        seed0 = zlib.crc32(iid.encode()) % (2**31)

        def run(prefix, seed, T=360):
            o = sim.restore_snapshot(sc.bddl, snap, build_seed=int(cfg.eval.seed))["obs"]
            fr = [dict(img=o["image"], wr=o["image2"], st=o["state"].copy(), seg=-1)]
            t = 0
            for a in (prefix if prefix is not None else []):
                r = sim.step(a)
                o, t = r.obs, t + 1
                fr.append(dict(img=o["image"], wr=o["image2"], st=o["state"].copy(), seg=0, a=a))
                if r.success:
                    return fr, True
            while t < T:
                ch = policy.predict_chunk([o], [task], seeds=[seed + t])[0][:10]
                for a in ch:
                    r = sim.step(a)
                    o, t = r.obs, t + 1
                    fr.append(dict(img=o["image"], wr=o["image2"], st=o["state"].copy(), seg=1, a=a))
                    if r.success:
                        return fr, True
            return fr, False

        A = run(None, seed0)
        rnd = [j for j in range(len(src)) if src[j] == "random"]
        B, b_idx = None, None
        for j in rnd:  # a random try that fails (what random usually does: 94% of random tries fail)
            res = run(acts[j], seed0 + 1)
            if not res[1]:
                B, b_idx = res, j
                break
        if B is None:
            b_idx = rnd[0]
            B = run(acts[b_idx], seed0 + 1)
        C, c_idx = None, None
        wins = [j for j in range(len(src)) if src[j] != "random" and ok[j]]
        for j in sorted(wins, key=lambda j: -float(v["pred_value"][j])):
            for s in range(3):
                res = run(acts[j], seed0 + 101 * s)
                if res[1]:
                    C, c_idx = res, j
                    break
            if C is not None:
                break
        if C is None:
            print("no reproducible WM success for", iid)
            continue
        cols = {"A": (A, None), "B": (B, acts[b_idx]), "C": (C, acts[c_idx])}
        print(iid, {c: (len(f), s) for c, ((f, s), _) in cols.items()}, "WM arm:", src[c_idx])

        W, PH = 512, 512 + 170  # panel width / height
        def panel(c, t, phase):
            (fr, succ), seg_a = cols[c]
            n = len(fr)
            f = fr[min(t, n - 1)]
            done = t >= n - 1
            cv = Image.new("RGB", (W, PH), (255, 255, 255))
            cv.paste(Image.fromarray(f["img"]).resize((W, W), Image.BICUBIC), (0, 60))
            cv.paste(Image.fromarray(f["wr"]).resize((150, 150), Image.BICUBIC), (W - 156, 60 + W - 156))
            d = ImageDraw.Draw(cv)
            d.rectangle([W - 158, 60 + W - 158, W - 4, 60 + W - 4], outline=(255, 255, 255), width=2)
            d.text((W - 154, 60 + W - 178), "camera cổ tay", fill=(255, 255, 255), font=F_SM)
            d.rectangle([0, 0, W, 58], fill=COLS[c])
            d.text((10, 6), TITLE[c], fill=(255, 255, 255), font=F_MED)
            if done:
                st, col = ("THÀNH CÔNG" if succ else "THẤT BẠI"), ((40, 160, 80) if succ else (200, 50, 50))
            elif f["seg"] == 0:
                st, col = "đang chạy 50 bước search", (230, 130, 20)
            elif f["seg"] == 1:
                st, col = "SmolVLA đang tự làm", (0, 130, 220)
            else:
                st, col = "trạng thái lỗi ban đầu", (90, 90, 90)
            d.text((10, 32), f"bước {min(t, n - 1)}  •  {st}", fill=(255, 255, 255), font=F_SM)
            if done:
                d.rectangle([0, 60, W - 1, 60 + W - 1], outline=col, width=8)
            # info rows
            y = 60 + W + 8
            gq = f["st"][6] - f["st"][7] if len(f["st"]) > 7 else f["st"][6]
            gtxt = "MỞ" if abs(gq) > 0.06 else "ĐÓNG/kẹp"
            d.text((10, y), f"gripper: {gtxt}    độ cao tay: {f['st'][2]:.2f} m", fill=(0, 0, 0), font=F_SM)
            if seg_a is not None:
                d.text((10, y + 22), "50 bước đầu: " + describe(seg_a), fill=(150, 80, 0), font=F_SM)
            else:
                d.text((10, y + 22), "không có đoạn recovery: SmolVLA làm tiếp từ đầu", fill=(80, 80, 80), font=F_SM)
            # timeline bar
            y2 = y + 52
            Tm = 360
            d.rectangle([10, y2, W - 10, y2 + 16], outline=(160, 160, 160))
            xs = lambda s: 10 + (W - 20) * min(s, Tm) / Tm
            seg_end = H if seg_a is not None else 0
            if seg_end and min(t, seg_end, n - 1) > 0:
                d.rectangle([10, y2, xs(min(t, seg_end, n - 1)), y2 + 16], fill=(230, 130, 20))
            if t > seg_end and n - 1 > seg_end:
                d.rectangle([xs(seg_end), y2, xs(min(t, n - 1)), y2 + 16], fill=(0, 130, 220))
            if done:
                d.text((xs(n - 1) + 4 if xs(n - 1) < W - 120 else xs(n - 1) - 110, y2 - 1),
                       "✓ xong" if succ else "✗ hết giờ", fill=col, font=F_SM)
            d.text((10, y2 + 22), "thanh thời gian: cam = 50 bước search, xanh = SmolVLA", fill=(110, 110, 110), font=F_SM)
            return np.asarray(cv)

        def frame(t, caption):
            row = np.concatenate([panel(c, t, None) for c in "ABC"], 1)
            cap = Image.new("RGB", (row.shape[1], 92), (35, 48, 59))
            d = ImageDraw.Draw(cap)
            d.text((14, 8), f"Nhiệm vụ: \"{task}\"", fill=(255, 255, 255), font=F_MED)
            d.text((14, 44), caption, fill=(255, 215, 120), font=F_BIG)
            return np.concatenate([np.asarray(cap), row], 0)

        T = max(len(cols[c][0][0]) for c in "ABC")
        cap0 = "1) Cả 3 cột bắt đầu từ CÙNG một trạng thái lỗi (SmolVLA đã làm hỏng trước đó)."
        cap1 = "2) 50 bước đầu: B chạy chuỗi action ngẫu nhiên, C chạy chuỗi mà WM tưởng tượng là tốt nhất."
        cap2 = "3) Hết 50 bước: từ đây cả 3 cột đều do SmolVLA tự điều khiển (không ai giúp nữa)."
        cap3 = "4) Kết quả: chỉ khác nhau ở 50 bước đầu, nhưng C đưa robot về trạng thái SmolVLA làm tiếp được."
        vid = [frame(0, cap0)] * 30
        for t in range(1, H + 1):
            vid += [frame(t, cap1)] * 2  # half speed during the searched segment
        vid += [frame(H, cap2)] * 30
        for t in range(H + 1, T, 2):
            vid.append(frame(t, cap2 if t < H + 40 else cap3))
        vid += [frame(T, cap3)] * 45
        short = tag.replace("pick_up_the_black_bowl_", "")[:40].strip("_") + "__" + gt
        od = OUT / short
        od.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(od / "explainer.mp4", vid, fps=15, macro_block_size=1)
        keys = [(0, "lỗi ban đầu"), (25, "giữa đoạn search"), (H, "hết 50 bước"), (H + 60, "SmolVLA làm tiếp"), (T, "kết quả")]
        rows = []
        for c in "ABC":
            rows.append(np.concatenate([panel(c, t, None) for t, _ in keys], 1))
        hdr = Image.new("RGB", (rows[0].shape[1], 40), (35, 48, 59))
        d = ImageDraw.Draw(hdr)
        for i, (t, name) in enumerate(keys):
            d.text((i * W + 12, 8), f"{name} (bước {t if t < T else 'cuối'})", fill=(255, 215, 120), font=F_MED)
        imageio.imwrite(od / "keyframes.png", np.concatenate([np.asarray(hdr)] + rows, 0))
        print("saved", od)


if __name__ == "__main__":
    main()
