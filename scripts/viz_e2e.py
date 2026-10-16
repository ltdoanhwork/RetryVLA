#!/usr/bin/env python
"""End-to-end (NO search) behaviour of the policies from one failure state: pi0 vs pi1 variants.

python scripts/viz_e2e.py <split>:<scene_tag>__<close|open><k> [...]     (split only used in the caption)

Per instance -> runs/_analysis/viz/e2e/<short>/e2e.mp4 + keyframes.png + seeds.json
Columns: pi0 (SmolVLA base) | pi1-nominal (demo SFT) | pi1-random (+208 random-search recoveries) |
pi1-WM (+208 WM recoveries, round-robin selection). Every policy is rolled out N_SEEDS times from the same
failure state; the video shows seed 0 for every column (not cherry-picked) and the header gives k/N successes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "runs/_analysis/viz/e2e"
N_SEEDS, T_MAX = 5, 520
POLICIES = [
    ("π0  SmolVLA gốc", "HuggingFaceVLA/smolvla_libero", (120, 120, 120)),
    ("π1-nominal  (+demo)", "runs/20261013-080655_M1_sft_nominal_only/checkpoints/006000/pretrained_model", (90, 110, 140)),
    ("π1-random  (+208 recovery random)", "@runs/*_M5_m5b_sft_random/checkpoints/006000/pretrained_model", (150, 110, 60)),
    ("π1-WM  (+208 recovery WM)", "@runs/*_M5_m5c_sft_wm_diverse_rr/checkpoints/006000/pretrained_model", (230, 130, 20)),
]


def main() -> None:
    import glob

    import imageio
    from PIL import Image, ImageDraw

    from retryvla.config import load_config
    from retryvla.envs.libero_env import LiberoSim, discover_scenes
    from retryvla.eval.eval_recover import noise_seed
    from retryvla.policy import SmolVLAWrapper
    from scripts.viz_explainer import F_BIG, F_MED, F_SM

    cfg = load_config(REPO / "configs/exp/m0_baseline_spatial.yaml")
    scenes = {s.tag: s for s in discover_scenes(REPO / cfg.recover.scene_root)}
    pols = []
    for name, path, col in POLICIES:
        if path.startswith("@"):
            path = sorted(glob.glob(str(REPO / path[1:])))[-1]
        elif not path.startswith("HuggingFace"):
            path = str(REPO / path)
        pols.append((name, SmolVLAWrapper(path, "cuda", 10), col))
    sim = LiberoSim()

    for arg in sys.argv[1:]:
        split, iid = arg.split(":", 1)
        tag, gt = iid.rsplit("__", 1)
        g0 = gt.rstrip("0123456789")
        k = int(gt[len(g0):])
        sc = scenes[tag]
        task = sc.instruction(cfg.recover.instruction)
        sim.load_scene(sc.bddl, sc.npy, gripper_close=(g0 == "close"), noise_xy=float(cfg.recover.obj_noise_xy),
                       noise_seed=noise_seed(int(cfg.eval.seed), sc.tag, k),
                       settle_steps=int(cfg.recover.gripper_settle_steps), build_seed=int(cfg.eval.seed))
        snap = sim.snapshot()

        def run(policy, seed):
            o = sim.restore_snapshot(sc.bddl, snap, build_seed=int(cfg.eval.seed))["obs"]
            fr, t = [dict(img=o["image"], wr=o["image2"], st=o["state"].copy())], 0
            while t < T_MAX:
                for a in policy.predict_chunk([o], [task], seeds=[seed * 7919 + t])[0][:10]:
                    r = sim.step(a)
                    o, t = r.obs, t + 1
                    fr.append(dict(img=o["image"], wr=o["image2"], st=o["state"].copy()))
                    if r.success:
                        return fr, True, t
            return fr, False, t

        res, stats = {}, {}
        for name, pol, _ in pols:
            outs = [run(pol, s) for s in range(N_SEEDS)]
            res[name] = outs[0]
            stats[name] = {"successes": int(sum(o[1] for o in outs)), "n": N_SEEDS,
                           "steps": [int(o[2]) for o in outs], "success_per_seed": [bool(o[1]) for o in outs]}
            print(iid[-45:], name, stats[name], flush=True)

        W, PH = 448, 448 + 96

        def panel(i, t):
            name, _, col = pols[i]
            fr, ok, n = res[name]
            f = fr[min(t, len(fr) - 1)]
            done = t >= len(fr) - 1
            cv = Image.new("RGB", (W, PH), (255, 255, 255))
            cv.paste(Image.fromarray(f["img"]).resize((W, W), Image.BICUBIC), (0, 56))
            cv.paste(Image.fromarray(f["wr"]).resize((128, 128), Image.BICUBIC), (W - 132, 56 + W - 132))
            d = ImageDraw.Draw(cv)
            d.rectangle([W - 134, 56 + W - 134, W - 3, 56 + W - 3], outline=(255, 255, 255), width=2)
            d.rectangle([0, 0, W, 54], fill=col)
            d.text((8, 5), name, fill=(255, 255, 255), font=F_MED)
            s = stats[name]
            d.text((8, 30), f"thành công {s['successes']}/{s['n']} seed  •  video: seed 0", fill=(255, 255, 255), font=F_SM)
            if done:
                c2 = (40, 160, 80) if ok else (200, 50, 50)
                d.rectangle([0, 56, W - 1, 56 + W - 1], outline=c2, width=8)
                d.text((12, 66), ("THÀNH CÔNG ở bước %d" % n) if ok else "THẤT BẠI (hết %d bước)" % T_MAX,
                       fill=c2, font=F_BIG)
            y = 56 + W + 6
            gq = f["st"][6] - f["st"][7]
            d.text((8, y), f"bước {min(t, len(fr) - 1)}   gripper {'MỞ' if abs(gq) > 0.06 else 'ĐÓNG/kẹp'}   "
                           f"độ cao tay {f['st'][2]:.2f} m", fill=(0, 0, 0), font=F_SM)
            d.rectangle([8, y + 24, W - 8, y + 36], outline=(160, 160, 160))
            d.rectangle([8, y + 24, 8 + (W - 16) * min(t, len(fr) - 1) / T_MAX, y + 36],
                        fill=(40, 160, 80) if done and ok else (200, 50, 50) if done else (0, 130, 220))
            return np.asarray(cv)

        def frame(t, cap):
            row = np.concatenate([panel(i, t) for i in range(len(pols))], 1)
            c = Image.new("RGB", (row.shape[1], 86), (35, 48, 59))
            d = ImageDraw.Draw(c)
            d.text((12, 6), f"[{split.upper()}] Nhiệm vụ: \"{task}\"   •   cả 4 policy bắt đầu từ CÙNG trạng thái lỗi, "
                            f"KHÔNG search, KHÔNG WM lúc chạy", fill=(255, 255, 255), font=F_SM)
            d.text((12, 36), cap, fill=(255, 215, 120), font=F_BIG)
            return np.concatenate([np.asarray(c), row], 0)

        T = max(len(res[n][0]) for n, _, _ in pols)
        vid = [frame(0, "Trạng thái lỗi ban đầu: tay robot đang ở chỗ mà SmolVLA gốc hay bị kẹt.")] * 30
        for t in range(1, T, 2):
            vid.append(frame(t, "Mỗi policy tự điều khiển từ đầu đến cuối (end-to-end)."))
        vid += [frame(T, "Kết quả. π1-WM đã HỌC được cách tự thoát ra rồi làm lại, không cần search.")] * 45
        short = tag.replace("pick_up_the_black_bowl_", "")[:34].strip("_") + "__" + tag.split("__")[1] + "_" + \
            tag.split("__")[2] + "__" + gt
        od = OUT / f"{split}_{short}"
        od.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(od / "e2e.mp4", vid, fps=20, macro_block_size=1)
        keys = [0, 40, 80, 140, 220, T]
        rows = [np.concatenate([panel(i, t) for t in keys], 1) for i in range(len(pols))]
        hdr = Image.new("RGB", (rows[0].shape[1], 36), (35, 48, 59))
        d = ImageDraw.Draw(hdr)
        for j, t in enumerate(keys):
            d.text((j * W + 10, 7), f"bước {t}" if t < T else "kết quả", fill=(255, 215, 120), font=F_MED)
        imageio.imwrite(od / "keyframes.png", np.concatenate([np.asarray(hdr)] + rows, 0))
        (od / "seeds.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
        print("saved", od, flush=True)


if __name__ == "__main__":
    main()
