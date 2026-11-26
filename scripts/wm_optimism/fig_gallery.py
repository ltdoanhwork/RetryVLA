#!/usr/bin/env python
"""WM over-optimism, figure 3: case gallery (false positives vs true positives), CPU.

python -m scripts.wm_optimism.fig_gallery [--decoder runs/_analysis/wm_optimism/cache/decoder.pt]

Per case (one row): start frame | TRUE end frame (agent, wrist) | per-token error ||z_imag - z_true||^2 over the 6x6
DINOv2 grid overlaid on the true end frame (agent, wrist; one colour scale for all rows) | nearest REAL frame to the
imagined latent (same-scene bank: every frame of every verified bridge of this instance, own trajectory excluded) |
optional decoder reconstruction of the imagined latent (+ of the true latent for fidelity).
Outputs: runs/_analysis/wm_optimism/figs/f3_gallery_{1,2,3}.png + cases.json
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from retryvla import REPO_ROOT  # noqa: E402
from scripts.wm_optimism.common import FAM_VI, grid_err, nn_retrieve, select_cases  # noqa: E402

OUT = REPO_ROOT / "runs/_analysis/wm_optimism"
KIND_VI = {"FP_wm": "FALSE POSITIVE (lỗi WM)", "FP_critic": "FALSE POSITIVE (lỗi critic)", "TP": "TRUE POSITIVE"}
KIND_COL = {"FP_wm": "#c0392b", "FP_critic": "#d35400", "TP": "#1e8449"}


def short_task(t: str) -> str:
    return t.replace("pick the akita black bowl ", "bowl ").replace(" and place it on the plate", " -> plate")


def overlay(ax, img, err, vmax):
    ax.imshow(img)
    h = img.shape[0]
    ax.imshow(err, cmap="inferno", alpha=0.55, vmin=0, vmax=vmax, extent=(0, h, h, 0), interpolation="nearest")
    for i in range(1, 6):
        ax.axhline(i * h / 6, color="w", lw=0.3, alpha=0.4); ax.axvline(i * h / 6, color="w", lw=0.3, alpha=0.4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/_analysis/wm_optimism/cache/gallery.pt")
    ap.add_argument("--decoder", default="runs/_analysis/wm_optimism/cache/decoder.pt")
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    cache = torch.load(REPO_ROOT / a.cache, weights_only=False)
    cases = select_cases(cache)
    dec = None
    dp = REPO_ROOT / a.decoder
    if dp.exists():
        from scripts.wm_optimism.train_decoder import TokDecoder

        ck = torch.load(dp, weights_only=False, map_location="cpu")
        dec = TokDecoder().eval()
        dec.load_state_dict(ck["model"])
        held = {tuple(x) for x in ck["held"]}
        assert all((c["stem"], c["k"]) in held for c in cases), "gallery cases must be held out of decoder training"

    def decode(tok, cam):
        with torch.no_grad():
            return dec(tok.float()[None], torch.tensor([cam]))[0].permute(1, 2, 0).numpy()

    errs = [grid_err(cache[c["stem"]], c["k"]) for c in cases]
    vmax_a = float(np.percentile(np.concatenate([e[0].ravel() for e in errs]), 97))   # one scale per camera
    vmax_w = float(np.percentile(np.concatenate([e[1].ravel() for e in errs]), 97))
    for c, e in zip(cases, errs):
        d = cache[c["stem"]]
        k = c["k"]
        lv = int(d["lv"][k])
        zq = d["zi"][k, max(lv - 1, 0)]
        kk, ff, dist, own, rank = nn_retrieve(cache, c["stem"], k, zq)
        c.update({"task": d["task"], "lv": lv, "err_agent": float(e[0].mean()), "err_wrist": float(e[1].mean()),
                  "nn_k": kk, "nn_f": ff, "nn_dist": dist, "own_end_dist": own, "own_end_rank": rank,
                  "nn_success": bool(d["success"][kk]), "nn_c_true_end": float(d["c_true"][kk]),
                  "nn_is_end": ff == int(d["lv"][kk])})
        if d.get("sym") is not None:
            from retryvla.critic.rich import SYM_IDX

            s = d["sym"][k].float()
            c["grasped_end"] = float(s[lv, SYM_IDX["grasped"]])
            c["grasped_any"] = float(s[: lv + 1, SYM_IDX["grasped"]].max())
            c["obj_lift_end"] = float(s[lv, SYM_IDX["obj_lift"]])
            c["d_grip_obj_end"] = float(s[lv, SYM_IDX["d_grip_obj"]])
            c["obj_goal_contact_end"] = float(s[lv, SYM_IDX["obj_goal_contact"]])
            c["grip_open_end"] = float(s[lv, SYM_IDX["grip_open"]])
    cols = ["start (t=0)", "TRUE end\n(agent)", "TRUE end\n(wrist)", "lỗi token WM\n(agent, 6x6)",
            "lỗi token WM\n(wrist, 6x6)", "frame THẬT gần nhất\nvới latent mơ (agent)", "… (wrist)"]
    if dec is not None:
        cols += ["DECODER(latent mơ)\nagent", "DECODER(latent mơ)\nwrist", "DECODER(latent thật)\nagent (độ trung thực)"]
    groups = [[c for c in cases if c["kind"] == "FP_wm"][:5], [c for c in cases if c["kind"] == "FP_wm"][5:],
              [c for c in cases if c["kind"] != "FP_wm"]]
    files = []
    for gi, grp in enumerate(groups, 1):
        if not grp:
            continue
        n = len(grp)
        fig, axs = plt.subplots(n, len(cols), figsize=(1.45 * len(cols), 1.95 * n + 0.45), squeeze=False)
        for r, c in enumerate(grp):
            d = cache[c["stem"]]
            k, lv = c["k"], c["lv"]
            e = grid_err(d, k)
            ims = [d["a256"][k, 0].numpy(), d["a256"][k, 1].numpy(), d["w256"][k, 1].numpy()]
            for j in range(3):
                axs[r, j].imshow(ims[j])
            overlay(axs[r, 3], d["a256"][k, 1].numpy(), e[0], vmax_a)
            overlay(axs[r, 4], d["w256"][k, 1].numpy(), e[1], vmax_w)
            axs[r, 5].imshow(d["img_a"][c["nn_k"], c["nn_f"]].numpy())
            axs[r, 6].imshow(d["img_w"][c["nn_k"], c["nn_f"]].numpy())
            axs[r, 5].text(3, 12, f"cand {c['nn_k']} t={5 * c['nn_f']}\n{'SUCCESS' if c['nn_success'] else 'fail'}",
                           fontsize=6.5, va="top", color="#1e8449" if c["nn_success"] else "#444444",
                           bbox=dict(facecolor="white", alpha=0.8, lw=0, pad=1))
            if dec is not None:
                zq = d["zi"][k, max(lv - 1, 0)]
                axs[r, 7].imshow(decode(zq[:36], 0))
                axs[r, 8].imshow(decode(zq[36:], 1))
                axs[r, 9].imshow(decode(d["zt"][k, lv][:36], 0))
            for j in range(len(cols)):
                axs[r, j].set_xticks([]); axs[r, j].set_yticks([])
                if r == 0:
                    axs[r, j].set_title(cols[j], fontsize=7.5)
            g = ""
            if "grasped_end" in c:
                g = (f" | sim: grasped(end)={int(c['grasped_end'] > 0.5)} any={int(c['grasped_any'] > 0.5)}"
                     f" lift={c['obj_lift_end']:.3f} m d(grip,bowl)={c['d_grip_obj_end']:.3f} m")
            txt = (f"#{cases.index(c) + 1} {KIND_VI[c['kind']]} | {short_task(c['task'])} | arm {c['arm']} | "
                   f"{FAM_VI.get(c['fam'], c['fam'])} | gripper flips {c['flips']}\n"
                   f"C^pi(mơ)={c['c_wm']:.2f}  C^pi(thật)={c['c_true']:.2f}  sim thật: "
                   f"{'SUCCESS' if c['success'] else 'FAIL'} | lỗi token TB agent {c['err_agent']:.2f} / wrist "
                   f"{c['err_wrist']:.2f}{g}")
            axs[r, 0].text(0, -0.04, txt, transform=axs[r, 0].transAxes, fontsize=7.3, va="top",
                           color=KIND_COL[c["kind"]], fontweight="bold")
        fig.suptitle("Gallery: 'WM mơ thấy gì' vs sự thật (latent WM v1: DINOv2-small 6x6 x 2 cam; heatmap = "
                     f"||z_mơ − z_thật||² từng token;\nthang chung mọi hàng: agent 0–{vmax_a:.2f}, wrist 0–{vmax_w:.2f}; "
                     "cột 'frame THẬT gần nhất' = frame thật (cùng cảnh, bridge khác) có latent gần latent mơ nhất)",
                     fontsize=8.5)
        fig.subplots_adjust(left=0.01, right=0.99, top=1 - 1.05 / (1.95 * n + 0.45), bottom=0.06, wspace=0.04,
                            hspace=0.42)
        p = OUT / "figs" / f"f3_gallery_{gi}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        files.append(str(p))
    (OUT / "cases.json").write_text(json.dumps(cases, indent=1, default=float))
    print(json.dumps([{k: c[k] for k in ("kind", "arm", "fam", "c_wm", "c_true", "success", "err_agent", "err_wrist",
                                          "nn_success", "own_end_rank")} for c in cases], indent=0)[:3000])
    print(files)


if __name__ == "__main__":
    main()
