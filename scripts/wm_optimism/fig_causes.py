#!/usr/bin/env python
"""WM over-optimism, figure 4-6 + causes.json: which families / events / horizons / token regions carry the optimism.

python -m scripts.wm_optimism.fig_causes
Inputs: runs/_analysis/wm_vs_true/rows.parquet (all verif_m5b candidates), runs/_analysis/wm_optimism/cache/
{rows.parquet, tok_err.npy} (cache_stats subset), data/critic3d/verif_m5b (privileged sym state of the same bridges),
data/wm/verif_m5b actions (mmap), cache/gallery.pt (same-scene bank for the off-manifold check).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from retryvla import REPO_ROOT  # noqa: E402
from scripts.wm_optimism.cache_stats import action_family  # noqa: E402

W = REPO_ROOT / "runs/_analysis/wm_optimism"
FIG = W / "figs"


def sym_and_family() -> pd.DataFrame:
    from retryvla.critic.rich import SYM_IDX as I

    out = []
    for f in sorted(glob.glob(str(REPO_ROOT / "data/wm/verif_m5b/*.pt"))):
        stem = Path(f).stem
        d = torch.load(f, weights_only=False, mmap=True)
        fam, flips, rough, _ = action_family(d["actions"].numpy().astype(np.float32))
        lv = d["valid"].bool().long().cumsum(1).argmax(1).numpy()
        rp = REPO_ROOT / "data/critic3d/verif_m5b" / f"{stem}.pt"
        s = torch.load(rp, weights_only=False)["sym"].float().numpy() if rp.exists() else None
        src = list(d["source"]); seen = {}
        for k in range(len(src)):
            r = seen.get(src[k], 0); seen[src[k]] = r + 1
            row = {"instance_id": d["instance_id"], "arm": src[k], "rank": r, "fam": fam[k], "flips": int(flips[k])}
            if s is not None:
                e = int(lv[k])
                row.update(grasp_end=s[k, e, I["grasped"]] > 0.5, grasp_any=s[k, : e + 1, I["grasped"]].max() > 0.5,
                           touch_any=s[k, : e + 1, I["finger_obj_contact"]].max() > 0.5,
                           d_grip_end=float(s[k, e, I["d_grip_obj"]]),
                           lift=float(s[k, e, I["obj_lift"]] - s[k, 0, I["obj_lift"]]),
                           off_table=s[k, e, I["off_table"]] > 0.5)
            out.append(row)
    return pd.DataFrame(out)


def end_type(r) -> str:
    if r.grasp_end:
        return "đang cầm bát"
    if r.touch_any:
        return "chạm/kẹp hụt"
    return "gần bát (<5 cm)" if r.d_grip_end < 0.05 else "xa bát"


def summ(g: pd.DataFrame) -> dict:
    pa = g.c_wm > 0.5
    return {"n": len(g), "sr": g.success.mean(), "c_wm": g.c_wm.mean(), "c_true": g.c_true.mean(),
            "gap_wm": (g.c_wm - g.c_true).mean(), "gap_critic": (g.c_true - g.success).mean(),
            "fp_a": float((~g.success[pa]).mean()) if pa.any() else np.nan,
            "share_pos_a": pa.mean()}


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    res = {}
    m5 = pd.read_parquet(REPO_ROOT / "runs/_analysis/wm_vs_true/rows.parquet")
    m5 = m5[m5.dir == "verif_m5b"]
    sf = sym_and_family()
    m5 = m5.merge(sf, on=["instance_id", "arm", "rank"], how="left")
    m5["sel"] = np.where(m5.arm == "random", "random (không chọn)", "search-selected")
    # ---------------- families x selection
    tab = m5.groupby(["fam", "sel"]).apply(lambda g: pd.Series(summ(g))).reset_index()
    res["family"] = tab.to_dict("records")
    # flips
    m5["flip_bin"] = pd.cut(m5.flips, [-1, 0, 2, 6, 100], labels=["0", "1-2", "3-6", ">6"])
    res["flips"] = m5.groupby(["flip_bin", "sel"], observed=True).apply(lambda g: pd.Series(summ(g))).reset_index().to_dict("records")
    # end-state type
    m5["etype"] = m5.apply(end_type, axis=1)
    res["end_type"] = m5.groupby(["etype", "sel"]).apply(lambda g: pd.Series(summ(g))).reset_index().to_dict("records")
    fpw = m5[(m5.c_wm > 0.5) & ~m5.success]
    res["fp_end_type_share"] = fpw.etype.value_counts(normalize=True).to_dict()
    res["tn_end_type_share"] = m5[(m5.c_wm <= 0.5) & ~m5.success].etype.value_counts(normalize=True).to_dict()
    res["succ_end_type_share"] = m5[m5.success].etype.value_counts(normalize=True).to_dict()
    # ---------------- figure 4: family / flips / end type
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for a, key, order, title in (
            (ax[0], "fam", ["prim", "prim+cem", "pi0_s1.0", "pi0_s1.5", "perturb"], "theo họ ứng viên (heuristic từ action)"),
            (ax[1], "flip_bin", ["0", "1-2", "3-6", ">6"], "theo số lần đổi dấu gripper"),
            (ax[2], "etype", ["đang cầm bát", "chạm/kẹp hụt", "gần bát (<5 cm)", "xa bát"], "theo trạng thái THẬT cuối bridge (sim)")):
        x = np.arange(len(order))
        for j, (sel, col) in enumerate((("random (không chọn)", "#7f7f7f"), ("search-selected", "#d62728"))):
            g = m5[m5.sel == sel]
            wm = [(g[g[key] == o].c_wm - g[g[key] == o].c_true).mean() for o in order]
            cr = [(g[g[key] == o].c_true - g[g[key] == o].success).mean() for o in order]
            n = [int((g[key] == o).sum()) for o in order]
            a.bar(x + (j - 0.5) * 0.38, wm, 0.36, color=col, label=f"{sel}: lỗi WM (C mơ − C thật)")
            a.bar(x + (j - 0.5) * 0.38, cr, 0.36, bottom=wm, color=col, alpha=0.35, hatch="//",
                  label=f"{sel}: lỗi critic (C thật − SR)")
            for xi, ni, w_, c_ in zip(x, n, wm, cr):
                a.text(xi + (j - 0.5) * 0.38, max(w_ + c_, w_, 0) + 0.01, f"n={ni}", ha="center", fontsize=6.5)
        a.axhline(0, color="k", lw=0.7)
        a.set_xticks(x, order, fontsize=8)
        a.set_title(title, fontsize=10)
        a.set_ylabel("độ lạc quan trung bình (ΔC^pi)")
        a.grid(alpha=0.2, axis="y")
    ax[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Ai gây lạc quan? verif_m5b, 256 instance x 64 ứng viên (random = 4096 không chọn; selected = "
                 "wm_diverse / wm_noise / wm_qd, 12288)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "f4_causes_family_event.png", dpi=140)
    plt.close(fig)
    # ---------------- figure 5: horizon + token map (cache_stats subset)
    cr = pd.read_parquet(W / "cache/rows.parquet")
    te = np.load(W / "cache/tok_err.npy").astype(np.float32)
    assert len(te) == len(cr)
    c5 = cr[cr.dir == "verif_m5b"].copy()
    c5["sel"] = np.where(c5.arm == "random", "random", "selected")
    c5["grp"] = np.select([(c5.c_wm > 0.5) & ~c5.success & (c5.c_true < 0.5), (c5.c_wm > 0.5) & c5.success,
                           (c5.c_wm <= 0.5) & ~c5.success], ["FP lỗi WM", "TP", "TN"], "khác")
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    hs = np.arange(1, 11)
    for (sel, grp), col, ls in ((("random", "TN"), "#7f7f7f", "-"), (("selected", "TN"), "#7f7f7f", "--"),
                                (("selected", "TP"), "#1e8449", "-"), (("selected", "FP lỗi WM"), "#c0392b", "-")):
        g = c5[(c5.sel == sel) & (c5.grp == grp)]
        if len(g) < 10:
            continue
        ax[0].plot(hs * 5, [g[f"ez{h}"].mean() for h in hs], ls, color=col, marker="o", ms=3, label=f"{sel} {grp} (n={len(g)})")
        ax[1].plot(hs * 5, [(g[f"cw{h}"] - g[f"ct{h}"]).mean() for h in hs], ls, color=col, marker="o", ms=3,
                   label=f"{sel} {grp}")
    ax[0].set_title("lỗi latent WM theo horizon (MSE token chuẩn hoá)")
    ax[1].set_title("C^pi(mơ) − C^pi(thật) theo horizon\n(critic huấn luyện trên trạng thái CUỐI; giữa chừng chỉ để chẩn đoán)", fontsize=9)
    for a in ax[:2]:
        a.set_xlabel("bước env trong bridge"); a.grid(alpha=0.25); a.legend(fontsize=7)
    m_fp = te[(cr.dir == "verif_m5b").to_numpy() & (cr.c_wm > 0.5).to_numpy() & ~cr.success.to_numpy() & (cr.c_true < 0.5).to_numpy()]
    m_tn = te[(cr.dir == "verif_m5b").to_numpy() & (cr.c_wm <= 0.5).to_numpy() & ~cr.success.to_numpy()]
    m_tp = te[(cr.dir == "verif_m5b").to_numpy() & (cr.c_wm > 0.5).to_numpy() & cr.success.to_numpy()]
    maps = np.stack([m.mean(0).reshape(2, 6, 6) for m in (m_tn, m_tp, m_fp)])
    big = np.concatenate([np.concatenate([maps[i, 0], np.full((6, 1), np.nan), maps[i, 1]], 1) for i in range(3)], 0)
    im = ax[2].imshow(big, cmap="inferno")
    ax[2].set_yticks([2.5, 8.5, 14.5], ["TN", "TP", "FP lỗi WM"])
    ax[2].set_xticks([2.5, 9.5], ["agent 6x6", "wrist 6x6"])
    ax[2].set_title("lỗi token TB ở cuối bridge (bản đồ 6x6)")
    plt.colorbar(im, ax=ax[2], fraction=0.04)
    fig.suptitle(f"Lỗi tăng theo horizon & nằm chủ yếu ở camera cổ tay (subset cache_stats: {len(c5)} ứng viên verif_m5b)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "f5_horizon_tokens.png", dpi=140)
    plt.close(fig)
    res["tokmap"] = {n: {"agent": float(m.reshape(-1, 2, 36)[:, 0].mean()), "wrist": float(m.reshape(-1, 2, 36)[:, 1].mean()),
                         "n": int(len(m))} for n, m in (("TN", m_tn), ("TP", m_tp), ("FP_wm", m_fp))}
    res["horizon"] = {f"{s}|{g}": {"ez": [float(c5[(c5.sel == s) & (c5.grp == g)][f"ez{h}"].mean()) for h in hs],
                                   "dc": [float((c5[(c5.sel == s) & (c5.grp == g)][f"cw{h}"] - c5[(c5.sel == s) & (c5.grp == g)][f"ct{h}"]).mean()) for h in hs]}
                      for s in ("random", "selected") for g in ("TN", "TP", "FP lỗi WM")}
    res["eef_err_m"] = c5.groupby("grp").eef_err.mean().to_dict()
    # ---------------- off-manifold check (gallery bank): is the imagined end latent further from ANY real frame?
    gc = torch.load(W / "cache/gallery.pt", weights_only=False)
    rows = []
    for stem, d in gc.items():
        zt = d["zt"].float(); K, F = zt.shape[:2]; lv = d["lv"]
        bank = torch.cat([zt[k, : int(lv[k]) + 1] for k in range(K)])
        owner = torch.cat([torch.full((int(lv[k]) + 1,), k) for k in range(K)])
        for k in range(K):
            e = int(lv[k])
            for kind, q in (("mơ", d["zi"][k, max(e - 1, 0)].float()), ("thật", zt[k, e])):
                dist = ((bank - q[None]) ** 2).mean((1, 2))
                dist[owner == k] = float("inf")
                rows.append({"stem": stem, "k": k, "kind": kind, "nn": float(dist.min()),
                             "c_wm": float(d["c_wm"][k]), "c_true": float(d["c_true"][k]), "success": bool(d["success"][k])})
    om = pd.DataFrame(rows)
    res["offmanifold"] = om.groupby(["kind"]).nn.describe().to_dict()
    fpm = om[(om.kind == "mơ") & (om.c_wm > 0.5) & ~om.success & (om.c_true < 0.5)]
    res["offmanifold_fp_imag_nn_mean"] = float(fpm.nn.mean())
    res["offmanifold_other_imag_nn_mean"] = float(om[(om.kind == "mơ")].drop(fpm.index).nn.mean())
    fig, a = plt.subplots(figsize=(6.5, 3.8))
    bins = np.linspace(0, om.nn.quantile(0.99), 40)
    a.hist(om[om.kind == "thật"].nn, bins, alpha=0.6, color="#1f77b4", density=True, label="latent THẬT cuối bridge")
    a.hist(om[om.kind == "mơ"].nn, bins, alpha=0.6, color="#d62728", density=True, label="latent MƠ (WM) cuối bridge")
    a.hist(fpm.nn, bins, histtype="step", lw=2, color="k", density=True, label=f"latent mơ của FP lỗi WM (n={len(fpm)})")
    a.set_xlabel("khoảng cách tới frame THẬT gần nhất (cùng cảnh, bridge khác)")
    a.set_title("Latent mơ có 'lạc khỏi đa tạp' các trạng thái thật không?", fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(FIG / "f6_offmanifold.png", dpi=140); plt.close(fig)
    (W / "causes.json").write_text(json.dumps(res, indent=1, default=lambda x: float(x) if np.isscalar(x) else str(x)))
    print(json.dumps(res, indent=1, default=lambda x: float(x) if np.isscalar(x) else str(x))[:6000])


if __name__ == "__main__":
    main()
