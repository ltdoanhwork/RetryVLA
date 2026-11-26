#!/usr/bin/env python
"""WM over-optimism, figure 1 + 2: exploitation curve and the winner's curse (CPU, existing data only).

python -m scripts.wm_optimism.fig_curve
Inputs: runs/_analysis/wm_vs_true/{rows,rows_rq3,rows_vj}.parquet (same candidates scored on WM-imagined vs TRUE end
latents) and the online RQ3 sweeps runs/*_RQ3_{verif,vj}_sweep_s*/per_instance.jsonl (V-JEPA's OWN picks).
Outputs: runs/_analysis/wm_optimism/figs/{f1_exploitation_curve,f2_winners_curse}.png + curve.json
"""

from __future__ import annotations

import glob
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from retryvla import REPO_ROOT  # noqa: E402

A = REPO_ROOT / "runs/_analysis/wm_vs_true"
OUT = REPO_ROOT / "runs/_analysis/wm_optimism/figs"
BUD = (8, 32, 128, 512)


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    r = np.random.default_rng(seed)
    m = r.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def load_online(pat):
    d = pd.concat([pd.read_json(f, lines=True) for f in sorted(glob.glob(str(REPO_ROOT / pat)))])
    return d.drop_duplicates(["instance_id", "arm"], keep="last")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    rq3 = pd.read_parquet(A / "rows_rq3.parquet")
    m5 = pd.read_parquet(A / "rows.parquet")
    vj = pd.read_parquet(A / "rows_vj.parquet")
    bud = rq3[rq3.arm == "wm_bud_l0"].copy()
    bud["N"] = bud.pick_budget.astype(int)
    key = ["dir", "instance_id", "arm", "rank"]
    bud = bud.merge(vj, on=key + ["success"], how="left")
    # ---- online curves (each verifier's OWN picks): imagined C^pi + real success
    on = {}
    for name, pat in (("WM v1", "runs/*_RQ3_verif_sweep_s*/per_instance.jsonl"),
                      ("V-JEPA 2-AC", "runs/*_RQ3_vj_sweep_s*/per_instance.jsonl")):
        d = load_online(pat)
        on[name] = d[d.arm == "wm_bud_l0"]
    common = sorted(set(on["WM v1"].instance_id) & set(on["V-JEPA 2-AC"].instance_id))
    curve = {}
    for name, g in on.items():
        g = g.set_index("instance_id").loc[common]
        y = np.concatenate(g.order_success.to_list()).astype(float)
        p = np.concatenate(g.order_pred.to_list())
        b = np.concatenate(g.order_budget.to_list())
        curve[name] = {int(N): {"imag": float(p[b == N].mean()), "imag_ci": boot_ci(p[b == N]),
                                "real": float(y[b == N].mean()), "real_ci": boot_ci(y[b == N]),
                                "fp": float((y[b == N][p[b == N] > 0.5] == 0).mean())} for N in BUD}
    # offline: SAME WM v1 picks re-simulated -> true C^pi (DINO critic) and V-JEPA critic on true / imagined
    off = {}
    for N in BUD:
        g = bud[bud.N == N]
        off[N] = {"c_wm": g.c_wm.mean(), "c_true": g.c_true.mean(), "sr": g.success.mean(),
                  "c_true_ci": boot_ci(g.c_true), "c_wm_ci": boot_ci(g.c_wm), "sr_ci": boot_ci(g.success),
                  "vj_true": g.c_vj_true.mean(), "n": len(g), "fp_a": float((~g.success[g.c_wm > 0.5]).mean()),
                  "fp_b": float((~g.success[g.c_true > 0.5]).mean())}
    # ------------------------------------------------------------------ figure 1
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    x = np.arange(len(BUD))
    for name, col, mk in (("WM v1", "#d62728", "o"), ("V-JEPA 2-AC", "#9467bd", "s")):
        c = curve[name]
        im = [c[N]["imag"] for N in BUD]
        lo = [c[N]["imag_ci"][0] for N in BUD]; hi = [c[N]["imag_ci"][1] for N in BUD]
        ax[0].plot(x, im, mk + "-", color=col, lw=2.2, label=f"{name}: C^pi tưởng tượng (WM chấm)")
        ax[0].fill_between(x, lo, hi, color=col, alpha=0.12)
        re = [c[N]["real"] for N in BUD]
        lo = [c[N]["real_ci"][0] for N in BUD]; hi = [c[N]["real_ci"][1] for N in BUD]
        ax[0].plot(x, re, mk + "--", color=col, lw=1.6, alpha=0.8, label=f"{name}: thành công THẬT (sim)")
        ax[0].fill_between(x, lo, hi, color=col, alpha=0.06)
    ax[0].plot(x, [off[N]["c_wm"] for N in BUD], "x:", color="#d62728", ms=8, lw=1,
               label="WM v1 picks (offline, chạy lại): C^pi tưởng tượng")
    ct = [off[N]["c_true"] for N in BUD]
    lo = [off[N]["c_true_ci"][0] for N in BUD]; hi = [off[N]["c_true_ci"][1] for N in BUD]
    ax[0].plot(x, ct, "D-", color="#1f77b4", lw=2, label="WM v1 picks: C^pi trên latent THẬT (critic, không WM)")
    ax[0].fill_between(x, lo, hi, color="#1f77b4", alpha=0.12)
    for i, N in enumerate(BUD):
        ax[0].annotate("", xy=(i + 0.06, off[N]["c_wm"]), xytext=(i + 0.06, off[N]["c_true"]),
                       arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1, alpha=0.6))
        ax[0].annotate("", xy=(i - 0.06, off[N]["c_true"]), xytext=(i - 0.06, off[N]["sr"]),
                       arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1, alpha=0.6))
    ax[0].text(3.1, (off[512]["c_wm"] + off[512]["c_true"]) / 2, "lỗi WM", color="#d62728", fontsize=9)
    ax[0].text(3.1, (off[512]["c_true"] + off[512]["sr"]) / 2, "lỗi critic", color="#1f77b4", fontsize=9)
    ax[0].set_xticks(x, [f"N={N}" for N in BUD])
    ax[0].set_xlabel("ngân sách tưởng tượng N (số ứng viên WM chấm trước khi chọn top-4)")
    ax[0].set_ylabel("trung bình trên top-4 / instance")
    ax[0].set_ylim(0, 1)
    ax[0].set_title("Exploitation curve: càng tìm nhiều trong mơ, WM càng 'tự tin',\n"
                    "nhưng thành công thật KHÔNG tăng", fontsize=10.5)
    ax[0].legend(fontsize=7.6, loc="upper left")
    ax[0].grid(alpha=0.25)
    # FP panel
    for name, col, mk in (("WM v1", "#d62728", "o"), ("V-JEPA 2-AC", "#9467bd", "s")):
        ax[1].plot(x, [curve[name][N]["fp"] for N in BUD], mk + "-", color=col, lw=2,
                   label=f"{name}: FP của C^pi tưởng tượng (online)")
    ax[1].plot(x, [off[N]["fp_b"] for N in BUD], "D-", color="#1f77b4", lw=2,
               label="WM v1 picks: FP nếu chấm trên latent THẬT")
    ax[1].set_xticks(x, [f"N={N}" for N in BUD])
    ax[1].set_ylim(0.4, 1)
    ax[1].set_ylabel("P(thất bại thật | C^pi > 0.5)")
    ax[1].set_xlabel("ngân sách tưởng tượng N")
    ax[1].set_title("False-positive rate: >70 % 'thành công trong mơ' là thất bại thật", fontsize=10.5)
    ax[1].legend(fontsize=8, loc="lower right")
    ax[1].grid(alpha=0.25)
    fig.suptitle(f"RQ3 wm_bud_l0, {len(common)} train instances (paired), top-4 by J = C^pi per N; "
                 "C^pi(true) = critic v2 trên latent DINOv2 THẬT sau khi chạy lại bridge trong sim", fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "f1_exploitation_curve.png", dpi=150)
    plt.close(fig)
    # ------------------------------------------------------------------ figure 2: winner's curse
    sets = [("random (không chọn)", m5[(m5.dir == "verif_m5b") & (m5.arm == "random")], "#7f7f7f")]
    for N in BUD:
        sets.append((f"top-4 @ N={N}", bud[bud.N == N], {8: "#fdae6b", 32: "#fd8d3c", 128: "#e6550d", 512: "#a63603"}[N]))
    sets.append(("wm_diverse (768 imag.)", m5[(m5.dir == "verif_m5b") & (m5.arm == "wm_diverse")], "#d62728"))
    vjm = m5.merge(vj, on=key + ["success"], how="left")
    vj_sets = {"random (không chọn)": vjm[(vjm.dir == "verif_m5b") & (vjm.arm == "random")],
               "wm_diverse (768 imag.)": vjm[(vjm.dir == "verif_m5b") & (vjm.arm == "wm_diverse")]}
    for N in BUD:
        vj_sets[f"top-4 @ N={N}"] = bud[bud.N == N]
    fig, ax = plt.subplots(2, 3, figsize=(13, 6.6), sharex=True)
    bins = np.linspace(-1, 1, 41)
    stats = {}
    for i, (lab, g, col) in enumerate(sets):
        a = ax.flat[i]
        dv = (g.c_wm - g.c_true).to_numpy()
        wts = np.full(len(dv), 1.0 / len(dv))
        a.hist(dv, bins=bins, color=col, alpha=0.85, weights=wts, label="WM v1 (DINOv2)")
        g2 = vj_sets[lab]
        dj = (g2.c_vj_wm - g2.c_vj_true).dropna().to_numpy()
        if len(dj):
            a.hist(dj, bins=bins, histtype="step", color="#5b2c83", lw=1.6, weights=np.full(len(dj), 1.0 / len(dj)),
                   label="V-JEPA 2-AC (cùng ứng viên)")
        c0 = float((np.abs(dv) < 0.05).mean())
        a.text(0.04, 0.235, f"|gap|<0.05: {c0:.0%} (cột giữa bị cắt)", fontsize=7.5)
        a.axvline(0, color="k", lw=0.8)
        a.axvline(dv.mean(), color="#d62728", ls="--", lw=1.4)
        pos = g.c_wm > 0.5
        stats[lab] = {"n": int(len(g)), "mean_gap": float(dv.mean()), "share_gap_gt_0.5": float((dv > 0.5).mean()),
                      "mean_gap_vj": float(dj.mean()) if len(dj) else None, "sr": float(g.success.mean()),
                      "fp_a": float((~g.success[pos]).mean()) if pos.any() else None}
        a.set_title(f"{lab}\nn={len(g)}  gap TB={dv.mean():+.2f}  P(gap>0.5)={(dv > 0.5).mean():.2f}  SR thật={g.success.mean():.2f}",
                    fontsize=9)
        a.set_ylim(0, 0.25)
        a.grid(alpha=0.2)
        if i == 0:
            a.legend(fontsize=7.5, loc="upper left")
    for a in ax[1]:
        a.set_xlabel("C^pi(tưởng tượng) − C^pi(thật)  [>0 = WM lạc quan]")
    for a in ax[:, 0]:
        a.set_ylabel("tỉ lệ ứng viên")
    fig.suptitle("Winner's curse: ứng viên ngẫu nhiên thì lỗi WM ~ đối xứng quanh 0; "
                 "search chọn top-k chính là chọn những ứng viên WM sai theo hướng lạc quan", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(OUT / "f2_winners_curse.png", dpi=150)
    plt.close(fig)
    js = {"online_curve": {k: {str(N): {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
                               for N, v in c.items()} for k, c in curve.items()},
          "offline_wm_v1_picks": {str(N): {k: (list(v) if isinstance(v, tuple) else float(v)) for k, v in o.items()}
                                  for N, o in off.items()},
          "winners_curse": stats, "n_common": len(common)}
    (OUT.parent / "curve.json").write_text(json.dumps(js, indent=1, default=float))
    print(json.dumps(js, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
