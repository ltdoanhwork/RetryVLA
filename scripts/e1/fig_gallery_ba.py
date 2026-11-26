#!/usr/bin/env python
"""E1 visual check: the SAME 15 gallery cases of WM_OPTIMISM_VISUAL (10 WM-caused FPs, 2 critic FPs, 3 TPs; all in
held-out groups) imagined by WM v1 (before) and by the E1 WMs (after). CPU.

python -m scripts.e1.fig_gallery_ba [--arms A0 A2 A2x3 A4] [--seed 1]

Per row: TRUE end (agent, wrist) | per arm: wrist token-error heatmap ||z_imag - z_true||^2 on the true wrist frame
+ DECODER(imagined wrist latent). Caption: C^pi(imag) per arm (seed mean over seeds 1-3 where available; the
picture uses --seed), C^pi(true), sim outcome. Output: docs/research/figs/e1/f_gallery_ba_{1,2}.jpg + ba.json.
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
from retryvla.critic.train_critic import load_wm  # noqa: E402
from retryvla.search.wm_search import WMCriticRanker  # noqa: E402
from scripts.wm_optimism.train_decoder import TokDecoder  # noqa: E402

OUT = REPO_ROOT / "docs/research/figs/e1"


@torch.no_grad()
def imagine(wm, d, k):
    lv = int(d["lv"][k])
    z0, q0 = d["zt"][k, 0].float()[None], d["qt"][k, 0].float()[None]
    _, _, _, zs, qs = wm.rollout(z0, q0, d["actions"][k].float()[None])
    h = max(lv - 1, 0)
    return zs[h][0], qs[h][0], lv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=["A0", "A2", "A2x3", "A4"])
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    cache = torch.load(REPO_ROOT / "runs/_analysis/wm_optimism/cache/gallery.pt", weights_only=False)
    cases = json.loads((REPO_ROOT / "runs/_analysis/wm_optimism/cases.json").read_text())
    rk = WMCriticRanker((REPO_ROOT / "data/wm/latest_wm.txt").read_text().strip(),
                        (REPO_ROOT / "data/wm/latest_critic_v2.txt").read_text().strip(), "facebook/dinov2-small", 6,
                        torch.device("cpu"), 0.0, 1.0)
    ck = torch.load(REPO_ROOT / "runs/_analysis/wm_optimism/cache/decoder.pt", weights_only=False, map_location="cpu")
    dec = TokDecoder().eval(); dec.load_state_dict(ck["model"])
    wms = {"WMv1": {0: rk.wm}}
    for arm in a.arms:
        for s in (1, 2, 3):
            p = REPO_ROOT / f"runs/e1/wm/{arm}_s{s}/wm.pt"
            if p.exists():
                wms.setdefault(arm, {})[s] = load_wm(str(p), "cpu")[0]
    names = [n for n in wms if wms[n]]

    def crit(z, q, task):
        t = torch.tensor([rk.vocab.get(task, 0)])
        return float(torch.sigmoid(rk.critic.logits(z[None], q[None], t)))

    rows = []
    for c in cases:
        d = cache[c["stem"]]
        k = c["k"]
        r = {"case": c, "img": {}, "c": {}}
        for n in names:
            vals = []
            for s, wm in wms[n].items():
                zi, qi, lv = imagine(wm, d, k)
                vals.append(crit(zi, qi, d["task"]))
                if s == (0 if n == "WMv1" else a.seed) or (s == min(wms[n]) and n not in r["img"]):
                    e = ((zi - d["zt"][k, lv].float()) ** 2).mean(-1).numpy().reshape(2, 6, 6)
                    with torch.no_grad():
                        im = dec(zi[36:].float()[None], torch.tensor([1]))[0].permute(1, 2, 0).numpy()
                    r["img"][n] = (e, im)
            r["c"][n] = float(np.mean(vals))
        r["lv"] = lv
        rows.append(r)
    vmax = float(np.percentile(np.concatenate([r["img"][n][0][1].ravel() for r in rows for n in names]), 97))
    cols = ["TRUE end agent", "TRUE end wrist"] + sum([[f"{n}: lỗi token wrist", f"{n}: DECODER(mơ) wrist"]
                                                       for n in names], [])
    out = []
    for gi, grp in enumerate([rows[:8], rows[8:]], 1):
        fig, axs = plt.subplots(len(grp), len(cols), figsize=(1.3 * len(cols), 1.75 * len(grp) + 0.5), squeeze=False)
        for i, r in enumerate(grp):
            c = r["case"]; d = cache[c["stem"]]; k = c["k"]
            axs[i, 0].imshow(d["a256"][k, 1].numpy()); axs[i, 1].imshow(d["w256"][k, 1].numpy())
            for j, n in enumerate(names):
                e, im = r["img"][n]
                ax = axs[i, 2 + 2 * j]
                w = d["w256"][k, 1].numpy(); h = w.shape[0]
                ax.imshow(w); ax.imshow(e[1], cmap="inferno", alpha=0.55, vmin=0, vmax=vmax, extent=(0, h, h, 0),
                                        interpolation="nearest")
                ax.text(3, 14, f"C mơ {r['c'][n]:.2f}\nerr {e[1].mean():.2f}", fontsize=6.5, va="top",
                        bbox=dict(facecolor="white", alpha=0.8, lw=0, pad=1))
                axs[i, 3 + 2 * j].imshow(im)
            for j in range(len(cols)):
                axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
                if i == 0:
                    axs[i, j].set_title(cols[j], fontsize=6.5)
            col = {"FP_wm": "#c0392b", "FP_critic": "#d35400", "TP": "#1e8449"}[c["kind"]]
            axs[i, 0].text(0, -0.05, f"#{rows.index(r) + 1} {c['kind']} | {c['fam']} | C thật {c['c_true']:.2f} | sim "
                           f"{'SUCCESS' if c['success'] else 'FAIL'} | C mơ: " +
                           ", ".join(f"{n} {r['c'][n]:.2f}" for n in names),
                           transform=axs[i, 0].transAxes, fontsize=7, va="top", color=col, fontweight="bold")
        fig.suptitle(f"E1 before / after trên cùng các ca của WM_OPTIMISM_VISUAL (held-out). Ảnh: seed {a.seed}; số C mơ = "
                     f"TB seed 1-3. Heatmap wrist, thang chung 0-{vmax:.2f}", fontsize=8)
        fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.9 / (1.75 * len(grp) + 0.5), bottom=0.05, wspace=0.04,
                            hspace=0.45)
        p = OUT / f"f_gallery_ba_{gi}.jpg"
        fig.savefig(p, dpi=100, pil_kwargs={"quality": 85}); plt.close(fig)
        out.append(str(p))
    (OUT / "ba.json").write_text(json.dumps([{"kind": r["case"]["kind"], "fam": r["case"]["fam"], "stem": r["case"]["stem"],
                                              "k": r["case"]["k"], "c_true": r["case"]["c_true"],
                                              "success": r["case"]["success"], "c_imag": r["c"],
                                              "err_wrist": {n: float(r["img"][n][0][1].mean()) for n in names}}
                                             for r in rows], indent=1))
    print(out)


if __name__ == "__main__":
    main()
