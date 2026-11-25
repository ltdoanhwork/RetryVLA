"""V2R visual check of a multi-root bank (user rule; wraps .cursor/skills/retryvla-visual-check/scripts/render_traj.py).

python scripts/v2r/visual_bank.py <name> "<root glob>" [<phys keep jsonl> | -] [--n 8]

Strata over ALL roots of the bank (bridge = segment 0): 2 worst bridge d3, 1 worst seam (bridge -> continuation),
1 most gripper flips (if > 1), then one per task (round-robin), random fill to n. With a keep list only kept
(physically plausible) trajectories are eligible (= the R2 survivors). Renders with stored frames (no replay):
<name>_*.mp4 / _sheet.png / .json into runs/_analysis/visual/recover_<name>/ + index.md (strata, metrics, files).
"""

from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
R = REPO / ".cursor/skills/retryvla-visual-check/scripts/render_traj.py"


def main():
    name, pat, keep_f = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "-")
    n = 8
    out = REPO / f"runs/_analysis/visual/recover_{name}"
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    keep = None
    if keep_f != "-":
        keep = {(str(Path(r["root"]).resolve()), int(r["episode_index"])) for r in map(json.loads, open(REPO / keep_f))
                if r["keep"]}
    rows = []
    for root in sorted(glob.glob(str(REPO / pat))):
        root = Path(root)
        tj = root / "retryvla/trajectories.jsonl"
        if not tj.exists():
            continue
        files = sorted((root / "data").rglob("*.parquet"))
        if not files:
            continue
        df = pd.concat([pd.read_parquet(f, columns=["action", "segment", "episode_index"]) for f in files])
        for t in map(json.loads, open(tj)):
            e = int(t["episode_index"])
            if keep is not None and (str(root.resolve()), e) not in keep:
                continue
            d = df[df.episode_index == e]
            A = np.stack(d.action.values).astype(np.float32)
            S = np.asarray(d.segment.values).reshape(-1)
            H = int((S == 0).sum())
            b = A[:H]
            d3 = float(np.linalg.norm(np.diff(b[:, :6], n=3, axis=0), axis=1).mean()) if H > 3 else 0.0
            seam = float(np.linalg.norm(A[H, :6] - A[H - 1, :6])) if 0 < H < len(A) else 0.0
            fl = int(np.sum(np.diff(np.sign(b[:, 6])) != 0)) if H > 1 else 0
            rows.append({"root": str(root), "ep": e, "task": t.get("task", "?"), "iid": t.get("instance_id"), "H": H,
                         "T": len(A), "d3": d3, "seam": seam, "flips": fl})
    if not rows:
        (out / "index.md").write_text(f"# {name}: no trajectories\n")
        return
    rng = np.random.default_rng(0)
    pick, why = [], {}

    def add(i, w):
        if i not in pick and len(pick) < n:
            pick.append(i)
            why[i] = w
    for i in np.argsort([-r["d3"] for r in rows])[:2]:
        add(int(i), "worst-jerk")
    add(int(np.argmax([r["seam"] for r in rows])), "worst-seam")
    fi = int(np.argmax([r["flips"] for r in rows]))
    if rows[fi]["flips"] > 1:
        add(fi, "most-flips")
    tasks = sorted({r["task"] for r in rows})
    rng.shuffle(tasks)
    for t in tasks:
        c = [i for i, r in enumerate(rows) if r["task"] == t and i not in pick]
        if c:
            add(int(rng.choice(c)), "per-task")
    for i in rng.permutation(len(rows)):
        add(int(i), "random")
    md = [f"# Visual check: {name} ({len(rows)} eligible trajectories, bank `{pat}`, keep list `{keep_f}`)", "",
          f"bank stats: d3 median {np.median([r['d3'] for r in rows]):.4f}, seam median "
          f"{np.median([r['seam'] for r in rows]):.3f}, flips>1 {np.mean([r['flips'] > 1 for r in rows]):.2f}", "",
          "| # | why | task | instance | ep | H / T | bridge d3 | seam | flips | sheet | mp4 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, i in enumerate(pick):
        r = rows[i]
        rel = Path(r["root"]).relative_to(REPO)
        before = set(out.glob("*_sheet.png"))
        subprocess.call([sys.executable, str(R), f"bank:{rel}#{r['ep']}", "--out", str(out)], cwd=REPO)
        new = sorted(set(out.glob("*_sheet.png")) - before)
        sheet = new[0] if new else None
        mp4 = sheet.with_name(sheet.name.replace("_sheet.png", ".mp4")) if sheet else None
        md.append(f"| {k} | {why[i]} | {r['task'][:50]} | {str(r['iid'])[-40:]} | {r['ep']} | {r['H']} / {r['T']} | "
                  f"{r['d3']:.4f} | {r['seam']:.3f} | {r['flips']} | {sheet if sheet else '-'} | {mp4 if mp4 else '-'} |")
    (out / "index.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
