#!/usr/bin/env python
"""Append (or update) a row in experiments/REGISTRY.md from a run's summary.json.

usage: python scripts/register_run.py runs/<run_id> [--note "free text"]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "experiments" / "REGISTRY.md"
HEADER = (
    "| run_id | milestone | tag | status | git | key results | note |\n"
    "|---|---|---|---|---|---|---|\n"
)
KEY_METRICS = (
    "success_rate", "rsr", "rsr_close", "rsr_open", "n_episodes",
    "n_scenes_solved", "verify_rate", "sim_episodes", "n_trajectories", "final_loss",
)


def _fmt_results(res: dict) -> str:
    parts = []
    for k in KEY_METRICS:
        if k in res:
            v = res[k]
            parts.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
    return ", ".join(parts) or "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    s = json.loads((Path(a.run_dir) / "summary.json").read_text())
    row = (
        f"| {s['run_id']} | {s['milestone']} | {s['tag']} | {s['status']} | {s['git_sha']} | "
        f"{_fmt_results(s.get('results', {}))} | {a.note} |\n"
    )
    text = REGISTRY.read_text() if REGISTRY.exists() else "# Run registry\n\n" + HEADER
    lines = [ln for ln in text.splitlines(keepends=True) if not ln.startswith(f"| {s['run_id']} |")]
    text = "".join(lines)
    if HEADER not in text:
        text += "\n" + HEADER
    REGISTRY.write_text(text.rstrip("\n") + "\n" + row)
    print(row.strip())


if __name__ == "__main__":
    main()
