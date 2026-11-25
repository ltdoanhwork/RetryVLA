#!/usr/bin/env python3
"""Register every finished V2R run not yet in experiments/REGISTRY.md (scripts/register_run.py)."""
import glob, json, subprocess, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
reg = (REPO / "experiments/REGISTRY.md").read_text()
for d in sorted(glob.glob(str(REPO / "runs/*_V2R_v2r_*"))):
    rid = Path(d).name
    if "smoke" in rid or rid in reg:
        continue
    try:
        if json.load(open(Path(d) / "summary.json")).get("status") != "done":
            continue
    except Exception:
        continue
    subprocess.call([sys.executable, str(REPO / "scripts/register_run.py"), d, "--note", "V2R (PLAN V2R, 2026-11-24)"])
