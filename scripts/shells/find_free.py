#!/usr/bin/env python3
"""Find GPUs RetryVLA can borrow on the Determined cluster.

python scripts/shells/find_free.py            # table of every RUNNING, non-avoided shell
python scripts/shells/find_free.py --json     # machine-readable list of free (shell, gpu) slots

A GPU slot is "free" if: shell not in shells.json:avoid, /pfss visible, GPU memory free >=
min_free_mib, peak util over 5 samples <= max_util_pct, and CPU load/core <= max_load_per_core
(MuJoCo sim workers are CPU-bound). Read-only: never launches or kills anything.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "shells.json").read_text())


def running_shells() -> list[tuple[str, str]]:
    out = subprocess.run(["det", "shell", "list"], capture_output=True, text=True, timeout=60).stdout
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 4 and parts[3] == "RUNNING":
            rows.append((parts[0], parts[2]))
    if not rows:
        raise SystemExit("no RUNNING shells (Determined token expired? run `det auth login`)")
    return rows


def probe(shell_id: str) -> dict:
    try:
        p = subprocess.run(["det", "shell", "open", shell_id, "--", "bash", str(HERE / "probe.sh")],
                           capture_output=True, text=True, timeout=120)
        txt = p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    info: dict = {"gpus": [], "tmux": [], "pfss": False}
    for line in txt.splitlines():
        f = line.split()
        if not f:
            continue
        if f[0] == "HOST" and len(f) >= 3:
            info["host"], info["node"] = f[1], f[2]
        elif f[0] == "GPU" and len(f) >= 6:
            info["gpus"].append({"idx": int(f[1]), "used": int(f[2]), "total": int(f[3]), "util": int(f[4]),
                                 "driver": f[5]})
        elif f[0] == "CPU" and len(f) >= 3:
            info["nproc"], info["load"] = int(f[1]), float(f[2])
        elif f[0] == "TMUX":
            info["tmux"] = f[1:]
        elif f[0] == "PFSS":
            info["pfss"] = f[1] == "ok"
    if "host" not in info:
        info["error"] = (txt.strip().splitlines() or ["no output"])[-1][:120]
    return info


def free_slots(sid: str, desc: str, info: dict, rules: dict, claimed: set) -> list[dict]:
    if info.get("error") or not info.get("pfss"):
        return []
    load_ok = info.get("load", 99) / max(1, info.get("nproc", 1)) <= rules["max_load_per_core"]
    out = []
    for g in info["gpus"]:
        ok = (g["total"] - g["used"] >= rules["min_free_mib"] and g["util"] <= rules["max_util_pct"] and load_ok
              and f"{sid[:8]}:{g['idx']}" not in claimed)
        if ok:
            out.append({"shell": sid, "short": sid[:8], "desc": desc, "gpu": g["idx"], "node": info.get("node"),
                        "free_mib": g["total"] - g["used"], "load": info.get("load"), "nproc": info.get("nproc")})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rules, avoid = CFG["borrow_rules"], set(CFG["avoid"])
    claimed = {k for k in CFG.get("claims", {})}
    shells = [(s, d) for s, d in running_shells() if s[:8] not in avoid]
    with cf.ThreadPoolExecutor(16) as ex:
        infos = dict(zip([s for s, _ in shells], ex.map(probe, [s for s, _ in shells])))
    slots = []
    for sid, desc in shells:
        slots += free_slots(sid, desc, infos[sid], rules, claimed)
    if a.json:
        print(json.dumps(slots, indent=1))
        return
    print(f"{'shell':9} {'desc':20} {'node':8} {'load/nproc':11} gpus (idx:usedMiB/util%)   tmux")
    for sid, desc in shells:
        i = infos[sid]
        if i.get("error"):
            print(f"{sid[:8]:9} {desc[:20]:20} ERROR {i['error']}")
            continue
        g = " ".join(f"{x['idx']}:{x['used']}/{x['util']}%" for x in i["gpus"]) or "no-gpu"
        print(f"{sid[:8]:9} {desc[:20]:20} {str(i.get('node')):8} {i.get('load')}/{i.get('nproc'):<6} {g:26} "
              f"{' '.join(i['tmux'])}{'' if i['pfss'] else '  [NO /pfss]'}")
    print(f"\nFREE slots ({len(slots)}): " + ", ".join(f"{s['short']}:{s['gpu']}({s['desc']})" for s in slots))


if __name__ == "__main__":
    main()
