"""Goal G2-phys (pre-registered PLAN 2026-11-24 "G2-phys / FC / AUX"): physical-plausibility scan + filter of verified
recovery trajectories, and per-step privileged geometry (also the AUX-head targets).

python -m retryvla.search.goal_phys --bank data/recovery/goal_search_onset_cur --audit_dir data/goal_rsi/onset_s7 \
    --seed 7 [--arms wm_diverse random --workers 10]

For every trajectory of the bank (bridge segment 0 + M0 continuation segment 1): restore the entry's exact snapshot
(goal_instances: replay M0's audit prefix), replay ALL recorded actions open-loop, and read sim_geom after every step.
Replay check: success flag must match (else `replay_mismatch`, rejected). Physical metrics (bridge = segment 0 and the
first 20 continuation steps = seam window):
- pen_cm: max contact penetration (cm);
- off_table: any free object off the table / below its top;
- obj_jump_cm: max per-step displacement of the manipulated object while NOT in finger contact (flying / impulses);
- d3: mean || third difference || of the bridge actions (dims 0-5);
- seam: || a_H - a_{H-1} || (dims 0-5) at the bridge -> continuation handoff;
- grip_flips: gripper sign changes in the bridge.
Reject rules (frozen here; thresholds from D0 = 300 train demos):
  pen_cm > 1.0 | off_table | obj_jump_cm > 3.0 | d3 > 2 x median(D0 d3 over 50-step windows) |
  seam > 3 x median(D0 || a_t - a_{t-1} ||) | grip_flips > 1.
Outputs <bank>/phys_<arm>.jsonl (one row per trajectory: root, episode_index, instance_id, task, metrics, reject
reasons, keep) and <bank>/phys/<arm>/<root-id>_<ep>.npz (geom [T + 1, G], success) for AUX targets;
<bank>/phys_summary.json (survivors, per-task balance, thresholds).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from retryvla import REPO_ROOT
from retryvla.envs.vec import SimPool
from retryvla.geom.features import IDX
from retryvla.search.goal_instances import build_goal_instances


def d3(a):
    a = np.asarray(a, np.float32)[:, :6]
    return float(np.linalg.norm(np.diff(a, n=3, axis=0), axis=1).mean()) if len(a) > 3 else 0.0


def demo_thresholds():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    d = LeRobotDataset("local/d0", root=REPO_ROOT / "data/goal_rsi/d0_train")
    hf = d.hf_dataset.with_format(None).select_columns(["action", "episode_index"])[:]
    a, ep = np.asarray(hf["action"], np.float32), np.asarray(hf["episode_index"])
    D3, D1 = [], []
    for e in np.unique(ep):
        x = a[ep == e]
        D1 += list(np.linalg.norm(np.diff(x[:, :6], axis=0), axis=1))
        for s in range(0, len(x) - 50, 25):
            D3.append(d3(x[s: s + 50]))
    return {"d3_med": float(np.median(D3)), "d1_med": float(np.median(D1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--audit_dir", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--arms", nargs="+", default=["wm_diverse", "random"])
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    bank = REPO_ROOT / a.bank
    thr = demo_thresholds()
    TH = {"pen_cm": 1.0, "obj_jump_cm": 3.0, "d3": 2 * thr["d3_med"], "seam": 3 * thr["d1_med"], "grip_flips": 1}
    ents = {json.loads(l)["entry_id"]: json.loads(l) for l in open(REPO_ROOT / a.audit_dir / f"entries_s{a.seed}.jsonl")}
    summ = {"thresholds": TH, "demo": thr, "arms": {}}
    with SimPool(a.workers, 128) as pool:
        for arm in a.arms:
            outp = bank / f"phys_{arm}.jsonl"
            done = set()
            if outp.exists():
                done = {(json.loads(l)["root"], json.loads(l)["episode_index"]) for l in open(outp)}
            (bank / "phys" / arm).mkdir(parents=True, exist_ok=True)
            items = []
            for root in sorted(glob.glob(str(bank / f"s*/p*/{arm}"))):
                rid = root.split(str(bank) + "/")[1].replace("/", "_")
                ds = LeRobotDataset("local/x", root=root)
                hf = ds.hf_dataset.with_format(None).select_columns(["action", "segment", "episode_index"])[:]
                A, S, E = np.asarray(hf["action"], np.float32), np.asarray(hf["segment"]).reshape(-1), np.asarray(hf["episode_index"])
                for l in open(Path(root) / "retryvla/trajectories.jsonl"):
                    t = json.loads(l)
                    if (root, t["episode_index"]) in done:
                        continue
                    m = E == t["episode_index"]
                    items.append((root, rid, t, A[m], S[m]))
            inst_cache = {}
            for b in range(0, len(items), pool.n):
                batch = items[b: b + pool.n]
                need = [it[2]["instance_id"] for it in batch if it[2]["instance_id"] not in inst_cache]
                for iid in dict.fromkeys(need):
                    inst_cache[iid] = build_goal_instances(pool, [dict(ents[iid])], a.audit_dir)[0]
                for w, (root, rid, t, acts, seg) in enumerate(batch):
                    ins = inst_cache[t["instance_id"]]
                    pool.send(w, "phys_scan", ins.bddl, ins.snapshot, acts)
                rows = []
                for w, (root, rid, t, acts, seg) in enumerate(batch):
                    r = pool.recv(w)
                    g, succ = r["geom"], r["success"]
                    H = int((seg == 0).sum())
                    win = slice(0, min(len(g), H + 21))
                    gw = g[win]
                    pen = float(gw[:, IDX["max_pen"]].max() * 100)
                    off = bool(gw[:, IDX["off_table"]].max() > 0.5)
                    pos = gw[:, IDX["obj_x"]: IDX["obj_z"] + 1]
                    free = gw[1:, IDX["finger_obj_contact"]] < 0.5
                    jump = float((np.linalg.norm(np.diff(pos, axis=0), axis=1)[free].max() * 100) if free.any() else 0.0)
                    dd3 = d3(acts[:H])
                    seam = float(np.linalg.norm(acts[H, :6] - acts[H - 1, :6])) if 0 < H < len(acts) else 0.0
                    gf = int(np.sum(np.diff(np.sign(acts[:H, 6])) != 0))
                    mism = bool(succ.any()) is not True
                    rej = [k for k, bad in (("replay_mismatch", mism), ("pen", pen > TH["pen_cm"]), ("off_table", off),
                                            ("obj_jump", jump > TH["obj_jump_cm"]), ("jerk", dd3 > TH["d3"]),
                                            ("seam", seam > TH["seam"]), ("grip_flips", gf > TH["grip_flips"])) if bad]
                    np.savez_compressed(bank / "phys" / arm / f"{rid}_{t['episode_index']}.npz", geom=g, success=succ)
                    rows.append({"root": root, "episode_index": t["episode_index"], "instance_id": t["instance_id"],
                                 "task": t["task"], "H": H, "pen_cm": pen, "off_table": off, "obj_jump_cm": jump,
                                 "d3": dd3, "seam": seam, "grip_flips": gf, "reject": rej, "keep": not rej})
                with open(outp, "a") as f:
                    for r_ in rows:
                        f.write(json.dumps(r_) + "\n")
                print(json.dumps({"arm": arm, "done": b + len(batch), "of": len(items)}), flush=True)
            allr = [json.loads(l) for l in open(outp)]
            by = {}
            for r_ in allr:
                by.setdefault(r_["task"], [0, 0])
                by[r_["task"]][0] += 1
                by[r_["task"]][1] += r_["keep"]
            reasons = {}
            for r_ in allr:
                for k in r_["reject"]:
                    reasons[k] = reasons.get(k, 0) + 1
            summ["arms"][arm] = {"n": len(allr), "kept": sum(r_["keep"] for r_ in allr),
                                 "kept_entries": len({r_["instance_id"] for r_ in allr if r_["keep"]}),
                                 "reject_reasons": reasons, "per_task_total_kept": by}
    (bank / "phys_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
