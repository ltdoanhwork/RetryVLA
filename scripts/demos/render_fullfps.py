"""Full-frame-rate demo videos (every sim step = one frame, 20 Hz = LIBERO control rate, H.264 yuv420p) for judging
smoothness, with per-clip jerk numbers in the caption. CPU only (MuJoCo OSMesa).

python -m scripts.demos.render_fullfps --entry <goal entry id> [--res 256] [--out runs/_analysis/visual/fullfps]

Per entry (LIBERO-Goal, M0 audit seed 7 failure entries, docs/research/RECOVERY_DEMOS.md):
  H  human demo of the same scene: the LIBERO hdf5 demo (= the d0_train episode) actions replayed from its init_state
  N  M0 nominal from a clean start: an audit rollout that SUCCEEDED (same demo with eval seed 8 when it succeeded,
     else another demo of the task with seed 7), logged actions replayed
  L  LRS v2 (data/recovery/lrs_goal_v2): teacher path + hand-off student run, replayed from the entry snapshot
  W  ours: verified WM bridge + M0 continuation (data/recovery/goal_search), replayed from the entry snapshot
Outputs: <entry>/{H_human,N_m0_nominal,L_lrs_v2,W_wm_bridge}.mp4 (one panel, from its own start), sbs_L_vs_W.mp4
(entry-aligned), quad.mp4 (4 panels on the episode clock: L / W show the M0 audit prefix 0..t first), *_sheet.png
(consecutive-frame contact sheets) and stats.json (jerk table).
Jerk: mean ||d3 a[:6]|| over the clip's own actions (prefix excluded), also d2 and per window segment; gripper flips
per 100 steps; EE jerk = mean ||d3 ee_xyz|| in mm (the motion the camera actually sees).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from retryvla import REPO_ROOT  # noqa: E402
from scripts.demos.goal_demo_common import build_entry, entry_meta, wm_success_traj  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "render_traj", REPO_ROOT / ".cursor/skills/retryvla-visual-check/scripts/render_traj.py")
RT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RT)
RT.SEG_COL.update({"human": (30, 150, 60), "M0 nominal": (90, 90, 90), "M0 prefix": (150, 150, 150),
                   "n/a": (200, 40, 40)})
FPS = 20
_wrap0 = RT.wrap
RT.wrap = lambda txt, n, k: [l[:n] for l in txt.split("\n")][:k] if "\n" in txt else _wrap0(txt, n, k)


def jstats(act: np.ndarray, state: np.ndarray, seg: list[str], window: tuple[str, ...]) -> dict:
    a = np.asarray(act, np.float64)
    out = {"T": len(a)}

    def one(x, st):
        if len(x) < 4:
            return {"d3": float("nan"), "d2": float("nan"), "flips": 0, "flips100": float("nan"), "ee_d3_mm": float("nan")}
        d2 = np.diff(x[:, :6], 2, axis=0)
        d3 = np.diff(x[:, :6], 3, axis=0)
        g = np.sign(x[:, 6])
        g = g[g != 0]
        fl = int((np.diff(g) != 0).sum())
        ee = np.diff(np.asarray(st, np.float64)[:, :3] * 1000, 3, axis=0)
        return {"d3": float(np.linalg.norm(d3, axis=1).mean()), "d2": float(np.linalg.norm(d2, axis=1).mean()),
                "flips": fl, "flips100": 100.0 * fl / len(x),
                "ee_d3_mm": float(np.linalg.norm(ee, axis=1).mean()) if len(ee) else float("nan")}
    out["all"] = one(a, state[: len(a) + 1])
    m = np.array([any(s.startswith(w) for w in window) for s in seg[: len(a)]], bool)
    if m.any() and not m.all():
        i0, i1 = int(np.argmax(m)), len(m) - int(np.argmax(m[::-1]))
        out["window"] = {"seg": window[0], "t0": i0, "t1": i1, **one(a[i0:i1], state[i0:i1 + 1])}
    return out


def jline(s: dict) -> str:
    """Two caption lines: whole clip, then the recovery window (bridge / teacher) when there is one."""
    a = s["all"]
    t = f"|d3a| {a['d3']:.3f} flip {a['flips100']:.1f}/100 EE {a['ee_d3_mm']:.2f}mm"
    w = s.get("window")
    t += "\n" + (f"{w['seg']} only: |d3a| {w['d3']:.3f} flip {w['flips100']:.0f}/100" if w else "(no window)")
    return t


def replay_from(sim, bddl, snap, acts, stop=True):
    o = sim.restore_snapshot(bddl, snap)["obs"]
    fr, st, succ = [o["image"]], [o["state"]], None
    for i, a in enumerate(np.asarray(acts, np.float32)):
        r = sim.step(a)
        fr.append(r.obs["image"])
        st.append(r.obs["state"])
        if r.success and succ is None:
            succ = i + 1
            if stop:
                break
    return fr, np.asarray(st, np.float32), succ


def replay_init(sim, bddl, init, acts, wait):
    o = sim.load_init_state(bddl, init, num_steps_wait=wait, seed=0, return_snapshot=False)["obs"]
    fr, st, succ = [o["image"]], [o["state"]], None
    for i, a in enumerate(np.asarray(acts, np.float32)):
        r = sim.step(a)
        fr.append(r.obs["image"])
        st.append(r.obs["state"])
        if r.success and succ is None:
            succ = i + 1
            break
    return fr, np.asarray(st, np.float32), succ


def hdf5_demo(task: str, demo: int):
    import h5py
    sp = json.loads((REPO_ROOT / "data/splits/goal_rsi_v1.json").read_text())["tasks"][task]
    with h5py.File(sp["hdf5"]) as h:
        d = h[f"data/demo_{demo}"]
        init = np.asarray(d.attrs["init_state"]) if "init_state" in d.attrs else np.asarray(d["states"][0])
        return init, np.asarray(d["actions"], np.float32)


def d0_episode(task: str, demo: int) -> int:
    rep = json.loads((REPO_ROOT / "data/goal_rsi/d0_train/replay_report.json").read_text())
    return next(i for i, r in enumerate(rep) if r["task"] == task and r["demo"] == demo)


def m0_nominal(task: str, demo: int):
    """(seed, demo, T) of an audit rollout that succeeded: same demo (seed 8) if possible, else seed-7 median-T."""
    rows = {s: [json.loads(l) for l in open(REPO_ROOT / f"data/goal_rsi/audit_m0_s{s}/rollouts.jsonl")] for s in (7, 8)}
    for s in (7, 8):
        for r in rows[s]:
            if r["task"] == task and r["demo"] == demo and r["success"]:
                return s, demo, r["T"]
    ok = sorted([r for r in rows[7] if r["task"] == task and r["success"]], key=lambda r: r["T"])
    r = ok[len(ok) // 2]
    return 7, r["demo"], r["T"]


def lrs_v2(iid: str):
    root = REPO_ROOT / "data/recovery/lrs_goal_v2"
    for p in sorted(root.glob("a*/l*/rows.jsonl")):
        for l in open(p):
            r = json.loads(l)
            if r["instance_id"] == iid:
                z = np.load(p.parent / "actions" / f"{iid}.npz")
                return r, z
    return None, None


def mk(title, task, sub, frames, state, act, seg, succ, fps=FPS, window=("bridge",), outcome=None, n_prefix=0):
    """Trajectory dict for RT.panel. Stats on the clip's own actions (frames/actions after n_prefix)."""
    s = jstats(act[n_prefix:], state[n_prefix:], seg[n_prefix:], window)
    tr = {"title": title, "task": jline(s), "entry": f"{sub} | {task}", "frames": frames, "state": state,
          "action": act, "seg": seg, "success": succ is not None, "success_step": succ, "info": {}, "jstats": s}
    if outcome:
        tr["outcome"] = outcome
    return tr


def contact_consecutive(tr: dict, path: Path, n: int = 12, rows: int = 2, tile: int = 128) -> None:
    """Two strips of n CONSECUTIVE frames (every sim step): around the largest |d3 a| of the clip and at the start of
    the window / middle of the clip -> to see frame-to-frame motion, not a subsampled overview."""
    a = np.asarray(tr["action"], np.float64)
    T = len(a)
    d3 = np.r_[np.zeros(3), np.linalg.norm(np.diff(a[:, :6], 3, axis=0), axis=1)] if T > 3 else np.zeros(T)
    starts = [max(0, min(int(np.argmax(d3)) - n // 2, len(tr["frames"]) - n))]
    w = tr["jstats"].get("window")
    starts.append(max(0, min((w["t0"] + 5) if w else T // 2, len(tr["frames"]) - n)))
    out = []
    for s0 in starts[:rows]:
        tiles = []
        for i in range(s0, min(s0 + n, len(tr["frames"]))):
            im = cv2.resize(tr["frames"][i], (tile, tile), interpolation=cv2.INTER_AREA)
            si = min(i, T - 1)
            col = RT.seg_color(tr["seg"][si])
            cv2.rectangle(im, (0, 0), (tile - 1, tile - 1), col, 2)
            cv2.rectangle(im, (0, 0), (tile - 1, 12), (255, 255, 255), -1)
            g = "C" if a[si, 6] > 0 else "o"
            cv2.putText(im, f"t{i} g:{g} d3 {d3[si]:.2f}", (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1, cv2.LINE_AA)
            tiles.append(im)
        out.append(np.concatenate(tiles, 1))
    W = max(o.shape[1] for o in out)
    strip, _, _ = RT.plot_strip(tr, W, 120)
    hdr = RT.caption_bar([(tr["title"], (20, 20, 20)), (tr["task"].replace("\n", " | "), (20, 20, 20)),
                          (tr["entry"], (90, 90, 90))], W, 1.0)
    body = [np.concatenate([o, np.full((o.shape[0], W - o.shape[1], 3), 255, np.uint8)], 1) for o in out]
    RT.save_png(path, np.concatenate([hdr] + body + [strip], 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--out", default="runs/_analysis/visual/fullfps")
    a = ap.parse_args()
    from libero.libero import get_libero_path
    from retryvla.envs.libero_env import LiberoSim

    iid = a.entry
    od = REPO_ROOT / a.out / iid
    od.mkdir(parents=True, exist_ok=True)
    sim = LiberoSim(a.res)
    e = entry_meta(iid)
    task, demo, t = e["task"], int(e["demo"]), int(e["t"])
    bddl = os.path.join(get_libero_path("bddl_files"), "libero_goal", f"{task}.bddl")
    lang = json.loads((REPO_ROOT / "data/splits/goal_rsi_v1.json").read_text())["tasks"][task]["language"]
    checks, trs = {}, {}

    # H: human demo (the d0_train episode = hdf5 demo, replayed in our sim from its init_state, like goal_rsi_build)
    init, ha = hdf5_demo(task, demo)
    ep = d0_episode(task, demo)
    import pandas as pd
    files = sorted((REPO_ROOT / "data/goal_rsi/d0_train/data").rglob("*.parquet"))
    d0a = None
    for f in files:
        df = pd.read_parquet(f, columns=["episode_index", "frame_index", "action"])
        df = df[df.episode_index == ep]
        if len(df):
            d0a = np.stack(df.sort_values("frame_index")["action"].to_numpy()).astype(np.float32)
            break
    checks["H_d0_episode"] = ep
    checks["H_d0_actions_match_hdf5"] = bool(d0a is not None and np.allclose(d0a, ha[1:len(d0a) + 1]))
    fr, st, succ = replay_init(sim, bddl, init, ha, wait=0)
    checks["H_replay_success_step"] = succ
    n = len(fr) - 1
    trs["H"] = mk(f"(1) HUMAN demo d{demo} (d0_train ep{ep})", lang, f"teleop, {n} steps", fr, st, ha[:n],
                  ["human"] * n, succ, window=("human",))

    # N: M0 nominal from a clean start
    s, dn, T = m0_nominal(task, demo)
    na = np.load(REPO_ROOT / f"data/goal_rsi/audit_m0_s{s}/actions_s{s}.npz")[f"{task}__demo_{dn}"].astype(np.float32)
    init_n, _ = hdf5_demo(task, dn)
    fr, st, succ = replay_init(sim, bddl, init_n, na[:T], wait=10)
    checks["N"] = {"seed": s, "demo": dn, "logged_T": T, "replay_success_step": succ}
    n = len(fr) - 1
    trs["N"] = mk(f"(2) M0 nominal d{dn} seed {s} (clean start)", lang, f"M0 audit, {n} steps", fr, st, na[:n],
                  ["M0 nominal"] * n, succ, window=("M0 nominal",))

    # entry snapshot + M0 audit prefix (for the episode-clock quad view)
    E = build_entry(sim, iid)
    pre_fr = [o["image"] for o in E["prefix_obs"]]
    pre_st = np.asarray([o["state"] for o in E["prefix_obs"]], np.float32)
    pre_a = E["audit"][:t]

    # L: LRS v2 teacher + hand-off
    r, z = lrs_v2(iid)
    if r is not None and len(z["path"]) and "handoff" in r["stages"][-1]:
        k = len(r["stages"]) - 1
        path = z["path"].astype(np.float32)
        stud = z[f"student_{k}"].astype(np.float32)
        h = r["stages"][-1]["handoff"]
        nt = int(h.get("n_test", 0))
        bounds = np.cumsum([s_["n_actions"] for s_ in r["stages"]])
        la = np.concatenate([path, stud], 0)
        fr, st, succ = replay_from(sim, E["bddl"], E["snap"], la)
        n = len(fr) - 1
        seg = [f"teacher st{int(np.searchsorted(bounds, i, side='right')) + 1}" if i < len(path)
               else ("handoff" if i < len(path) + nt else "continuation") for i in range(n)]
        checks["L"] = {"status": r["status"], "judge_accept": h.get("accept"), "gt_success": h.get("gt_success"),
                       "replay_success_step": succ, "teacher_len": len(path), "n_test": nt}
        trs["L"] = mk(f"(3) LRS v2 teacher {len(r['stages'])}st + handoff", lang, f"entry t{t}", fr, st, la[:n], seg,
                      succ, window=("teacher",),
                      outcome=f"judge {'ACCEPT' if h.get('accept') else 'REJECT'} | SIM "
                              f"{'SUCCESS @ ' + str(succ) if succ else 'FAIL'}")
    else:
        fr, st, _ = replay_from(sim, E["bddl"], E["snap"], np.zeros((0, 7), np.float32))
        why = r["status"] if r else "no record"
        checks["L"] = {"status": why, "n_actions": 0}
        trs["L"] = {"title": "(3) LRS v2: no teacher actions", "task": f"status {why}:\n0 actions executed",
                    "entry": f"entry t{t} | {lang}", "frames": fr * 40,
                    "state": np.repeat(st, 40, 0), "action": np.zeros((40, 7), np.float32), "seg": ["n/a"] * 40,
                    "success": False, "success_step": None, "info": {}, "outcome": f"LRS v2 {why}: nothing to show",
                    "jstats": {"T": 0, "all": {"d3": float("nan"), "d2": float("nan"), "flips": 0,
                                               "flips100": float("nan"), "ee_d3_mm": float("nan")}}}

    # W: ours, WM bridge + M0
    W = wm_success_traj(iid)
    nb = int(W["recovery_len"])
    fr, st, succ = replay_from(sim, E["bddl"], E["snap"], W["action"])
    m = min(len(st) - 1, len(W["state"]))
    checks["W"] = {"state_maxdiff": float(np.abs(st[:m] - W["state"][:m]).max()), "replay_success_step": succ,
                   "logged_success_step": int(W["success_step"]), "cpi": W["pred_value_verif"], "bridge_len": nb}
    n = len(fr) - 1
    seg = ["bridge" if i < nb else "continuation" for i in range(n)]
    trs["W"] = mk(f"(4) ours: WM bridge + M0 (C^pi {W['pred_value_verif']:.2f})", lang, f"entry t{t}", fr, st,
                  W["action"][:n], seg, succ)

    # single clips + sbs
    res = a.res
    names = {"H": "H_human", "N": "N_m0_nominal", "L": "L_lrs_v2", "W": "W_wm_bridge"}
    P = {}
    for key, tr in trs.items():
        P[key] = RT.panel(tr, res)
        RT.write_mp4(od / f"{names[key]}.mp4", P[key] + [P[key][-1]] * FPS, fps=FPS)
        contact_consecutive(tr, od / f"{names[key]}_consec_sheet.png")
        RT.save_png(od / f"{names[key]}_sheet.png", RT.sheet({**tr, "task": tr["task"].replace("\n", " | "),
                                                            "stats": RT.stats(tr["action"], tr["seg"])}, 8))
    sbs = RT.hstack([P["L"], P["W"]])
    RT.write_mp4(od / "sbs_L_vs_W.mp4", sbs + [sbs[-1]] * FPS, fps=FPS)

    # quad on the episode clock: L / W = M0 audit prefix (0..t) then their own part
    def with_prefix(tr):
        n = len(tr["action"]) if tr["seg"][0] != "n/a" else 0
        return {**tr, "frames": pre_fr[:-1] + tr["frames"], "state": np.concatenate([pre_st[:-1], tr["state"]], 0),
                "action": np.concatenate([pre_a, tr["action"]], 0),
                "seg": ["M0 prefix"] * t + list(tr["seg"]),
                "success_step": (t + tr["success_step"]) if tr.get("success_step") else None,
                "outcome": tr.get("outcome") or None, "_n": n}
    quad_tr = [trs["H"], trs["N"], with_prefix(trs["L"]), with_prefix(trs["W"])]
    for q in quad_tr[2:]:
        if not q["outcome"]:
            q.pop("outcome")
    Q = RT.hstack([RT.panel(q, res) for q in quad_tr])
    RT.write_mp4(od / "quad.mp4", Q + [Q[-1]] * FPS, fps=FPS)
    cv2.imwrite(str(od / "quad_frame_t.png"), cv2.cvtColor(Q[min(t + 10, len(Q) - 1)], cv2.COLOR_RGB2BGR))

    out = {"entry": iid, "task": task, "demo": demo, "t": t, "checks": checks,
           "jerk": {k: v["jstats"] for k, v in trs.items()}}
    (od / "stats.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, default=float), flush=True)


if __name__ == "__main__":
    main()
