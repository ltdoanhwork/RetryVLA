"""ONE full video per example, end to end, every sim step (20 Hz, H.264): the original M0 attempt from t = 0 -> the cut
-> our recovery bridge (WM + C^pi, sim-verified) -> M0 continues until BDDL success; next to it, M0 ALONE from the same
cut (its own logged actions, fails). FAILURE ONSET (onset scan: first probe step from which M0 re-rollouts succeed 0 %)
and the CUT are flagged on the frames. Every frame is a deterministic sim replay of logged actions (CPU, OSMesa).

python -m scripts.demos.render_full_episode --entry <entry_id> [--bank <root>#<ep>] [--tag NAME]
python -m scripts.demos.render_full_episode --entry <s8 entry> --policy L.npz:LABEL R.npz:LABEL   # closed-loop clips
Outputs runs/_analysis/visual/fullfps/full_episode/<tag>.mp4 + _sheet.png (consecutive frames at cut / bridge end)
+ .json (jerk per segment: mean ||d3 a[:6]||, flips / 100 steps, EE ||d3 xyz|| mm, seam |da| at the cut / handoff).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from retryvla import REPO_ROOT  # noqa: E402
from scripts.demos.render_fullfps import RT, jstats  # noqa: E402

FPS = 20
COL = {"orig": (110, 110, 110), "bridge": (230, 120, 0), "cont": (30, 90, 200), "alone": (200, 40, 40),
       "distilled": (0, 140, 90), "ok": (30, 160, 60), "fail": (200, 40, 40), "onset": (210, 0, 120),
       "cut": (20, 20, 20)}
RT.SEG_COL.update({"M0 original": COL["orig"], "RECOVERY": COL["bridge"], "M0 continues": COL["cont"],
                   "M0 alone": COL["alone"], "distilled": COL["distilled"]})


def _seg_color(label: str):
    ks = [k for k in RT.SEG_COL if label.startswith(k)]
    return RT.SEG_COL[max(ks, key=len)] if ks else (120, 120, 120)


RT.seg_color = _seg_color


def onset_of(scene: str, seed: int):
    for d in (f"data/goal_rsi/onset_s{seed}",):
        f = REPO_ROOT / d / "scan.jsonl"
        if f.exists():
            for l in open(f):
                r = json.loads(l)
                if r["scene"] == scene:
                    return r.get("t_on"), r.get("t_last"), r.get("p")
    return None, None, None


def load_bank_ep(spec: str):
    root, ep = spec.rsplit("#", 1)
    tr = RT.load_bank(root, int(ep), False, 128)   # stored frames not needed; we only use actions / states
    row = next(r for r in RT.bank_rows(Path(root)) if int(r["episode_index"]) == int(ep))
    nb = int(row["recovery_len"])
    return {"action": tr["action"], "state": tr["state"], "nb": nb, "success_step": row.get("success_step"),
            "cpi": row.get("pred_value", row.get("score")), "src": f"{Path(root).relative_to(REPO_ROOT) if Path(root).is_absolute() else root}#{ep}"}


def build(sim, iid: str):
    """Init state (10 settle steps) -> audit actions[:t] with frames, as goal_instances / eval_goal_entries."""
    import h5py
    from libero.libero import get_libero_path
    e, d = RT.goal_entry(iid)
    sp = json.loads((REPO_ROOT / "data/splits/goal_rsi_v1.json").read_text())["tasks"][e["task"]]
    with h5py.File(sp["hdf5"]) as h:
        dd = h[f"data/demo_{e['demo']}"]
        init = np.asarray(dd.attrs["init_state"]) if "init_state" in dd.attrs else np.asarray(dd["states"][0])
    bddl = os.path.join(get_libero_path("bddl_files"), "libero_goal", f"{e['task']}.bddl")
    audit = RT.audit_actions(e, d)
    o = sim.load_init_state(bddl, init, num_steps_wait=10, seed=0, return_snapshot=False)["obs"]
    fr, st = [o["image"]], [o["state"]]
    for a in audit[: e["t"]]:
        r = sim.step(a)
        fr.append(r.obs["image"])
        st.append(r.obs["state"])
    return {"e": e, "lang": sp["language"], "bddl": bddl, "snap": sim.snapshot(), "audit": audit,
            "pre_fr": fr, "pre_st": np.asarray(st, np.float32)}


def run(sim, B, acts):
    o = sim.restore_snapshot(B["bddl"], B["snap"])["obs"]
    fr, st, succ = [o["image"]], [o["state"]], None
    for i, a in enumerate(np.asarray(acts, np.float32)):
        r = sim.step(a)
        fr.append(r.obs["image"])
        st.append(r.obs["state"])
        if r.success and succ is None:
            succ = i + 1
            break
    return fr, np.asarray(st, np.float32), succ


def seg_stats(act, st, seg) -> dict:
    """jerk etc. per contiguous segment + seam |da| between segments."""
    out, i0 = {}, 0
    for i in range(1, len(seg) + 1):
        if i == len(seg) or seg[i] != seg[i0]:
            s = jstats(act[i0:i], st[i0:i + 1], seg[i0:i], ("__none__",))["all"]
            out[seg[i0]] = {"t0": i0, "t1": i, "T": i - i0, **s}
            if i < len(seg):
                out[seg[i0]]["seam_da_out"] = float(np.linalg.norm(act[i, :6] - act[i - 1, :6]))
            i0 = i
    return out


ABBR = {"M0 original attempt": "orig", "RECOVERY": "bridge", "M0 continues": "cont", "M0 alone": "alone",
        "distilled": "policy"}


def short_line(s: dict) -> str:
    """after-the-cut segments only (the M0 prefix is identical in both panels)."""
    def ab(k):
        return next((v for kk, v in ABBR.items() if k.startswith(kk)), k[:6])
    return " | ".join(f"{ab(k)} d3 {v['d3']:.2f} fl{v['flips100']:.0f}" for k, v in s.items()
                      if v["T"] >= 4 and not k.startswith("M0 original"))


def fit_text(im, txt, y, scale, color=(255, 255, 255)):
    """putText shrunk to fit the frame width (labels must never be cut off)."""
    w = im.shape[1] - 12
    tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    if tw > w:
        scale *= w / tw
    cv2.putText(im, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def panel(title, lang, act, st, fr, seg, events, outcome, ok, res, jl):
    """caption (title / task / jerk) + frame with segment banner on top and event banner at the bottom + strip."""
    T = len(act)
    tr = {"action": act, "state": st, "seg": seg}
    strip, x0, x1 = RT.plot_strip(tr, res, 104)
    sc = res / 256
    out = []
    for i, f in enumerate(fr):
        si = min(i, T - 1)
        lab = seg[si] if i < T else seg[-1]
        col = RT.seg_color(lab)
        bar = RT.caption_bar([(title, (20, 20, 20)), (lang, (60, 60, 60)), (jl[0], (60, 60, 60)),
                              (jl[1], (60, 60, 60))], res, sc)
        im = cv2.resize(f, (res, res), interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(im, (0, 0), (res - 1, res - 1), col, 4)
        cv2.rectangle(im, (0, 0), (res - 1, int(22 * sc)), col, -1)
        fit_text(im, f"{lab}  t={i}", int(16 * sc), 0.45 * sc)
        ev = None
        for (t0, t1, txt, c) in events:
            if t0 <= i < t1:
                ev = (txt, c)
        if i == len(fr) - 1:
            ev = (outcome, COL["ok"] if ok else COL["fail"])
        if ev:
            cv2.rectangle(im, (0, res - int(24 * sc)), (res - 1, res - 1), ev[1], -1)
            fit_text(im, ev[0], res - int(8 * sc), 0.45 * sc)
        s = strip.copy()
        x = int(x0 + (x1 - x0) * min(i, T) / max(T, 1))
        cv2.line(s, (x, 0), (x, s.shape[0] - 1), (220, 0, 0), 2)
        out.append(np.concatenate([bar, im, s], 0))
    return out


def consec_sheet(panels_fr, centers, path, n=10, tile=120):
    rows = []
    for c in centers:
        for p in panels_fr:
            lo = max(0, min(c - n // 2, len(p) - n))
            rows.append(np.concatenate([cv2.resize(p[j], (tile, int(p[j].shape[0] * tile / p[j].shape[1])),
                                                   interpolation=cv2.INTER_AREA) for j in range(lo, lo + n)], 1))
    W = max(r.shape[1] for r in rows)
    rows = [np.concatenate([r, np.full((r.shape[0], W - r.shape[1], 3), 255, np.uint8)], 1) for r in rows]
    RT.save_png(path, np.concatenate(rows, 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    ap.add_argument("--bank", default="", help="<root>#<ep> of the bridge + continuation (default: goal_search best)")
    ap.add_argument("--policy", nargs=2, default=None, help="L.npz:LABEL R.npz:LABEL closed-loop clips instead")
    ap.add_argument("--tag", default="")
    ap.add_argument("--note", default="", help="extra source note in the title")
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--out", default="runs/_analysis/visual/fullfps/full_episode")
    a = ap.parse_args()
    from retryvla.envs.libero_env import LiberoSim
    sim = LiberoSim(a.res)
    B = build(sim, a.entry)
    e, t = B["e"], int(B["e"]["t"])
    t_on, t_last, p = onset_of(e["scene"], int(e["seed"]))
    pre_a = B["audit"][:t]
    pre_seg = ["M0 original attempt"] * t
    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    tag = a.tag or a.entry
    lang = B["lang"]
    rec = {"entry": a.entry, "cut_t": t, "onset_t": t_on, "onset_p": p}

    def events_for(path_follows_m0_after_cut: bool):
        ev = [(t, t + 30, f"CUT at t={t}: snapshot", COL["cut"])]
        if t_on is not None and (t_on < t or path_follows_m0_after_cut):
            ev.append((t_on, t_on + 40, f"FAILURE ONSET t={t_on} (M0 0% from here)", COL["onset"]))
        return ev

    if a.policy:
        sides = []
        for spec in a.policy:
            f, lab = spec.rsplit(":", 1)
            if f == "audit":   # M0's own logged continuation of the failed audit rollout (bit-exact, fails)
                sides.append((lab, B["audit"][t:].astype(np.float32), -1))
                continue
            z = np.load(f)
            sides.append((lab, z["actions"].astype(np.float32), int(z["success_step"])))
        L_lab, L_act, L_s = sides[0]
        R_lab, R_act, R_s = sides[1]
        frL, stL, sL = run(sim, B, L_act)
        frR, stR, sR = run(sim, B, R_act)
        rec.update({"L": L_lab, "R": R_lab, "L_logged_success": L_s, "L_replay_success": sL,
                    "R_logged_success": R_s, "R_replay_success": sR})
        segL = pre_seg + [f"distilled {L_lab}"] * (len(frL) - 1)
        segR = pre_seg + [f"M0 alone ({R_lab})"] * (len(frR) - 1)
        titleL, titleR = f"(a) DISTILLED {L_lab}, closed loop", f"(b) {R_lab} alone from the same cut"
        actL, actR = np.concatenate([pre_a, L_act[: len(frL) - 1]]), np.concatenate([pre_a, R_act[: len(frR) - 1]])
        evL, evR = events_for(False), events_for(True)
    else:
        if a.bank:
            W = load_bank_ep(a.bank)
        else:
            from scripts.demos.goal_demo_common import wm_success_traj
            w = wm_success_traj(a.entry)
            W = {"action": w["action"], "state": w["state"], "nb": int(w["recovery_len"]),
                 "success_step": int(w["success_step"]), "cpi": w["pred_value_verif"],
                 "src": f"goal_search s{w['shard']}/wm_diverse#{w['episode_index']}"}
        frL, stL, sL = run(sim, B, W["action"])
        m = min(len(stL) - 1, len(W["state"]))
        rec.update({"bridge_src": W["src"], "cpi": W["cpi"], "bridge_len": W["nb"],
                    "logged_success_step": W["success_step"], "replay_success_step": sL,
                    "replay_state_maxdiff": float(np.abs(stL[:m] - W["state"][:m]).max())})
        R_act = B["audit"][t:]
        frR, stR, sR = run(sim, B, R_act)
        rec["m0_alone_success_step"] = sR
        nL = len(frL) - 1
        segL = pre_seg + ["RECOVERY BRIDGE (WM + C^pi, sim-verified)" if i < W["nb"] else "M0 continues"
                          for i in range(nL)]
        segR = pre_seg + ["M0 alone (no recovery)"] * (len(frR) - 1)
        titleL = f"(a) OURS: M0 -> WM bridge -> M0  {a.note}".strip()
        titleR = "(b) M0 ALONE from the same cut"
        actL, actR = np.concatenate([pre_a, W["action"][:nL]]), np.concatenate([pre_a, R_act[: len(frR) - 1]])
        evL = events_for(False) + [(t + W["nb"], t + W["nb"] + 25, "HANDOFF: bridge -> M0", COL["cont"])]
        evR = events_for(True)
    stLf = np.concatenate([B["pre_st"][:-1], stL])
    stRf = np.concatenate([B["pre_st"][:-1], stR])
    frLf, frRf = B["pre_fr"][:-1] + frL, B["pre_fr"][:-1] + frR
    sL_ep = (t + sL) if sL else None
    sR_ep = (t + sR) if sR else None
    SL, SR = seg_stats(actL, stLf, segL), seg_stats(actR, stRf, segR)
    rec["jerk_L"], rec["jerk_R"] = SL, SR
    aL = jstats(actL, stLf, segL, ("__none__",))["all"]
    aR = jstats(actR, stRf, segR, ("__none__",))["all"]
    rec["jerk_L_all"], rec["jerk_R_all"] = aL, aR

    def jl(S, A):
        return (f"all |d3a| {A['d3']:.2f} flip {A['flips100']:.1f}/100 EE {A['ee_d3_mm']:.1f}mm", short_line(S))
    PL = panel(titleL, lang, actL, stLf, frLf, segL, evL,
               f"SUCCESS @ t={sL_ep}" if sL_ep else "FAIL (300 steps)", bool(sL_ep), a.res, jl(SL, aL))
    PR = panel(titleR, lang, actR, stRf, frRf, segR, evR,
               f"SUCCESS @ t={sR_ep}" if sR_ep else f"FAIL (t={len(frRf) - 1})", bool(sR_ep), a.res, jl(SR, aR))
    V = RT.hstack([PL, PR])
    RT.write_mp4(out / f"{tag}.mp4", V + [V[-1]] * FPS, fps=FPS)
    centers = [t, t + (rec.get("bridge_len") or 25)]
    consec_sheet([PL, PR], centers, out / f"{tag}_consec_sheet.png")
    rec["mp4"] = str(out / f"{tag}.mp4")
    rec["n_frames"] = len(V) + FPS
    (out / f"{tag}.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(rec, default=float), flush=True)


if __name__ == "__main__":
    main()
