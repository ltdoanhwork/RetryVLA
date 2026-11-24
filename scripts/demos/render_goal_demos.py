"""Render the LIBERO-Goal recovery demos (docs/research/RECOVERY_DEMOS.md) by deterministic sim replay of LOGGED actions.

python -m scripts.demos.render_goal_demos [--entries ...] [--res 256] [--out runs/_analysis/demos/goal]

Per entry (same snapshot for every panel):
  A  M0 alone: audit actions[t:T] (M0's own continuation, logged outcome FAIL)
  B  WM bridge (verified): bridge + M0 continuation from data/recovery/goal_search (actions logged; check: replayed
     8-d state == logged observation.state at every frame, success at the logged step)
  C  WM false positive (optional): bridge from data/wm/goal_verif + M0 continuation re-run with the logged policy seed
     (scripts/demos/goal_fp_rollout.py, GPU) -> actions file fp_<entry>.npz
  L  LRS (2610.05151 re-run, retryvla.baselines.lrs_goal): teacher path (all stages) + the final hand-off student run
     (judge window, then M0 continues; logged), check: success at the logged GT step
Outputs: <entry>/{A,B,C,L}.mp4, <entry>/side_by_side.mp4 + .gif, checks.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from retryvla import REPO_ROOT
from scripts.demos.goal_demo_common import ENTRIES, build_entry, wm_success_traj

COL = {"m0": (90, 90, 90), "bridge": (230, 140, 20), "teacher": (40, 120, 230), "handoff": (150, 60, 200),
       "cont": (40, 90, 170), "ok": (30, 160, 60), "fail": (200, 40, 40), "pre": (120, 120, 120)}


def caption(img: np.ndarray, lines: list[str], color, res: int, banner: str | None = None, bcol=None) -> np.ndarray:
    """RGB frame -> frame with a caption bar on top (and an optional outcome banner at the bottom)."""
    img = cv2.resize(img, (res, res), interpolation=cv2.INTER_NEAREST if img.shape[0] >= res else cv2.INTER_LINEAR)
    lh = 15
    bar = np.full((lh * len(lines) + 6, res, 3), 255, np.uint8)
    cv2.rectangle(bar, (0, 0), (5, bar.shape[0]), color, -1)
    for i, l in enumerate(lines):
        cv2.putText(bar, l, (9, 13 + lh * i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20) if i else color, 1,
                    cv2.LINE_AA)
    img = img.copy()
    cv2.rectangle(img, (0, 0), (res - 1, res - 1), color, 3)
    if banner:
        cv2.rectangle(img, (0, res - 22), (res - 1, res - 1), bcol, -1)
        cv2.putText(img, banner, (6, res - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def play(sim, E, acts: np.ndarray, labels: list[str]) -> dict:
    """Restore the entry snapshot, step the actions; frames = obs before each action + the final obs."""
    sim.restore_snapshot(E["bddl"], E["snap"])
    frames, states, succ_step = [E["obs"]["image"]], [E["obs"]["state"]], None
    for i, a in enumerate(acts):
        r = sim.step(a)
        frames.append(r.obs["image"])
        states.append(r.obs["state"])
        if r.success and succ_step is None:
            succ_step = i + 1
            break   # eval protocol: the episode ends at success
    return {"frames": frames, "states": np.asarray(states), "labels": labels[: len(frames) - 1] + labels[-1:],
            "success_step": succ_step}


def write_mp4(path: Path, frames: list[np.ndarray], fps: int = 20) -> None:
    import imageio.v2 as iio

    with iio.get_writer(str(path), fps=fps, codec="libx264", quality=7, macro_block_size=8) as w:
        for f in frames:
            w.append_data(f)


def write_gif(path: Path, frames: list[np.ndarray], every: int = 3, width: int = 0, fps: int = 10) -> None:
    from PIL import Image

    fr = frames[::every] + [frames[-1]] * 8
    if width:
        h = int(fr[0].shape[0] * width / fr[0].shape[1])
        fr = [cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA) for f in fr]
    ims = [Image.fromarray(f).quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
           for f in fr]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=True)


def panel_frames(run: dict, title: str, sub: str, seg_of, outcome: str, ok: bool, res: int) -> list[np.ndarray]:
    out = []
    n = len(run["frames"])
    for i, f in enumerate(run["frames"]):
        seg, col = seg_of(min(i, n - 2))
        last = i == n - 1
        out.append(caption(f, [title, sub, f"{seg}  step {i}"], col, res,
                           banner=outcome if last else None, bcol=COL["ok"] if ok else COL["fail"]))
    out += [out[-1]] * 20
    return out


def hstack(panels: list[list[np.ndarray]]) -> list[np.ndarray]:
    T = max(len(p) for p in panels)
    H = max(p[0].shape[0] for p in panels)
    out = []
    for t in range(T):
        row = []
        for p in panels:
            f = p[min(t, len(p) - 1)]
            if f.shape[0] < H:
                f = np.concatenate([f, np.full((H - f.shape[0], f.shape[1], 3), 255, np.uint8)], 0)
            row.append(f)
            row.append(np.full((H, 4, 3), 255, np.uint8))
        out.append(np.concatenate(row[:-1], 1))
    return out


def lrs_row(iid: str, root: Path):
    for p in sorted(root.glob("a*/l*/rows.jsonl")):
        for r in map(json.loads, p.read_text().splitlines()):
            if r["instance_id"] == iid:
                return r, np.load(p.parent / "actions" / f"{iid}.npz")
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entries", nargs="*", default=ENTRIES)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--out", default="runs/_analysis/demos/goal")
    ap.add_argument("--lrs_root", default="data/recovery/lrs_goal")
    ap.add_argument("--fp_dir", default="runs/_analysis/demos/goal/fp")
    a = ap.parse_args()
    from retryvla.envs.libero_env import LiberoSim

    out = REPO_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    sim = LiberoSim(a.res)
    checks = {}
    for iid in a.entries:
        E = build_entry(sim, iid)
        e, lang, t = E["entry"], E["lang"], int(E["entry"]["t"])
        short = f"{e['task'][:28]} d{e['demo']} t{t}"
        od = out / iid
        od.mkdir(exist_ok=True)
        ck = {}
        panels = []
        # A: M0 alone (audit continuation)
        A = E["audit"][t:]
        ra = play(sim, E, A, ["M0 continues (no recovery)"] * len(A))
        ck["A_success_step"] = ra["success_step"]
        pa = panel_frames(ra, "(a) M0 alone from the failure entry", short, lambda i: ("M0 continues", COL["m0"]),
                          f"SIM: {'SUCCESS' if ra['success_step'] else 'FAIL'} ({len(A)} steps)", bool(ra["success_step"]),
                          a.res)
        write_mp4(od / "A_m0_alone.mp4", pa)
        panels.append(pa)
        # B: WM bridge verified
        W = wm_success_traj(iid)
        nb = int(W["recovery_len"])
        rb = play(sim, E, W["action"], [])
        n = min(len(rb["states"]) - 1, len(W["state"]))
        ck["B_state_maxdiff"] = float(np.abs(rb["states"][:n] - W["state"][:n]).max())
        ck["B_success_step"], ck["B_logged_success_step"] = rb["success_step"], int(W["success_step"])
        ck["B_verif_match"], ck["B_cpi"] = W["verif_match"], W["pred_value_verif"]
        pb = panel_frames(rb, f"(b) WM bridge + M0 (C^pi {W['pred_value_verif']:.2f})", short,
                          lambda i: ("RECOVERY BRIDGE", COL["bridge"]) if i < nb else ("M0 continues", COL["cont"]),
                          f"SIM: {'SUCCESS @ ' + str(rb['success_step']) if rb['success_step'] else 'FAIL'}",
                          bool(rb["success_step"]), a.res)
        write_mp4(od / "B_wm_bridge.mp4", pb)
        panels.append(pb)
        # C: WM false positive (needs the GPU re-run of the continuation)
        fp = REPO_ROOT / a.fp_dir / f"{iid}.npz"
        if fp.exists():
            z = np.load(fp)
            rc = play(sim, E, z["actions"], [])
            nbr = int(z["n_bridge"])
            ck["C_success_step"], ck["C_rollout_success"] = rc["success_step"], bool(z["success"])
            ck["C_cpi"] = float(z["pred_value"])
            pc = panel_frames(rc, f"(c) WM false positive (C^pi {float(z['pred_value']):.2f})", short,
                              lambda i: ("WM CANDIDATE BRIDGE", COL["fail"]) if i < nbr else ("M0 continues", COL["cont"]),
                              f"SIM: {'SUCCESS' if rc['success_step'] else 'FAIL'} (WM said success)",
                              bool(rc["success_step"]), a.res)
            write_mp4(od / "C_wm_false_positive.mp4", pc)
        # L: LRS teacher + hand-off
        r, z = lrs_row(iid, REPO_ROOT / a.lrs_root)
        if r is not None and r["stages"] and "handoff" in r["stages"][-1]:
            st = r["stages"][-1]
            h = st["handoff"]
            k = len(r["stages"]) - 1
            path = z["path"]
            stud = z[f"student_{k}"]
            bounds = np.cumsum([s_["n_actions"] for s_ in r["stages"]])
            acts = np.concatenate([path, stud], 0)
            rl = play(sim, E, acts, [])
            np_, nt = len(path), int(h["n_test"])
            ck["L_success_step"], ck["L_logged_gt_step"] = rl["success_step"], (np_ + h["gt_step"]) if h["gt_success"] else None
            ck["L_status"], ck["L_judge_accept"], ck["L_gt"] = r["status"], bool(h["accept"]), bool(h["gt_success"])

            def seg(i):
                if i < np_:
                    s_ = int(np.searchsorted(bounds, i, side="right"))
                    return f"LRS TEACHER stage {s_ + 1}", COL["teacher"]
                if i < np_ + nt:
                    return "HANDOFF TEST (LLM judge)", COL["handoff"]
                return "M0 continues", COL["cont"]
            judge = "ACCEPT" if h["accept"] else "REJECT"
            pl = panel_frames(rl, f"(L) LRS teacher ({len(r['stages'])} st) + M0", short, seg,
                              f"judge {judge} | SIM {'SUCCESS' if h['gt_success'] else 'FAIL'}", bool(h["gt_success"]),
                              a.res)
            write_mp4(od / "L_lrs_teacher.mp4", pl)
            panels.append(pl)
        sbs = hstack(panels)
        write_mp4(od / "side_by_side.mp4", sbs)
        write_gif(od / "side_by_side.gif", sbs, every=4, width=min(900, sbs[0].shape[1]), fps=8)
        checks[iid] = ck
        print(iid, json.dumps(ck), flush=True)
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=float))


if __name__ == "__main__":
    main()
