"""V2R chain (PLAN "V2R"): runs ON a RetryVLA-owned 2-GPU shell inside tmux `rv_v2r_chain`; several shells share one
queue on /pfss (atomic claim files). Every task is resumable / skip-if-done (marker written last by the task itself).

python scripts/v2r/chain.py            (started by scripts/v2r/chain.sh; one per shell)

Slots per shell: on each GPU at most one `sft` task and one `sim` task (search / eval / replay / D_P), sim workers
11 per task (32 CPUs per shell). Claims: runs/v2r/claims/<task> (host, pid, t); a claim is stale when its host's
heartbeat (runs/v2r/hb_<host>) is > 20 min old or (same host) its pid is gone. Failed tasks are retried <= 2 times.
Stages register themselves: after a group of tasks is done, scripts/v2r/analyze_v2r.py writes
runs/_analysis/v2r/{v2r.md, v2r.json}; the chain appends the table to RESULTS / a line to NOTES and commits + pushes
(explicit paths). Visual-check gate: an SFT of a bank waits for runs/_analysis/visual/recover_<bank>/LOOKED (written
by the agent after looking at the sheets) up to VISUAL_WAIT_H hours after the render, then proceeds and logs the
deviation.
"""

from __future__ import annotations

import glob
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
V = REPO / "runs/v2r"
CL = V / "claims"
LOGD = V / "logs"
HOST = socket.gethostname()
VISUAL_WAIT_H = 0.75
NW = 10
PI1NOM = "runs/20261013-080655_M1_sft_nominal_only/checkpoints/006000/pretrained_model"
R0 = {0: "runs/20261016-042447_M5_m5c_sft_wm_diverse_rr/checkpoints/006000/pretrained_model",
      1: "runs/20261018-232806_M8_seed_pi1_s1/checkpoints/006000/pretrained_model",
      2: "runs/20261019-004004_M8_seed_pi1_s2/checkpoints/006000/pretrained_model"}
R1B = "data/recovery/v2r_r1/s*/p*/wm_diverse"
PHYS1 = "data/recovery/v2r_r1/phys_wm_diverse_hard.jsonl"   # deviation 2026-11-25 03:35 UTC (NOTES)
DPB = "data/recovery/v2r_dp/sh*/bank*"


def log(**kw):
    kw = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": HOST, **kw}
    print(json.dumps(kw), flush=True)
    try:
        with open(V / "chain.jsonl", "a") as f:
            f.write(json.dumps(kw) + "\n")
    except OSError:
        pass


def g(p):
    return sorted(glob.glob(str(REPO / p)))


def run_dir(tag):
    """newest finished run dir of a tag (summary status done)"""
    for d in reversed(g(f"runs/*_V2R_{tag}")):
        try:
            if json.load(open(Path(d) / "summary.json")).get("status") == "done":
                return Path(d)
        except Exception:
            continue
    return None


# ------------------------------------------------------------------ readiness helpers
def r1_search_done():
    """8 shards DONE; a shard taken over by the second shell (CLAIM2) also needs FIN2"""
    for k in range(8):
        d = REPO / f"data/recovery/v2r_r1/s{k}"
        if not (d / "DONE").exists() or ((d / "CLAIM2").exists() and not (d / "FIN2").exists()):
            return False
    return True


def dp_done():
    return len(g("data/recovery/v2r_dp/sh*/CLOSED")) == 2


def n_dp():
    n = 0
    for b in g(DPB):
        if (Path(b) / "CLOSED").exists() and (Path(b) / "retryvla/trajectories.jsonl").exists():
            n += sum(1 for _ in open(Path(b) / "retryvla/trajectories.jsonl"))
    return n


def looked(bank, t_ready):
    m = REPO / f"runs/_analysis/visual/recover_{bank}"
    if (m / "LOOKED").exists():
        return True
    idx = m / "index.md"
    if idx.exists() and time.time() - idx.stat().st_mtime > VISUAL_WAIT_H * 3600:
        f = m / "PROCEEDED_WITHOUT_LOOK"
        if not f.exists():
            f.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
            log(event="visual_gate_timeout", bank=bank)
        return True
    return False


def sft_done(tag):
    d = run_dir(tag)
    return d is not None and (d / "checkpoints/006000/pretrained_model").exists()


def ckpt(tag):
    return str(run_dir(tag) / "checkpoints/006000/pretrained_model")


def eval_done(tag, kind):
    d = run_dir(f"{tag}_{kind}")
    return d is not None and (d / "events_summary.json").exists()


def best_arm():
    """PLAN: highest mean RSR among R1-R3 passing the nominal guard (mean >= 0.776); within 2 pt -> simpler arm."""
    import numpy as np
    sc = {}
    for arm in ("r1", "r2", "r3"):
        rs, ns = [], []
        for s in (0, 1, 2):
            dr, dn = run_dir(f"v2r_{arm}_s{s}_recover_test"), run_dir(f"v2r_{arm}_s{s}_nominal")
            if dr and dn:
                rs.append(json.load(open(dr / "summary.json"))["results"]["rsr"])
                ns.append(json.load(open(dn / "summary.json"))["results"]["success_rate"])
        if rs and np.mean(ns) >= 0.796 - 0.02:
            sc[arm] = float(np.mean(rs))
    if not sc:
        return "r1"
    top = max(sc.values())
    return [a for a in ("r1", "r2", "r3") if a in sc and sc[a] >= top - 0.02][0]


def n2():
    """R2 trajectory count = min(208, physically plausible survivors) (PLAN: if < 208, all survivors)."""
    f = REPO / PHYS1   # D-V1: survivors of the HARD physical rules (strict jerk / seam rules -> ranking)
    return min(208, sum(json.loads(l)["keep"] for l in open(f))) if f.exists() else 208


def arm_groups(arm):
    """train overrides (key=value list) of an arm's recovery mix"""
    N2 = n2()
    n_p = min(n_dp(), 70, N2 // 2)
    g2 = f"{{datasets: ['{R1B}'], n: {N2}, keep_list: ['{PHYS1}'], rank: d3}}"
    if arm in ("r1", "r1m"):
        return [f"train.recovery_datasets=['{R1B}']", f"train.max_recovery_trajectories={208 if arm == 'r1' else N2}",
                "train.subset_mode=roundrobin"]
    if arm == "r2":
        return [f"train.recovery_groups=[{g2}]"]
    if arm == "r3":
        g2b = g2.replace(f"n: {N2}", f"n: {N2 - n_p}")
        return [f"train.recovery_groups=[{g2b}, {{datasets: ['{DPB}'], n: {n_p}, require_closed: true}}]"]
    raise ValueError(arm)


def r3_seeds():
    return (0, 1, 2) if min(n_dp(), 70) >= 20 else (0,)


# ------------------------------------------------------------------ task list
def tasks():
    """(name, kind, ready(), cmd list, done()) ; kinds: sft | sim"""
    T = []
    sh = "scripts/v2r/job.sh"

    def ev(tag, pol, part):
        if part == "nominal":
            return [sh, "eval_nominal", tag, pol]
        return [sh, "eval_recover", tag, pol]

    # R0 nominal re-evals with saved actions (event metrics only) + pi1-nominal
    for s, p in R0.items():
        T.append((f"r0_s{s}_nominal", "sim", lambda: True, ev(f"v2r_r0_s{s}", p, "nominal"),
                  lambda s=s: eval_done(f"v2r_r0_s{s}", "nominal")))
    T.append(("pi1nom_nominal", "sim", lambda: True, ev("v2r_pi1nom", PI1NOM, "nominal"),
              lambda: eval_done("v2r_pi1nom", "nominal")))
    # D_P (2 shards)
    for k in (0, 1):
        T.append((f"dp_sh{k}", "sim", lambda: True, [sh, "dp", str(k)],
                  lambda k=k: (REPO / f"data/recovery/v2r_dp/sh{k}/CLOSED").exists()))
    # phys scans + visual renders
    T.append(("r1_phys", "sim", r1_search_done, [sh, "phys", "data/recovery/v2r_r1", "wm_diverse", "-"],
              lambda: (REPO / "data/recovery/v2r_r1/phys_summary_wm_diverse.json").exists()))
    T.append(("dp_phys", "sim", dp_done, [sh, "phys", "data/recovery/v2r_dp", "dp", "sh*/bank*"],
              lambda: (REPO / "data/recovery/v2r_dp/phys_summary_dp.json").exists()))
    T.append(("vis_r1", "sim", r1_search_done, [sh, "visual", "v2r_r1", R1B, "-"],
              lambda: (REPO / "runs/_analysis/visual/recover_v2r_r1/index.md").exists()))
    T.append(("vis_r2", "sim", lambda: (REPO / "data/recovery/v2r_r1/phys_summary_wm_diverse.json").exists(),
              [sh, "visual", "v2r_r2h", R1B, PHYS1],
              lambda: (REPO / "runs/_analysis/visual/recover_v2r_r2h/index.md").exists()))
    T.append(("vis_dp", "sim", dp_done, [sh, "visual", "v2r_dp", DPB, "-"],
              lambda: (REPO / "runs/_analysis/visual/recover_v2r_dp/index.md").exists()))
    gates = {"r1": lambda: r1_search_done() and looked("v2r_r1", 0),
             "r2": lambda: (REPO / "data/recovery/v2r_r1/phys_summary_wm_diverse.json").exists() and looked("v2r_r2h", 0),
             "r3": lambda: (REPO / "data/recovery/v2r_r1/phys_summary_wm_diverse.json").exists() and dp_done()
             and looked("v2r_dp", 0)}
    if (REPO / "data/recovery/v2r_r1/phys_summary_wm_diverse.json").exists() and n2() < 208:
        gates["r1m"] = gates["r2"]   # PLAN: < 208 survivors -> R1 re-matched to the same count, 1 seed
        tag = "v2r_r1m_s0"
        T.append((f"{tag}_sft", "sft", gates["r1m"], [sh, "sft", tag, "0", "@r1m"], lambda tag=tag: sft_done(tag)))
        for part in ("recover_test", "nominal"):
            T.append((f"{tag}_{part}", "sim", lambda tag=tag: sft_done(tag),
                      [sh, "eval_" + ("recover" if part != "nominal" else "nominal"), tag, "@ckpt:" + tag],
                      lambda tag=tag, part=part: eval_done(tag, part)))
    order = [("r1", 0), ("r1", 1), ("r1", 2), ("r2", 0), ("r2", 1), ("r2", 2), ("r3", 0), ("r3", 1), ("r3", 2)]
    for arm, s in order:
        tag = f"v2r_{arm}_s{s}"
        if arm == "r3" and s not in r3_seeds() and dp_done():
            continue
        T.append((f"{tag}_sft", "sft", gates[arm], [sh, "sft", tag, str(s), "@" + arm], lambda tag=tag: sft_done(tag)))
        for part in ("recover_test", "nominal"):
            T.append((f"{tag}_{part}", "sim", lambda tag=tag: sft_done(tag),
                      [sh, "eval_" + ("recover" if part != "nominal" else "nominal"), tag, "@ckpt:" + tag],
                      lambda tag=tag, part=part: eval_done(tag, part)))

    def r123_done():
        return all(eval_done(f"v2r_{a}_s{s}", p) for a in ("r1", "r2") for s in (0, 1, 2)
                   for p in ("recover_test", "nominal")) and dp_done() and \
            all(eval_done(f"v2r_r3_s{s}", p) for s in r3_seeds() for p in ("recover_test", "nominal"))
    for s in (0, 1, 2):
        tag = f"v2r_r4_s{s}"
        T.append((f"{tag}_sft", "sft", lambda: r123_done() and (REPO / "data/recovery/v2r_dp/phys_summary_dp.json").exists(),
                  [sh, "sft", tag, str(s), "@r4"], lambda tag=tag: sft_done(tag)))
        for part in ("recover_test", "nominal"):
            T.append((f"{tag}_{part}", "sim", lambda tag=tag: sft_done(tag),
                      [sh, "eval_" + ("recover" if part != "nominal" else "nominal"), tag, "@ckpt:" + tag],
                      lambda tag=tag, part=part: eval_done(tag, part)))
    # priority: bank products + visual checks first (they gate the SFTs), R0 / pi1-nominal event re-evals last
    first = ("vis_r1", "r1_phys", "dp_sh0", "dp_sh1", "vis_r2", "vis_dp", "dp_phys")
    T = [t for t in T if t[0] in first] + [t for t in T if t[0] not in first and not t[0].startswith(("r0_", "pi1nom"))] \
        + [t for t in T if t[0].startswith(("r0_", "pi1nom"))]
    return T


STAGES = [("R1", lambda: all(eval_done(f"v2r_r1_s{s}", p) for s in (0, 1, 2) for p in ("recover_test", "nominal"))),
          ("R2", lambda: all(eval_done(f"v2r_r2_s{s}", p) for s in (0, 1, 2) for p in ("recover_test", "nominal"))),
          ("R3", lambda: dp_done() and all(eval_done(f"v2r_r3_s{s}", p) for s in r3_seeds()
                                           for p in ("recover_test", "nominal"))),
          ("R4", lambda: all(eval_done(f"v2r_r4_s{s}", p) for s in (0, 1, 2) for p in ("recover_test", "nominal")))]


def resolve(cmd):
    out = []
    for c in cmd:
        if c.startswith("@ckpt:"):
            out.append(ckpt(c[6:]))
        elif c == "@r4":
            b = best_arm()
            (V / "r4_base.txt").write_text(b)
            out += ["@" + b, "train.aux_beta=0.1",
                    "train.aux_phys_dirs=['data/recovery/v2r_r1/phys/wm_diverse','data/recovery/v2r_dp/phys/dp']"]
        elif c in ("@r1", "@r1m", "@r2", "@r3"):
            out += [c]
        else:
            out.append(c)
    fin = []
    for c in out:
        if c in ("@r1", "@r1m", "@r2", "@r3"):
            fin += arm_groups(c[1:])
        else:
            fin.append(c)
    return fin


TMUX = "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin/tmux"


def external_sim_busy() -> bool:
    """R1 search lanes (rv_v2r_s0..s3, launched before the chain) occupy all sim CPU of this shell while they run."""
    try:
        out = subprocess.run([TMUX, "ls"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return False
    names = [l.split(":")[0] for l in out.splitlines()]
    return sum(n.startswith("rv_v2r_s") and not n.startswith("rv_v2r_smoke") for n in names) >= 2


def claim(name) -> bool:
    CL.mkdir(parents=True, exist_ok=True)
    p = CL / name
    if p.exists():
        try:
            c = json.loads(p.read_text())
        except Exception:
            c = {}
        hb = V / f"hb_{c.get('host')}"
        stale = (not hb.exists()) or time.time() - hb.stat().st_mtime > 1200
        if c.get("host") == HOST and not Path(f"/proc/{c.get('pid')}").exists():
            stale = True
        if not stale:
            return False
        p.unlink(missing_ok=True)
        log(event="stale_claim_removed", task=name, was=c)
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, json.dumps({"host": HOST, "pid": -1, "t": time.time()}).encode())
    os.close(fd)
    return True


def register(stage):
    """registration (analysis -> RESULTS / NOTES / REGISTRY / review -> commit + push) runs on the AGENT box (working
    git credentials): request it via a marker that scripts/v2r/watch.py picks up (scripts/v2r/register_stage.py)."""
    (V / f"register_req_{stage}").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
    log(event="register_requested", stage=stage)


def main():
    V.mkdir(parents=True, exist_ok=True)
    LOGD.mkdir(parents=True, exist_ok=True)
    (V / "fails").mkdir(exist_ok=True)
    procs = {}   # name -> (Popen, gpu, kind)
    log(event="start")
    while True:
        (V / f"hb_{HOST}").write_text(str(time.time()))
        for n, (p, gpu, kind) in list(procs.items()):
            rc = p.poll()
            if rc is None:
                lf_ = LOGD / f"{n}.log"
                if lf_.exists() and time.time() - lf_.stat().st_mtime > 2700:   # 45-min stall rule
                    log(event="stall_kill", task=n)
                    try:
                        os.killpg(p.pid, 15)
                        time.sleep(20)
                        os.killpg(p.pid, 9)
                    except OSError:
                        pass
                continue
            log(event="finished", task=n, rc=rc, gpu=gpu)
            if rc != 0:
                f = V / "fails" / n
                f.write_text(str(int(f.read_text()) + 1 if f.exists() else 1))
            (CL / n).unlink(missing_ok=True)
            del procs[n]
        T = tasks()
        all_done = all(t[4]() for t in T)
        used = {(gpu, kind) for _, gpu, kind in procs.values()}
        kinds = {t[0]: t[1] for t in T}
        for c in CL.glob("*"):   # tasks of this host started by an earlier chain process (still alive) keep their slot
            try:
                cc = json.loads(c.read_text())
            except Exception:
                continue
            if cc.get("host") == HOST and c.name not in procs and Path(f"/proc/{cc.get('pid')}").exists() \
                    and "gpu" in cc:
                used.add((int(cc["gpu"]), kinds.get(c.name, "sim")))
        orphan = set()
        for cf in CL.glob("*") if CL.exists() else []:   # orphans of a previous chain process on this host
            try:
                c = json.loads(cf.read_text())
            except Exception:
                continue
            if c.get("host") == HOST and cf.name not in procs and c.get("pid", -1) > 0 and Path(f"/proc/{c['pid']}").exists():
                used.add((c.get("gpu", 0), c.get("kind", "sim")))
                orphan.add(cf.name)
        if external_sim_busy():
            used |= {(0, "sim"), (1, "sim")}
        for name, kind, ready, cmd, done in T:
            if name in procs or name in orphan or done():
                continue
            fails = int((V / "fails" / name).read_text()) if (V / "fails" / name).exists() else 0
            if fails > 2 or not ready():
                continue
            free = [gp for gp in (0, 1) if (gp, kind) not in used]
            if not free:
                continue
            if not claim(name):
                continue
            gpu = free[0]
            try:
                args = resolve(cmd)
            except Exception as e:   # noqa: BLE001
                log(event="resolve_failed", task=name, err=str(e)[:200])
                (CL / name).unlink(missing_ok=True)
                continue
            lf = open(LOGD / f"{name}.log", "a")
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "MUJOCO_EGL_DEVICE_ID": str(gpu), "TMPDIR": "/tmp",
                   "V2R_NW": str(NW)}
            p = subprocess.Popen(["bash", *args], cwd=REPO, stdout=lf, stderr=subprocess.STDOUT, env=env,
                                 start_new_session=True)
            (CL / name).write_text(json.dumps({"host": HOST, "pid": p.pid, "t": time.time(), "gpu": gpu, "kind": kind}))
            procs[name] = (p, gpu, kind)
            used.add((gpu, kind))
            log(event="launch", task=name, gpu=gpu, kind=kind, args=args[:6])
        for st, ok in STAGES:
            m = V / f"registered_{st}"
            if ok() and not m.exists():
                try:
                    fd = os.open(m, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)
                except FileExistsError:
                    continue
                register(st)
        if all_done and not procs and not orphan:
            log(event="all_done")
            (V / "ALL_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
            return
        idle_f = V / f"idle_{HOST}"
        if not procs and not orphan:
            if not idle_f.exists():
                idle_f.write_text(str(time.time()))
            log(event="idle_waiting")
        else:
            idle_f.unlink(missing_ok=True)
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
