"""CRIT-V: train every critic variant (retryvla.critic.variants) on one suite's training bank and predict on its
held-out verified bank, on TRUE and WM-IMAGINED end states.

python -m retryvla.critic.train_variants --suite goal [--kinds base pc_true ...] [--seeds 0 1 2] [--steps 8000]

Suites (train split only):
  goal     train data/wm/goal_collect (+ data/critic3d/goal_collect), eval data/wm/goal_verif; WM = goal_latest_wm
  spatial  train collect_v1 + verif_r1 (C^pi v2 data; M5 round-2 held-out instances excluded from verif_r1),
           eval verif_m5b; WM = WM v1 (latest_wm)
Recipe = C^pi (m2_train_critic): width 256, depth 3, dropout 0.1, AdamW lr 2e-4 wd 0.05, bs 128, 8000 steps,
50 % imagined end states per batch, pos_weight = neg / pos (clamped 1..20). Same seeds / batches for every kind.
References (no training): the shipped C^pi of the suite and (spatial) the B2 generic success critic.
Output: runs/<ts>_CRITV_<suite>/{preds.pt, ckpt/<kind>_s<seed>.pt, log.json}; preds = per eval candidate:
instance, source, success, and for each (kind, seed, view in {true, imag}) the predicted probability.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from retryvla import REPO_ROOT
from retryvla.critic.train_critic import load_wm
from retryvla.critic.variants import CHAIN_FLAT, CriticV, chain_loss, chain_targets, tts_class
from retryvla.utils import setup_logging
from retryvla.wm.data import task_vocab

log = logging.getLogger("train_variants")

SUITES = {
    "goal": {"train": ["goal_collect"], "eval": "goal_verif", "wm": "data/wm/goal_latest_wm.txt",
             "refs": {"cpi_shipped": "data/wm/goal_latest_critic.txt"}, "exclude": {}},
    "spatial": {"train": ["collect_v1", "verif_r1"], "eval": "verif_m5b", "wm": "data/wm/latest_wm.txt",
                "refs": {"cpi_shipped": "data/wm/latest_critic_v2.txt", "success_generic": "data/wm/latest_critic_success.txt"},
                "exclude": {"verif_r1": "data/splits/m5_r2_heldout_instances.txt"}},
}
KINDS = ["base", "pc_true", "pc_pred", "cot", "cot_nochain", "impl", "impl_noalign"]


@torch.no_grad()
def load_bank(name: str, wm, norm, dev, exclude: str | None = None) -> dict:
    """Latents (data/wm/<name>) + rich records (data/critic3d/<name>) -> normalized z0 / real end / imagined end
    (fp16, CPU), q's, end sym [N, S], end point cloud, actions, labels."""
    ex = set()
    if exclude:
        ex = {l.strip() for l in open(REPO_ROOT / exclude) if l.strip()}
    files = sorted(glob.glob(str(REPO_ROOT / "data/wm" / name / "*.pt")))
    out = {k: [] for k in ("z0", "q0", "zr", "qr", "zi", "qi", "sym", "sym0", "pc", "pcl", "act", "y", "ss", "inst",
                           "task", "src")}
    miss = 0
    for f in files:
        rf = REPO_ROOT / "data/critic3d" / name / Path(f).name
        if not rf.exists():
            miss += 1
            continue
        d = torch.load(f, weights_only=False)
        if d["instance_id"] in ex or d.get("slimmed"):
            continue
        r = torch.load(rf, weights_only=False)
        K = len(r["success"])
        z, q, v = d["z"][:K], d["q"][:K].float(), d["valid"][:K].bool()
        lv = v.long().cumsum(1).argmax(1)
        ar = torch.arange(K)
        zn0, qn0 = norm.z(z[:, 0].to(dev)), norm.q(q[:, 0].to(dev))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            _, _, _, zs, qs = wm.rollout(zn0, qn0, d["actions"][:K, :50].float().to(dev))
        Zs, Qs = torch.stack(zs, 1).float(), torch.stack(qs, 1).float()
        hi = (lv - 1).clamp_min(0).to(dev)
        ard = ar.to(dev)
        out["z0"].append(zn0.half().cpu()); out["q0"].append(qn0.float().cpu())
        out["zr"].append(norm.z(z[ar, lv].to(dev)).half().cpu()); out["qr"].append(norm.q(q[ar, lv].to(dev)).cpu())
        out["zi"].append(Zs[ard, hi].half().cpu()); out["qi"].append(Qs[ard, hi].cpu())
        out["sym"].append(r["sym"][:, -1].float()); out["sym0"].append(r["sym"][:, 0].float())
        out["pc"].append(r["pc"][:, -1]); out["pcl"].append(r["pcl"][:, -1])
        out["act"].append(d["actions"][:K, :50].float())
        out["y"].append(r["success"].float()); out["ss"].append(r["success_step"].long())
        out["inst"] += [d["instance_id"]] * K; out["task"] += [d["task"]] * K; out["src"] += list(r["source"])
    log.info("bank %s: %d candidates from %d files (%d without rich record)", name, sum(len(x) for x in out["y"]),
             len(out["y"]), miss)
    return {k: (torch.cat(v) if k not in ("inst", "task", "src") else v) for k, v in out.items()}


def cat_banks(bs: list[dict]) -> dict:
    return {k: (torch.cat([b[k] for b in bs]) if torch.is_tensor(bs[0][k]) else sum([b[k] for b in bs], []))
            for k in bs[0]}


def make_model(kind, D, P, n_tasks, dev, c_mean=None, c_std=None):
    m = CriticV(kind, dim=D, n_patch=P, n_tasks=n_tasks).to(dev)
    if m.n_chain and c_mean is not None:
        m.c_mean.copy_(c_mean); m.c_std.copy_(c_std)
    return m


def train_one(kind, seed, tr, T, dev, steps, bs, teacher=None, lam_chain=1.0, lam_aux=1.0, lam_tok=1.0):
    torch.manual_seed(seed)
    tr = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in tr.items()}   # whole bank on the GPU (~1 GB)
    T = T.to(dev)
    D, P = tr["zr"].shape[-1], tr["zr"].shape[1]
    ct, cm = chain_targets(tr["sym"].cpu())
    mu = torch.stack([ct[cm[:, i], i].mean() if cm[:, i].any() else torch.tensor(0.) for i in range(ct.shape[1])])
    sd = torch.stack([ct[cm[:, i], i].std().clamp_min(1e-3) if cm[:, i].sum() > 1 else torch.tensor(1.)
                      for i in range(ct.shape[1])])
    model = make_model(kind, D, P, max(16, int(T.max()) + 1), dev, mu.to(dev), sd.to(dev))
    ct, cm = ct.to(dev), cm.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.05)
    y = tr["y"]
    pos_w = torch.tensor((1 - y.mean()) / y.mean().clamp_min(1e-3)).clamp(1, 20).to(dev)
    tts = tts_class(tr["y"], tr["ss"])
    pos_w = pos_w.to(dev)
    rng = np.random.default_rng(seed)           # same batches / view masks for every kind at a given seed
    N = len(y)
    lam_c = 0.0 if kind == "cot_nochain" else lam_chain
    t0 = time.time()
    hist = []
    for step in range(1, steps + 1):
        b = torch.as_tensor(rng.choice(N, bs), device=dev)
        im = torch.as_tensor(rng.random(bs) < 0.5, device=dev)
        z = torch.where(im[:, None, None], tr["zi"][b], tr["zr"][b]).to(dev).float()
        q = torch.where(im[:, None], tr["qi"][b], tr["qr"][b]).to(dev)
        t = T[b].to(dev)
        xyz, lab = tr["pc"][b].to(dev).float(), tr["pcl"][b].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            e = model.encode(z, q, t, xyz, lab)
            lg = model.logits_from(e).float()
            loss = Fn.binary_cross_entropy_with_logits(lg, y[b].to(dev), pos_weight=pos_w)
            aux = {}
            if model.n_chain and lam_c > 0:
                aux["chain"] = lam_c * chain_loss(model, e["chain"].float(), ct[b].to(dev), cm[b].to(dev))
            if kind == "pc_pred" and teacher is not None:
                with torch.no_grad():
                    tt = teacher.pc(xyz, lab).float()
                aux["tok"] = lam_tok * Fn.mse_loss(e["ptok"].float(), tt)
            if kind.startswith("impl"):
                zr, qr = tr["zr"][b].to(dev).float(), tr["qr"][b].to(dev)
                zi, qi = tr["zi"][b].to(dev).float(), tr["qi"][b].to(dev)
                with torch.no_grad():
                    f_true = model.encode(zr, qr, t)["feat"].float()
                fs = model.encode(tr["z0"][b].to(dev).float(), tr["q0"][b].to(dev), t)["feat"]
                pred_end = model.dyn(torch.cat([fs, model.a_emb(tr["act"][b].to(dev).flatten(1))], 1))
                aux["dyn"] = lam_aux * (1 - Fn.cosine_similarity(pred_end.float(), f_true, dim=-1)).mean()
                aux["tts"] = lam_aux * Fn.cross_entropy(model.tts(e["feat"]).float(), tts[b].to(dev))
                if kind == "impl":
                    f_im = model.encode(zi, qi, t)["feat"]
                    aux["align"] = lam_aux * (1 - Fn.cosine_similarity(model.align(f_im).float(), f_true, dim=-1)).mean()
            tot = loss + sum(aux.values()) if aux else loss
        opt.zero_grad(set_to_none=True)
        tot.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 1000 == 0 or step == steps:
            hist.append({"step": step, "bce": float(loss), **{k: float(v) for k, v in aux.items()}})
            log.info("%s s%d step %d bce %.3f %s | %.0fs", kind, seed, step, float(loss),
                     " ".join(f"{k} {float(v):.3f}" for k, v in aux.items()), time.time() - t0)
    return model.eval(), hist


@torch.no_grad()
def predict(model, ev, T, dev, view, bs=512):
    z, q = (ev["zr"], ev["qr"]) if view == "true" else (ev["zi"], ev["qi"])
    out, chains = [], []
    for i in range(0, len(z), bs):
        sl = slice(i, i + bs)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            e = model.encode(z[sl].to(dev).float(), q[sl].to(dev), T[sl].to(dev), ev["pc"][sl].to(dev).float(),
                             ev["pcl"][sl].to(dev))
            out.append(torch.sigmoid(model.logits_from(e).float()).cpu())
            if model.n_chain:
                chains.append(model.chain_values(e["chain"].float()).cpu())
    return torch.cat(out).numpy(), (torch.cat(chains).numpy() if chains else None)


@torch.no_grad()
def predict_ref(path, ev, vocab_ev, dev, view, bs=512):
    from retryvla.critic.value_model import RecoveryValue

    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = RecoveryValue(dim=ck["dim"], n_patch=ck["n_patch"], n_tasks=max(16, len(ck["vocab"])),
                      width=int(ck["cfg"]["width"]), depth=int(ck["cfg"]["depth"]), dropout=0.0).to(dev).eval()
    m.load_state_dict(ck["model"])
    t = torch.as_tensor([ck["vocab"].get(x, 0) for x in ev["task"]])
    z, q = (ev["zr"], ev["qr"]) if view == "true" else (ev["zi"], ev["qi"])
    out = []
    for i in range(0, len(z), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out.append(torch.sigmoid(m.logits(z[i:i + bs].to(dev).float(), q[i:i + bs].to(dev),
                                              t[i:i + bs].to(dev)).float()).cpu())
    return torch.cat(out).numpy()


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=list(SUITES), required=True)
    ap.add_argument("--kinds", nargs="*", default=KINDS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda")
    S = SUITES[a.suite]
    wm, norm, _ = load_wm((REPO_ROOT / S["wm"]).read_text().strip(), dev)
    tr = cat_banks([load_bank(n, wm, norm, dev, S["exclude"].get(n)) for n in S["train"]])
    ev = load_bank(S["eval"], wm, norm, dev)
    del wm
    vocab = task_vocab(tr["task"] + ev["task"])
    Ttr = torch.as_tensor([vocab[x] for x in tr["task"]])
    Tev = torch.as_tensor([vocab[x] for x in ev["task"]])
    run = REPO_ROOT / (a.out or f"runs/{time.strftime('%Y%m%d-%H%M%S')}_CRITV_{a.suite}")
    (run / "ckpt").mkdir(parents=True, exist_ok=True)
    preds = {"inst": ev["inst"], "src": ev["src"], "task": ev["task"], "y": ev["y"].numpy(), "ss": ev["ss"].numpy(),
             "sym": ev["sym"].numpy(), "p": {}, "chain": {}, "train_n": len(tr["y"]), "train_pos": float(tr["y"].mean())}
    for name, link in S["refs"].items():
        ck = (REPO_ROOT / link).read_text().strip()
        for view in ("true", "imag"):
            preds["p"][(name, 0, view)] = predict_ref(ck, ev, vocab, dev, view)
        log.info("ref %s: %s", name, ck)
    hists = {}
    for seed in a.seeds:
        teacher = None
        for kind in a.kinds:
            m, h = train_one(kind, seed, tr, Ttr, dev, a.steps, a.bs, teacher=teacher)
            hists[f"{kind}_s{seed}"] = h
            if kind == "pc_true":
                teacher = m
            for view in ("true", "imag"):
                p, ch = predict(m, ev, Tev, dev, view)
                preds["p"][(kind, seed, view)] = p
                if ch is not None:
                    preds["chain"][(kind, seed, view)] = ch
            torch.save({"kind": kind, "seed": seed, "model": m.state_dict(), "vocab": vocab, "dim": tr["zr"].shape[-1],
                        "n_patch": tr["zr"].shape[1], "wm": (REPO_ROOT / S["wm"]).read_text().strip(),
                        "chain_keys": CHAIN_FLAT}, run / "ckpt" / f"{kind}_s{seed}.pt")
            torch.save(preds, run / "preds.pt")
            json.dump(hists, open(run / "log.json", "w"), indent=1)
    log.info("done -> %s", run)


if __name__ == "__main__":
    main()
