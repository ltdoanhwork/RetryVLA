"""P2: policy-agnostic value critics with the SAME architecture (RecoveryValue) and the SAME WM as C^pi.

python -m retryvla.critic.train_value_baseline configs/exp/b2_train_progress.yaml   # B3 progress P(s, g)
python -m retryvla.critic.train_value_baseline configs/exp/b2_train_success.yaml    # B4 task-success value

vcritic.kind = progress: soft label t/(T-1) on nominal demo frames (real + WM-imagined), benchmark failure states
  (frame 0 of the LIBERO-Recover train instances) labelled vcritic.fail_label, and terminal states of segments that
  finished the task by themselves (prefix success, sim-detected) labelled 1. No continuation outcome of ANY policy.
vcritic.kind = success: binary "task completed from here" pooled over several continuation policies
  (segment logs listed in vcritic.segment_sources with a `policy` name) + expert demo states (label 1); batches are
  drawn with per-source weights vcritic.mix so no single policy dominates.
Both: BCE (soft targets for progress), imagined_frac of each batch uses the WM-imagined version of a state.
Validation (held out: the WM's val failure episodes, 15% of demo episodes) + a common diagnostic on the C^pi
validation set (collect_v1 val instances, pi0 continuation labels) for this critic AND C^pi (reference), so the
table compares all critics on the same held-out states. Checkpoint format = train_critic (WMCriticRanker loads it).
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
import torch.nn.functional as Fn

from retryvla import REPO_ROOT, tracking
from retryvla.config import cli_config, to_dict
from retryvla.critic import value_data as VD
from retryvla.critic.train_critic import load_wm, predict, ranking_lift
from retryvla.critic.value_model import RecoveryValue, auc, expected_calibration_error
from retryvla.utils import setup_logging
from retryvla.wm.data import save_ckpt

log = logging.getLogger("train_value_baseline")


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def build(c, wm, norm, wck, dev):
    S = VD.Samples()
    val_groups = {VD.group_of(i) for i in wck.get("val_instances", [])}
    excl = VD.eval_instances() if c.get("exclude_eval_instances", True) else set()
    if c.get("exclude_instances_file"):
        excl |= {l.strip() for l in open(REPO_ROOT / c.exclude_instances_file) if l.strip()}
    log.info("excluding %d search-evaluation instance ids from every source", len(excl))
    # demos
    if c.get("demo_latents"):
        for zr, qr, zi, qi, y, task, grp, ep in VD.demo_samples(c.demo_latents, wm, norm, dev, c.kind,
                                                                 int(c.get("demo_horizon", 50))):
            S.add(zr, qr, zi, qi, y, task, grp, "demo", grp, int(VD.demo_is_val(ep, float(c.demo_val_frac))))
    # benchmark failure states (frame 0), unique per instance
    seen = set()
    for r in c.get("fail_roots") or []:
        for d in VD.load_segment_root(r, wm, norm, dev, exclude=excl, frame0_only=True):
            if d["iid"] in seen:
                continue
            seen.add(d["iid"])
            g = VD.group_of(d["iid"])
            S.add(d["z0"], d["q0"], None, None, float(c.fail_label), d["task"], g, "fail", d["iid"], int(g in val_groups))
    # segment logs (labels from the policy that continued; for kind=progress only prefix successes + diagnostics)
    for s in c.get("segment_sources") or []:
        kt = set(s["keep_trials"]) if s.get("keep_trials") is not None else None
        for d in VD.load_segment_root(s["root"], wm, norm, dev, keep_trials=kt, exclude=excl):
            g = VD.group_of(d["iid"])
            isv = int(g in val_groups)
            for k in range(len(d["y"])):
                if c.kind == "success":
                    S.add(d["zr"][k], d["qr"][k], d["zi"][k], d["qi"][k], d["y"][k], d["task"], g, s["policy"], d["iid"], isv)
                elif bool(d["prefix"][k]):  # progress: the segment itself finished the task (sim-detected)
                    S.add(d["zr"][k], d["qr"][k], d["zi"][k], d["qi"][k], 1.0, d["task"], g, "prefix", d["iid"], isv)
    return S


def load_diag(c, wm, norm, wck, dev):
    """C^pi validation set: collect_v1 candidates of the WM val instances, pi0 continuation labels (eval only)."""
    val = set(wck.get("val_instances", []))
    zr, qr, zi, qi, y, task, inst, pre = [], [], [], [], [], [], [], []
    for d in VD.load_segment_root(c.get("diag_root", "data/wm/collect_v1"), wm, norm, dev):
        if d["iid"] not in val:
            continue
        zr.append(d["zr"]); qr.append(d["qr"]); zi.append(d["zi"]); qi.append(d["qi"]); y.append(d["y"])
        pre.append(d["prefix"])
        task += [d["task"]] * len(d["y"]); inst += [d["iid"]] * len(d["y"])
    return (torch.cat(zr), torch.cat(qr), torch.cat(zi), torch.cat(qi), torch.cat(y).numpy(), task, inst,
            torch.cat(pre).numpy())


def diag_metrics(model, vocab, diag, norm, dev, topk=4) -> dict:
    zr, qr, zi, qi, y, task, inst, pre = diag
    T = torch.as_tensor([vocab.get(t, 0) for t in task])
    pr = predict(model, norm.z(zr), norm.q(qr), T, dev)
    pi = predict(model, zi.float(), qi, T, dev)
    rl = ranking_lift(pi, y, inst, k=topk)
    npf = ~pre  # candidates whose segment did NOT finish the task itself (the competence-relevant part)
    return {"pi0_auc_real": auc(pr, y), "pi0_auc_imagined": auc(pi, y),
            "pi0_auc_real_noprefix": auc(pr[npf], y[npf]), "pi0_auc_imag_noprefix": auc(pi[npf], y[npf]), "pi0_topk_success": rl["topk_success"],
            "pi0_mean_success": rl["mean_success"], "pi0_topk_hit": rl["topk_hit"], "pi0_random_k_hit": rl["random_k_hit"],
            "_pred_imag": pi}


def load_critic(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = RecoveryValue(dim=ck["dim"], n_patch=ck["n_patch"], n_tasks=max(16, len(ck["vocab"])), width=int(ck["cfg"]["width"]),
                      depth=int(ck["cfg"]["depth"]), dropout=0.0)
    m.load_state_dict(ck["model"])
    return m.to(dev).eval(), ck["vocab"]


def main() -> None:
    setup_logging()
    cfg = cli_config()
    c = cfg.vcritic
    torch.manual_seed(int(c.seed))
    dev = torch.device(cfg.policy.device)
    wm_path = (REPO_ROOT / c.wm_link).read_text().strip()
    wm, norm, wck = load_wm(wm_path, dev)
    t0 = time.time()
    S = build(c, wm, norm, wck, dev)
    srcs = sorted(set(S.src))
    log.info("built %d samples in %.0fs: %s", len(S), time.time() - t0,
             {s: int(sum(1 for x in S.src if x == s)) for s in srcs})
    P, D = S.zr[0].shape
    zr = torch.stack(S.zr).half()
    qr = torch.stack([torch.as_tensor(x).float() for x in S.qr])
    has_i = torch.as_tensor([x is not None for x in S.zi])
    zi = torch.stack([x if x is not None else torch.zeros(P, D, dtype=torch.float16) for x in S.zi])
    qi = torch.stack([x if x is not None else torch.zeros(8) for x in S.qi])
    y = torch.as_tensor(S.y)
    is_val = np.asarray(S.extra, bool)
    vocab = {t: i for i, t in enumerate(sorted(set(S.task)))}
    T = torch.as_tensor([vocab[t] for t in S.task])
    src = np.asarray(S.src)
    mix = {k: float(v) for k, v in dict(c.mix).items()}
    pools = {s: np.nonzero((src == s) & ~is_val)[0] for s in mix if mix[s] > 0}
    pools = {s: v for s, v in pools.items() if len(v)}
    w = np.array([mix[s] for s in pools]); w = w / w.sum()
    names = list(pools)
    log.info("train pools %s, weights %s, val %d", {s: len(v) for s, v in pools.items()}, dict(zip(names, w.round(3))),
             int(is_val.sum()))
    diag = load_diag(c, wm, norm, wck, dev)
    model = RecoveryValue(dim=D, n_patch=P, n_tasks=max(16, len(vocab)), width=int(c.width), depth=int(c.depth),
                          dropout=float(c.dropout)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(c.lr), weight_decay=float(c.weight_decay))
    rng = np.random.default_rng(int(c.seed))
    steps, bs = int(c.steps), int(c.batch_size)

    def batch_inputs(b, imag_frac):
        b = torch.as_tensor(b)
        use = (torch.rand(len(b)) < imag_frac) & has_i[b]
        z = torch.where(use[:, None, None].to(dev), zi[b].to(dev).float(), norm.z(zr[b].to(dev)))
        q = torch.where(use[:, None].to(dev), qi[b].to(dev), norm.q(qr[b].to(dev)))
        return z, q, T[b].to(dev)

    def val_metrics() -> dict:
        model.eval()
        m = {}
        vi = np.nonzero(is_val)[0]
        for s in sorted(set(src[vi])):
            ii = vi[src[vi] == s]
            pr = predict(model, norm.z(zr[ii]), norm.q(qr[ii]), T[ii], dev)
            yy = y[ii].numpy()
            jj = ii[has_i[ii].numpy()]
            pim = predict(model, zi[jj].float(), qi[jj], T[jj], dev) if len(jj) else np.zeros(0)
            if c.kind == "progress" and s == "demo":
                m["demo_spearman_real"] = spearman(pr, yy)
                m["demo_pearson_real"] = float(np.corrcoef(pr, yy)[0, 1])
                m["demo_spearman_imag"] = spearman(pim, y[jj].numpy()) if len(jj) > 2 else float("nan")
                m["demo_mae_real"] = float(np.abs(pr - yy).mean())
            elif yy.min() != yy.max():
                m[f"{s}_auc_real"] = auc(pr, yy > 0.5)
                if len(jj):
                    m[f"{s}_auc_imag"] = auc(pim, y[jj].numpy() > 0.5)
                    m[f"{s}_ece_imag"] = expected_calibration_error(pim, y[jj].numpy())
            m[f"{s}_mean_pred"] = float(pr.mean())
            m[f"{s}_n_val"] = int(len(ii))
        if c.kind == "progress":  # failure states vs held-out demo states (should rank failures low)
            f_ = vi[src[vi] == "fail"]
            d_ = vi[src[vi] == "demo"]
            if len(f_) and len(d_):
                pf = predict(model, norm.z(zr[f_]), norm.q(qr[f_]), T[f_], dev)
                pdm = predict(model, norm.z(zr[d_]), norm.q(qr[d_]), T[d_], dev)
                m["demo_vs_fail_auc"] = auc(np.r_[pdm, pf], np.r_[np.ones(len(pdm)), np.zeros(len(pf))].astype(bool))
        dm = diag_metrics(model, vocab, diag, norm, dev, int(c.topk))
        m.update({k: v for k, v in dm.items() if not k.startswith("_")})
        model.train()
        return m, dm["_pred_imag"]

    with tracking.start(cfg) as run:
        run.summary.update({"n_samples": {s: int((src == s).sum()) for s in srcs},
                            "n_val": {s: int(((src == s) & is_val).sum()) for s in srcs}, "wm_ckpt": wm_path})
        t0 = time.time()
        m = {}
        for step in range(1, steps + 1):
            ks = rng.choice(len(names), size=bs, p=w)
            b = np.array([pools[names[k]][rng.integers(len(pools[names[k]]))] for k in ks])
            z, q, tt = batch_inputs(b, float(c.imagined_frac))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                lg = model.logits(z, q, tt)
            loss = Fn.binary_cross_entropy_with_logits(lg.float(), y[b].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % int(c.log_every) == 0:
                run.log({"vcritic/loss": float(loss)}, step=step)
            if step % int(c.eval_every) == 0 or step == steps:
                m, _ = val_metrics()
                run.log({f"val/{k}": v for k, v in m.items()}, step=step)
                log.info("step %d loss %.4f | %s | %.0fs", step, float(loss),
                         " ".join(f"{k}={v:.3f}" for k, v in m.items() if isinstance(v, float)), time.time() - t0)
        # reference: C^pi on the same diagnostic set + rank agreement with this critic on imagined states
        m, p_new = val_metrics()
        if c.get("reference_critic_link"):
            ref, rvocab = load_critic((REPO_ROOT / c.reference_critic_link).read_text().strip(), dev)
            dr = diag_metrics(ref, rvocab, diag, norm, dev, int(c.topk))
            m.update({f"ref_{k}": v for k, v in dr.items() if not k.startswith("_")})
            m["spearman_vs_ref_imag"] = spearman(p_new, dr["_pred_imag"])
        ck = run.path("critic.pt")
        save_ckpt(ck, model=model.state_dict(), vocab=vocab, cfg=to_dict(c), wm_ckpt=str(wm_path), n_patch=P, dim=D,
                  kind=str(c.kind))
        run.summary.update(m | {"checkpoint": str(ck)})
        log.info("final: %s", {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
        (REPO_ROOT / c.out_link).write_text(str(ck) + "\n")


if __name__ == "__main__":
    main()
