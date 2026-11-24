"""CRIT-V critic variants on the C^pi input (end-state DINOv2 tokens of both cams + proprio + task).

kind (CriticV(kind=...)):
  base         RecoveryValue recipe (CLS / task / q / 72 patch tokens) -> P(pi finishes | s). CONTROL.
  pc_true      + 20 point-cloud tokens (16 local FPS/kNN PointNet tokens + 4 per-class tokens) from the TRUE labelled
               point cloud of the state (sim depth + segmentation). Upper bound: needs privileged 3D at test time.
  pc_pred      + 20 point tokens PREDICTED from the latent (z, q) by a token predictor, trained jointly with an MSE to
               the (frozen) tokens of a trained pc_true critic on the TRUE cloud (like the RGR geometry head, works on
               WM-imagined states).
  cot          + 4 causal chain queries (contact -> object state -> relation -> BDDL predicates), each supervised with
               sim labels; success head = MLP([CLS, embedding of the predicted chain]).
  cot_nochain  same architecture, chain loss weight 0 (capacity control).
  impl         base + imagination-consistency (trunk feature of the imagined end state -> stop-grad trunk feature of
               the TRUE end state of the same candidate) + latent dynamics in the critic's own space (start-state
               feature + action chunk embedding -> stop-grad true end feature) + distributional time-to-success head.
  impl_noalign impl without the imagination-consistency term.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from retryvla.critic.rich import PC_CLASSES, SYM_IDX

# chain groups (order = causal chain). Binary entries use BCE, continuous use smooth-L1 on standardized targets.
CHAIN = [
    ("contact", [("finger_obj_contact", "b"), ("both_finger_contact", "b"), ("grasped", "b"),
                 ("robot_other_contact", "b")]),
    ("object", [("obj_z", "c"), ("obj_upz", "c"), ("off_table", "b"), ("max_pen", "c")]),
    ("relation", [("d_grip_obj", "c"), ("obj_goal_contact", "b"), ("ogxy", "c"), ("ogz", "c")]),
    ("predicates", [("pred0", "p"), ("pred1", "p"), ("pred2", "p"), ("pred_frac", "c")]),
]
CHAIN_FLAT = [k for _, g in CHAIN for k in g]
NC = len(CHAIN_FLAT)
TTS_BINS = (50, 100, 150, 200, 250, 10 ** 6)   # time-to-success bins (env steps incl. the 50-step bridge) + fail


def chain_targets(sym: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """sym [N, S] (end state) -> chain targets [N, NC] float, mask [N, NC] (pred entries of -1 are masked)."""
    s = sym.float()
    cols, mask = [], []
    for k, t in CHAIN_FLAT:
        if k == "ogxy":
            v = torch.linalg.norm(s[:, SYM_IDX["obj_x"]:SYM_IDX["obj_y"] + 1] - s[:, SYM_IDX["goal_x"]:SYM_IDX["goal_y"] + 1], dim=1)
        elif k == "ogz":
            v = s[:, SYM_IDX["obj_z"]] - s[:, SYM_IDX["goal_z"]]
        else:
            v = s[:, SYM_IDX[k]]
        m = (v >= 0) if t == "p" else torch.ones_like(v, dtype=torch.bool)
        if k == "off_table":
            v = (v > 0.5).float()
        cols.append(v.clamp(min=0) if t == "p" else v); mask.append(m)
    return torch.stack(cols, 1), torch.stack(mask, 1)


def tts_class(success: torch.Tensor, success_step: torch.Tensor) -> torch.Tensor:
    edges = torch.as_tensor(TTS_BINS[:-1], device=success_step.device)
    c = torch.bucketize(success_step.clamp_min(0), edges, right=False)
    return torch.where(success.bool(), c, torch.full_like(c, len(TTS_BINS)))


# ---------------------------------------------------------------- point-cloud tokenizer
def fps(x: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    """Farthest point sampling. x [B, N, 3], valid [B, N] -> indices [B, k] (deterministic start: first valid)."""
    B, N, _ = x.shape
    d = torch.where(valid, torch.full((B, N), 1e9, device=x.device), torch.full((B, N), -1.0, device=x.device))
    idx = torch.zeros(B, k, dtype=torch.long, device=x.device)
    cur = valid.float().argmax(1)
    for i in range(k):
        idx[:, i] = cur
        c = x[torch.arange(B, device=x.device), cur][:, None]
        d = torch.minimum(d, torch.where(valid, ((x - c) ** 2).sum(-1), torch.full_like(d, -1.0)))
        cur = d.argmax(1)
    return idx


class PointTokenizer(nn.Module):
    """Labelled point cloud -> 16 local tokens (FPS centres, 32-NN groups, PointNet on [local xyz, class one-hot])
    + 4 per-class global tokens (PointNet max / mean pool). xyz in metres (table frame), label 255 = padding."""

    def __init__(self, width: int = 256, n_centres: int = 16, k: int = 32):
        super().__init__()
        nc = len(PC_CLASSES)
        self.nc, self.M, self.k = nc, n_centres, k
        self.local = nn.Sequential(nn.Linear(3 + nc, 128), nn.GELU(), nn.Linear(128, width))
        self.centre = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, width))
        self.glob = nn.Sequential(nn.Linear(3 + nc, 128), nn.GELU(), nn.Linear(128, width))
        self.cls_emb = nn.Embedding(nc, width)
        self.out = nn.LayerNorm(width)
        self.n_tokens = n_centres + nc

    def forward(self, xyz: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        x = xyz.float() * 5.0                                   # ~[-1, 1] over the workspace
        lab = lab.long()
        valid = lab < self.nc
        oh = Fn.one_hot(lab.clamp(max=self.nc - 1), self.nc).float() * valid[..., None]
        B = x.shape[0]
        ci = fps(x, valid, self.M)                              # [B, M]
        c = torch.gather(x, 1, ci[..., None].expand(-1, -1, 3))  # [B, M, 3]
        d = ((x[:, None] - c[:, :, None]) ** 2).sum(-1)         # [B, M, N]
        d = d.masked_fill(~valid[:, None], 1e9)
        nn_i = d.topk(self.k, dim=-1, largest=False).indices    # [B, M, k]
        ar = torch.arange(B, device=x.device)[:, None, None]
        loc = torch.cat([x[ar, nn_i] - c[:, :, None], oh[ar, nn_i]], -1)
        okn = valid[ar, nn_i]
        f = self.local(loc).masked_fill(~okn[..., None], -1e4).amax(2)
        f = torch.where(okn.any(-1, keepdim=True), f, torch.zeros_like(f))
        tok_l = f + self.centre(c)
        g = self.glob(torch.cat([x, oh], -1))                   # [B, N, W]
        toks = []
        for k in range(self.nc):
            m = (lab == k)
            gm = g.masked_fill(~m[..., None], -1e4).amax(1)
            gm = torch.where(m.any(1, keepdim=True), gm, torch.zeros_like(gm))
            toks.append(gm + self.cls_emb.weight[k])
        return self.out(torch.cat([tok_l, torch.stack(toks, 1)], 1))


class TokenPredictor(nn.Module):
    """(z, q) normalized latent -> n point tokens (for imagined states; pc_pred)."""

    def __init__(self, dim: int, n_patch: int, n_tokens: int, width: int = 256, depth: int = 2, heads: int = 4):
        super().__init__()
        self.inp = nn.Linear(dim, width)
        self.q = nn.Linear(8, width)
        self.queries = nn.Parameter(torch.randn(1, n_tokens, width) * 0.02)
        self.pos = nn.Parameter(torch.zeros(1, n_patch + 1 + n_tokens, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.1, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, depth)
        self.out = nn.LayerNorm(width)
        self.n = n_tokens

    def forward(self, z, q):
        B = z.shape[0]
        x = torch.cat([self.inp(z), self.q(q)[:, None], self.queries.expand(B, -1, -1)], 1) + self.pos
        return self.out(self.enc(x)[:, -self.n:])


# ---------------------------------------------------------------- critic
class CriticV(nn.Module):
    def __init__(self, kind: str = "base", dim: int = 384, n_patch: int = 72, n_tasks: int = 16, width: int = 256,
                 depth: int = 3, heads: int = 4, dropout: float = 0.1, action_dim: int = 7, horizon: int = 50):
        super().__init__()
        self.kind, self.width = kind, width
        self.in_proj = nn.Linear(dim, width)
        self.q_proj = nn.Linear(8, width)
        self.task = nn.Embedding(n_tasks, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        n_extra = 0
        if kind in ("pc_true", "pc_pred"):
            self.pc = PointTokenizer(width)
            n_extra = self.pc.n_tokens
            self.pc_type = nn.Parameter(torch.zeros(1, 1, width))
            if kind == "pc_pred":
                self.pc_pred = TokenPredictor(dim, n_patch, n_extra, width)
        self.n_chain = len(CHAIN) if kind.startswith("cot") else 0
        if self.n_chain:
            self.chain_q = nn.Parameter(torch.randn(1, self.n_chain, width) * 0.02)
            self.chain_heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, len(g)))
                                              for _, g in CHAIN])
            self.chain_emb = nn.Sequential(nn.Linear(NC, width), nn.GELU(), nn.Linear(width, width))
            self.register_buffer("c_mean", torch.zeros(NC))
            self.register_buffer("c_std", torch.ones(NC))
        self.n_extra = n_extra
        self.pos = nn.Parameter(torch.zeros(1, n_patch + 3 + n_extra + self.n_chain, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.trunk = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width if not self.n_chain else 2 * width, 1)
        if kind.startswith("impl"):
            self.a_emb = nn.Sequential(nn.Linear(action_dim * horizon, width), nn.GELU(), nn.Linear(width, width))
            self.dyn = nn.Sequential(nn.Linear(2 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
            self.align = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
            self.tts = nn.Linear(width, len(TTS_BINS) + 1)
        P = n_patch + 3 + n_extra
        if self.n_chain:  # causal chain: chain query k sees all state tokens and chain queries < k
            L = P + self.n_chain
            m = torch.zeros(L, L, dtype=torch.bool)
            for k in range(self.n_chain):
                m[P + k, P + k + 1:] = True
            m[:P, P:] = True                                    # state tokens do not see the chain queries
            self.register_buffer("attn_mask", m)
        else:
            self.attn_mask = None

    def point_tokens(self, z, q, xyz=None, lab=None):
        if self.kind == "pc_true":
            return self.pc(xyz, lab)
        return self.pc_pred(z, q)

    def encode(self, z, q, t, xyz=None, lab=None, ptok=None):
        """-> dict(feat = CLS feature [N, W], chain = raw chain outputs [N, NC] or None, ptok)."""
        n = z.shape[0]
        parts = [self.cls.expand(n, -1, -1), self.task(t)[:, None], self.q_proj(q)[:, None], self.in_proj(z)]
        if self.n_extra:
            if ptok is None:
                ptok = self.point_tokens(z, q, xyz, lab)
            parts.append(ptok + self.pc_type)
        if self.n_chain:
            parts.append(self.chain_q.expand(n, -1, -1))
        x = torch.cat(parts, 1) + self.pos
        y = self.trunk(x, mask=self.attn_mask)
        feat = self.norm(y[:, 0])
        chain = None
        if self.n_chain:
            chain = torch.cat([h(y[:, -self.n_chain + k]) for k, h in enumerate(self.chain_heads)], 1)
        return {"feat": feat, "chain": chain, "ptok": ptok}

    def chain_values(self, chain_raw):
        """raw chain outputs -> standardized values fed back to the success head (binary: probabilities)."""
        v = chain_raw.clone()
        b = torch.as_tensor([t != "c" for _, t in CHAIN_FLAT], device=v.device)
        v[:, b] = torch.sigmoid(chain_raw[:, b])
        return v

    def logits_from(self, e):
        if self.n_chain:
            return self.head(torch.cat([e["feat"], self.chain_emb(self.chain_values(e["chain"]))], 1)).squeeze(-1)
        return self.head(e["feat"]).squeeze(-1)

    def logits(self, z, q, t, xyz=None, lab=None):
        return self.logits_from(self.encode(z, q, t, xyz, lab))


def chain_loss(model: CriticV, raw: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    ls = []
    for i, (_, typ) in enumerate(CHAIN_FLAT):
        m = mask[:, i]
        if not m.any():
            continue
        if typ == "c":
            y = (tgt[:, i] - model.c_mean[i]) / model.c_std[i]
            ls.append(Fn.smooth_l1_loss(raw[m, i], y[m], beta=0.5))
        else:
            ls.append(Fn.binary_cross_entropy_with_logits(raw[m, i], tgt[m, i].clamp(0, 1)))
    return torch.stack(ls).mean()
