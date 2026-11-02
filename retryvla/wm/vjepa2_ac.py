"""V-JEPA 2-AC (Meta, github.com/facebookresearch/vjepa2, MIT; weights dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt)
as a drop-in latent world model for RetryVLA recovery search.

Pieces (same interface the search / scorers use for retryvla.wm.dynamics.LatentWM + WMCriticRanker):
  VJEncoder      frozen ViT-g/16 (1B, 256 px, tubelet 2; one frame = the image repeated twice, as in the AC
                 training / world_model_wrapper) on the AGENT view only (AC was trained on one exocentric DROID cam).
                 16x16x1408 tokens -> layer_norm -> 2x2 avg-pool -> 8x8 = 64 tokens -> layer_norm  ("pooled latents").
  VJWorldModel   the AC predictor (24 x 1024, ~300M; frame-causal, RoPE) on the 64 pooled tokens. The attention keeps
                 the pretrained grid_size 16 (positions are snapped to the 16-grid: an 8x8 token at 2x stride).
                 Action space: one model step = 5 LIBERO env steps (20 Hz -> 4 Hz = the DROID AC frame rate).
                 DROID action = [d xyz (m), d euler (rad), d gripper closedness] per frame; LIBERO commands are
                 normalized OSC deltas. The action token takes the 5x7 command chunk through a Linear(35 -> 1024)
                 initialised to W_AC @ M (M = physical map: sum over the chunk x OSC scale (0.05 m, 0.5 rad), gripper
                 command -> closedness rate), then fine-tuned. State token: [eef xyz, euler(axis-angle), closedness].
                 The AC predictor does not predict proprioception -> q_head (MLP on pooled prediction, q, chunk) gives
                 q_{t+1} for the critic and the next state token.
  rollout(z0, q0, actions) -> (zH, qH, unc=0, zs, qs) like LatentWM.rollout (z in pooled-token space, q normalized).
  VJRanker       WMCriticRanker look-alike (encode / score / score_real / ensemble_values) with a C^pi critic
                 (retryvla.critic.value_model.RecoveryValue, dim 1408, 64 patches) trained on V-JEPA latents.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from retryvla import REPO_ROOT

VJ_REPO = REPO_ROOT / "third_party/vjepa2"           # git clone --depth 1 (commit in experiments/NOTES.md)
VJ_DIR = REPO_ROOT / "data/foundation/vjepa2"
GRID, TOK, DIM, K = 8, 64, 1408, 5                    # pooled grid, tokens / frame, ViT-g width, env steps / model step
POS_SCALE, ROT_SCALE = 0.05, 0.5                      # robosuite OSC_POSE output_max (m, rad) per unit command


def _import_vjepa():
    if "timm" not in sys.modules:  # only timm.models.layers.drop_path is used (stochastic depth, 0 here)
        try:
            import timm  # noqa: F401
        except ImportError:
            def drop_path(x, drop_prob=0.0, training=False):
                if drop_prob == 0.0 or not training:
                    return x
                keep = 1 - drop_prob
                m = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
                return x * m / keep
            t, tm, tl = (types.ModuleType(n) for n in ("timm", "timm.models", "timm.models.layers"))
            tl.drop_path = drop_path
            t.models, tm.layers = tm, tl
            sys.modules.update({"timm": t, "timm.models": tm, "timm.models.layers": tl})
    if str(VJ_REPO) not in sys.path:
        sys.path.insert(0, str(VJ_REPO))
    from src.models import ac_predictor, vision_transformer  # noqa: E402

    return vision_transformer, ac_predictor


def _clean(sd: dict) -> dict:
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}


def split_checkpoint(src: Path = VJ_DIR / "vjepa2-ac-vitg.pt") -> None:
    """11.7 GB fp32 release checkpoint -> encoder (bf16, ~2 GB) + predictor (fp32, ~1.2 GB) files (atomic)."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    for key, dt in (("encoder", torch.bfloat16), ("predictor", torch.float32)):
        sd = {k: v.to(dt) for k, v in _clean(ck[key]).items()}
        out = VJ_DIR / f"vjepa2_ac_vitg_{key}.pt"
        torch.save(sd, str(out) + ".tmp")
        Path(str(out) + ".tmp").replace(out)


def build_encoder(dev, dtype=torch.bfloat16) -> nn.Module:
    vit, _ = _import_vjepa()
    enc = vit.vit_giant_xformers(patch_size=16, img_size=(256, 256), num_frames=64, tubelet_size=2, use_sdpa=True,
                                 use_SiLU=False, wide_SiLU=True, uniform_power=False, use_rope=True)
    enc.load_state_dict(torch.load(VJ_DIR / "vjepa2_ac_vitg_encoder.pt", map_location="cpu"), strict=False)
    return enc.to(dev, dtype).eval().requires_grad_(False)


class VJEncoder:
    """uint8 [B, 256, 256, 3] (RetryVLA obs convention, already upright) -> [B, 64, 1408] pooled, layer-normed."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)

    def __init__(self, dev, bs: int = 32):
        self.dev, self.bs = torch.device(dev), bs
        self.net = build_encoder(self.dev)

    @torch.no_grad()
    def encode_images(self, img: torch.Tensor | np.ndarray) -> torch.Tensor:
        img = torch.as_tensor(img)
        out = []
        for i in range(0, len(img), self.bs):
            x = img[i: i + self.bs].to(self.dev).permute(0, 3, 1, 2).float().div(255.0)
            if x.shape[-1] != 256:
                x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            x = x[:, :, None].repeat(1, 1, 2, 1, 1)                       # [B, 3, T=2, H, W]
            x = (x - self.mean.to(x)) / self.std.to(x)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                h = self.net(x.to(torch.bfloat16))                         # [B, 256, 1408]
            out.append(pool_tokens(h.float()))
        return torch.cat(out)


def pool_tokens(h: torch.Tensor) -> torch.Tensor:
    """[B, 256, D] encoder tokens -> layer_norm -> 2x2 avg-pool -> [B, 64, D] -> layer_norm."""
    b, n, d = h.shape
    g = int(math.isqrt(n))
    h = F.layer_norm(h, (d,)).transpose(1, 2).reshape(b, d, g, g)
    h = F.avg_pool2d(h, g // GRID).flatten(2).transpose(1, 2)
    return F.layer_norm(h, (d,))


# ---------------------------------------------------------------- action / state maps (LIBERO -> DROID AC format)
def _aa_to_euler(aa: torch.Tensor) -> torch.Tensor:
    """axis-angle [..., 3] -> extrinsic xyz euler [..., 3] (scipy 'xyz' convention, as DROID cartesian_position)."""
    th = aa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = aa / th
    kx, ky, kz = k.unbind(-1)
    c, s = torch.cos(th[..., 0]), torch.sin(th[..., 0])
    C = 1 - c
    r00, r10, r20 = c + kx * kx * C, ky * kx * C + kz * s, kz * kx * C - ky * s
    r21, r22 = kz * ky * C + kx * s, c + kz * kz * C
    return torch.stack([torch.atan2(r21, r22), torch.asin((-r20).clamp(-1, 1)), torch.atan2(r10, r00)], -1)


def droid_state(q: torch.Tensor) -> torch.Tensor:
    """LIBERO state [..., 8] = eef pos, axis-angle, 2 gripper finger qpos -> [..., 7] xyz, euler, closedness."""
    clo = (1 - (q[..., 6] - q[..., 7]) / 0.08).clamp(0, 1)
    return torch.cat([q[..., :3], _aa_to_euler(q[..., 3:6]), clo[..., None]], -1)


def physical_map() -> torch.Tensor:
    """M [7, 35]: 5x7 normalized command chunk -> DROID delta (sum x OSC scale; gripper +1 close = closing rate)."""
    M = torch.zeros(7, K * 7)
    for t in range(K):
        for j in range(3):
            M[j, t * 7 + j] = POS_SCALE
            M[3 + j, t * 7 + 3 + j] = ROT_SCALE
        M[6, t * 7 + 6] = 0.1   # ~ +0.5 closedness per 0.25 s at a constant close command
    return M


class ChunkActionEncoder(nn.Module):
    """Linear(35 -> 1024) = pretrained AC action_encoder (7 -> 1024) composed with the physical map at init."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.lin = nn.Linear(K * 7, base.out_features)
        with torch.no_grad():
            self.lin.weight.copy_(base.weight @ physical_map())
            self.lin.bias.copy_(base.bias)

    def forward(self, a):
        return self.lin(a)


class VJWorldModel(nn.Module):
    k = K
    P = TOK

    def __init__(self, q_mean: torch.Tensor | None = None, q_std: torch.Tensor | None = None, max_frames: int = 11,
                 pretrained: bool = True, activation_checkpointing: bool = False):
        super().__init__()
        _, acp = _import_vjepa()
        pred = acp.vit_ac_predictor(img_size=(256, 256), patch_size=16, num_frames=64, tubelet_size=2, embed_dim=DIM,
                                    use_activation_checkpointing=activation_checkpointing)
        if pretrained:
            pred.load_state_dict(torch.load(VJ_DIR / "vjepa2_ac_vitg_predictor.pt", map_location="cpu"), strict=True)
        # pooled 8x8 tokens: reshape on the 8-grid, attention positions stay on the pretrained 16-grid (snapped)
        pred.grid_height = pred.grid_width = GRID
        _, mm = sys.modules["src.models.ac_predictor"], sys.modules["src.models.utils.modules"]
        pred.attn_mask = mm.build_action_block_causal_attention_mask(max_frames, GRID, GRID, add_tokens=2)
        pred.action_encoder = ChunkActionEncoder(pred.action_encoder)
        self.pred = pred
        self.q_head = nn.Sequential(nn.Linear(DIM + 8 + K * 7, 512), nn.GELU(), nn.Linear(512, 8))
        nn.init.zeros_(self.q_head[-1].weight); nn.init.zeros_(self.q_head[-1].bias)
        self.register_buffer("q_mean", torch.zeros(8) if q_mean is None else q_mean.float())
        self.register_buffer("q_std", torch.ones(8) if q_std is None else q_std.float())

    def _state_tok(self, qn: torch.Tensor) -> torch.Tensor:
        return droid_state(qn * self.q_std + self.q_mean)

    def predict_next(self, zs: torch.Tensor, qs: torch.Tensor, acs: torch.Tensor) -> torch.Tensor:
        """zs [N, T, P, D] context, qs [N, T, 8] (normalized), acs [N, T, 35] -> next-frame tokens for every
        context frame [N, T, P, D] (frame-causal: output t predicts frame t+1), layer-normed."""
        n, T = zs.shape[:2]
        out = self.pred(zs.flatten(1, 2), acs, self._state_tok(qs))
        return F.layer_norm(out.view(n, T, TOK, DIM), (DIM,))

    def next_q(self, z_next: torch.Tensor, q: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return q + self.q_head(torch.cat([z_next.float().mean(-2), q.float(), a.float()], -1)).to(q.dtype)

    def rollout(self, z0: torch.Tensor, q0: torch.Tensor, actions: torch.Tensor, window: int = 11):
        """Open-loop imagination like LatentWM.rollout: z0 [N, P, D], q0 [N, 8] normalized, actions [N, H, 7]."""
        n, H, _ = actions.shape
        ac = actions[:, : H - H % K].reshape(n, -1, K * 7)
        Z, Q = [z0], [q0]
        for t in range(ac.shape[1]):
            lo = max(0, t + 1 - window)
            zt = self.predict_next(torch.stack(Z[lo:], 1), torch.stack(Q[lo:], 1), ac[:, lo: t + 1])[:, -1]
            Z.append(zt)
            Q.append(self.next_q(zt, Q[-1], ac[:, t]))
        return Z[-1], Q[-1], torch.zeros(n, device=z0.device), Z[1:], Q[1:]


def load_vj_wm(path, dev) -> tuple[VJWorldModel, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    wm = VJWorldModel(ck["q_mean"], ck["q_std"], pretrained=False)
    wm.load_state_dict(ck["model"])
    return wm.to(dev).eval(), ck


class _VJNorm:
    """Normalizer stand-in: tokens are already layer-normed (identity); q standardized with the WM's stats."""

    def __init__(self, q_mean, q_std):
        self.q_mean, self.q_std = q_mean, q_std

    def z(self, z):
        return z.float()

    def q(self, q):
        return (q.float() - self.q_mean.to(q.device)) / self.q_std.to(q.device)


class VJRanker:
    """Drop-in for retryvla.search.wm_search.WMCriticRanker (encode / score / score_real / ensemble_values)."""

    def __init__(self, wm_ckpt: str, critic_ckpt: str, dev, lam_unc: float = 0.0, beta_rough: float = 1.0,
                 encoder: VJEncoder | None = None, bs: int = 64):
        from retryvla.critic.value_model import RecoveryValue

        self.dev = torch.device(dev)
        self.wm, wck = load_vj_wm(wm_ckpt, self.dev)
        self.norm = _VJNorm(wck["q_mean"], wck["q_std"])
        ck = torch.load(critic_ckpt, map_location="cpu", weights_only=False)
        c = ck["cfg"]
        self.vocab = ck["vocab"]
        self.critic = RecoveryValue(dim=ck["dim"], n_patch=ck["n_patch"], n_tasks=max(16, len(self.vocab)),
                                    width=int(c["width"]), depth=int(c["depth"]), dropout=0.0).to(self.dev).eval()
        self.critic.load_state_dict(ck["model"])
        self.enc = encoder or VJEncoder(self.dev)
        self.lam, self.beta, self.bs = lam_unc, beta_rough, bs
        self.mc_dropout = 0.1

    @torch.no_grad()
    def encode(self, obs: dict):
        z = self.enc.encode_images(torch.as_tensor(obs["image"])[None])
        return z, self.norm.q(torch.as_tensor(obs["state"], device=self.dev)[None])

    @torch.no_grad()
    def _values(self, zH, qH, tid, critic=None):
        c = critic or self.critic
        t = torch.full((len(zH),), tid, device=self.dev, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
            return torch.sigmoid(c.logits(zH.float(), qH.float(), t)).float()

    @torch.no_grad()
    def score_real(self, zH, qH, task: str) -> np.ndarray:
        return self._values(zH, qH, self.vocab.get(task, 0)).cpu().numpy()

    @torch.no_grad()
    def imagine(self, z0, q0, cands: np.ndarray):
        Z, Q = [], []
        for i in range(0, len(cands), self.bs):
            a = torch.as_tensor(np.ascontiguousarray(cands[i: i + self.bs]), device=self.dev)
            n = len(a)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                zH, qH, _, _, _ = self.wm.rollout(z0.expand(n, -1, -1), q0.expand(n, -1), a)
            Z.append(zH.float()); Q.append(qH.float())
        return torch.cat(Z), torch.cat(Q)

    @torch.no_grad()
    def score(self, z0, q0, task: str, cands: np.ndarray, bs: int = 0, goal=None) -> dict:
        from retryvla.search.cem import roughness

        zH, qH = self.imagine(z0, q0, cands)
        V = self._values(zH, qH, self.vocab.get(task, 0)).cpu().numpy()
        R = roughness(cands)
        n = len(cands)
        return {"value": V, "unc": np.zeros(n), "unc_n": np.zeros(n), "score": V - self.beta * R, "rough": R,
                "zfeat": torch.cat([zH.mean(1), qH], 1).cpu().numpy()}

    @torch.no_grad()
    def ensemble_values(self, z0, q0, task: str, cands: np.ndarray) -> dict:
        """scorers.ensemble_values contract. No WM ensemble (single predictor): u_wmb = 0; u_ens = u_crit = std of
        C^pi over 4 MC-dropout (p 0.1) critic masks on the imagined end state."""
        from retryvla.search.cem import roughness
        from retryvla.search.scorers import _mc_critic

        zH, qH = self.imagine(z0, q0, cands)
        tid = self.vocab.get(task, 0)
        V = self._values(zH, qH, tid)
        with torch.random.fork_rng(devices=[self.dev] if self.dev.type == "cuda" else []):
            torch.manual_seed(12345)
            mc = _mc_critic(self, self.mc_dropout)
            VM = torch.stack([self._values(zH, qH, tid, mc) for _ in range(4)], 1).cpu().numpy()
        n = len(cands)
        V = V.cpu().numpy()
        return {"value": V, "v_members": VM, "u_ens": VM.std(1), "u_wmb": np.zeros(n), "u_crit": VM.std(1),
                "v_wmb_mean": V, "u_wm": np.zeros(n), "rough": roughness(cands),
                "zfeat": torch.cat([zH.mean(1), qH], 1).cpu().numpy()}
