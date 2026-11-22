"""pi0 (lerobot PyTorch port) with openpi-`*_low_mem`-style adapters, for the 2610.05151 reverse-direction track.

Deviation D2 (logged in docs/related_work/COMPARE_2610.05151.md): openpi JAX `pi0_libero_low_mem_finetune` =
LoRA r16 on Gemma-2B + LoRA r32 on the 300M action expert, LLM base frozen, SigLIP + projections trainable
(~468 M of 3.288 B). Here: peft LoRA injected into the same linear layers (attention q/k/v/o + MLP gate/up/down),
r16 (alpha 16) on the PaliGemma language model and r32 (alpha 32) on the expert; SigLIP vision tower, multimodal
projector and the state / action / time projections fully trainable; everything else frozen.

Checkpoint format (small): <dir>/trainable.pt (trainable params only, fp32) + <dir>/meta.json (base path, policy
config overrides, parent checkpoint for continuation) + <dir>/{policy_preprocessor,policy_postprocessor}.json.
Loading = base pi0 -> (parent chain, merged) -> inject LoRA -> load trainable -> merge.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

LLM_RE = r".*paligemma_with_expert\.paligemma\..*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
EXP_RE = r".*paligemma_with_expert\.gemma_expert\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
FULL_TRAIN = ("vision_tower", "multi_modal_projector", "state_proj", "action_in_proj", "action_out_proj",
              "action_time_mlp_in", "action_time_mlp_out")
BASE = "lerobot/pi0_base"


def make_config(res: int = 128, n_action_steps: int = 5, base: str = BASE):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature

    cfg = PreTrainedConfig.from_pretrained(base)
    cfg.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, res, res)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, res, res)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    cfg.n_action_steps = int(n_action_steps)
    cfg.pretrained_path = base
    return cfg


def _inject(model: torch.nn.Module) -> None:
    from peft import LoraConfig, inject_adapter_in_model

    inject_adapter_in_model(LoraConfig(r=16, lora_alpha=16, target_modules=LLM_RE, lora_dropout=0.0), model,
                            adapter_name="llm")
    inject_adapter_in_model(LoraConfig(r=32, lora_alpha=32, target_modules=EXP_RE, lora_dropout=0.0), model,
                            adapter_name="exp")


def set_trainable(policy) -> dict:
    n_tr = n_all = 0
    for name, p in policy.named_parameters():
        tr = ("lora_" in name) or any(f".{k}." in f".{name}" or name.startswith(k) for k in FULL_TRAIN)
        p.requires_grad_(tr)
        n_all += p.numel()
        n_tr += p.numel() * tr
    return {"trainable": n_tr, "total": n_all}


def _merge(model: torch.nn.Module) -> None:
    from peft.tuners.lora import LoraLayer

    for m in model.modules():
        if isinstance(m, LoraLayer):
            m.merge(safe_merge=False)
    # strip adapters: replace each LoRA wrapper by its merged base layer
    for name, m in list(model.named_modules()):
        for cn, c in list(m.named_children()):
            if isinstance(c, LoraLayer):
                setattr(m, cn, c.get_base_layer())


def load_policy(ckpt: str | None, res: int = 128, n_action_steps: int = 5, device: str = "cuda",
                for_training: bool = False):
    """ckpt None -> pi0 base with our features. Returns (policy, cfg). for_training: inject a FRESH LoRA on top of
    the (merged) checkpoint and mark the trainable set."""
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    chain = []
    c = ckpt
    while c:
        chain.append(Path(c))
        c = json.loads((Path(c) / "meta.json").read_text()).get("parent")
    cfg = make_config(res, n_action_steps)
    policy = PI0Policy.from_pretrained(BASE, config=cfg)
    for d in reversed(chain):
        _inject(policy)
        sd = torch.load(d / "trainable.pt", map_location="cpu", weights_only=True)
        missing = policy.load_state_dict(sd, strict=False)
        bad = [k for k in sd if k in missing.unexpected_keys]
        assert not bad, f"unexpected keys in {d}: {bad[:5]}"
        _merge(policy)
    if for_training:
        _inject(policy)
        info = set_trainable(policy)
    else:
        for p in policy.parameters():
            p.requires_grad_(False)
        info = {}
    policy.to(device)
    return policy, cfg, info


def save_ckpt(policy, out: Path, meta: dict, pre=None, post=None, dtype=torch.float32) -> None:
    """dtype bf16 (opt-in) halves trainable.pt; load_policy casts it back into the fp32 parameters."""
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().to(dtype).cpu() for k, v in policy.named_parameters() if v.requires_grad}
    torch.save(sd, out / "trainable.pt.tmp")
    (out / "trainable.pt.tmp").replace(out / "trainable.pt")
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    if pre is not None:
        pre.save_pretrained(out)
    if post is not None:
        post.save_pretrained(out)
