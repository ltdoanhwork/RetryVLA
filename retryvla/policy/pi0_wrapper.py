"""Pi0Wrapper: same predict_chunk interface as SmolVLAWrapper (run_jobs / search / verification work unchanged) for
pi0 checkpoints of retryvla.train.pi0_sft (2610.05151 reverse-direction track)."""

from __future__ import annotations

import torch

from retryvla.policy.smolvla_wrapper import SmolVLAWrapper


class Pi0Wrapper(SmolVLAWrapper):
    def __init__(self, ckpt: str, device: str = "cuda", n_action_steps: int = 5, res: int = 128):
        from lerobot.policies.factory import make_pre_post_processors

        from retryvla.policy import pi0_lora as P

        self.path = ckpt
        self.device = torch.device(device)
        self.policy, cfg, _ = P.load_policy(ckpt, res=res, n_action_steps=n_action_steps, device=device)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=ckpt,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}})
        c = self.policy.config
        self.chunk_size, self.max_action_dim = c.chunk_size, c.max_action_dim
        self.action_dim = c.output_features["action"].shape[0]

    @torch.no_grad()
    def predict_chunk(self, obs, tasks, seeds=None, noise_scale: float = 1.0):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return super().predict_chunk(obs, tasks, seeds=seeds, noise_scale=noise_scale)
