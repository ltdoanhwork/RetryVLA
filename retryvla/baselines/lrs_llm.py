"""Local VLM teacher backend for the LRS re-implementation (arXiv 2610.05151): Qwen3-VL-8B-Instruct via HF transformers.

No API LLM is available in this environment (no OpenAI / Anthropic / Gemini key), so the paper's gpt-6.1-sol teacher is
replaced by the strongest local multimodal model that fits next to SmolVLA on one A100 (documented deviation).
"""

from __future__ import annotations

import json
import re
import time

import numpy as np
import torch
from PIL import Image


class VLMTeacher:
    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct", device: str = "cuda",
                 max_new_tokens: int = 400, temperature: float = 0.7, top_p: float = 0.9):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.max_new_tokens, self.temperature, self.top_p = int(max_new_tokens), float(temperature), float(top_p)
        self.n_calls, self.t_total, self.tok_in, self.tok_out = 0, 0.0, 0, 0

    @torch.no_grad()
    def chat(self, messages: list[dict], seed: int, greedy: bool = False, max_new_tokens: int | None = None) -> str:
        """messages: [{"role", "content": [{"type": "text"|"image", ...}]}]; images as uint8 HxWx3 arrays."""
        msgs = []
        for m in messages:
            content = []
            for c in m["content"]:
                if c["type"] == "image":
                    content.append({"type": "image", "image": Image.fromarray(np.asarray(c["image"], np.uint8))})
                else:
                    content.append(c)
            msgs.append({"role": m["role"], "content": content})
        inputs = self.proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                               return_tensors="pt").to(self.model.device)
        torch.manual_seed(int(seed))
        t0 = time.time()
        mnt = int(max_new_tokens or self.max_new_tokens)
        if greedy:   # LRS-G observer / evaluator (opt-in): deterministic decoding
            out = self.model.generate(**inputs, max_new_tokens=mnt, do_sample=False)
        else:
            out = self.model.generate(**inputs, max_new_tokens=mnt, do_sample=True,
                                      temperature=self.temperature, top_p=self.top_p)
        n_in = int(inputs["input_ids"].shape[1])
        text = self.proc.batch_decode(out[:, n_in:], skip_special_tokens=True)[0]
        self.n_calls += 1
        self.t_total += time.time() - t0
        self.tok_in += n_in
        self.tok_out += int(out.shape[1] - n_in)
        return text

    def stats(self) -> dict:
        return {"llm_calls": self.n_calls, "llm_s": round(self.t_total, 1), "tok_in": self.tok_in,
                "tok_out": self.tok_out}


def parse_commands(text: str) -> list[dict] | None:
    """Last JSON object with a "commands" list in the reply (fenced or bare)."""
    cands = re.findall(r"\{[^{}]*\"commands\"\s*:\s*\[.*?\]\s*\}", text, flags=re.S)
    for c in reversed(cands):
        try:
            obj = json.loads(c)
            if isinstance(obj.get("commands"), list):
                return [x for x in obj["commands"] if isinstance(x, dict)]
        except json.JSONDecodeError:
            continue
    # fallback: outermost braces
    i, j = text.find("{"), text.rfind("}")
    if 0 <= i < j:
        try:
            obj = json.loads(text[i: j + 1])
            if isinstance(obj, dict) and isinstance(obj.get("commands"), list):
                return [x for x in obj["commands"] if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
    return None
