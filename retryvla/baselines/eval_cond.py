"""B3.2: evaluate an advantage-CONDITIONED policy (retryvla.baselines.sft_rise mode=cond) with the standard protocol.

python -m retryvla.baselines.eval_cond recover <config.yaml> [overrides]   # == retryvla.eval.eval_recover
python -m retryvla.baselines.eval_cond nominal <config.yaml> [overrides]   # == retryvla.eval.eval_nominal

Identical to the standard evaluators except that the positive-advantage suffix stored in
<policy.path>/rise_condition.json is appended to every instruction given to the policy (test-time conditioning on
"positive", as in RISE / RECAP). The suffix is also written into the run config (key rise_cond_suffix).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from retryvla import REPO_ROOT


def main() -> None:
    target = sys.argv.pop(1)
    assert target in ("recover", "nominal"), target
    path = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("policy.path=")), None)
    assert path, "policy.path=<ckpt> is required"
    p = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    sfx = json.loads((p / "rise_condition.json").read_text())["positive_suffix"]
    sys.argv.append(f"rise_cond_suffix={json.dumps(sfx)}")

    from retryvla.policy import smolvla_wrapper as sw

    orig = sw.SmolVLAWrapper._batch

    def _batch(self, obs, tasks):
        return orig(self, obs, [t + sfx for t in tasks])

    sw.SmolVLAWrapper._batch = _batch  # covers predict_chunk / sample_chunks / chunks_from_noise
    if target == "recover":
        from retryvla.eval.eval_recover import main as m
    else:
        from retryvla.eval.eval_nominal import main as m
    print(f"[eval_cond] appending {sfx!r} to every instruction", flush=True)
    m()


if __name__ == "__main__":
    main()
