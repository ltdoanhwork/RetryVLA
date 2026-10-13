"""M2 skeleton: train the latent world model.

Planned pipeline (not implemented this round):
  1. ``encode`` stage: run frozen DINOv2 over nominal LIBERO-Spatial demos + M0/M1 rollouts
     (pi0 failures, CEM branches from ``branches.parquet`` re-simulated with recording) on
     the *train* split only; cache tokens to ``data/wm_cache/<name>/*.pt``.
  2. ``train`` stage: ``wm_loss`` (1-step z, q, open-loop multistep) with sequence windows.
  3. ``eval`` stage: ``horizon_errors`` at 1/4/8/16 chunk steps, reported separately on
     nominal vs failure states. Gate for M3: failure-state error must not blow up vs nominal.
"""

from __future__ import annotations

from retryvla.config import cli_config


def main() -> None:
    cfg = cli_config()
    raise NotImplementedError(f"M2 world-model training is a skeleton (milestone {cfg.milestone}); see module doc")


if __name__ == "__main__":
    main()
