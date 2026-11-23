"""Back on Track (arXiv 2610.19579) re-implementation as a TEST-TIME baseline arm (no distillation).

liveness.py  policy-conditioned discounted liveness value (arXiv 2605.11479), two-stage, ensemble of MLPs
collect.py   on-policy rollouts of the frozen policy on the deployment distribution (train split only)
retryvla/policy/bot.py  monitor (smoothed ensemble value vs sigmoid threshold) + PCA-basis CEM in our latent WM
See docs/related_work/BACK_ON_TRACK_REPRO.md for every deviation.
"""
