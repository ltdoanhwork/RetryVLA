"""B3.3 residual recovery RL baseline (ReCoVLA-inspired; NOT a ReCoVLA reproduction).

Frozen SmolVLA pi0 + a small residual BRIDGE controller that is active only during the first H env steps of an
episode (the same 50-step bridge window as RetryVLA's recovery segments), trained with PPO in the simulator from
the same train failure states as the PoC. After the bridge, pi0 alone continues until success / the step cap.

- policy.py    : features (frozen DINOv2 + state + hashed instruction + pi0's chunk) and the actor-critic.
- bridge.py    : ResidualBridgePolicy, a drop-in `predict_chunk` wrapper (used by eval_recover / eval_nominal).
- env.py       : batched episode runner over SimPool (pi0 + residual batched across workers, like rollout.py).
- train_ppo.py : PPO trainer with a matched simulator-transition budget, learning curve, resumable checkpoints.
"""
