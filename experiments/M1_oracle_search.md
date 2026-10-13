# M1: Oracle (simulator) recovery search + distillation

## Hypothesis
If a recovery segment found by search in the true simulator (scored by whether frozen pi0 then completes
the task) is distilled into SmolVLA, pi1 recovers more often on held-out failure scenes than pi0, and more often
than pi1 trained on the same number of random-search recoveries. If this fails, fix the action representation, loss or
policy interface before investing in a world model (M2).

## Setup
- Search on `pilot` failure instances only (scene x {close, open}, fixed protocol seed). Skip instances
  where pi0 succeeds in both pre-check rollouts.
- Proposals: 24 SmolVLA chunk samples (noise 1.0/1.5) + 24 smooth perturbations + 16 primitives.
- CEM: 64 samples, 8 elites, 4 iterations, H=50, momentum 0.7. Oracle scorer: segment + pi0 continuation (300 steps).
- Baseline: `method=random` (fresh proposals every iteration, same budget).
- Data: verified successes only, up to 4 per instance; segments tagged recovery(0) / continuation(1).
- SFT: from smolvla_libero, 6k steps, bs 32, lr 5e-5 cosine, 50% recovery / 50% original Spatial demos.
- Equal trajectory count across arms via `train.match_trajectory_count_of`.

## Results

| Arm | Instances solved | Trajectories | Sim episodes | test RSR | close | open | nominal SR | run |
|---|---|---|---|---|---|---|---|---|
| pi0 | - | - | - | | | | | |
| random search + SFT | | | | | | | | |
| oracle CEM + SFT | | | | | | | | |

## Decision
(go/no-go for M2)
