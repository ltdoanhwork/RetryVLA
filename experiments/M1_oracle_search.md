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

### Search (pilot, interim: CEM shard s0b at 35/42)

| | CEM | random |
|---|---|---|
| instances searched | 100 | 108 |
| instances solved (>=1 verified recovery) | 72 (0.72) | 74 (0.69) |
| verified trajectories | 268 | 241 |
| sim episodes spent | 15,232 | 6,912 |
| branch success, iteration 0 (shared proposal pool) | 0.11 | 0.11 |
| branch success, CEM refit iterations 1-3 | 0.009 | - |

Findings (`experiments/figs/m1_search_overview.png`, `scripts/analyze_m1.py`):
1. **The CEM refit does not work as implemented.** Success drops from 11% to about 1% after the first refit. The elites mix
   different strategies (primitives, perturbed SmolVLA chunks, gripper-flip variants), so a per-timestep Gaussian
   fitted to them averages different modes into actions that do nothing useful. All CEM gains come from iteration 0, which equals random.
   Fix before claiming anything about search: refit around single elites (local perturbation of each elite / iCEM-style
   elite carry-over), or a per-source mixture.
2. **The random arm got the same solve rate with less than half the simulator budget** (its config runs only the proposal pool, so the budgets
   are not matched. Match budgets in the next run).
3. **Many "recoveries" are trivial.** About a third of the instances are solved by a segment of 15 steps or fewer, often 2-14 steps of a
   perturbed chunk that just opens the gripper while the bowl is already held over the plate (e.g. `on_the_stove ep020 s0200 close0`).
   Half of the verified trajectories have no SmolVLA continuation at all (the prefix alone reaches the goal).
4. Proposal sources: primitives 0.14, perturbed 0.11, SmolVLA samples 0.08-0.09 branch success.
5. Hard tasks stay hard: drawer (about 0.4 solved) and next-to-cookie-box (0-0.25).

### Policies (test split, 560 episodes, official protocol)

| Policy | test RSR | close | open |
|---|---|---|---|
| pi0 (smolvla_libero) | 0.118 | | |
| pi1-nominal (SFT on Spatial demos only, 6k steps) | 0.138 | 0.107 | 0.168 |
| pi1-random | running | | |
| pi1-CEM | waiting for search | | |

## Decision
(go/no-go for M2)
