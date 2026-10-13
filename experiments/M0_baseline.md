# M0: SmolVLA baseline on LIBERO-Spatial / LIBERO-Recover Spatial

## Hypothesis
`HuggingFaceVLA/smolvla_libero`, evaluated as-is, solves nominal LIBERO-Spatial at roughly the reported rate (about 90%) but
fails most LIBERO-Recover spatial failure states. That gap is the room for recovery learning.

## Setup
- Policy: `HuggingFaceVLA/smolvla_libero` (chunk 50, `n_action_steps=10`, 10 flow steps, noise scale 1.0).
- Nominal: libero_spatial, 10 tasks x 50 init states, 280 steps (`configs/exp/m0_baseline_spatial.yaml`).
- Scenes: `selected_bddl_scenes_spatial`: 168 scenes / 84 failure episodes / 9 of the 10 spatial tasks
  ("from table center" has no failure scenes). The scenes are heavily skewed: ramekin 48, drawer 46, between 24, ...
- Split `data/splits/spatial_v1.json`, grouped by failure episode: train 112 scenes (pilot 60), test 56.
- RSR protocol: official LIBERO-Recovery (5 close + 5 open gripper trials, ±2 cm noise, 520 steps).
- Instruction: `bddl` (official paraphrase, e.g. "pick the akita black bowl ...") vs `canonical`
  (the original LIBERO wording that SmolVLA was trained on).

## Results
(filled from `experiments/REGISTRY.md`)

M0b (20 scenes, borrowed GPU ca18f440): obs ok 20/20. Physics replay is bit-exact for 20/20 (reload and snapshot).
The rendered pixels differ by at most 1 intensity level on 4 scenes (GL jitter, harmless). 2/20 scenes already satisfy the BDDL goal
at the failure state (`cookie_box ep044 s0180`, `stove ep044 s0180`). They count as trivial successes under the official
protocol. Flag them when reporting RSR.

| Eval | Split | Instruction | SR / RSR | close | open | run |
|---|---|---|---|---|---|---|
| nominal | - | canonical | | | | |
| M0b verify | 20 scenes | - | deterministic / obs ok | | | |
| recover | test | bddl | | | | |
| recover | pilot | bddl | | | | |
| recover | test | canonical | | | | |

## Decision
(go/no-go for M1)
