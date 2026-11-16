# T2Mem PoC 0: repeated failed grasps within an episode

[by=touch] tol = 0.03 m (EEF pose relative to bowl at attempt onset), attempt = close held >= 5 steps (re-open < 10 merged) with EEF within 0.1 m of the bowl. Failed episodes only; `f_infeasible` excluded. Within = attempt within tol of an EARLIER failed attempt of the same episode; cross (null) = same number of failed attempts drawn from OTHER failure episodes of the same task.

| model | failed eps | attempts / ep | eps >= 2 attempts | eps with repeat | within repeat rate | cross-episode rate (null) | a2 eps with repeat |
|---|---|---|---|---|---|---|---|
| pi1_s0 | 40 | 1.38 | 14 (35%) | 6 (15%) | 10/25 (40%) | 4/18 (22%) | 5/20 |
| m7base_s0 | 33 | 1.24 | 11 (33%) | 5 (15%) | 8/19 (42%) | 0/10 (0%) | 4/14 |

Per mode (episodes with >= 1 repeat / episodes):

- pi1_s0: a1_wrong_place 0/13, a2_failed_grasp 5/20, c_misplace 1/6, d_stuck 0/1
- m7base_s0: a1_wrong_place 0/11, a2_failed_grasp 4/14, b_drop 1/1, c_misplace 0/4, d_stuck 0/3

Pooled: 11/73 = 15% failed episodes contain a repeated failed grasp -> **DROP approach A (< 20 %)**.
Within-episode repeat rate 41% vs cross-episode null 14% (ratio 2.86): ratio ~1 means the repeats reflect a fixed task prior, not an episodic loop.
