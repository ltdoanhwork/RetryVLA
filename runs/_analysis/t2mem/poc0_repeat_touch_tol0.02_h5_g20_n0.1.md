# T2Mem PoC 0: repeated failed grasps within an episode

[by=touch] tol = 0.02 m (EEF pose relative to bowl at attempt onset), attempt = close held >= 5 steps (re-open < 20 merged) with EEF within 0.1 m of the bowl. Failed episodes only; `f_infeasible` excluded. Within = attempt within tol of an EARLIER failed attempt of the same episode; cross (null) = same number of failed attempts drawn from OTHER failure episodes of the same task.

| model | failed eps | attempts / ep | eps >= 2 attempts | eps with repeat | within repeat rate | cross-episode rate (null) | a2 eps with repeat |
|---|---|---|---|---|---|---|---|
| pi1_s0 | 40 | 1.20 | 14 (35%) | 3 (8%) | 3/18 (17%) | 0/14 (0%) | 3/20 |
| m7base_s0 | 33 | 1.18 | 11 (33%) | 3 (9%) | 5/17 (29%) | 0/9 (0%) | 2/14 |

Per mode (episodes with >= 1 repeat / episodes):

- pi1_s0: a1_wrong_place 0/13, a2_failed_grasp 3/20, c_misplace 0/6, d_stuck 0/1
- m7base_s0: a1_wrong_place 0/11, a2_failed_grasp 2/14, b_drop 1/1, c_misplace 0/4, d_stuck 0/3

Pooled: 6/73 = 8% failed episodes contain a repeated failed grasp -> **DROP approach A (< 20 %)**.
Within-episode repeat rate 23% vs cross-episode null 0% (ratio 228571428.57): ratio ~1 means the repeats reflect a fixed task prior, not an episodic loop.
