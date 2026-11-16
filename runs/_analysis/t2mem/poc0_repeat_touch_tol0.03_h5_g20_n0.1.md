# T2Mem PoC 0: repeated failed grasps within an episode

[by=touch] tol = 0.03 m (EEF pose relative to bowl at attempt onset), attempt = close held >= 5 steps (re-open < 20 merged) with EEF within 0.1 m of the bowl. Failed episodes only; `f_infeasible` excluded. Within = attempt within tol of an EARLIER failed attempt of the same episode; cross (null) = same number of failed attempts drawn from OTHER failure episodes of the same task.

| model | failed eps | attempts / ep | eps >= 2 attempts | eps with repeat | within repeat rate | cross-episode rate (null) | a2 eps with repeat |
|---|---|---|---|---|---|---|---|
| pi1_s0 | 40 | 1.20 | 14 (35%) | 5 (12%) | 6/18 (33%) | 0/14 (0%) | 4/20 |
| m7base_s0 | 33 | 1.18 | 11 (33%) | 4 (12%) | 6/17 (35%) | 0/9 (0%) | 3/14 |

Per mode (episodes with >= 1 repeat / episodes):

- pi1_s0: a1_wrong_place 0/13, a2_failed_grasp 4/20, c_misplace 1/6, d_stuck 0/1
- m7base_s0: a1_wrong_place 0/11, a2_failed_grasp 3/14, b_drop 1/1, c_misplace 0/4, d_stuck 0/3

Pooled: 9/73 = 12% failed episodes contain a repeated failed grasp -> **DROP approach A (< 20 %)**.
Within-episode repeat rate 34% vs cross-episode null 0% (ratio 342857142.86): ratio ~1 means the repeats reflect a fixed task prior, not an episodic loop.
