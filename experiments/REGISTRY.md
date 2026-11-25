# Run registry

One row per run. Rows are appended with `python scripts/register_run.py runs/<run_id> --note "..."`.
Run directories (`runs/`) are gitignored; this file plus the `M*.md` docs are the tracked record.

| run_id | milestone | tag | status | git | key results | note |
|---|---|---|---|---|---|---|
| 20261013-054057_M0_baseline_spatial_nominal | M0 | baseline_spatial_nominal | done | 6373f75 | success_rate=0.656, n_episodes=500 | pi0 nominal, n_action_steps=10 |
| 20261013-063336_M0_verify_scenes | M0 | verify_scenes | done | 144499e | - | M0b 20 scenes; physics exact 20/20, 1px GL jitter on 4; 2 scenes already-success |
| 20261013-063416_M0_nas1_full | M0 | nas1_full | done | 38de7b2 | success_rate=0.715, n_episodes=200 | pi0 nominal n_action_steps=1 (20/task) |
| 20261013-063621_M0_recover_bddl | M0 | recover_bddl | done | 1a6cce1 | rsr=0.152, rsr_close=0.153, rsr_open=0.150, n_episodes=1160 | pi0 RSR, bddl instruction, test+pilot |
| 20261013-063631_M0_recover_canonical | M0 | recover_canonical | done | 7ac9bce | rsr=0.155, rsr_close=0.162, rsr_open=0.148, n_episodes=1160 | recover_canonical |
| 20261013-080655_M1_sft_nominal_only | M1 | sft_nominal_only | done | 7ac9bce | final_loss=0.337 | sft_nominal_only |
| 20261013-093503_M1_pi1nom_recover_test | M1 | pi1nom_recover_test | done | 021e66f | rsr=0.138, rsr_close=0.107, rsr_open=0.168, n_episodes=560 | pi1-nominal (SFT demos only) test RSR |
| 20261013-151228_M1_pi1nom_nominal | M1 | pi1nom_nominal | done | 57b6695 | success_rate=0.796, n_episodes=500 | pi1-nominal nominal SR (pi0 0.656) |
| 20261013-164210_M1_pi1cem_recover_test | M1 | pi1cem_recover_test | done | b737a4b | rsr=0.286, rsr_close=0.261, rsr_open=0.311, n_episodes=560 | pi1cem_recover_test |
| 20261013-164219_M1_pi1rand_recover_test | M1 | pi1rand_recover_test | done | b737a4b | rsr=0.279, rsr_close=0.246, rsr_open=0.311, n_episodes=560 | pi1rand_recover_test |
| 20261013-164233_M1_pi1cem_nominal | M1 | pi1cem_nominal | done | b737a4b | success_rate=0.690, n_episodes=500 | pi1cem_nominal |
| 20261013-164238_M1_pi1rand_nominal | M1 | pi1rand_nominal | done | b737a4b | success_rate=0.698, n_episodes=500 | pi1rand_nominal |
| 20261014-021046_M1_random_r25_recover_test | M1 | random_r25_recover_test | done | 18a35eb | rsr=0.254, rsr_close=0.225, rsr_open=0.282, n_episodes=560 | M1-v2 random_r25_recover_test |
| 20261014-021105_M1_random_r25_nominal | M1 | random_r25_nominal | done | 18a35eb | success_rate=0.726, n_episodes=500 | M1-v2 random_r25_nominal |
| 20261014-030113_M1_random_r25_nontrivial_recover_test | M1 | random_r25_nontrivial_recover_test | done | 18a35eb | rsr=0.220, rsr_close=0.207, rsr_open=0.232, n_episodes=560 | M1-v2 random_r25_nontrivial_recover_test |
| 20261014-061522_M1_random_r25_nontrivial_nominal | M1 | random_r25_nontrivial_nominal | done | 18a35eb | success_rate=0.766, n_episodes=500 | M1-v2 random_r25_nontrivial_nominal |
| 20261014-103001_M2_train_wm_v1 | M2 | train_wm_v1 | done | dd2c699 | - | train_wm_v1 |
| 20261014-120238_M2_train_critic_v1 | M2 | train_critic_v1 | done | dd2c699 | - | train_critic_v1 |
| 20261014-121255_M3_wm_search_hard_s0of2 | M3 | wm_search_hard_s0of2 | done | dd2c699 | - | wm_search_hard_s0of2 |
| 20261014-121313_M3_wm_search_hard_s1of2 | M3 | wm_search_hard_s1of2 | done | dd2c699 | - | wm_search_hard_s1of2 |
| 20261014-151922_M4_sft_A_base_nom_random | M4 | sft_A_base_nom_random | done | 451ef58 | final_loss=0.512 | M4 sft_A_base_nom_random |
| 20261014-151933_M4_sft_B_base_nom_random_wm | M4 | sft_B_base_nom_random_wm | done | 451ef58 | final_loss=0.482 | M4 sft_B_base_nom_random_wm |
| 20261014-200803_M4_sft_A_base_nom_random_recover_test | M4 | sft_A_base_nom_random_recover_test | done | 451ef58 | rsr=0.293, rsr_close=0.254, rsr_open=0.332, n_episodes=560 | M4 sft_A_base_nom_random_recover_test |
| 20261014-200814_M4_sft_B_base_nom_random_wm_recover_test | M4 | sft_B_base_nom_random_wm_recover_test | done | 451ef58 | rsr=0.298, rsr_close=0.257, rsr_open=0.339, n_episodes=560 | M4 sft_B_base_nom_random_wm_recover_test |
| 20261015-035127_M4_sft_A_base_nom_random_nominal | M4 | sft_A_base_nom_random_nominal | done | e2f6a82 | success_rate=0.756, n_episodes=500 | M4 sft_A_base_nom_random_nominal |
| 20261015-035136_M4_sft_B_base_nom_random_wm_nominal | M4 | sft_B_base_nom_random_wm_nominal | done | e2f6a82 | success_rate=0.748, n_episodes=500 | M4 sft_B_base_nom_random_wm_nominal |
| 20261124-160200_V2R_v2r_r1_search_s0of8 | V2R | v2r_r1_search_s0of8 | done | a547a63 | n_trajectories={'wm_diverse': 49} | V2R (PLAN V2R, 2026-11-24) |
| 20261124-160200_V2R_v2r_r1_search_s1of8 | V2R | v2r_r1_search_s1of8 | done | a547a63 | n_trajectories={'wm_diverse': 50} | V2R (PLAN V2R, 2026-11-24) |
| 20261124-160200_V2R_v2r_r1_search_s2of8 | V2R | v2r_r1_search_s2of8 | done | a547a63 | n_trajectories={'wm_diverse': 32} | V2R (PLAN V2R, 2026-11-24) |
| 20261124-160200_V2R_v2r_r1_search_s3of8 | V2R | v2r_r1_search_s3of8 | done | a547a63 | n_trajectories={'wm_diverse': 62} | V2R (PLAN V2R, 2026-11-24) |
| 20261124-173027_V2R_v2r_r1_search_s6of8 | V2R | v2r_r1_search_s6of8 | done | 7145e8d | n_trajectories={'wm_diverse': 65} | V2R (PLAN V2R, 2026-11-24) |
| 20261124-173312_V2R_v2r_r1_search_s4of8 | V2R | v2r_r1_search_s4of8 | done | 7145e8d | n_trajectories={'wm_diverse': 64} | V2R (PLAN V2R, 2026-11-24) |
| 20261124-174440_V2R_v2r_r1_search_s5of8 | V2R | v2r_r1_search_s5of8 | done | 7145e8d | n_trajectories={'wm_diverse': 59} | V2R (PLAN V2R, 2026-11-24) |
| 20261124-174649_V2R_v2r_r1_search_s7of8 | V2R | v2r_r1_search_s7of8 | done | 7145e8d | n_trajectories={'wm_diverse': 55} | V2R (PLAN V2R, 2026-11-24) |
| 20261125-031125_V2R_v2r_r0_s0_nominal | V2R | v2r_r0_s0_nominal | done | 2baf51f | success_rate=0.804, n_episodes=500 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-031354_V2R_v2r_r1_s0 | V2R | v2r_r1_s0 | done | c0c896e | final_loss=0.460 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-031354_V2R_v2r_r1_s1 | V2R | v2r_r1_s1 | done | c0c896e | final_loss=0.461 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-031402_V2R_v2r_r1_s2 | V2R | v2r_r1_s2 | done | c0c896e | final_loss=0.472 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-032628_V2R_v2r_r0_s1_nominal | V2R | v2r_r0_s1_nominal | done | c0c896e | success_rate=0.770, n_episodes=500 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-034001_V2R_v2r_r2_s0 | V2R | v2r_r2_s0 | done | c0c896e | final_loss=0.447 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-043724_V2R_v2r_r2_s1 | V2R | v2r_r2_s1 | done | c0c896e | final_loss=0.464 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-053326_V2R_v2r_r2_s2 | V2R | v2r_r2_s2 | done | b4d88d6 | final_loss=0.461 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-083923_V2R_v2r_r1_s0_nominal | V2R | v2r_r1_s0_nominal | done | 0c6900a | success_rate=0.788, n_episodes=500 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-083924_V2R_v2r_r1_s0_recover_test | V2R | v2r_r1_s0_recover_test | done | 7aaf42c | rsr=0.386, rsr_close=0.371, rsr_open=0.400, n_episodes=560 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-084021_V2R_v2r_r1_s1_recover_test | V2R | v2r_r1_s1_recover_test | done | 6852583 | rsr=0.398, rsr_close=0.382, rsr_open=0.414, n_episodes=560 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-084145_V2R_v2r_r1_s1_nominal | V2R | v2r_r1_s1_nominal | done | 7aaf42c | success_rate=0.778, n_episodes=500 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-084149_V2R_v2r_r1_s2_recover_test | V2R | v2r_r1_s2_recover_test | done | 7aaf42c | rsr=0.425, rsr_close=0.418, rsr_open=0.432, n_episodes=560 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-084405_V2R_v2r_r1_s2_nominal | V2R | v2r_r1_s2_nominal | done | 7aaf42c | success_rate=0.778, n_episodes=500 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-084433_V2R_v2r_r3_s2 | V2R | v2r_r3_s2 | done | 6852583 | final_loss=0.424 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-084731_V2R_v2r_r3_s0 | V2R | v2r_r3_s0 | done | 6852583 | final_loss=0.415 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-084731_V2R_v2r_r3_s1 | V2R | v2r_r3_s1 | done | 6852583 | final_loss=0.407 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-094408_V2R_v2r_r2_s0_nominal | V2R | v2r_r2_s0_nominal | done | 6852583 | success_rate=0.788, n_episodes=500 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-091605_V2R_v2r_r2_s0_recover_test | V2R | v2r_r2_s0_recover_test | done | 134cc61 | rsr=0.427, rsr_close=0.404, rsr_open=0.450, n_episodes=560 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-094734_V2R_v2r_r2_s1_recover_test | V2R | v2r_r2_s1_recover_test | done | 134cc61 | rsr=0.412, rsr_close=0.379, rsr_open=0.446, n_episodes=560 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-095219_V2R_v2r_r2_s2_recover_test | V2R | v2r_r2_s2_recover_test | done | 134cc61 | rsr=0.421, rsr_close=0.411, rsr_open=0.432, n_episodes=560 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-095301_V2R_v2r_r2_s1_nominal | V2R | v2r_r2_s1_nominal | done | 134cc61 | success_rate=0.808, n_episodes=500 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-101519_V2R_v2r_r2_s2_nominal | V2R | v2r_r2_s2_nominal | done | 134cc61 | success_rate=0.776, n_episodes=500 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-102020_V2R_v2r_r3_s0_recover_test | V2R | v2r_r3_s0_recover_test | done | 134cc61 | rsr=0.427, rsr_close=0.368, rsr_open=0.486, n_episodes=560 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-102218_V2R_v2r_r3_s0_nominal | V2R | v2r_r3_s0_nominal | done | 134cc61 | success_rate=0.802, n_episodes=500 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-105328_V2R_v2r_r3_s1_nominal | V2R | v2r_r3_s1_nominal | done | 134cc61 | success_rate=0.800, n_episodes=500 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-103824_V2R_v2r_r3_s1_recover_test | V2R | v2r_r3_s1_recover_test | done | 3effc78 | rsr=0.448, rsr_close=0.396, rsr_open=0.500, n_episodes=560 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-105627_V2R_v2r_r3_s2_recover_test | V2R | v2r_r3_s2_recover_test | done | 3effc78 | rsr=0.425, rsr_close=0.386, rsr_open=0.464, n_episodes=560 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-105729_V2R_v2r_r3_s2_nominal | V2R | v2r_r3_s2_nominal | done | 3effc78 | success_rate=0.810, n_episodes=500 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-111605_V2R_v2r_r0_s2_nominal | V2R | v2r_r0_s2_nominal | done | 3effc78 | success_rate=0.776, n_episodes=500 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-111936_V2R_v2r_pi1nom_nominal | V2R | v2r_pi1nom_nominal | done | 3effc78 | success_rate=0.774, n_episodes=500 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-115508_V2R_v2r_r4_s2 | V2R | v2r_r4_s2 | done | 34879e5 | final_loss=0.476 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-115702_V2R_v2r_r4_s0 | V2R | v2r_r4_s0 | done | 34879e5 | final_loss=0.463 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-115702_V2R_v2r_r4_s1 | V2R | v2r_r4_s1 | done | 34879e5 | final_loss=0.472 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-125102_V2R_v2r_r4_s2_nominal | V2R | v2r_r4_s2_nominal | done | 34879e5 | success_rate=0.798, n_episodes=500 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-125103_V2R_v2r_r4_s2_recover_test | V2R | v2r_r4_s2_recover_test | done | 34879e5 | rsr=0.425, rsr_close=0.404, rsr_open=0.446, n_episodes=560 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-125348_V2R_v2r_r4_s0_nominal | V2R | v2r_r4_s0_nominal | done | 34879e5 | success_rate=0.798, n_episodes=500 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-125348_V2R_v2r_r4_s0_recover_test | V2R | v2r_r4_s0_recover_test | done | 34879e5 | rsr=0.425, rsr_close=0.421, rsr_open=0.429, n_episodes=560 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-125531_V2R_v2r_r4_s1_recover_test | V2R | v2r_r4_s1_recover_test | done | 34879e5 | rsr=0.445, rsr_close=0.389, rsr_open=0.500, n_episodes=560 | V2R (PLAN V2R, 2026-11-24) |
| 20261125-125550_V2R_v2r_r4_s1_nominal | V2R | v2r_r4_s1_nominal | done | 34879e5 | success_rate=0.788, n_episodes=500 | V2R (PLAN V2R, 2026-11-24) |
