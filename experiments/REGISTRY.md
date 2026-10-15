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
