# Run registry

One row per run. Rows are appended with `python scripts/register_run.py runs/<run_id> --note "..."`.
Run directories (`runs/`) are gitignored; this file plus the `M*.md` docs are the tracked record.

| run_id | milestone | tag | status | git | key results | note |
|---|---|---|---|---|---|---|
| 20261013-054057_M0_baseline_spatial_nominal | M0 | baseline_spatial_nominal | done | 6373f75 | success_rate=0.656, n_episodes=500 | pi0 nominal, n_action_steps=10 |
| 20261013-063336_M0_verify_scenes | M0 | verify_scenes | done | 144499e | - | M0b 20 scenes; physics exact 20/20, 1px GL jitter on 4; 2 scenes already-success |
| 20261013-063416_M0_nas1_full | M0 | nas1_full | done | 38de7b2 | success_rate=0.715, n_episodes=200 | pi0 nominal n_action_steps=1 (20/task) |
