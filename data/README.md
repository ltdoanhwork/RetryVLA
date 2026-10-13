# Data

Gitignored except `splits/`. Rebuild with `scripts/01_download_assets.sh` and `scripts/make_split.py`.

- `assets/scenes/scenes/selected_bddl_scenes_*`: LIBERO-Recover benchmark scenes (ModelScope `ataier/LIBERO_Recovery_Assets`, `scenes.tar.gz`)
- `assets/scenes_full/assets_times/bddl_scenes_*`: dense per-20-step states of the same failure episodes (`scenes_full.tar.gz`)
- `splits/spatial_v1.json`: tracked scene-level split (group = task folder + failure episode)
- `recovery/<name>/`: generated recovery LeRobotDatasets (+ `retryvla/` side metadata)
