#!/usr/bin/env bash
S=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/scripts/shells/tmux_start.sh
bash $S rv_v2r_sb7 scripts/v2r/search_lane_b.sh 0 7
bash $S rv_v2r_sb6 scripts/v2r/search_lane_b.sh 0 6
bash $S rv_v2r_sb5 scripts/v2r/search_lane_b.sh 1 5
bash $S rv_v2r_sb4 scripts/v2r/search_lane_b.sh 1 4
