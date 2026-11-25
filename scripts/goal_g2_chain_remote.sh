#!/usr/bin/env bash
# Runs the Goal G2 / P2-3 orchestrator ON the RetryVLA-owned shell itself (2026-11-25: the agent box changes container
# and its local tmux died with it). det works from inside the shell. Watchdog: restart up to 5 times.
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA
for k in 1 2 3 4 5; do python3 -u scripts/goal_g2_chain.py && break; echo "CHAIN restart $k"; sleep 300; done
