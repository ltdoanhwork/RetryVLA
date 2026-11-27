#!/usr/bin/env bash
# LRS-G debug lane B (2026-11-27): queue of control runs on one GPU.
# usage: dbg_laneB.sh <gpu> "<name>,<seed>,<root>" ...
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA
G=$1; shift
for x in "$@"; do bash scripts/lrs/dbg_run.sh $G ${x//,/ }; done
echo "[dbg] LANE_DONE B $(date -Is)"
