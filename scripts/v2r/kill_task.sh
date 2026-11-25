#!/usr/bin/env bash
# kill one V2R task's process group by the pid in its claim file (runs ON the task's shell): kill_task.sh <task>
C=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/v2r/claims/$1
P=$(python3 -c "import json;print(json.load(open('$C'))['pid'])")
kill -TERM -- -$P 2>/dev/null; sleep 20; kill -KILL -- -$P 2>/dev/null; rm -f $C; echo "KILLED_TASK $1 pgid $P"
