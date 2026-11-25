#!/usr/bin/env bash
H=$(hostname); echo "HOST $H"
for c in /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/v2r/claims/*; do
  read h p <<<$(python3 -c "import json;d=json.load(open('$c'));print(d['host'],d['pid'])")
  [ "$h" = "$H" ] || continue
  echo "CLAIM $(basename $c) pid $p: $(ps -o pid,stat,etime,cmd -p $p --no-headers 2>/dev/null | cut -c1-150)"
done
ps -eo pid,stat,etime,cmd | grep -E "retryvla|job.sh" | grep -v grep | cut -c1-160 | head -20
