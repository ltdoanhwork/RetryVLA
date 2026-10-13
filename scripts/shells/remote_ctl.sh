#!/usr/bin/env bash
# Status / stop for RetryVLA jobs on remote shells.
# usage: scripts/shells/remote_ctl.sh status            # every claim: tmux alive? last log line
#        scripts/shells/remote_ctl.sh stop <shell8:gpu> # kill that rv_ session + release claim
set -euo pipefail
# det -> ssh breaks with the conda env libs on LD_LIBRARY_PATH (OpenSSL mismatch)
unset LD_LIBRARY_PATH
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd)
CMD=${1:-status}
export PATH="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin:$PATH"
full() { det shell list | awk -F'|' -v p="$1" '$1 ~ p && $4 ~ /RUNNING/ {gsub(/ /,"",$1); print $1; exit}'; }
claims() { python3 -c "import json; [print(k, v['tmux']) for k, v in json.load(open('$HERE/shells.json')).get('claims', {}).items()]"; }
if [ "$CMD" = status ]; then
  claims | while read -r key sess; do
    sid=${key%%:*}; f=$(full "$sid")
    if [ -z "$f" ]; then echo "$key $sess SHELL_GONE"; continue; fi
    alive=$(det shell open "$f" -- /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin/tmux has-session -t "=$sess" </dev/null >/dev/null 2>&1 && echo ALIVE || echo ENDED)
    log=$(ls -t "$REPO"/runs/_logs/${sess}_*.log 2>/dev/null | head -1)
    echo "$key $sess $alive | $(grep -a -E 'RV_DONE|done \||SR =|RSR =|step [0-9]+/' "$log" 2>/dev/null | tail -1 | cut -c1-140)"
  done
elif [ "$CMD" = stop ]; then
  key=$2; sid=${key%%:*}
  sess=$(python3 -c "import json; print(json.load(open('$HERE/shells.json'))['claims']['$key']['tmux'])")
  [[ "$sess" == rv_* ]] || { echo "refusing to kill non-rv_ session $sess"; exit 1; }
  f=$(full "$sid"); [ -n "$f" ] && det shell open "$f" -- /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/bin/tmux kill-session -t "=$sess" || true
  python3 - "$HERE/shells.json" "$key" <<'PY'
import json, sys
p, k = sys.argv[1:]; d = json.load(open(p)); d.get("claims", {}).pop(k, None); json.dump(d, open(p, "w"), indent=1)
PY
  echo "stopped $key ($sess)"
fi
