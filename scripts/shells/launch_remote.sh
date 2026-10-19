#!/usr/bin/env bash
# Launch a RetryVLA entrypoint on a (borrowed) Determined shell GPU, inside tmux.
# usage: scripts/shells/launch_remote.sh <shell_id_prefix> <gpu_idx> <session> <python.module> <config.yaml> [k=v ...]
#   session names are auto-prefixed with "rv_" so they never collide with the owner's sessions.
# e.g. scripts/shells/launch_remote.sh 22d385b6 0 m0c retryvla.eval.eval_recover configs/exp/m0_baseline_spatial.yaml
set -euo pipefail
# det -> ssh breaks with the conda env libs on LD_LIBRARY_PATH (OpenSSL mismatch)
unset LD_LIBRARY_PATH
HERE=$(cd "$(dirname "$0")" && pwd)
PREFIX=$1; GPU=$2; SESS=$3; shift 3
[[ "$SESS" == rv_* ]] || SESS="rv_$SESS"
if python3 -c "import json,sys; sys.exit(0 if '${PREFIX:0:8}' in json.load(open('$HERE/shells.json'))['avoid'] else 1)"; then
  echo "REFUSED: ${PREFIX:0:8} is in shells.json:avoid"; exit 1
fi
if ! python3 -c "import json,sys; a=json.load(open('$HERE/shells.json')).get('allow_only'); sys.exit(0 if (not a or '${PREFIX:0:8}' in a) else 1)"; then
  echo "REFUSED: ${PREFIX:0:8} not in shells.json:allow_only (user: no borrowing)"; exit 1
fi
FULL=$(det shell list | awk -F'|' -v p="$PREFIX" '$1 ~ p && $4 ~ /RUNNING/ {gsub(/ /,"",$1); print $1; exit}')
[ -n "$FULL" ] || { echo "shell $PREFIX not RUNNING"; exit 1; }
# guard: GPU must still be free right now
STATE=$(det shell open "$FULL" -- bash "$HERE/probe.sh" 2>/dev/null | awk -v g="$GPU" '$1=="GPU" && $2==g {print $3, $4, $5}')
read -r USED TOT UTIL <<<"$STATE"
[ -n "${USED:-}" ] || { echo "probe failed for $PREFIX gpu $GPU"; exit 1; }
SHARE=$(python3 -c "import json; print(1 if json.load(open('$HERE/shells.json')).get('share_ok',{}).get('${PREFIX:0:8}') else 0)")
if [ "$SHARE" = 1 ]; then  # user explicitly allowed sharing this (their own) GPU: only require free memory
  [ $((TOT-USED)) -lt 20000 ] && { echo "REFUSED: $PREFIX gpu $GPU has <20 GB free"; exit 1; }
elif [ $((TOT-USED)) -lt 40000 ] || [ "$UTIL" -gt 30 ]; then
  echo "REFUSED: $PREFIX gpu $GPU busy (used=${USED}MiB util=${UTIL}%)"; exit 1
fi
OUT=$(det shell open "$FULL" -- bash "$HERE/remote_tmux.sh" "$SESS" "$GPU" "$@" 2>&1 | grep -E "LAUNCHED|EXISTS|NO_TMUX|QUOTA|rror" || true)
echo "$OUT"
[[ "$OUT" == *LAUNCHED* ]] || { echo "launch failed; no claim recorded"; exit 1; }
# record the claim
python3 - "$HERE/shells.json" "${PREFIX:0:8}:$GPU" "$SESS" "$*" <<'PY'
import json, sys, datetime
p, key, sess, args = sys.argv[1:]
d = json.load(open(p)); d.setdefault("claims", {})[key] = {"tmux": sess, "args": args,
    "since": datetime.datetime.now().isoformat(timespec="seconds")}
json.dump(d, open(p, "w"), indent=1)
PY
