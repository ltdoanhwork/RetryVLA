#!/usr/bin/env bash
# Poll free GPU slots (shells.json avoid-list respected) every $1 seconds (default 300).
# Writes runs/_gpu_watch/{status.txt,history.log}; prints a wake sentinel only when the free set changes.
set -uo pipefail
unset LD_LIBRARY_PATH
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT="$REPO/runs/_gpu_watch"; mkdir -p "$OUT"
INTERVAL=${1:-300}
PREV=""
while true; do
  [ -f "$OUT/STOP" ] && { echo "watch stopped (STOP file)"; exit 0; }
  TABLE=$(python3 "$REPO/scripts/shells/find_free.py" 2>&1)
  FREE=$(echo "$TABLE" | grep '^FREE' | sed 's/^FREE slots ([0-9]*): //')
  { echo "# $(date -Is)"; echo "$TABLE"; } > "$OUT/status.txt"
  if [ "$FREE" != "$PREV" ]; then
    echo "$(date -Is) FREE: $FREE" >> "$OUT/history.log"
    echo "AGENT_LOOP_WAKE_gpus {\"prompt\":\"free GPU set changed: $FREE\"}"
    PREV=$FREE
  fi
  sleep "$INTERVAL"
done
