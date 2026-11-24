#!/usr/bin/env bash
# Commit ONLY what is already staged (no git add), as the repo user, no trailers. Used when a shared file has other
# agents' uncommitted hunks: stage our hunk via a blob (git update-index --cacheinfo) and commit just that.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MSG=$1
NAME=$(git config user.name); EMAIL=$(git config user.email)
export GIT_AUTHOR_NAME=$NAME GIT_AUTHOR_EMAIL=$EMAIL GIT_COMMITTER_NAME=$NAME GIT_COMMITTER_EMAIL=$EMAIL
TREE=$(git write-tree); PARENT=$(git rev-parse HEAD)
[ "$TREE" != "$(git rev-parse HEAD^{tree})" ] || { echo "nothing to commit"; exit 0; }
C=$(printf '%s\n' "$MSG" | git commit-tree "$TREE" -p "$PARENT")
git update-ref HEAD "$C" "$PARENT"
git log -1 --format='%h %an | %s'
