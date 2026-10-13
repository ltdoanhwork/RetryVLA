#!/usr/bin/env bash
# Commit staged + tracked changes as the repo's configured user, via plumbing (no tool trailers).
# usage: scripts/git_commit.sh "message" [paths...]   (no paths = all changes, incl. new files)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MSG=$1; shift
if [ $# -gt 0 ]; then git add -- "$@"; else git add -A; fi
NAME=$(git config user.name); EMAIL=$(git config user.email)
export GIT_AUTHOR_NAME=$NAME GIT_AUTHOR_EMAIL=$EMAIL GIT_COMMITTER_NAME=$NAME GIT_COMMITTER_EMAIL=$EMAIL
TREE=$(git write-tree)
PARENT=$(git rev-parse HEAD)
[ "$TREE" != "$(git rev-parse HEAD^{tree})" ] || { echo "nothing to commit"; exit 0; }
C=$(printf '%s\n' "$MSG" | git commit-tree "$TREE" -p "$PARENT")
git update-ref HEAD "$C" "$PARENT"
git log -1 --format='%h %an | %s'
