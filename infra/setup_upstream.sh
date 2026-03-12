#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

UPSTREAM_PATH="${1:-../asciiswarm_upstream.git}"

if [ -d "$UPSTREAM_PATH" ]; then
    echo "Bare repo already exists at $UPSTREAM_PATH"
    exit 0
fi

git clone --bare . "$UPSTREAM_PATH"

if git remote get-url upstream &>/dev/null; then
    git remote set-url upstream "$UPSTREAM_PATH"
else
    git remote add upstream "$UPSTREAM_PATH"
fi

git push upstream main
echo "Upstream bare repo ready at $UPSTREAM_PATH"
