#!/bin/bash
set -euo pipefail

AGENT_ID="${HOSTNAME:-agent_unknown}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20}"
ITERATION=0

git config --global --add safe.directory /upstream
git config --global --add safe.directory /workspace/repo

git clone /upstream /workspace/repo
cd /workspace/repo
pip install -e . --quiet

git config user.name "SwarmAgent-${AGENT_ID}"
git config user.email "agent-${AGENT_ID}@asciiswarm.local"

mkdir -p /agent_logs/${AGENT_ID}

while [ "$ITERATION" -lt "$MAX_ITERATIONS" ]; do
    ITERATION=$((ITERATION + 1))

    # --- Rebase-hell guard ---
    # If a previous iteration left the repo mid-rebase or with dirty state,
    # nuke it so Claude doesn't wake up in a paralyzed git tree.
    git rebase --abort 2>/dev/null || true
    git reset --hard HEAD 2>/dev/null || true
    git clean -fd 2>/dev/null || true

    git pull --rebase origin main || {
        # If pull --rebase itself fails (conflict), abort and retry clean
        git rebase --abort 2>/dev/null || true
        git reset --hard origin/main 2>/dev/null || true
    }

    COMMIT=$(git rev-parse --short=6 HEAD)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOGFILE="/agent_logs/${AGENT_ID}/run_${ITERATION}_${COMMIT}_${TIMESTAMP}.log"

    echo "[${AGENT_ID}] Iteration ${ITERATION}/${MAX_ITERATIONS} at ${COMMIT}"

    claude --dangerously-skip-permissions \
           --model claude-opus-4-6 \
           -p "$(cat AGENT_PROMPT.md)" \
           &> "${LOGFILE}" || true

    git push origin main 2>/dev/null || true
    sleep 5
done

echo "[${AGENT_ID}] Reached MAX_ITERATIONS (${MAX_ITERATIONS}). Exiting."
