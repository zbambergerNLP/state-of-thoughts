#!/usr/bin/env bash
# Sync project to a remote server via rsync.
#
# Set REMOTE_HOST and REMOTE_PATH. Optionally set REMOTE_PASSWORD for SSH password
# authentication (requires sshpass). Without REMOTE_PASSWORD, uses SSH keys.
# Must be invoked from the project root.
#
# Usage:
#   REMOTE_HOST=user@host.example.edu REMOTE_PATH=/home/user/project/ ./scripts/sync_to_remote.sh
#
# With password:
#   REMOTE_HOST=user@host.example.edu REMOTE_PATH=/home/user/project/ REMOTE_PASSWORD=secret ./scripts/sync_to_remote.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${REMOTE_HOST:-}" ]]; then
  echo "Error: REMOTE_HOST is not set (e.g., user@host.example.edu)" >&2
  exit 1
fi

if [[ -z "${REMOTE_PATH:-}" ]]; then
  echo "Error: REMOTE_PATH is not set (e.g., /home/user/project/)" >&2
  exit 1
fi

if [[ -n "${REMOTE_PASSWORD:-}" ]]; then
  if ! command -v sshpass &>/dev/null; then
    echo "Error: REMOTE_PASSWORD is set but sshpass is not installed" >&2
    exit 1
  fi
  RSYNC_CMD=(sshpass -p "$REMOTE_PASSWORD" rsync)
else
  RSYNC_CMD=(rsync)
fi

cd "${PROJECT_ROOT}"

if [[ ! -f "README.md" ]] || [[ ! -f "pyproject.toml" ]]; then
  echo "Error: Must run from project root (README.md and pyproject.toml not found)" >&2
  exit 1
fi

"${RSYNC_CMD[@]}" -avz \
  --exclude='.git/' \
  --exclude='.mypy_cache/' \
  --exclude='__pycache__/' \
  --exclude='.cursor/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.vscode/' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='*_env/' \
  --exclude='paper/' \
  --exclude='experiments/results' \
  --exclude='.idea/' \
  ./ "${REMOTE_HOST}:${REMOTE_PATH}"

echo "✓ Synced to ${REMOTE_HOST}:${REMOTE_PATH}"
