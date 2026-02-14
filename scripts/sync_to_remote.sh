#!/usr/bin/env bash
# Sync project to a remote server via rsync.
#
# Required: REMOTE_HOST, REMOTE_PATH
# Optional: SOURCE_PATH (default: project root), REMOTE_PASSWORD (for sshpass)
#
# Usage:
#   REMOTE_HOST=user@host.example.edu REMOTE_PATH=/remote/path/to/state-of-thoughts/ ./scripts/sync_to_remote.sh
#
# With password:
#   REMOTE_HOST=user@host.example.edu REMOTE_PATH=/remote/path/to/state-of-thoughts/ REMOTE_PASSWORD=secret ./scripts/sync_to_remote.sh
# 
# Sync specific subdirectory (REMOTE_PATH must be the matching directory on the server):
#   REMOTE_HOST=user@host.example.edu REMOTE_PATH=/remote/path/to/state-of-thoughts/experiments SOURCE_PATH=experiments ./scripts/sync_to_remote.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${REMOTE_HOST:-}" ]]; then
  echo "Error: REMOTE_HOST is not set (e.g., user@host.example.edu)" >&2
  exit 1
fi

if [[ -z "${REMOTE_PATH:-}" ]]; then
  echo "Error: REMOTE_PATH is not set (e.g., /home/user/state-of-thoughts/)" >&2
  exit 1
fi

# Source path: use SOURCE_PATH if set, else project root
SOURCE_PATH="${SOURCE_PATH:-${PROJECT_ROOT}}"
if [[ "${SOURCE_PATH}" != /* ]]; then
  SOURCE_PATH="${PROJECT_ROOT}/${SOURCE_PATH}"
fi
if [[ ! -d "${SOURCE_PATH}" ]]; then
  echo "Error: SOURCE_PATH is not a directory: ${SOURCE_PATH}" >&2
  exit 1
fi
SOURCE_PATH="$(cd "${SOURCE_PATH}" && pwd)"

# Ensure trailing slashes so rsync copies contents into the remote path (not nested dirs)
SOURCE_PATH="${SOURCE_PATH}/"
REMOTE_PATH="${REMOTE_PATH%/}/"

if [[ -n "${REMOTE_PASSWORD:-}" ]]; then
  if ! command -v sshpass &>/dev/null; then
    echo "Error: REMOTE_PASSWORD is set but sshpass is not installed" >&2
    exit 1
  fi
  RSYNC_CMD=(sshpass -p "$REMOTE_PASSWORD" rsync)
else
  RSYNC_CMD=(rsync)
fi

if [[ "${SOURCE_PATH}" == "${PROJECT_ROOT}/" ]] && { [[ ! -f "${PROJECT_ROOT}/README.md" ]] || [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]]; }; then
  echo "Error: Source appears to be project root but README.md and pyproject.toml not found" >&2
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
  "${SOURCE_PATH}" "${REMOTE_HOST}:${REMOTE_PATH}"

echo "✓ Synced from ${SOURCE_PATH%/} to ${REMOTE_HOST}:${REMOTE_PATH}"
