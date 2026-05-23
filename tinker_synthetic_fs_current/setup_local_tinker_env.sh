#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

export TMPDIR="${TMPDIR:-/tmp/user/${UID}}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$TMPDIR/pip-cache}"
export PIP_PROGRESS_BAR="${PIP_PROGRESS_BAR:-off}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-wrap}"
unset PYTHONHOME

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

CONDA_ENV_NAME="${TINKER_SYNTHFS_CONDA_ENV:-tinker_synthfs_qwen35}"
CONDA_BIN="${TINKER_SYNTHFS_CONDA_BIN:-conda}"
PYTHON_CMD=("$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV_NAME" python)

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$CONDA_ENV_NAME"; then
  echo "Conda env not found: $CONDA_ENV_NAME" >&2
  echo "Create it with: conda create -n $CONDA_ENV_NAME python=3.11 pip -y" >&2
  exit 2
fi

if [ "${SKIP_TINKER_ENV_INSTALL:-0}" != "1" ]; then
  if ! "${PYTHON_CMD[@]}" -c "import chz, tinker_cookbook, wandb, torch" >/dev/null 2>&1; then
    if ! "${PYTHON_CMD[@]}" -c "import torch" >/dev/null 2>&1; then
      "${PYTHON_CMD[@]}" -m pip install \
        --prefer-binary \
        --index-url https://download.pytorch.org/whl/cpu \
        torch
    fi
    "${PYTHON_CMD[@]}" -m pip install --prefer-binary -r "$SCRIPT_DIR/requirements.txt"
  fi
fi

"${PYTHON_CMD[@]}" -m py_compile \
  "$SCRIPT_DIR/prepare_synthetic_fs_index.py" \
  "$SCRIPT_DIR/synthetic_fs_env.py" \
  "$SCRIPT_DIR/train_synthetic_fs_rl.py" \
  "$SCRIPT_DIR/make_nonexcluded_eval50.py"

echo "Using Conda env: $CONDA_ENV_NAME"
"${PYTHON_CMD[@]}" -c "import sys; print('Using Python:', sys.executable)"
