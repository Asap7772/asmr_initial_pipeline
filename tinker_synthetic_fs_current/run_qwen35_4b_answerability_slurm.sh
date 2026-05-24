#!/usr/bin/env bash
# Slurm wrapper for run_qwen35_4b_answerability_local.sh.
#
# Default sweep has 3 entries, one per dataset:
#   sbatch --array=0-2 --export=ALL run_qwen35_4b_answerability_slurm.sh
#
# If MAX_TURNS_SWEEP expands the sweep, set --array to
#   num_datasets * num_max_turns - 1
#
# Without --array, this runs sweep index 1 by default, matching:
#   bash run_qwen35_4b_answerability_local.sh 1
#
# Override with SLURM_SWEEP_INDEX=0, 1, 2, ... or SLURM_SWEEP_INDEX=all.
#SBATCH --job-name=synthfsAns
#SBATCH --account=iris
#SBATCH --partition=iris-hi
#SBATCH --cpus-per-task=8
#SBATCH --mem=384G
#SBATCH --time=72:00:00
#SBATCH --output=/iris/u/asap7772/asmr_private/tinker_runs/slurm-%x-%j.out
#SBATCH --error=/iris/u/asap7772/asmr_private/tinker_runs/slurm-%x-%j.err

set -euo pipefail

if [ -n "${TINKER_SYNTHFS_SCRIPT_DIR:-}" ]; then
  SCRIPT_DIR="$(cd "$TINKER_SYNTHFS_SCRIPT_DIR" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/run_qwen35_4b_answerability_local.sh" ]; then
  SCRIPT_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
elif [ -f "/iris/u/asap7772/asmr_private/tinker_synthetic_fs_current/run_qwen35_4b_answerability_local.sh" ]; then
  SCRIPT_DIR="/iris/u/asap7772/asmr_private/tinker_synthetic_fs_current"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_SCRIPT="$SCRIPT_DIR/run_qwen35_4b_answerability_local.sh"

cd "$SCRIPT_DIR"

export TMPDIR="${TMPDIR:-/tmp/user/${UID}}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$TMPDIR/pip-cache}"
export PIP_PROGRESS_BAR="${PIP_PROGRESS_BAR:-off}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-wrap}"
export RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/tinker_runs}"
export SHELL_LOG_DIR="${SHELL_LOG_DIR:-$RUN_ROOT/shell_logs}"
SLURM_SWEEP_INDEX="${SLURM_SWEEP_INDEX:-1}"
if [ -z "${TINKER_SYNTHFS_CONDA_BIN:-}" ] && [ -x "/iris/u/asap7772/miniconda3/condabin/conda" ]; then
  export TINKER_SYNTHFS_CONDA_BIN="/iris/u/asap7772/miniconda3/condabin/conda"
fi

if [ "$SLURM_SWEEP_INDEX" != "all" ] && ! [[ "$SLURM_SWEEP_INDEX" =~ ^[0-9]+$ ]]; then
  echo "SLURM_SWEEP_INDEX must be a non-negative integer or 'all', got: $SLURM_SWEEP_INDEX" >&2
  exit 2
fi

if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
  export SWEEP_ID="${SWEEP_ID:-slurm${SLURM_ARRAY_JOB_ID}}"
elif [ -n "${SLURM_JOB_ID:-}" ]; then
  export SWEEP_ID="${SWEEP_ID:-slurm${SLURM_JOB_ID}}"
fi

mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$RUN_ROOT" "$SHELL_LOG_DIR"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}"
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "RUN_ROOT=$RUN_ROOT"
echo "SHELL_LOG_DIR=$SHELL_LOG_DIR"
echo "SWEEP_ID=${SWEEP_ID:-}"
echo "SWEEP_INDEX=${SWEEP_INDEX:-}"
echo "SLURM_SWEEP_INDEX=$SLURM_SWEEP_INDEX"
echo "TARGET_SCRIPT=$TARGET_SCRIPT"

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -z "${SWEEP_INDEX:-}" ]; then
  if [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    exec "$TARGET_SCRIPT" "$@"
  fi
  exec "$TARGET_SCRIPT" "$SLURM_ARRAY_TASK_ID" "$@"
fi

if [ -z "${SWEEP_INDEX:-}" ] && [ "$SLURM_SWEEP_INDEX" != "all" ]; then
  if [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    exec "$TARGET_SCRIPT" "$@"
  fi
  exec "$TARGET_SCRIPT" "$SLURM_SWEEP_INDEX" "$@"
fi

exec "$TARGET_SCRIPT" "$@"
