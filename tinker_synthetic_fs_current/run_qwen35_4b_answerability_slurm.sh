#!/usr/bin/env bash
# Slurm wrapper for run_qwen35_4b_answerability_local.sh.
#
# Default sweep has 3 entries, one per dataset:
#   sbatch --array=0-2 --export=ALL run_qwen35_4b_answerability_slurm.sh
#
# If MAX_TURNS_SWEEP expands the sweep, set --array to
#   num_datasets * num_max_turns - 1
#
# Without --array, this delegates to the local launcher unchanged, which runs
# every selected sweep entry in the same Slurm allocation.
#SBATCH --job-name=synthfsAns
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --cpus-per-task=8
#SBATCH --mem=384G
#SBATCH --time=72:00:00
#SBATCH --output=/iris/u/asap7772/asmr_private/tinker_runs/slurm-%x-%j.out
#SBATCH --error=/iris/u/asap7772/asmr_private/tinker_runs/slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
if [ -z "${TINKER_SYNTHFS_CONDA_BIN:-}" ] && [ -x "/iris/u/asap7772/miniconda3/condabin/conda" ]; then
  export TINKER_SYNTHFS_CONDA_BIN="/iris/u/asap7772/miniconda3/condabin/conda"
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
echo "RUN_ROOT=$RUN_ROOT"
echo "SHELL_LOG_DIR=$SHELL_LOG_DIR"
echo "SWEEP_ID=${SWEEP_ID:-}"
echo "TARGET_SCRIPT=$TARGET_SCRIPT"

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -z "${SWEEP_INDEX:-}" ]; then
  if [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    exec "$TARGET_SCRIPT" "$@"
  fi
  exec "$TARGET_SCRIPT" "$SLURM_ARRAY_TASK_ID" "$@"
fi

exec "$TARGET_SCRIPT" "$@"
