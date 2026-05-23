#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi
export WANDB_MODE="${WANDB_MODE:-online}"

BUILDER_EXECUTOR_BACKEND="${BUILDER_EXECUTOR_BACKEND:-vllm}"
BUILDER_EXECUTOR_MODEL="${BUILDER_EXECUTOR_MODEL:-Qwen/Qwen3.5-35B-A3B}"
BUILDER_EXECUTOR_BASE_URL="${BUILDER_EXECUTOR_BASE_URL:-https://iris-lab-ws--lateral-vllm-qwen3-5-35b-a3b.modal.run/v1/chat/completions}"
BUILDER_EXECUTOR_API_KEY_ENV="${BUILDER_EXECUTOR_API_KEY_ENV:-}"
if [ "$BUILDER_EXECUTOR_BACKEND" = "openrouter" ] && [ -z "$BUILDER_EXECUTOR_API_KEY_ENV" ]; then
  BUILDER_EXECUTOR_API_KEY_ENV=OPENROUTER_API_KEY
fi

required_keys=(TINKER_API_KEY GEMINI_API_KEY)
if [ -n "$BUILDER_EXECUTOR_API_KEY_ENV" ]; then
  required_keys+=("$BUILDER_EXECUTOR_API_KEY_ENV")
fi

missing=()
for key in "${required_keys[@]}"; do
  if [ -z "${!key:-}" ]; then
    missing+=("$key")
  fi
done
if [ "${WANDB_MODE:-online}" != "disabled" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  missing+=("WANDB_API_KEY")
fi
if [ "${#missing[@]}" -gt 0 ]; then
  echo "Missing required env vars: ${missing[*]}" >&2
  echo "Export them in your shell or add them to $REPO_ROOT/.env, then rerun this script." >&2
  exit 2
fi

# Validates the Conda env and installs missing requirements there.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/setup_local_tinker_env.sh"

CONDA_ENV_NAME="${TINKER_SYNTHFS_CONDA_ENV:-tinker_synthfs_qwen35}"
CONDA_BIN="${TINKER_SYNTHFS_CONDA_BIN:-conda}"
PYTHON_CMD=("$CONDA_BIN" run --no-capture-output -n "$CONDA_ENV_NAME" python)

DATASET_OUT_DIR="${DATASET_OUT_DIR:-$REPO_ROOT/data/tinker_synthetic_fs_alltrain}"
INDEX_JSONL="${INDEX_JSONL:-$DATASET_OUT_DIR/index.jsonl}"
"${PYTHON_CMD[@]}" "$SCRIPT_DIR/prepare_synthetic_fs_index.py" \
  --agent-dir "$REPO_ROOT/data/train" \
  --privileged-dir "$REPO_ROOT/data/train_privileged" \
  --out-dir "$DATASET_OUT_DIR"

LOCAL_RUN_ID="${LOCAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/tinker_runs}"
RUN_NAME="${RUN_NAME:-synthfs_qwen35_4b_pi_alltrain_bs16_gs4_mt32_ans32_ansrep4_probe4int8_answerability_only_g31litepreview_seed2_local${LOCAL_RUN_ID}}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$RUN_NAME}"
SHELL_LOG_DIR="${SHELL_LOG_DIR:-$RUN_ROOT/shell_logs}"
SHELL_LOG="$SHELL_LOG_DIR/${RUN_NAME}.train.log"

mkdir -p "$RUN_ROOT" "$SHELL_LOG_DIR"

echo "RUN_NAME=$RUN_NAME"
echo "RUN_DIR=$RUN_DIR"
echo "SHELL_LOG=$SHELL_LOG"
echo "MODEL_NAME=$MODEL_NAME"
echo "INDEX_JSONL=$INDEX_JSONL"
echo "WANDB_MODE=${WANDB_MODE:-online}"
echo "CONDA_ENV_NAME=$CONDA_ENV_NAME"
echo "BUILDER_EXECUTOR_BACKEND=$BUILDER_EXECUTOR_BACKEND"
echo "BUILDER_EXECUTOR_MODEL=$BUILDER_EXECUTOR_MODEL"
echo "BUILDER_EXECUTOR_BASE_URL=$BUILDER_EXECUTOR_BASE_URL"
echo "BUILDER_EXECUTOR_API_KEY_ENV=$BUILDER_EXECUTOR_API_KEY_ENV"
echo "Extra train args: $*"

"${PYTHON_CMD[@]}" "$SCRIPT_DIR/train_synthetic_fs_rl.py" \
  index_jsonl="$INDEX_JSONL" \
  excluded_qids_jsonl="" \
  eval_index_jsonl="" \
  batch_size=16 \
  group_size=4 \
  max_turns=32 \
  builder_compaction_trigger_tokens=3000 \
  answerer_model=gemini-3.1-flash-lite-preview \
  judge_model=gemini-3.1-flash-lite-preview \
  builder_compaction_model=gemini-3.1-flash-lite-preview \
  builder_executor_backend="$BUILDER_EXECUTOR_BACKEND" \
  builder_executor_model="$BUILDER_EXECUTOR_MODEL" \
  builder_executor_base_url="$BUILDER_EXECUTOR_BASE_URL" \
  builder_executor_api_key_env="$BUILDER_EXECUTOR_API_KEY_ENV" \
  reward_mode=hybrid \
  terminal_answerer_repeats=4 \
  answerability_delta_reward_scale=1.0 \
  answerability_probe_repeats=4 \
  answerability_probe_max_per_episode=4 \
  answerability_probe_interval_turns=8 \
  filesystem_maturity_scale=0.0 \
  step_filesystem_maturity_delta_scale=0.0 \
  step_construction_action_bonus=0.0 \
  step_non_construction_turn_penalty=0.0 \
  step_non_construction_streak_penalty=0.0 \
  step_tool_error_penalty=0.0 \
  termination_penalty=0.0 \
  empty_synthetic_penalty=0.0 \
  answerer_retrieval_cost_scale=0.0 \
  answerer_synthetic_read_cost_scale=0.0 \
  synthetic_success_bonus=0.0 \
  synthetic_usage_bonus=0.0 \
  raw_usage_ratio_penalty=0.0 \
  mature_stop_bonus=0.0 \
  save_every=5 \
  rolling_save_every=1 \
  rolling_ttl_seconds=604800 \
  log_path="$RUN_DIR" \
  wandb_project=synthetic-fs-rl \
  wandb_name="$RUN_NAME" \
  behavior_if_log_dir_exists=raise \
  model_name="$MODEL_NAME" \
  "$@" \
  2>&1 | tee -a "$SHELL_LOG"
