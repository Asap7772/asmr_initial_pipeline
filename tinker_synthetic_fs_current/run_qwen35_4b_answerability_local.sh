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

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
AGENT_DIR="${AGENT_DIR:-$DATA_ROOT/train}"
PRIVILEGED_DIR="${PRIVILEGED_DIR:-$DATA_ROOT/train_privileged}"
HELDOUT_QUESTIONS_JSON="${HELDOUT_QUESTIONS_JSON:-$DATA_ROOT/heldout_50_questions.json}"
INDEX_JSONL="${INDEX_JSONL:-/iris/u/asap7772/asap7772/asmr_private/data/tinker_synthetic_fs_alltrain_supponly/index.jsonl}"
EVAL_INDEX_JSONL="${EVAL_INDEX_JSONL:-/iris/u/asap7772/asap7772/asmr_private/data/tinker_synthetic_fs_alltest_supponly/index.jsonl}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/tinker_runs}"
DEFAULT_RUN_NAME="synth_fs_allfiles_supponly"
RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/$RUN_NAME}"
RAM_SPOOL_DIR="${RAM_SPOOL_DIR:-/scr/asap7772/tinker_synthfs_spool/$RUN_NAME}"
SHELL_LOG_DIR="${SHELL_LOG_DIR:-$RUN_ROOT/shell_logs}"
SHELL_LOG="$SHELL_LOG_DIR/${RUN_NAME}.train.log"

mkdir -p "$RUN_ROOT" "$SHELL_LOG_DIR"

echo "RUN_NAME=$RUN_NAME"
echo "RUN_DIR=$RUN_DIR"
echo "RAM_SPOOL_DIR=$RAM_SPOOL_DIR"
echo "SHELL_LOG=$SHELL_LOG"
echo "MODEL_NAME=$MODEL_NAME"
echo "DATA_ROOT=$DATA_ROOT"
echo "AGENT_DIR=$AGENT_DIR"
echo "PRIVILEGED_DIR=$PRIVILEGED_DIR"
echo "HELDOUT_QUESTIONS_JSON=$HELDOUT_QUESTIONS_JSON"
echo "INDEX_JSONL=$INDEX_JSONL"
echo "EVAL_INDEX_JSONL=$EVAL_INDEX_JSONL"
echo "WANDB_MODE=${WANDB_MODE:-online}"
echo "CONDA_ENV_NAME=$CONDA_ENV_NAME"
echo "BUILDER_EXECUTOR_BACKEND=$BUILDER_EXECUTOR_BACKEND"
echo "BUILDER_EXECUTOR_MODEL=$BUILDER_EXECUTOR_MODEL"
echo "BUILDER_EXECUTOR_BASE_URL=$BUILDER_EXECUTOR_BASE_URL"
echo "BUILDER_EXECUTOR_API_KEY_ENV=$BUILDER_EXECUTOR_API_KEY_ENV"
echo "Extra train args: $*"

"${PYTHON_CMD[@]}" "$SCRIPT_DIR/train_synthetic_fs_rl.py" \
  model_name="$MODEL_NAME" \
  renderer_name=qwen3_5 \
  lora_rank=32 \
  learning_rate=0.00004 \
  max_steps=110 \
  eval_every=0 \
  max_tokens=4096 \
  ttl_seconds=604800 \
  num_substeps=1 \
  loss_fn=importance_sampling \
  index_jsonl="$INDEX_JSONL" \
  eval_index_jsonl="$EVAL_INDEX_JSONL" \
  seed=2 \
  limit=0 \
  eval_size=0 \
  batch_size=16 \
  group_size=4 \
  max_turns=32 \
  max_trajectory_tokens=140000 \
  model_context_window_tokens=144352 \
  step_penalty=0.0 \
  raw_docs_penalty=0.0 \
  builder_compaction_trigger_tokens=3000 \
  answerer_model=gemini-3.1-flash-lite-preview \
  judge_model=gemini-3.1-flash-lite-preview \
  builder_compaction_model=gemini-3.1-flash-lite-preview \
  builder_executor_enabled=true \
  builder_batch_tools_enabled=true \
  builder_executor_max_source_chars=16000 \
  builder_executor_max_output_tokens=512 \
  builder_compaction_enabled=true \
  builder_compaction_keep_recent_turns=1 \
  builder_compaction_max_output_tokens=800 \
  builder_compaction_input_max_chars=60000 \
  builder_executor_backend="$BUILDER_EXECUTOR_BACKEND" \
  builder_executor_model="$BUILDER_EXECUTOR_MODEL" \
  builder_executor_base_url="$BUILDER_EXECUTOR_BASE_URL" \
  builder_executor_api_key_env="$BUILDER_EXECUTOR_API_KEY_ENV" \
  reward_mode=hybrid \
  answerer_max_turns=32 \
  answerer_workspace_mode=synthetic_only \
  answerer_final_answer_max_tokens=128 \
  answerer_retrieval_cost_token_unit=1000 \
  answerer_retrieval_cost_correct_only=true \
  answerer_synthetic_read_cost_unit=10 \
  terminal_answerer_repeats=4 \
  answerability_delta_reward_scale=1.0 \
  answerability_delta_min_abs=0.25 \
  answerability_delta_allow_negative=true \
  answerability_probe_repeats=4 \
  answerability_probe_max_per_episode=4 \
  answerability_probe_interval_turns=8 \
  answerability_probe_min_maturity=0.45 \
  filesystem_maturity_scale=0.0 \
  filesystem_coverage_weight=0.35 \
  filesystem_expansion_weight=0.3 \
  filesystem_organization_weight=0.35 \
  filesystem_stop_weight=0.0 \
  step_filesystem_maturity_delta_scale=0.0 \
  step_construction_action_bonus=0.0 \
  step_non_construction_turn_penalty=0.0 \
  step_non_construction_streak_penalty=0.0 \
  step_non_construction_streak_free=3 \
  step_tool_error_penalty=0.0 \
  termination_penalty=0.0 \
  empty_synthetic_penalty=0.0 \
  answerer_retrieval_cost_scale=0.0 \
  answerer_synthetic_read_cost_scale=0.0 \
  synthetic_success_bonus=0.0 \
  synthetic_usage_bonus=0.0 \
  raw_usage_ratio_penalty=0.0 \
  mature_stop_bonus=0.0 \
  mature_stop_min_score=0.8 \
  terminal_reward_clip_min=-1 \
  terminal_reward_clip_max=3 \
  judge_max_output_tokens=64 \
  log_step_details=false \
  log_compaction_summaries=false \
  retain_reward_tool_messages=false \
  trim_terminal_history_for_memory=true \
  return_empty_terminal_observation=true \
  clear_state_on_terminal_for_memory=true \
  num_groups_to_log=0 \
  rollout_json_export=false \
  save_every=5 \
  rolling_save_every=1 \
  rolling_ttl_seconds=604800 \
  ram_spool_enabled=true \
  ram_spool_dir="$RAM_SPOOL_DIR" \
  ram_spool_minibatch_groups=4 \
  ram_spool_cleanup=true \
  log_path="$RUN_DIR" \
  wandb_project=synthetic-fs-rl \
  wandb_name="$RUN_NAME" \
  behavior_if_log_dir_exists=raise \
  "$@" \
  2>&1 | tee -a "$SHELL_LOG"
