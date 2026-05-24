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
INDEX_JSONL="${INDEX_JSONL:-}"

DEFAULT_DATASET_DIRS=(
  "/afs/cs.stanford.edu/u/asap7772/asap7772/asmr_private/data/tinker_synthetic_fs_alltest"
  "/afs/cs.stanford.edu/u/asap7772/asap7772/asmr_private/data/tinker_synthetic_fs_alltest_max50files"
  "/afs/cs.stanford.edu/u/asap7772/asap7772/asmr_private/data/tinker_synthetic_fs_alltest_supponly"
)
if [ -n "${DATASET_DIRS:-}" ]; then
  read -r -a SWEEP_DATASET_PATHS <<< "$DATASET_DIRS"
else
  SWEEP_DATASET_PATHS=("${DEFAULT_DATASET_DIRS[@]}")
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
MODEL_CONTEXT_WINDOW_TOKENS="${MODEL_CONTEXT_WINDOW_TOKENS:-65536}"
CONTEXT_WINDOW_SAFETY_TOKENS="${CONTEXT_WINDOW_SAFETY_TOKENS:-256}"
MAX_TRAJECTORY_TOKENS="${MAX_TRAJECTORY_TOKENS:-140000}"
RAM_SPOOL_MINIBATCH_GROUPS="${RAM_SPOOL_MINIBATCH_GROUPS:-4}"
RAM_SPOOL_MAX_CONCURRENT_GROUPS="${RAM_SPOOL_MAX_CONCURRENT_GROUPS:-4}"
MAX_TURNS="${MAX_TURNS:-32}"
if [ -n "${MAX_TURNS_SWEEP:-}" ]; then
  read -r -a SWEEP_MAX_TURNS_VALUES <<< "$MAX_TURNS_SWEEP"
elif [ -n "${SWEEP_MAX_TURNS:-}" ]; then
  read -r -a SWEEP_MAX_TURNS_VALUES <<< "$SWEEP_MAX_TURNS"
else
  SWEEP_MAX_TURNS_VALUES=("$MAX_TURNS")
fi
if [ "${#SWEEP_MAX_TURNS_VALUES[@]}" -eq 0 ]; then
  echo "MAX_TURNS_SWEEP must contain at least one positive integer." >&2
  exit 2
fi
for max_turns_value in "${SWEEP_MAX_TURNS_VALUES[@]}"; do
  if ! [[ "$max_turns_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_TURNS_SWEEP values must be positive integers, got: $max_turns_value" >&2
    exit 2
  fi
done
if [ -n "${ANSWERER_MAX_TURNS:-}" ] && ! [[ "$ANSWERER_MAX_TURNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ANSWERER_MAX_TURNS must be a positive integer, got: $ANSWERER_MAX_TURNS" >&2
  exit 2
fi
if ! [[ "$RAM_SPOOL_MAX_CONCURRENT_GROUPS" =~ ^[0-9]+$ ]]; then
  echo "RAM_SPOOL_MAX_CONCURRENT_GROUPS must be a non-negative integer, got: $RAM_SPOOL_MAX_CONCURRENT_GROUPS" >&2
  exit 2
fi
if ! [[ "$RAM_SPOOL_MINIBATCH_GROUPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "RAM_SPOOL_MINIBATCH_GROUPS must be a positive integer, got: $RAM_SPOOL_MINIBATCH_GROUPS" >&2
  exit 2
fi
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/tinker_runs}"
SHELL_LOG_DIR="${SHELL_LOG_DIR:-$RUN_ROOT/shell_logs}"

RAND_CHARS=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
RAND=""
for _ in 1 2 3 4; do
  RAND_INDEX=$((RANDOM % ${#RAND_CHARS}))
  RAND+="${RAND_CHARS:RAND_INDEX:1}"
done
unset RAND_CHARS RAND_INDEX

SWEEP_ID="${SWEEP_ID:-$RAND}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-synth_fs_answerability_fixes}"
SWEEP_INDEX="${SWEEP_INDEX:-}"
if [ "$#" -gt 0 ] && [[ "$1" =~ ^[0-9]+$ ]]; then
  SWEEP_INDEX="$1"
  shift
fi
if [ -n "$SWEEP_INDEX" ] && ! [[ "$SWEEP_INDEX" =~ ^[0-9]+$ ]]; then
  echo "SWEEP_INDEX must be a non-negative integer, got: $SWEEP_INDEX" >&2
  exit 2
fi
EXTRA_ARGS=("$@")

mkdir -p "$RUN_ROOT" "$SHELL_LOG_DIR"

index_jsonl_for_dataset() {
  local dataset_path="$1"
  if [[ "$dataset_path" == *.jsonl ]]; then
    printf '%s\n' "$dataset_path"
  else
    printf '%s/index.jsonl\n' "$dataset_path"
  fi
}

train_index_jsonl_for_eval_dataset() {
  local dataset_path="$1"
  local dataset_dir
  local dataset_base
  local train_base

  if [[ "$dataset_path" == *.jsonl ]]; then
    dataset_dir="$(dirname "$dataset_path")"
  else
    dataset_dir="$dataset_path"
  fi
  dataset_base="$(basename "$dataset_dir")"
  if [[ "$dataset_base" != *alltest* ]]; then
    echo "Cannot infer matching train dataset from $dataset_path; expected a path containing alltest." >&2
    return 1
  fi
  train_base="${dataset_base/alltest/alltrain}"
  printf '%s/%s/index.jsonl\n' "$(dirname "$dataset_dir")" "$train_base"
}

slug_for_dataset() {
  local dataset_path="$1"
  local dataset_base
  dataset_base="$(basename "$dataset_path")"
  if [[ "$dataset_base" == "index.jsonl" ]]; then
    dataset_base="$(basename "$(dirname "$dataset_path")")"
  fi
  dataset_base="${dataset_base#tinker_synthetic_fs_}"
  dataset_base="${dataset_base//[^A-Za-z0-9_]/_}"
  printf '%s\n' "$dataset_base"
}

run_one_dataset() {
  local dataset_path="$1"
  local dataset_slug="$2"
  local train_index_jsonl="$3"
  local eval_index_jsonl="$4"
  local max_turns="$5"
  local answerer_max_turns="$6"
  local run_name="$7"
  local run_dir="$8"
  local ram_spool_dir="$9"
  local shell_log="${10}"

  mkdir -p "$(dirname "$ram_spool_dir")" "$(dirname "$shell_log")"

  echo "RUN_NAME=$run_name"
  echo "RUN_DIR=$run_dir"
  echo "RAM_SPOOL_DIR=$ram_spool_dir"
  echo "SHELL_LOG=$shell_log"
  echo "DATASET_PATH=$dataset_path"
  echo "DATASET_SLUG=$dataset_slug"
  echo "TRAIN_INDEX_JSONL=$train_index_jsonl"
  echo "EVAL_INDEX_JSONL=$eval_index_jsonl"
  echo "MODEL_NAME=$MODEL_NAME"
  echo "MAX_TURNS=$max_turns"
  echo "ANSWERER_MAX_TURNS=$answerer_max_turns"
  echo "MODEL_CONTEXT_WINDOW_TOKENS=$MODEL_CONTEXT_WINDOW_TOKENS"
  echo "CONTEXT_WINDOW_SAFETY_TOKENS=$CONTEXT_WINDOW_SAFETY_TOKENS"
  echo "MAX_TRAJECTORY_TOKENS=$MAX_TRAJECTORY_TOKENS"
  echo "RAM_SPOOL_MINIBATCH_GROUPS=$RAM_SPOOL_MINIBATCH_GROUPS"
  echo "RAM_SPOOL_MAX_CONCURRENT_GROUPS=$RAM_SPOOL_MAX_CONCURRENT_GROUPS"
  echo "DATA_ROOT=$DATA_ROOT"
  echo "AGENT_DIR=$AGENT_DIR"
  echo "PRIVILEGED_DIR=$PRIVILEGED_DIR"
  echo "HELDOUT_QUESTIONS_JSON=$HELDOUT_QUESTIONS_JSON"
  echo "INDEX_JSONL_OVERRIDE=${INDEX_JSONL:-}"
  echo "WANDB_MODE=${WANDB_MODE:-online}"
  echo "CONDA_ENV_NAME=$CONDA_ENV_NAME"
  echo "BUILDER_EXECUTOR_BACKEND=$BUILDER_EXECUTOR_BACKEND"
  echo "BUILDER_EXECUTOR_MODEL=$BUILDER_EXECUTOR_MODEL"
  echo "BUILDER_EXECUTOR_BASE_URL=$BUILDER_EXECUTOR_BASE_URL"
  echo "BUILDER_EXECUTOR_API_KEY_ENV=$BUILDER_EXECUTOR_API_KEY_ENV"
  echo "Extra train args: ${EXTRA_ARGS[*]}"

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
    index_jsonl="$train_index_jsonl" \
    eval_index_jsonl="$eval_index_jsonl" \
    seed=2 \
    limit=0 \
    eval_size=0 \
    batch_size=16 \
    group_size=4 \
    max_turns="$max_turns" \
    max_trajectory_tokens="$MAX_TRAJECTORY_TOKENS" \
    model_context_window_tokens="$MODEL_CONTEXT_WINDOW_TOKENS" \
    context_window_safety_tokens="$CONTEXT_WINDOW_SAFETY_TOKENS" \
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
    answerer_max_turns="$answerer_max_turns" \
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
    ram_spool_dir="$ram_spool_dir" \
    ram_spool_minibatch_groups="$RAM_SPOOL_MINIBATCH_GROUPS" \
    ram_spool_max_concurrent_groups="$RAM_SPOOL_MAX_CONCURRENT_GROUPS" \
    ram_spool_cleanup=true \
    log_path="$run_dir" \
    wandb_project=synthetic-fs-rl \
    wandb_name="$run_name" \
    behavior_if_log_dir_exists=raise \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$shell_log"
}

echo "SWEEP_ID=$SWEEP_ID"
echo "RUN_NAME_PREFIX=$RUN_NAME_PREFIX"
echo "SWEEP_INDEX=${SWEEP_INDEX:-all}"
echo "SWEEP_DATASETS=${SWEEP_DATASET_PATHS[*]}"
echo "MAX_TURNS_SWEEP=${SWEEP_MAX_TURNS_VALUES[*]}"
echo "ANSWERER_MAX_TURNS=${ANSWERER_MAX_TURNS:-follow_max_turns}"
echo "MODEL_NAME=$MODEL_NAME"
echo "RAM_SPOOL_MINIBATCH_GROUPS=$RAM_SPOOL_MINIBATCH_GROUPS"
echo "RAM_SPOOL_MAX_CONCURRENT_GROUPS=$RAM_SPOOL_MAX_CONCURRENT_GROUPS"
echo "DATA_ROOT=$DATA_ROOT"
echo "AGENT_DIR=$AGENT_DIR"
echo "PRIVILEGED_DIR=$PRIVILEGED_DIR"
echo "HELDOUT_QUESTIONS_JSON=$HELDOUT_QUESTIONS_JSON"
echo "INDEX_JSONL_OVERRIDE=${INDEX_JSONL:-}"
echo "WANDB_MODE=${WANDB_MODE:-online}"
echo "CONDA_ENV_NAME=$CONDA_ENV_NAME"
echo "BUILDER_EXECUTOR_BACKEND=$BUILDER_EXECUTOR_BACKEND"
echo "BUILDER_EXECUTOR_MODEL=$BUILDER_EXECUTOR_MODEL"
echo "BUILDER_EXECUTOR_BASE_URL=$BUILDER_EXECUTOR_BASE_URL"
echo "BUILDER_EXECUTOR_API_KEY_ENV=$BUILDER_EXECUTOR_API_KEY_ENV"
echo "Extra train args: ${EXTRA_ARGS[*]}"

pids=()
run_names=()
selected_dataset_paths=()
selected_max_turns=()
turn_sweep_size="${#SWEEP_MAX_TURNS_VALUES[@]}"
total_sweep_size=$((${#SWEEP_DATASET_PATHS[@]} * turn_sweep_size))
echo "TOTAL_SWEEP_ENTRIES=$total_sweep_size"
if [ -n "$SWEEP_INDEX" ]; then
  if [ "$SWEEP_INDEX" -ge "$total_sweep_size" ]; then
    echo "Sweep index $SWEEP_INDEX is out of range for $total_sweep_size sweep entries." >&2
    exit 4
  fi
  dataset_index=$((SWEEP_INDEX / turn_sweep_size))
  turn_index=$((SWEEP_INDEX % turn_sweep_size))
  echo "SELECTED_DATASET_INDEX=$dataset_index"
  echo "SELECTED_TURN_INDEX=$turn_index"
  selected_dataset_paths=("${SWEEP_DATASET_PATHS[$dataset_index]}")
  selected_max_turns=("${SWEEP_MAX_TURNS_VALUES[$turn_index]}")
else
  for dataset_path in "${SWEEP_DATASET_PATHS[@]}"; do
    for max_turns in "${SWEEP_MAX_TURNS_VALUES[@]}"; do
      selected_dataset_paths+=("$dataset_path")
      selected_max_turns+=("$max_turns")
    done
  done
fi

for sweep_i in "${!selected_dataset_paths[@]}"; do
  dataset_path="${selected_dataset_paths[$sweep_i]}"
  max_turns="${selected_max_turns[$sweep_i]}"
  answerer_max_turns="${ANSWERER_MAX_TURNS:-$max_turns}"
  if [ -n "${INDEX_JSONL:-}" ]; then
    train_index_jsonl="$INDEX_JSONL"
  else
    train_index_jsonl="$(train_index_jsonl_for_eval_dataset "$dataset_path")"
  fi
  eval_index_jsonl="$(index_jsonl_for_dataset "$dataset_path")"
  if [ ! -f "$train_index_jsonl" ]; then
    echo "Missing train dataset index: $train_index_jsonl" >&2
    exit 3
  fi
  if [ ! -f "$eval_index_jsonl" ]; then
    echo "Missing eval dataset index: $eval_index_jsonl" >&2
    exit 3
  fi

  dataset_slug="$(slug_for_dataset "$dataset_path")"
  turn_slug="mt${max_turns}_ans${answerer_max_turns}"
  if [ -n "${RUN_NAME:-}" ]; then
    run_name="${RUN_NAME}_${dataset_slug}_${turn_slug}"
  else
    run_name="${RUN_NAME_PREFIX}_${dataset_slug}_${turn_slug}_${SWEEP_ID}"
  fi
  run_dir="${RUN_DIR:-$RUN_ROOT/$run_name}"
  if [ -n "${RUN_DIR:-}" ]; then
    run_dir="$RUN_DIR/${dataset_slug}_${turn_slug}"
  fi
  ram_spool_dir="${RAM_SPOOL_DIR:-${TMPDIR:-/tmp}/tinker_synthfs_spool/$run_name}"
  if [ -n "${RAM_SPOOL_DIR:-}" ]; then
    ram_spool_dir="$RAM_SPOOL_DIR/${dataset_slug}_${turn_slug}"
  fi
  shell_log="$SHELL_LOG_DIR/${run_name}.train.log"

  echo "Launching $run_name with max_turns=$max_turns answerer_max_turns=$answerer_max_turns train_index_jsonl=$train_index_jsonl eval_index_jsonl=$eval_index_jsonl"
  run_one_dataset \
    "$dataset_path" \
    "$dataset_slug" \
    "$train_index_jsonl" \
    "$eval_index_jsonl" \
    "$max_turns" \
    "$answerer_max_turns" \
    "$run_name" \
    "$run_dir" \
    "$ram_spool_dir" \
    "$shell_log" &
  pids+=("$!")
  run_names+=("$run_name")
done

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "Finished ${run_names[$i]}"
  else
    echo "FAILED ${run_names[$i]}" >&2
    status=1
  fi
done

exit "$status"
