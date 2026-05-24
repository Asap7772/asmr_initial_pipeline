# Tinker Synthetic Filesystem Current Setup

- Primary weaker-model RL target: `Qwen/Qwen3.5-4B`.
- Optimizer/loss: Tinker `loss_fn=ppo`.
- Reward for the next weaker-model runs: answerability-focused
- Epoch semantics: true reshuffled passes over the filtered train docsets.
- Held-out set: current 50 BrowseComp+ questions in `heldout_50_questions_browsecomp_plus_query_ids.json`.
- Note: Tinker PPO does not expose a critic/value function by default, so value-function metrics are not logged.

## Files

- `synthetic_fs_env.py`: synthetic filesystem environment, builder/answerer tools, reward logic, dataset splitting, and true epoch reshuffling.
- `train_synthetic_fs_rl.py`: Tinker RL entrypoint with PPO config, epoch controls, and RL diagnostics.
- `run_weaker_qwen35_4b_ppo_epoch10_wandb.sh`: main PI-requested weaker-model PPO epoch-10 launcher. Configure with `MODEL_NAME`, `LORA_RANK`, and `LEARNING_RATE`.
- `run_base_passk_wandb.sh`: pass@k evaluator for base or fine-tuned checkpoints.
- `analyze_passk_rollouts.py`: pass@k summary/per-question analysis helper.
- `run_eval_heldout_wandb.sh`: single-trajectory held-out eval helper for a trained checkpoint.
- `run_streaming_summary_passk.py`: streaming summarization baseline implementation.
- `run_streaming_summary_passk_wandb.sh`: SLURM wrapper for streaming-summary pass@k.
- `prepare_synthetic_fs_index.py`: builds the local Tinker `index.jsonl` from `../data/train` and `../data/train_privileged`.
- `setup_local_tinker_env.sh`: creates/activates the local Python environment and installs `requirements.txt`.
- `run_qwen35_4b_answerability_local.sh`: local launcher for the answerability-only Qwen3.5-4B run.
- `run_qwen35_4b_answerability_slurm.sh`: SLURM wrapper for the answerability-only Qwen3.5-4B run.
- `make_nonexcluded_eval50.py`: legacy helper for regenerating a non-excluded held-out 50 split.
- `heldout_50_questions.json`: current 50 held-out questions.
- `heldout_50_questions_browsecomp_plus_query_ids.json`: same 50 held-out questions with both `query_id` and `question_id`.
- `requirements.txt`: minimal runtime dependencies.

## PPO Diagnostics Logged

- `optim/advantage_mean`
- `optim/advantage_std`
- `optim/advantage_min`
- `optim/advantage_max`
- `optim/advantage_abs_mean`
- `optim/zero_advantage_ratio`
- `optim/trajectory_turns_max`
- `optim/trajectory_turns_mean`
- `optim/action_tokens_max`
- `optim/action_tokens_mean`
- `optim/ppo_clip_low_ratio`
- `optim/ppo_clip_high_ratio`
- `optim/entropy` from Tinker/cookbook KL metrics

## Local Answerability-Only Run

```bash
cd /afs/cs.stanford.edu/u/asap7772/asap7772/asmr_private/tinker_synthetic_fs_current

for k in TINKER_API_KEY OPENROUTER_API_KEY GEMINI_API_KEY WANDB_API_KEY; do
  echo "$k: ${!k:+set}"
done

export MODEL_NAME=Qwen/Qwen3.5-4B

./run_qwen35_4b_answerability_local.sh
```

Useful local overrides:

```bash
export RUN_ROOT=/tmp/user/$UID/tinker_runs
export TINKER_SYNTHFS_CONDA_ENV=tinker_synthfs_qwen35
export SKIP_TINKER_ENV_INSTALL=1
export WANDB_MODE=disabled
export RAM_SPOOL_MINIBATCH_GROUPS=1
export RAM_SPOOL_MAX_CONCURRENT_GROUPS=4
export ANSWERER_REPEAT_CONCURRENCY=2
export MEMORY_GUARD_MAX_CGROUP_USED_FRACTION=0.70
```

The launcher writes a fresh local all-train index to `../data/tinker_synthetic_fs_alltrain/index.jsonl`
before starting training.
It sources `../.env` automatically if that file exists.
The launcher runs Python through `conda run -n ${TINKER_SYNTHFS_CONDA_ENV:-tinker_synthfs_qwen35}`.
`RAM_SPOOL_MAX_CONCURRENT_GROUPS` caps live rollout groups during the disk-spooled
sampling phase. The local launcher default is `4` for memory headroom; raise it
only after checking memory metrics, or set it to `0` to run all groups in the
batch concurrently if you want maximum speed.
`RAM_SPOOL_MINIBATCH_GROUPS` caps how many spooled groups are reloaded for each
training chunk. Lower values reduce peak RAM at the cost of slower sampling or
training. The memory guard defaults to `MEMORY_GUARD_MAX_CGROUP_USED_FRACTION=0.70`.
`ANSWERER_REPEAT_CONCURRENCY` keeps the same terminal/probe repeat counts but
runs independent answerer reward repeats concurrently; the launcher default is
`2` to reduce reward latency without changing the averaged reward target. Lower
it to `1` if Gemini or judge rate limits become the bottleneck, or raise it
carefully if remote capacity has headroom.
Peak reward-call pressure scales roughly with
`RAM_SPOOL_MAX_CONCURRENT_GROUPS * group_size * ANSWERER_REPEAT_CONCURRENCY`.
With the default launcher settings and `group_size=4`, this is about `32`
in-flight answerer repeats.

## Launch Answerability-Only Run On SLURM

```bash
cd /iris/u/asap7772/asmr_private/tinker_synthetic_fs_current

# Default non-array run: sweep index 1, matching the local launcher example
# `bash run_qwen35_4b_answerability_local.sh 1`.
sbatch --export=ALL run_qwen35_4b_answerability_slurm.sh

# Explicitly run sweep index 1.
sbatch --export=ALL,SLURM_SWEEP_INDEX=1 run_qwen35_4b_answerability_slurm.sh

# Array sweep: one array task per dataset.
sbatch --array=0-2 --export=ALL run_qwen35_4b_answerability_slurm.sh
```

Useful overrides are the same as the local launcher. For example:

```bash
export MAX_TURNS_SWEEP="16 32"
sbatch --array=0-5 --export=ALL run_qwen35_4b_answerability_slurm.sh
```

To preserve the full reward setup while increasing reward-call parallelism:

```bash
sbatch --export=ALL,SLURM_SWEEP_INDEX=1,ANSWERER_REPEAT_CONCURRENCY=2 run_qwen35_4b_answerability_slurm.sh
```

Without `--array`, the wrapper passes `${SLURM_SWEEP_INDEX:-1}` to the local
launcher. Set `SLURM_SWEEP_INDEX=all` to run every selected sweep entry in the
same SLURM allocation, or pass a numeric first argument such as
`run_qwen35_4b_answerability_slurm.sh 0` to override the default.

## Launch Streaming-Summarization Pass@4 Baseline

```bash
cd /sailhome/teresaz/BrowseCompV2/exact_pipeline/tinker_synthetic_fs_current

export PASSK_K=4
export PASSK_SEED=2
export STREAM_MODEL=qwen/qwen3.5-35b-a3b

sbatch --export=ALL run_streaming_summary_passk_wandb.sh
```
