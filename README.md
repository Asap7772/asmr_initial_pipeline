# ASMR Private

Research utilities for filesystem-style question answering, retrieval/reranking experiments, Modal-hosted vLLM serving, and Tinker RL runs over synthetic filesystem tasks.

This repository contains private benchmark data and generated artifacts. Treat `decrypted.jsonl`, `data/`, `retrieval/*.jsonl`, `inference/*.jsonl`, and `tinker_runs/` as local/private experiment outputs unless you have explicitly cleared them for sharing.

## Repository Layout

- `scripts/decrypt_browsecomp.py`: decrypts `Tevatron/browsecomp-plus` into `decrypted.jsonl`.
- `inference/make_filesystem.py`: converts decrypted BrowseComp+ rows into filesystem task directories.
- `inference/collect_baseline.py`: runs tool-calling file-reading inference against `data/train`.
- `inference/collect_llm.py`: provider-agnostic async LLM client helpers with prompt caching.
- `proc_bright.py`: builds BRIGHT gold/negative filesystem tasks under `data/bright`.
- `retrieval/retrieve_proc_heldout.py`: retrieves documents with Qwen embedding models through vLLM.
- `retrieval/retrieve_proc_heldout_moderncolbert.py`: retrieves documents with LightOn Reason-ModernColBERT through PyLate.
- `retrieval/reranker_proc_heldout.py`: retrieves then reranks with a Qwen reranker through vLLM.
- `retrieval/answer_proc_heldout.py`: answers retrieved/reranked records and optionally judges against gold answers.
- `modal_launch_vllm.py`: configurable Modal vLLM deployment entrypoint.
- `modal_launch_vllm_qwen35_35b_a3b_long_context.py`: Modal deployment for Qwen3.5-35B-A3B with a 262k-token context.
- `deploy_qwen35_35b_a3b_long_context_262k.sh`: convenience wrapper for the long-context Modal deployment.
- `arxiv_paper/`: arXiv and Hugging Face Paper Pages citation graph builders. See `arxiv_paper/README.md`.
- `prompts/`: prompt templates for cluster-bank and best-of-many workflows.
- `tinker_synthetic_fs_current/`: Tinker synthetic filesystem RL package. See its local README for detailed training and evaluation commands.
- `tinker_runs/`: local run outputs, logs, metrics, configs, code diffs, and checkpoints.

## Environment

Start from a Python environment with GPU/CUDA support when using vLLM:

```bash
python -m venv .venv
source .venv/bin/activate
pip install datasets pandas tqdm openai torch transformers vllm modal
pip install pylate
```

Copy the environment template and fill in only the services you use:

```bash
cp .env.example .env
source .env
```

Common keys:

- `HF_TOKEN`: Hugging Face model and dataset access.
- `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `TOGETHER_API_KEY`, `ANTHROPIC_API_KEY`: API inference and judging.
- `WANDB_API_KEY`: experiment tracking.
- `TINKER_API_KEY`: Tinker training runs.

The Tinker subproject has its own dependency path:

```bash
cd tinker_synthetic_fs_current
./setup_local_tinker_env.sh
```

## Data Layout

The filesystem tasks use two parallel views:

- `data/train/<question_id>/*.txt`: files visible to an agent or retriever.
- `data/train_privileged/<question_id>/query.txt`: the hidden question text.
- `data/train_privileged/<question_id>/answer.txt`: the hidden gold answer.
- `data/train_privileged/<question_id>/manifest.json`: document metadata and labels.
- `data/train_privileged/<question_id>/{evidence_docs,gold_docs,negative_docs}/`: labeled privileged copies.

BRIGHT-derived tasks follow the same shape under `data/bright/train` and `data/bright/train_privileged`.

Tinker index directories such as `data/tinker_synthetic_fs_alltrain*/` contain generated `index.jsonl` and `manifest.json` files for training/eval launchers.

## Build BrowseComp+ Filesystem Tasks

Decrypt BrowseComp+ into local JSONL:

```bash
python scripts/decrypt_browsecomp.py
```

Convert the decrypted rows into the agent/privileged filesystem layout:

```bash
python -m inference.make_filesystem
```

`inference/make_filesystem.py` currently uses repo-local absolute paths for `SOURCE_FILE`, `AGENT_SAVE_DIR`, and `PRIVILEGED_SAVE_DIR`.

## Build BRIGHT Filesystem Tasks

Create a BRIGHT filesystem sample targeting up to 50 documents per domain. The
selected gold documents are always included, even if they exceed the target, and
any remaining slots are filled with noisy documents from the same BRIGHT domain:

```bash
python proc_bright.py --output-root data/bright --queries-per-domain 10 --max-documents-per-domain 50 --seed 0
```

Optional domain filtering:

```bash
python proc_bright.py --output-root data/bright --queries-per-domain 5 --domains biology economics robotics
```

To build the restricted gold-document corpus without added noisy documents:

```bash
python proc_bright.py --output-root data/bright --queries-per-domain 10 --gold-docs-only --seed 0
```

The script writes `train/`, `train_privileged/`, and `heldout_<n>_questions.json` under the selected output root.

## Run File-Reading Baseline

Run iterative tool-calling inference over `data/train`:

```bash
python -m inference.collect_baseline \
  --provider openrouter \
  --model-name qwen/qwen3.5-35b-a3b \
  --N 4 \
  --max-concurrent 4 \
  --resume
```

Useful controls:

- `--query-id <id>`: run only one question; may be repeated.
- `--limit <n>` or `--max-problems <n>`: restrict the run.
- `--max-steps <n>`: cap file-read iterations.
- `--output-path <path>`: default is `inference/single_layer_outputs.jsonl`.

## Run Retrieval, Reranking, And Answering

Embedding-only retrieval:

```bash
python -m retrieval.retrieve_proc_heldout \
  --questions-path data/heldout_50_questions.json \
  --docs-dir data/train \
  --privileged-dir data/train_privileged \
  --top-k 5 \
  --output-path retrieval/heldout_retrieval_supponly.jsonl
```

Reason-ModernColBERT late-interaction retrieval:

```bash
python -m retrieval.retrieve_proc_heldout_moderncolbert \
  --questions-path data/heldout_50_questions.json \
  --docs-dir data/train \
  --privileged-dir data/train_privileged \
  --top-k 5 \
  --output-path retrieval/heldout_retrieval_reason_moderncolbert.jsonl
```

Retrieve then rerank:

```bash
python -m retrieval.reranker_proc_heldout \
  --questions-path data/heldout_50_questions.json \
  --docs-dir data/train \
  --privileged-dir data/train_privileged \
  --retrieval-top-k 10 \
  --rerank-top-k 5 \
  --output-path retrieval/heldout_retrieval_reranked_supponly.jsonl
```

Answer and judge retrieved records:

```bash
python -m retrieval.answer_proc_heldout \
  --input-path retrieval/heldout_retrieval_reranked_supponly.jsonl \
  --output-path retrieval/heldout_answers_reranked_supponly_gemini.jsonl \
  --provider gemini \
  --judge \
  --resume
```

Retrieval and reranking load local vLLM models and generally expect CUDA-visible GPUs. Answering defaults to a non-local API answerer and Gemini judging; configure provider keys in `.env`.

## Modal vLLM Serving

Deploy the default Qwen3.5-35B-A3B long-context endpoint:

```bash
./deploy_qwen35_35b_a3b_long_context_262k.sh
```

Or deploy the configurable entrypoint with explicit overrides:

```bash
MODAL_VLLM_MODEL_NAME=Qwen/Qwen3.5-4B \
MODAL_VLLM_APP_NAME=lateral-vllm-qwen35-4b \
MODAL_VLLM_WEB_LABEL=lateral-vllm-qwen35-4b \
MODAL_VLLM_REQUIRES_PROXY_AUTH=0 \
MODAL_VLLM_MIN_CONTAINERS=1 \
modal deploy --name lateral-vllm-qwen35-4b modal_launch_vllm.py
```

Both launchers expose an OpenAI-compatible `/v1/chat/completions` endpoint and use the served model alias `llm` by default.

## Tinker Synthetic Filesystem Runs

Detailed Tinker workflow notes live in `tinker_synthetic_fs_current/README.md`. The common local entrypoint is:

```bash
cd tinker_synthetic_fs_current
export MODEL_NAME=Qwen/Qwen3.5-4B
./run_qwen35_4b_answerability_local.sh
```

For SLURM:

```bash
cd tinker_synthetic_fs_current
sbatch --array=0-2 --export=ALL run_qwen35_4b_answerability_slurm.sh
```

Run outputs are written under `tinker_runs/` by default.

## Notes

- Several scripts intentionally write JSONL outputs in place; use `--output-path` when comparing runs.
- `--resume` skips already completed question IDs or sample IDs for supported scripts.
- `inference.collect_llm` caches provider responses under `inference/.prompt_cache_*`, which is ignored by git.
- The checked-in notebooks are for inspection and plotting; the command-line scripts are the canonical run paths.
