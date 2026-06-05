#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
SOURCE="${SOURCE:-hf-search}"
QUERY="${QUERY:-retrieval augmented generation}"
OUTPUT_DIR="${OUTPUT_DIR:-data/hf_paper_page_citation_graph}"
MAX_PAPERS="${MAX_PAPERS:-50}"
HF_SEARCH_LIMIT="${HF_SEARCH_LIMIT:-120}"
MAX_REFERENCES_PER_PAPER="${MAX_REFERENCES_PER_PAPER:-250}"
REQUEST_SLEEP="${REQUEST_SLEEP:-0.2}"
ENV_PATH="${ENV_PATH:-$REPO_ROOT/.env}"

ARGS=(
  --source "$SOURCE"
  --query "$QUERY"
  --max-papers "$MAX_PAPERS"
  --hf-search-limit "$HF_SEARCH_LIMIT"
  --max-references-per-paper "$MAX_REFERENCES_PER_PAPER"
  --request-sleep "$REQUEST_SLEEP"
  --env-path "$ENV_PATH"
  --output-dir "$OUTPUT_DIR"
)

if [[ "${FETCH_REFERENCE_METADATA:-0}" == "1" ]]; then
  ARGS+=(--fetch-reference-metadata)
fi

if [[ "${FETCH_REFERENCE_MARKDOWN:-0}" == "1" ]]; then
  ARGS+=(--fetch-reference-markdown)
fi

echo "Writing Hugging Face paper-page citation graph to: $OUTPUT_DIR"
echo "Source: $SOURCE"
echo "Query: $QUERY"
echo "Max papers: $MAX_PAPERS"
echo "Extra builder args: $*"

"$PYTHON" arxiv_paper/build_hf_paper_pages_citation_graph.py "${ARGS[@]}" "$@"

echo
echo "Done."
echo "Markdown files: $OUTPUT_DIR/paper_markdown/"
echo "Markdown index: $OUTPUT_DIR/markdown_index.md"
