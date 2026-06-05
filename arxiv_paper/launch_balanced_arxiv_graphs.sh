#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
TARGET_TOTAL="${TARGET_TOTAL:-1000}"
MIN_CITATIONS="${MIN_CITATIONS:-100}"
MAX_ARXIV_RESULTS="${MAX_ARXIV_RESULTS:-2000}"
ARXIV_BATCH_SIZE="${ARXIV_BATCH_SIZE:-100}"
S2_BATCH_SIZE="${S2_BATCH_SIZE:-100}"
ARXIV_SLEEP="${ARXIV_SLEEP:-5}"
S2_SLEEP="${S2_SLEEP:-1}"
MAX_RETRIES="${MAX_RETRIES:-8}"
SORT_BY="${SORT_BY:-relevance}"
SORT_ORDER="${SORT_ORDER:-descending}"
ENV_PATH="${ENV_PATH:-$REPO_ROOT/.env}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/arxiv_citation_graph_balanced_1000}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

# Twenty representative arXiv categories across CS, statistics, math, EESS,
# economics, biology, finance, physics, astronomy, condensed matter, and HEP.
# TARGET_TOTAL=1000 gives a default quota of 50 seed papers per category.
CATEGORIES=(
  "cs.LG"
  "cs.CL"
  "cs.CV"
  "cs.AI"
  "stat.ML"
  "stat.ME"
  "math.OC"
  "math.PR"
  "math.ST"
  "eess.SP"
  "eess.IV"
  "econ.EM"
  "q-bio.QM"
  "q-fin.ST"
  "physics.data-an"
  "physics.comp-ph"
  "astro-ph.CO"
  "cond-mat.stat-mech"
  "quant-ph"
  "hep-th"
)

category_count="${#CATEGORIES[@]}"
base_quota="$((TARGET_TOTAL / category_count))"
remainder="$((TARGET_TOTAL % category_count))"

mkdir -p "$OUTPUT_ROOT"

echo "Writing balanced arXiv citation graphs under: $OUTPUT_ROOT"
echo "Target seed papers: $TARGET_TOTAL across $category_count categories"
echo "Minimum citations per seed paper: $MIN_CITATIONS"
echo "arXiv request sleep: ${ARXIV_SLEEP}s; max retries: $MAX_RETRIES"
echo "Extra builder args: $*"

for idx in "${!CATEGORIES[@]}"; do
  category="${CATEGORIES[$idx]}"
  quota="$base_quota"
  if (( idx < remainder )); then
    quota="$((quota + 1))"
  fi

  safe_category="${category//./_}"
  output_dir="$OUTPUT_ROOT/$safe_category"

  if [[ "$SKIP_EXISTING" == "1" && -f "$output_dir/manifest.json" ]]; then
    echo "Skipping $category; manifest already exists at $output_dir/manifest.json"
    continue
  fi

  echo
  echo "[$((idx + 1))/$category_count] $category -> quota $quota"
  "$PYTHON" arxiv_paper/build_arxiv_citation_graph.py \
    --query "cat:$category" \
    --max-arxiv-results "$MAX_ARXIV_RESULTS" \
    --arxiv-batch-size "$ARXIV_BATCH_SIZE" \
    --sort-by "$SORT_BY" \
    --sort-order "$SORT_ORDER" \
    --min-citations "$MIN_CITATIONS" \
    --max-seed-papers "$quota" \
    --s2-batch-size "$S2_BATCH_SIZE" \
    --arxiv-sleep "$ARXIV_SLEEP" \
    --s2-sleep "$S2_SLEEP" \
    --max-retries "$MAX_RETRIES" \
    --env-path "$ENV_PATH" \
    --output-dir "$output_dir" \
    "$@"
done

echo
echo "Done. Per-category manifests are in $OUTPUT_ROOT/*/manifest.json"
