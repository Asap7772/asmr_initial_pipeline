#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
TARGET_TOTAL="${TARGET_TOTAL:-800}"
MAX_CANDIDATES="${MAX_CANDIDATES:-5000}"
MIN_CITATIONS="${MIN_CITATIONS:-0}"
MAX_SEED_PAPERS="${MAX_SEED_PAPERS:-0}"
TARGET_NODE_COUNT="${TARGET_NODE_COUNT:-10000}"
CITATION_DEPTH="${CITATION_DEPTH:-2}"
MAX_REFERENCES_PER_PAPER="${MAX_REFERENCES_PER_PAPER:-80}"
MAX_EXPANSION_PAPERS_PER_DEPTH="${MAX_EXPANSION_PAPERS_PER_DEPTH:-3000}"
S2_BATCH_SIZE="${S2_BATCH_SIZE:-100}"
S2_CONCURRENCY="${S2_CONCURRENCY:-1}"
S2_SLEEP="${S2_SLEEP:-1}"
S2_SEARCH_BATCH_SIZE="${S2_SEARCH_BATCH_SIZE:-1000}"
S2_SEARCH_MAX_PAGES="${S2_SEARCH_MAX_PAGES:-20}"
S2_SEARCH_SORT="${S2_SEARCH_SORT:-citationCount:desc}"
MAX_RETRIES="${MAX_RETRIES:-10}"
ENV_PATH="${ENV_PATH:-$REPO_ROOT/.env}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/arxiv_citation_graph_dense_tcs_s2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
CLUSTER_REGEX="${CLUSTER_REGEX:-}"
CLUSTER_LIMIT="${CLUSTER_LIMIT:-1}"
S2_PUBLICATION_TYPES="${S2_PUBLICATION_TYPES:-}"
SEED_PUBLICATION_TYPES="${SEED_PUBLICATION_TYPES:-}"
REQUIRE_SEED_PUBLICATION="${REQUIRE_SEED_PUBLICATION:-0}"
SEED_INCLUDE_KEYWORDS="${SEED_INCLUDE_KEYWORDS:-}"
SEED_EXCLUDE_KEYWORDS="${SEED_EXCLUDE_KEYWORDS:-}"
REFERENCE_INCLUDE_KEYWORDS="${REFERENCE_INCLUDE_KEYWORDS:-}"
REFERENCE_EXCLUDE_KEYWORDS="${REFERENCE_EXCLUDE_KEYWORDS:-}"
MIN_YEAR="${MIN_YEAR:-2014}"
MAX_YEAR="${MAX_YEAR:-}"
ALLOW_MISSING_S2_KEY="${ALLOW_MISSING_S2_KEY:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
FALLBACK_ON_FAILURE="${FALLBACK_ON_FAILURE:-1}"
FAILURE_LOG="${FAILURE_LOG:-$OUTPUT_ROOT/failed_clusters.tsv}"
SUCCESS_LOG="${SUCCESS_LOG:-$OUTPUT_ROOT/succeeded_clusters.tsv}"

# Dense TCS clusters are intentionally broad and overlapping. Each row is:
#   output-slug | Semantic Scholar keyword query | Semantic Scholar field filter
#
# The graph builder orients reference edges as:
#   cited parent/reference paper -> citing seed/frontier paper
#
# These broad recent-paper clusters are designed to place many related "child"
# seed papers around shared foundational parents.
DENSE_TCS_CLUSTERS=(
  "tcs_core|algorithms|Computer Science|computational complexity;cryptography;formal methods;programming languages;graph theory;information theory"
  "algorithms_complexity|computational complexity|Computer Science|algorithms;approximation algorithms;randomized algorithms;streaming algorithms;property testing"
  "complexity_lower_bounds|circuit complexity|Computer Science|computational complexity;proof complexity;communication complexity;lower bounds"
  "crypto_complexity|cryptography|Computer Science|zero knowledge;secure computation;pseudorandomness;cryptographic protocols"
  "learning_optimization_theory|learning theory|Computer Science|online learning;bandits;reinforcement learning theory;convex optimization;generalization"
  "discrete_math_algorithms|graph theory|Computer Science,Mathematics|combinatorics;combinatorial optimization;computational geometry;discrete mathematics"
  "logic_verification_pl|formal methods|Computer Science|model checking;program verification;type theory;programming languages;automata theory"
  "information_coding_quantum_tcs|information theory|Computer Science|coding theory;quantum information;quantum computing;quantum algorithms"
  "game_mechanism_tcs|algorithmic game theory|Computer Science,Economics|mechanism design;auction theory;social choice;equilibrium computation"
  "distributed_parallel_tcs|distributed computing|Computer Science|parallel algorithms;consensus;fault tolerance;streaming algorithms"
)

SELECTED_CLUSTERS=()
for cluster in "${DENSE_TCS_CLUSTERS[@]}"; do
  IFS="|" read -r slug _query _fields _fallbacks <<< "$cluster"
  if [[ -n "$CLUSTER_REGEX" && ! "$slug" =~ $CLUSTER_REGEX ]]; then
    continue
  fi
  SELECTED_CLUSTERS+=("$cluster")
  if (( CLUSTER_LIMIT > 0 && ${#SELECTED_CLUSTERS[@]} >= CLUSTER_LIMIT )); then
    break
  fi
done

cluster_count="${#SELECTED_CLUSTERS[@]}"
if (( cluster_count == 0 )); then
  echo "No clusters selected. Check CLUSTER_REGEX=$CLUSTER_REGEX or CLUSTER_LIMIT=$CLUSTER_LIMIT." >&2
  exit 1
fi

base_quota="$((TARGET_TOTAL / cluster_count))"
remainder="$((TARGET_TOTAL % cluster_count))"

mkdir -p "$OUTPUT_ROOT"
: > "$SUCCESS_LOG"
: > "$FAILURE_LOG"

echo "Writing dense TCS S2-search citation graphs under: $OUTPUT_ROOT"
echo "Target seed papers: $TARGET_TOTAL across $cluster_count dense clusters"
echo "Max candidates per cluster: $MAX_CANDIDATES"
echo "Min citations per seed: $MIN_CITATIONS"
echo "Graph shape: target_nodes=$TARGET_NODE_COUNT depth=$CITATION_DEPTH refs_per_paper=$MAX_REFERENCES_PER_PAPER expansion_cap=$MAX_EXPANSION_PAPERS_PER_DEPTH"
echo "S2 search: batch_size=$S2_SEARCH_BATCH_SIZE max_pages=$S2_SEARCH_MAX_PAGES sort=$S2_SEARCH_SORT"
echo "S2 batch: batch_size=$S2_BATCH_SIZE concurrency=$S2_CONCURRENCY sleep=${S2_SLEEP}s"
echo "Year filter: min=${MIN_YEAR:-none} max=${MAX_YEAR:-none}"
if [[ -n "$S2_PUBLICATION_TYPES" ]]; then
  echo "S2 search publication types: $S2_PUBLICATION_TYPES"
fi
if [[ -n "$SEED_PUBLICATION_TYPES" ]]; then
  echo "Seed publication types: $SEED_PUBLICATION_TYPES"
fi
if [[ "$REQUIRE_SEED_PUBLICATION" == "1" ]]; then
  echo "Require seed publication signal: 1"
fi
if [[ -n "$SEED_INCLUDE_KEYWORDS" ]]; then
  echo "Seed include keywords: $SEED_INCLUDE_KEYWORDS"
fi
if [[ -n "$SEED_EXCLUDE_KEYWORDS" ]]; then
  echo "Seed exclude keywords: $SEED_EXCLUDE_KEYWORDS"
fi
if [[ -n "$REFERENCE_INCLUDE_KEYWORDS" ]]; then
  echo "Reference include keywords: $REFERENCE_INCLUDE_KEYWORDS"
fi
if [[ -n "$REFERENCE_EXCLUDE_KEYWORDS" ]]; then
  echo "Reference exclude keywords: $REFERENCE_EXCLUDE_KEYWORDS"
fi
if [[ -n "${DEPTH_PRESET:-}" ]]; then
  echo "Depth preset: $DEPTH_PRESET"
fi
echo "Skip existing: $SKIP_EXISTING; dry run: $DRY_RUN; continue on error: $CONTINUE_ON_ERROR; fallback on failure: $FALLBACK_ON_FAILURE"
echo "Extra builder args: $*"

run_cluster() {
  local slug="$1"
  local query="$2"
  local fields_of_study="$3"
  local quota="$4"
  local output_dir="$5"
  shift 5

  local cmd=(
    "$PYTHON" arxiv_paper/build_arxiv_citation_graph.py
    --candidate-source s2-search
    --query "$query"
    --s2-search-query "$query"
    --s2-fields-of-study "$fields_of_study"
    --s2-search-sort "$S2_SEARCH_SORT"
    --s2-search-batch-size "$S2_SEARCH_BATCH_SIZE"
    --s2-search-max-pages "$S2_SEARCH_MAX_PAGES"
    --max-arxiv-results "$MAX_CANDIDATES"
    --min-citations "$MIN_CITATIONS"
    --max-seed-papers "$quota"
    --target-node-count "$TARGET_NODE_COUNT"
    --citation-depth "$CITATION_DEPTH"
    --max-references-per-paper "$MAX_REFERENCES_PER_PAPER"
    --max-expansion-papers-per-depth "$MAX_EXPANSION_PAPERS_PER_DEPTH"
    --s2-batch-size "$S2_BATCH_SIZE"
    --s2-concurrency "$S2_CONCURRENCY"
    --s2-sleep "$S2_SLEEP"
    --max-retries "$MAX_RETRIES"
    --env-path "$ENV_PATH"
    --output-dir "$output_dir"
  )

  if [[ -n "$S2_PUBLICATION_TYPES" ]]; then
    cmd+=(--s2-publication-types "$S2_PUBLICATION_TYPES")
  fi
  if [[ -n "$SEED_PUBLICATION_TYPES" ]]; then
    cmd+=(--seed-publication-types "$SEED_PUBLICATION_TYPES")
  fi
  if [[ "$REQUIRE_SEED_PUBLICATION" == "1" ]]; then
    cmd+=(--require-seed-publication)
  fi
  if [[ -n "$SEED_INCLUDE_KEYWORDS" ]]; then
    cmd+=(--seed-include-keywords "$SEED_INCLUDE_KEYWORDS")
  fi
  if [[ -n "$SEED_EXCLUDE_KEYWORDS" ]]; then
    cmd+=(--seed-exclude-keywords "$SEED_EXCLUDE_KEYWORDS")
  fi
  if [[ -n "$REFERENCE_INCLUDE_KEYWORDS" ]]; then
    cmd+=(--reference-include-keywords "$REFERENCE_INCLUDE_KEYWORDS")
  fi
  if [[ -n "$REFERENCE_EXCLUDE_KEYWORDS" ]]; then
    cmd+=(--reference-exclude-keywords "$REFERENCE_EXCLUDE_KEYWORDS")
  fi
  if [[ -n "$MIN_YEAR" ]]; then
    cmd+=(--min-year "$MIN_YEAR")
  fi
  if [[ -n "$MAX_YEAR" ]]; then
    cmd+=(--max-year "$MAX_YEAR")
  fi
  if [[ "$ALLOW_MISSING_S2_KEY" == "1" ]]; then
    cmd+=(--allow-missing-s2-key)
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    printf "Command:"
    printf " %q" "${cmd[@]}" "$@"
    printf "\n"
    return 0
  fi

  "${cmd[@]}" "$@"
}

for idx in "${!SELECTED_CLUSTERS[@]}"; do
  IFS="|" read -r slug query fields_of_study fallback_queries <<< "${SELECTED_CLUSTERS[$idx]}"

  if (( MAX_SEED_PAPERS > 0 )); then
    quota="$MAX_SEED_PAPERS"
  else
    quota="$base_quota"
    if (( idx < remainder )); then
      quota="$((quota + 1))"
    fi
  fi
  if (( quota > MAX_CANDIDATES )); then
    quota="$MAX_CANDIDATES"
  fi

  output_dir="$OUTPUT_ROOT/$slug"
  if [[ "$SKIP_EXISTING" == "1" && -f "$output_dir/manifest.json" ]]; then
    echo "Skipping $slug; manifest already exists at $output_dir/manifest.json"
    continue
  fi

  echo
  echo "[$((idx + 1))/$cluster_count] $slug -> seed quota $quota"
  echo "Query: $query"
  echo "Fields: $fields_of_study"

  if run_cluster "$slug" "$query" "$fields_of_study" "$quota" "$output_dir" "$@"; then
    printf "%s\t%s\t%s\n" "$slug" "$query" "$output_dir" >> "$SUCCESS_LOG"
    continue
  fi

  retry_status=1
  if [[ "$FALLBACK_ON_FAILURE" == "1" && -n "$fallback_queries" ]]; then
    IFS=";" read -r -a fallback_query_array <<< "$fallback_queries"
    for fallback_query in "${fallback_query_array[@]}"; do
      if [[ -z "$fallback_query" ]]; then
        continue
      fi
      echo "Primary run failed for $slug; retrying with fallback query: $fallback_query"
      if run_cluster "$slug" "$fallback_query" "$fields_of_study" "$quota" "$output_dir" "$@"; then
        printf "%s\t%s\t%s\tfallback_query\n" "$slug" "$fallback_query" "$output_dir" >> "$SUCCESS_LOG"
        retry_status=0
        break
      fi
    done
  fi

  if [[ "$retry_status" == "0" ]]; then
    continue
  fi

  printf "%s\t%s\t%s\n" "$slug" "$query" "$output_dir" >> "$FAILURE_LOG"
  if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
    echo "Continuing after failed cluster: $slug"
  else
    echo "Stopping after failed cluster: $slug" >&2
    exit 1
  fi
done

echo
echo "Done. Dense cluster manifests are in $OUTPUT_ROOT/*/manifest.json"
echo "Success log: $SUCCESS_LOG"
echo "Failure log: $FAILURE_LOG"
