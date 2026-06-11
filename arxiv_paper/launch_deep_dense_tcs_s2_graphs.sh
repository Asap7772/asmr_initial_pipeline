#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Larger/deeper variant of launch_dense_tcs_s2_graphs.sh.
#
# Defaults are tuned for one broad TCS graph with substantially more depth than
# the dense launcher. DEPTH_PRESET controls the graph shape:
#   depth5  - default; large enough to reach several reference generations
#   depth10 - deeper sampled crawl with a larger node budget and narrower
#             branch factor, so the run can reach later hops
#   depth3  - older smaller deep default
#   custom  - only generic defaults are set; pass your own env values
#
# All variables remain overrideable from the environment.

set_default() {
  local name="$1"
  local value="$2"
  if [[ -z "${!name+x}" ]]; then
    export "$name=$value"
  fi
}

if [[ -z "${DEPTH_PRESET+x}" ]]; then
  case "${CITATION_DEPTH:-}" in
    10) export DEPTH_PRESET="depth10" ;;
    5) export DEPTH_PRESET="depth5" ;;
    3) export DEPTH_PRESET="depth3" ;;
    *) export DEPTH_PRESET="depth5" ;;
  esac
fi

case "$DEPTH_PRESET" in
  depth3)
    set_default OUTPUT_ROOT "data/arxiv_citation_graph_deep3_dense_tcs_s2"
    set_default TARGET_TOTAL "500"
    set_default MAX_CANDIDATES "10000"
    set_default TARGET_NODE_COUNT "100000"
    set_default CITATION_DEPTH "3"
    set_default MAX_REFERENCES_PER_PAPER "100"
    set_default MAX_EXPANSION_PAPERS_PER_DEPTH "5000"
    set_default S2_SEARCH_MAX_PAGES "50"
    ;;
  depth5)
    set_default OUTPUT_ROOT "data/arxiv_citation_graph_deep5_dense_tcs_s2"
    set_default TARGET_TOTAL "350"
    set_default MAX_CANDIDATES "20000"
    set_default TARGET_NODE_COUNT "750000"
    set_default CITATION_DEPTH "5"
    set_default MAX_REFERENCES_PER_PAPER "50"
    set_default MAX_EXPANSION_PAPERS_PER_DEPTH "3000"
    set_default S2_SEARCH_MAX_PAGES "80"
    ;;
  depth10)
    set_default OUTPUT_ROOT "data/arxiv_citation_graph_deep10_dense_tcs_s2"
    set_default TARGET_TOTAL "200"
    set_default MAX_CANDIDATES "25000"
    set_default TARGET_NODE_COUNT "1250000"
    set_default CITATION_DEPTH "10"
    set_default MAX_REFERENCES_PER_PAPER "40"
    set_default MAX_EXPANSION_PAPERS_PER_DEPTH "2500"
    set_default S2_SEARCH_MAX_PAGES "100"
    ;;
  custom)
    set_default OUTPUT_ROOT "data/arxiv_citation_graph_deep_custom_dense_tcs_s2"
    set_default TARGET_TOTAL "250"
    set_default MAX_CANDIDATES "10000"
    set_default TARGET_NODE_COUNT "250000"
    set_default CITATION_DEPTH "5"
    set_default MAX_REFERENCES_PER_PAPER "50"
    set_default MAX_EXPANSION_PAPERS_PER_DEPTH "2500"
    set_default S2_SEARCH_MAX_PAGES "50"
    ;;
  *)
    echo "Unknown DEPTH_PRESET=$DEPTH_PRESET. Use depth3, depth5, depth10, or custom." >&2
    exit 1
    ;;
esac

set_default MIN_CITATIONS "0"
set_default S2_SEARCH_BATCH_SIZE "1000"
set_default S2_SEARCH_SORT "citationCount:desc"
set_default S2_BATCH_SIZE "100"
set_default S2_CONCURRENCY "1"
set_default S2_SLEEP "1"
set_default MAX_RETRIES "16"
set_default MIN_YEAR "2005"
set_default CLUSTER_LIMIT "1"
set_default SKIP_EXISTING "1"
set_default CONTINUE_ON_ERROR "1"
set_default FALLBACK_ON_FAILURE "1"

export DEPTH_PRESET

exec "$SCRIPT_DIR/launch_dense_tcs_s2_graphs.sh" "$@"
