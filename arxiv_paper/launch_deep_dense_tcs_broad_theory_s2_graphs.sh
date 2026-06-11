#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Broad theory-focused version of launch_deep_dense_tcs_theory_s2_graphs.sh.
#
# This wrapper keeps the same deep/backward-reference crawl budget as the deep
# dense TCS launcher, but uses the wider premier-TCS cluster list covering topic
# areas commonly represented at SODA/STOC/FOCS and adjacent theory venues.
#
# Override CLUSTER_REGEX to run a subset, for example:
#   CLUSTER_REGEX='^(graph_algorithms|approximation_hardness|fine_grained_complexity|computational_complexity)$'
#
# Override DEPTH_PRESET with depth3, depth5, depth10, or custom. The graph-shape
# defaults below mirror launch_deep_dense_tcs_s2_graphs.sh, while TARGET_TOTAL is
# scaled for many topic clusters.

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
    set_default OUTPUT_ROOT "data/arxiv_citation_graph_deep3_dense_tcs_broad_theory_s2"
    set_default TARGET_TOTAL "2200"
    set_default MAX_CANDIDATES "25000"
    set_default TARGET_NODE_COUNT "100000"
    set_default CITATION_DEPTH "3"
    set_default MAX_REFERENCES_PER_PAPER "100"
    set_default MAX_EXPANSION_PAPERS_PER_DEPTH "5000"
    set_default S2_SEARCH_MAX_PAGES "60"
    ;;
  depth5)
    set_default OUTPUT_ROOT "data/arxiv_citation_graph_deep5_dense_tcs_broad_theory_s2"
    set_default TARGET_TOTAL "1600"
    set_default MAX_CANDIDATES "30000"
    set_default TARGET_NODE_COUNT "750000"
    set_default CITATION_DEPTH "5"
    set_default MAX_REFERENCES_PER_PAPER "50"
    set_default MAX_EXPANSION_PAPERS_PER_DEPTH "3000"
    set_default S2_SEARCH_MAX_PAGES "80"
    ;;
  depth10)
    set_default OUTPUT_ROOT "data/arxiv_citation_graph_deep10_dense_tcs_broad_theory_s2"
    set_default TARGET_TOTAL "1000"
    set_default MAX_CANDIDATES "30000"
    set_default TARGET_NODE_COUNT "1250000"
    set_default CITATION_DEPTH "10"
    set_default MAX_REFERENCES_PER_PAPER "40"
    set_default MAX_EXPANSION_PAPERS_PER_DEPTH "2500"
    set_default S2_SEARCH_MAX_PAGES "100"
    ;;
  custom)
    set_default OUTPUT_ROOT "data/arxiv_citation_graph_deep_custom_dense_tcs_broad_theory_s2"
    set_default TARGET_TOTAL "1200"
    set_default MAX_CANDIDATES "25000"
    set_default TARGET_NODE_COUNT "750000"
    set_default CITATION_DEPTH "5"
    set_default MAX_REFERENCES_PER_PAPER "50"
    set_default MAX_EXPANSION_PAPERS_PER_DEPTH "2500"
    set_default S2_SEARCH_MAX_PAGES "80"
    ;;
  *)
    echo "Unknown DEPTH_PRESET=$DEPTH_PRESET. Use depth3, depth5, depth10, or custom." >&2
    exit 1
    ;;
esac

BROAD_THEORY_INCLUDE_DEFAULT="algorithm,algorithms,data structure,data structures,lower bound,lower bounds,hardness,hardness of approximation,inapproximability,complexity,approximation algorithm,approximation algorithms,online algorithm,online algorithms,dynamic algorithm,dynamic algorithms,streaming algorithm,streaming algorithms,sublinear algorithm,property testing,randomized algorithm,randomized algorithms,derandomization,pseudorandomness,graph algorithm,graph algorithms,shortest path,shortest paths,matching algorithm,matching algorithms,network flow,minimum cut,graph sparsification,matroid,submodular,combinatorial optimization,parameterized complexity,fixed parameter,kernelization,fine-grained complexity,conditional lower bound,conditional lower bounds,circuit complexity,proof complexity,communication complexity,query complexity,algebraic complexity,cryptography,zero knowledge,secure computation,multiparty computation,differential privacy,coding theory,error correcting code,list decoding,information theory,quantum algorithm,quantum algorithms,quantum complexity,quantum computing,algorithmic game theory,mechanism design,auction theory,social choice,fair division,market design,learning theory,PAC learning,sample complexity,online learning,bandit,regret,convex optimization,integer programming,linear programming,semidefinite programming,polyhedral combinatorics,computational geometry,discrete geometry,computational topology,distributed algorithm,distributed algorithms,consensus,parallel algorithm,parallel algorithms,external memory,automata,formal language,finite model theory,descriptive complexity,database theory,logic,semantics,type system,type systems,type theory,lambda calculus,model checking,program verification,formal methods,theorem proving,probabilistic method"
BROAD_THEORY_EXCLUDE_DEFAULT="dataset,datasets,benchmark,benchmarks,empirical,experiment,experiments,experimental,evaluation,survey,surveys,tutorial,tutorials,deep learning,neural,neural network,graph neural,transformer,large language model,language model,llm,diffusion,image,vision,object detection,segmentation,natural language processing,speech,medical,clinical,biology,genomics,proteomics,robotics,wireless,sensor,recommender,anomaly detection,case study,real world,real-world,implementation study,performance evaluation"

set_default CLUSTER_REGEX ""
set_default CLUSTER_LIMIT "0"
set_default MIN_CITATIONS "0"
set_default MIN_YEAR "2000"
set_default S2_SEARCH_SORT "publicationDate:desc"
set_default S2_SEARCH_BATCH_SIZE "1000"
set_default S2_BATCH_SIZE "100"
set_default S2_CONCURRENCY "1"
set_default S2_SLEEP "1"
set_default MAX_RETRIES "16"
set_default SKIP_EXISTING "1"
set_default CONTINUE_ON_ERROR "1"
set_default FALLBACK_ON_FAILURE "1"
set_default NO_FIELD_FALLBACK_ON_FAILURE "1"
set_default S2_PUBLICATION_TYPES "Conference,JournalArticle"
set_default SEED_PUBLICATION_TYPES "Conference,JournalArticle"
set_default REQUIRE_SEED_PUBLICATION "1"
set_default SEED_INCLUDE_KEYWORDS "$BROAD_THEORY_INCLUDE_DEFAULT"
set_default SEED_EXCLUDE_KEYWORDS "$BROAD_THEORY_EXCLUDE_DEFAULT"
set_default REFERENCE_EXCLUDE_KEYWORDS "$BROAD_THEORY_EXCLUDE_DEFAULT"

exec "$SCRIPT_DIR/launch_premier_tcs_s2_graphs.sh" "$@"
