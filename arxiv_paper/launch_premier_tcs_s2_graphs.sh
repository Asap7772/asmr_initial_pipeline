#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Broad, high-quality TCS launcher inspired by topic areas recurring across
# STOC/FOCS/SODA plus adjacent theory venues such as ICALP, CCC, LICS, SoCG,
# PODC, SPAA, ESA, COLT, TCC, and IPCO.
#
# Each cluster is intentionally a short S2 query. Long conjunctive queries tend
# to return very few arXiv-backed papers from S2 bulk search.
#
# DEPTH_PRESET controls graph size defaults:
#   depth3  - broad and cheaper
#   depth5  - default; broad with multi-hop parent structure
#   depth10 - deeper sampled crawl with fewer seeds per cluster
#   custom  - only generic defaults are set; pass your own env values

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
    set_default OUTPUT_ROOT "data/arxiv_citation_graph_premier_tcs_depth3_s2"
    set_default TARGET_TOTAL "2200"
    set_default TARGET_NODE_COUNT "50000"
    set_default CITATION_DEPTH "3"
    set_default MAX_REFERENCES_PER_PAPER "60"
    set_default MAX_EXPANSION_PAPERS_PER_DEPTH "2500"
    set_default S2_SEARCH_MAX_PAGES "60"
    ;;
  depth5)
    set_default OUTPUT_ROOT "data/arxiv_citation_graph_premier_tcs_depth5_s2"
    set_default TARGET_TOTAL "1600"
    set_default TARGET_NODE_COUNT "100000"
    set_default CITATION_DEPTH "5"
    set_default MAX_REFERENCES_PER_PAPER "45"
    set_default MAX_EXPANSION_PAPERS_PER_DEPTH "2500"
    set_default S2_SEARCH_MAX_PAGES "80"
    ;;
  depth10)
    set_default OUTPUT_ROOT "data/arxiv_citation_graph_premier_tcs_depth10_s2"
    set_default TARGET_TOTAL "1000"
    set_default TARGET_NODE_COUNT "180000"
    set_default CITATION_DEPTH "10"
    set_default MAX_REFERENCES_PER_PAPER "30"
    set_default MAX_EXPANSION_PAPERS_PER_DEPTH "2000"
    set_default S2_SEARCH_MAX_PAGES "100"
    ;;
  custom)
    set_default OUTPUT_ROOT "data/arxiv_citation_graph_premier_tcs_custom_s2"
    set_default TARGET_TOTAL "1200"
    set_default TARGET_NODE_COUNT "75000"
    set_default CITATION_DEPTH "5"
    set_default MAX_REFERENCES_PER_PAPER "45"
    set_default MAX_EXPANSION_PAPERS_PER_DEPTH "2000"
    set_default S2_SEARCH_MAX_PAGES "60"
    ;;
  *)
    echo "Unknown DEPTH_PRESET=$DEPTH_PRESET. Use depth3, depth5, depth10, or custom." >&2
    exit 1
    ;;
esac

PYTHON="${PYTHON:-python}"
MAX_CANDIDATES="${MAX_CANDIDATES:-30000}"
MIN_CITATIONS="${MIN_CITATIONS:-50}"
MAX_SEED_PAPERS="${MAX_SEED_PAPERS:-0}"
S2_BATCH_SIZE="${S2_BATCH_SIZE:-100}"
S2_CONCURRENCY="${S2_CONCURRENCY:-1}"
S2_SLEEP="${S2_SLEEP:-1}"
S2_SEARCH_BATCH_SIZE="${S2_SEARCH_BATCH_SIZE:-1000}"
S2_SEARCH_SORT="${S2_SEARCH_SORT:-citationCount:desc}"
MAX_RETRIES="${MAX_RETRIES:-16}"
ENV_PATH="${ENV_PATH:-$REPO_ROOT/.env}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
CLUSTER_REGEX="${CLUSTER_REGEX:-}"
CLUSTER_LIMIT="${CLUSTER_LIMIT:-0}"
MIN_YEAR="${MIN_YEAR:-2000}"
MAX_YEAR="${MAX_YEAR:-}"
ALLOW_MISSING_S2_KEY="${ALLOW_MISSING_S2_KEY:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
FALLBACK_ON_FAILURE="${FALLBACK_ON_FAILURE:-1}"
NO_FIELD_FALLBACK_ON_FAILURE="${NO_FIELD_FALLBACK_ON_FAILURE:-1}"
FAILURE_LOG="${FAILURE_LOG:-$OUTPUT_ROOT/failed_clusters.tsv}"
SUCCESS_LOG="${SUCCESS_LOG:-$OUTPUT_ROOT/succeeded_clusters.tsv}"

THEORY_INCLUDE_DEFAULT="algorithm,algorithms,data structure,data structures,lower bound,lower bounds,hardness,complexity,approximation algorithm,approximation algorithms,online algorithm,online algorithms,dynamic algorithm,dynamic algorithms,streaming algorithm,streaming algorithms,sublinear algorithm,property testing,randomized algorithm,randomized algorithms,derandomization,pseudorandomness,graph algorithm,graph algorithms,shortest path,matching algorithm,network flow,matroid,submodular,parameterized complexity,fixed parameter,fine-grained complexity,circuit complexity,proof complexity,communication complexity,query complexity,algebraic complexity,cryptography,zero knowledge,secure computation,multiparty computation,differential privacy,coding theory,error correcting code,information theory,quantum algorithm,quantum complexity,quantum computing,game theory,mechanism design,auction theory,social choice,fair division,learning theory,PAC learning,sample complexity,online learning,bandit,regret,optimization,convex optimization,combinatorial optimization,integer programming,linear programming,semidefinite programming,computational geometry,discrete geometry,computational topology,distributed algorithm,distributed algorithms,consensus,parallel algorithm,parallel algorithms,automata,formal language,finite model theory,descriptive complexity,database theory,logic,semantics,type system,type systems,type theory,lambda calculus,model checking,program verification,formal methods,theorem proving"
THEORY_EXCLUDE_DEFAULT="dataset,datasets,benchmark,benchmarks,empirical,experiment,experiments,experimental,evaluation,survey,surveys,tutorial,tutorials,deep learning,neural,neural network,graph neural,transformer,large language model,language model,llm,diffusion,image,vision,object detection,segmentation,natural language processing,speech,medical,clinical,biology,genomics,proteomics,robotics,wireless,sensor,recommender,anomaly detection,case study,real world,real-world,implementation study,performance evaluation"

S2_PUBLICATION_TYPES="${S2_PUBLICATION_TYPES:-Conference,JournalArticle}"
SEED_PUBLICATION_TYPES="${SEED_PUBLICATION_TYPES:-Conference,JournalArticle}"
REQUIRE_SEED_PUBLICATION="${REQUIRE_SEED_PUBLICATION:-1}"
SEED_INCLUDE_KEYWORDS="${SEED_INCLUDE_KEYWORDS:-$THEORY_INCLUDE_DEFAULT}"
SEED_EXCLUDE_KEYWORDS="${SEED_EXCLUDE_KEYWORDS:-$THEORY_EXCLUDE_DEFAULT}"
REFERENCE_INCLUDE_KEYWORDS="${REFERENCE_INCLUDE_KEYWORDS:-}"
REFERENCE_EXCLUDE_KEYWORDS="${REFERENCE_EXCLUDE_KEYWORDS:-$THEORY_EXCLUDE_DEFAULT}"

# Row format:
#   output-slug | Semantic Scholar keyword query | S2 field filter | fallback queries
PREMIER_TCS_CLUSTERS=(
  "algorithms_data_structures|data structures|Computer Science|algorithms;succinct data structures;lower bounds data structures"
  "graph_algorithms|graph algorithms|Computer Science|shortest paths;matching algorithms;network flow algorithms;dynamic graph algorithms"
  "shortest_paths_flows|shortest paths|Computer Science|maximum flow;minimum cut;graph sparsification"
  "matching_matroids|matching algorithms|Computer Science,Mathematics|matroid algorithms;submodular optimization;combinatorial optimization"
  "approximation_hardness|approximation algorithms|Computer Science|hardness of approximation;inapproximability;PCP theorem"
  "online_algorithms|online algorithms|Computer Science|competitive analysis;secretary problem;prophet inequalities"
  "streaming_sublinear|streaming algorithms|Computer Science|sublinear algorithms;property testing;sketching algorithms"
  "randomized_derandomization|randomized algorithms|Computer Science|derandomization;pseudorandomness;expander graphs"
  "dynamic_algorithms|dynamic algorithms|Computer Science|incremental algorithms;decremental algorithms;data structures"
  "fine_grained_complexity|fine-grained complexity|Computer Science|conditional lower bounds;SETH lower bounds;dynamic lower bounds"
  "parameterized_algorithms|parameterized complexity|Computer Science|fixed parameter tractability;kernelization;parameterized algorithms"
  "computational_complexity|computational complexity|Computer Science|complexity classes;hardness;lower bounds"
  "circuit_complexity|circuit complexity|Computer Science|Boolean circuits;AC0 lower bounds;lower bounds"
  "proof_complexity|proof complexity|Computer Science,Mathematics|proof systems;resolution lower bounds;sum of squares lower bounds"
  "communication_query_complexity|communication complexity|Computer Science|query complexity;decision tree complexity;lower bounds"
  "algebraic_complexity|algebraic complexity|Computer Science,Mathematics|arithmetic circuits;polynomial identity testing;geometric complexity theory"
  "quantum_algorithms_complexity|quantum algorithms|Computer Science,Physics|quantum complexity;quantum computing;quantum lower bounds"
  "cryptography_foundations|cryptography|Computer Science|zero knowledge;secure computation;cryptographic protocols"
  "zero_knowledge_proofs|zero knowledge proofs|Computer Science|interactive proofs;proof systems;succinct arguments"
  "secure_multiparty_computation|secure multiparty computation|Computer Science|secure computation;threshold cryptography;cryptographic protocols"
  "privacy_theory|differential privacy|Computer Science|privacy theory;private algorithms;privacy lower bounds"
  "coding_information_theory|coding theory|Computer Science,Mathematics|error correcting codes;list decoding;information theory"
  "algorithmic_game_theory|algorithmic game theory|Computer Science,Economics|mechanism design;auction theory;equilibrium computation"
  "social_choice_fair_division|computational social choice|Computer Science,Economics|fair division;voting theory;matching markets"
  "learning_theory|learning theory|Computer Science,Mathematics|PAC learning;sample complexity;generalization bounds"
  "online_learning_bandits|online learning|Computer Science,Mathematics|bandit algorithms;regret minimization;prediction with expert advice"
  "optimization_theory|convex optimization|Computer Science,Mathematics|combinatorial optimization;semidefinite programming;linear programming"
  "discrete_optimization_ipco|integer programming|Computer Science,Mathematics|polyhedral combinatorics;combinatorial optimization;cutting planes"
  "computational_geometry|computational geometry|Computer Science,Mathematics|discrete geometry;geometric algorithms;computational topology"
  "topology_algorithms|computational topology|Computer Science,Mathematics|persistent homology;geometric topology;topological algorithms"
  "distributed_computing|distributed algorithms|Computer Science|consensus;fault tolerance;distributed lower bounds"
  "parallel_external_memory|parallel algorithms|Computer Science|parallel complexity;external memory algorithms;PRAM algorithms"
  "automata_formal_languages|automata theory|Computer Science,Mathematics|formal languages;regular languages;tree automata"
  "logic_finite_model_theory|finite model theory|Computer Science,Mathematics|descriptive complexity;logic in computer science;database theory"
  "programming_language_semantics|programming language semantics|Computer Science|type systems;lambda calculus;program semantics"
  "verification_model_checking|model checking|Computer Science|program verification;formal methods;temporal logic"
  "database_theory|database theory|Computer Science|constraint satisfaction;query complexity;finite model theory"
  "theorem_proving_type_theory|type theory|Computer Science,Mathematics|proof theory;dependent types;theorem proving"
  "randomness_probabilistic_method|probabilistic method|Computer Science,Mathematics|random graphs;randomness in computation;pseudorandomness"
  "ml_foundations_fairness|foundations of machine learning|Computer Science,Mathematics|computational learning theory;fairness theory;learning-augmented algorithms"
  "market_design_matching|market design|Computer Science,Economics|matching theory;stable matching;mechanism design"
  "security_theory|algorithmic security privacy|Computer Science|cryptographic protocols;formal security;privacy theory"
)

SELECTED_CLUSTERS=()
for cluster in "${PREMIER_TCS_CLUSTERS[@]}"; do
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

echo "Writing premier TCS S2-search citation graphs under: $OUTPUT_ROOT"
echo "Target seed papers: $TARGET_TOTAL across $cluster_count clusters"
echo "Max candidates per cluster: $MAX_CANDIDATES"
echo "Min citations per seed: $MIN_CITATIONS"
echo "Seed publication types: ${SEED_PUBLICATION_TYPES:-none}; require publication signal: $REQUIRE_SEED_PUBLICATION"
echo "S2 search publication types: ${S2_PUBLICATION_TYPES:-none}"
echo "Graph shape: target_nodes=$TARGET_NODE_COUNT depth=$CITATION_DEPTH refs_per_paper=$MAX_REFERENCES_PER_PAPER expansion_cap=$MAX_EXPANSION_PAPERS_PER_DEPTH"
echo "S2 search: batch_size=$S2_SEARCH_BATCH_SIZE max_pages=$S2_SEARCH_MAX_PAGES sort=$S2_SEARCH_SORT"
echo "S2 batch: batch_size=$S2_BATCH_SIZE concurrency=$S2_CONCURRENCY sleep=${S2_SLEEP}s"
echo "Year filter: min=${MIN_YEAR:-none} max=${MAX_YEAR:-none}"
echo "Depth preset: $DEPTH_PRESET"
echo "Skip existing: $SKIP_EXISTING; dry run: $DRY_RUN; continue on error: $CONTINUE_ON_ERROR"
echo "Fallback on failure: $FALLBACK_ON_FAILURE; no-field fallback: $NO_FIELD_FALLBACK_ON_FAILURE"
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

  if [[ -n "$fields_of_study" ]]; then
    cmd+=(--s2-fields-of-study "$fields_of_study")
  fi
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
  echo "Fields: ${fields_of_study:-none}"

  if run_cluster "$slug" "$query" "$fields_of_study" "$quota" "$output_dir" "$@"; then
    printf "%s\t%s\t%s\n" "$slug" "$query" "$output_dir" >> "$SUCCESS_LOG"
    continue
  fi

  retry_status=1
  retry_queries=("$query")
  if [[ -n "$fallback_queries" ]]; then
    IFS=";" read -r -a fallback_query_array <<< "$fallback_queries"
    for fallback_query in "${fallback_query_array[@]}"; do
      if [[ -n "$fallback_query" ]]; then
        retry_queries+=("$fallback_query")
      fi
    done
  fi

  if [[ "$FALLBACK_ON_FAILURE" == "1" ]]; then
    for retry_query in "${retry_queries[@]:1}"; do
      echo "Primary run failed for $slug; retrying with fallback query: $retry_query"
      if run_cluster "$slug" "$retry_query" "$fields_of_study" "$quota" "$output_dir" "$@"; then
        printf "%s\t%s\t%s\tfallback_query\n" "$slug" "$retry_query" "$output_dir" >> "$SUCCESS_LOG"
        retry_status=0
        break
      fi
    done
  fi

  if [[ "$retry_status" != "0" && "$NO_FIELD_FALLBACK_ON_FAILURE" == "1" ]]; then
    for retry_query in "${retry_queries[@]}"; do
      echo "Field-filtered run failed for $slug; retrying without fields: $retry_query"
      if run_cluster "$slug" "$retry_query" "" "$quota" "$output_dir" "$@"; then
        printf "%s\t%s\t%s\tfallback_no_fields\n" "$slug" "$retry_query" "$output_dir" >> "$SUCCESS_LOG"
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
echo "Done. Premier TCS cluster manifests are in $OUTPUT_ROOT/*/manifest.json"
echo "Success log: $SUCCESS_LOG"
echo "Failure log: $FAILURE_LOG"
