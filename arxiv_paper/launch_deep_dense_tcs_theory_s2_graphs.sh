#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# More theory-focused version of launch_deep_dense_tcs_s2_graphs.sh.
#
# This wrapper keeps the same deep/backward-reference crawl machinery, but
# avoids broad seed queries such as "algorithms" that can pull in applied or
# empirical CS papers. It defaults to the complexity/lower-bounds cluster and
# adds strict title/abstract include/exclude filters for seed selection.
#
# Override CLUSTER_REGEX to run other dense clusters, for example:
#   CLUSTER_REGEX='^(algorithms_complexity|complexity_lower_bounds|crypto_complexity|logic_verification_pl)$'

THEORY_INCLUDE_DEFAULT="computational complexity,circuit complexity,proof complexity,communication complexity,lower bound,lower bounds,upper bound,upper bounds,hardness,hardness of approximation,approximation algorithm,randomized algorithm,derandomization,property testing,streaming algorithm,sublinear algorithm,parameterized complexity,fixed parameter,cryptography,zero knowledge,secure computation,pseudorandomness,automata,formal language,finite model theory,logic,model checking,program verification,type system,type systems,semantics,lambda calculus,information theory,coding theory,quantum complexity,quantum computing,algorithmic game theory,mechanism design,online learning,regret,bandit,convex optimization"
THEORY_EXCLUDE_DEFAULT="dataset,datasets,benchmark,benchmarks,empirical,experiment,experiments,experimental,evaluation,deep learning,neural,neural network,graph neural,transformer,large language model,language model,llm,diffusion,image,vision,object detection,segmentation,natural language processing,speech,medical,clinical,biology,genomics,proteomics,robotics,wireless,sensor,recommender,social network,anomaly detection,case study,real world,real-world"

export OUTPUT_ROOT="${OUTPUT_ROOT:-data/arxiv_citation_graph_deep_dense_tcs_theory_s2}"
export DEPTH_PRESET="${DEPTH_PRESET:-depth5}"
export CLUSTER_REGEX="${CLUSTER_REGEX:-^complexity_lower_bounds$}"
export CLUSTER_LIMIT="${CLUSTER_LIMIT:-0}"
export TARGET_TOTAL="${TARGET_TOTAL:-250}"
export MAX_CANDIDATES="${MAX_CANDIDATES:-25000}"
export MIN_CITATIONS="${MIN_CITATIONS:-0}"
export MIN_YEAR="${MIN_YEAR:-2000}"
export S2_SEARCH_SORT="${S2_SEARCH_SORT:-publicationDate:desc}"
export SEED_INCLUDE_KEYWORDS="${SEED_INCLUDE_KEYWORDS:-$THEORY_INCLUDE_DEFAULT}"
export SEED_EXCLUDE_KEYWORDS="${SEED_EXCLUDE_KEYWO[RDS:-$THEORY_EXCLUDE_DEFAULT}"
export REFERENCE_EXCLUDE_KEYWORDS="${REFERENCE_EXCLUDE_KEYWORDS:-$THEORY_EXCLUDE_DEFAULT}"

exec "$SCRIPT_DIR/launch_deep_dense_tcs_s2_graphs.sh" "$@"
