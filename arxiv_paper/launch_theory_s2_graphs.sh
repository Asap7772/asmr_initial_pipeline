#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
TARGET_TOTAL="${TARGET_TOTAL:-2400}"
MIN_CITATIONS="${MIN_CITATIONS:-50}"
MAX_CANDIDATES="${MAX_CANDIDATES:-1000}"
S2_BATCH_SIZE="${S2_BATCH_SIZE:-100}"
S2_CONCURRENCY="${S2_CONCURRENCY:-1}"
S2_SLEEP="${S2_SLEEP:-1}"
S2_SEARCH_BATCH_SIZE="${S2_SEARCH_BATCH_SIZE:-1000}"
S2_SEARCH_MAX_PAGES="${S2_SEARCH_MAX_PAGES:-10}"
S2_SEARCH_SORT="${S2_SEARCH_SORT:-citationCount:desc}"
MAX_RETRIES="${MAX_RETRIES:-10}"
ENV_PATH="${ENV_PATH:-$REPO_ROOT/.env}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/arxiv_citation_graph_theory_s2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
DOMAIN_REGEX="${DOMAIN_REGEX:-}"
DOMAIN_LIMIT="${DOMAIN_LIMIT:-0}"
S2_PUBLICATION_TYPES="${S2_PUBLICATION_TYPES:-}"
MIN_YEAR="${MIN_YEAR:-}"
MAX_YEAR="${MAX_YEAR:-}"
ALLOW_MISSING_S2_KEY="${ALLOW_MISSING_S2_KEY:-0}"

# Mostly theoretical seed domains, chosen to resemble arXiv theory-heavy
# subject areas rather than empirical/application-first areas. Each row is:
#   output-slug | Semantic Scholar keyword query | Semantic Scholar field filter
THEORY_DOMAINS=(
  # CS theory and formal foundations
  "algorithms|algorithms data structures computational complexity|Computer Science"
  "data_structures|data structures succinct data structures lower bounds|Computer Science"
  "complexity_theory|computational complexity theory lower bounds|Computer Science"
  "parameterized_complexity|parameterized complexity fixed parameter tractability kernelization|Computer Science"
  "proof_complexity|proof complexity circuit complexity lower bounds|Computer Science"
  "randomized_algorithms|randomized algorithms probabilistic method derandomization|Computer Science"
  "approximation_algorithms|approximation algorithms hardness of approximation|Computer Science"
  "sublinear_algorithms|sublinear algorithms streaming algorithms property testing|Computer Science"
  "computational_geometry|computational geometry discrete geometry algorithms|Computer Science"
  "distributed_computing_theory|distributed computing theory consensus lower bounds|Computer Science"
  "parallel_algorithms|parallel algorithms parallel complexity|Computer Science"
  "database_theory|database theory finite model theory query languages|Computer Science"
  "algorithmic_game_theory|algorithmic game theory mechanism design equilibria|Computer Science"
  "computational_social_choice|computational social choice voting theory|Computer Science"
  "multiagent_theory|multiagent systems game theory equilibrium theory|Computer Science"
  "cryptography|cryptography secure computation zero knowledge|Computer Science"
  "secure_computation|secure multiparty computation cryptographic protocols|Computer Science"
  "zero_knowledge|zero knowledge proofs proof systems cryptography|Computer Science"
  "differential_privacy_theory|differential privacy theory privacy guarantees|Computer Science"
  "coding_theory|coding theory error correcting codes|Computer Science"
  "information_theory|information theory entropy channel coding|Computer Science"
  "formal_methods|formal methods model checking program verification|Computer Science"
  "program_verification|program verification Hoare logic separation logic|Computer Science"
  "model_checking|model checking temporal logic verification|Computer Science"
  "programming_languages|programming languages type systems semantics|Computer Science"
  "type_theory|type theory dependent types lambda calculus|Computer Science"
  "program_semantics|program semantics denotational semantics operational semantics|Computer Science"
  "automata_logic|automata theory formal languages finite model theory|Computer Science"
  "logic_in_cs|logic in computer science proof theory type theory|Computer Science"
  "symbolic_computation|symbolic computation computer algebra algorithms|Computer Science"

  # ML/statistical theory, without empirical benchmark domains
  "learning_theory|statistical learning theory generalization bounds|Computer Science"
  "computational_learning_theory|computational learning theory PAC learning sample complexity|Computer Science"
  "online_learning|online learning regret minimization bandit theory|Computer Science"
  "bandit_theory|multi armed bandits regret lower bounds|Computer Science"
  "rl_theory|reinforcement learning theory Markov decision processes|Computer Science"
  "causal_theory|causal inference identification graphical models theory|Computer Science"
  "probabilistic_graphical_models|probabilistic graphical models variational inference theory|Computer Science"
  "bayesian_theory|Bayesian theory posterior contraction nonparametric Bayes|Mathematics"
  "statistical_decision_theory|statistical decision theory minimax risk|Mathematics"
  "information_geometry|information geometry statistical manifolds|Mathematics"

  # Optimization, control, and operations research theory
  "optimization_theory|convex optimization duality variational analysis|Mathematics"
  "nonconvex_optimization_theory|nonconvex optimization landscape convergence theory|Mathematics"
  "discrete_optimization|combinatorial optimization integer programming polyhedra|Mathematics"
  "optimal_transport|optimal transport Wasserstein geometry|Mathematics"
  "variational_analysis|variational analysis monotone operators convex analysis|Mathematics"
  "control_theory|control theory optimal control dynamical systems|Mathematics,Engineering"
  "systems_theory|systems theory stability controllability observability|Engineering"

  # Discrete math, probability, and statistics
  "graph_theory|graph theory extremal combinatorics|Mathematics"
  "combinatorics|combinatorics extremal set theory|Mathematics"
  "extremal_combinatorics|extremal combinatorics Ramsey theory|Mathematics"
  "algebraic_combinatorics|algebraic combinatorics symmetric functions|Mathematics"
  "additive_combinatorics|additive combinatorics sumsets arithmetic progressions|Mathematics"
  "probability_theory|probability theory stochastic processes martingales|Mathematics"
  "stochastic_processes|stochastic processes Markov processes martingales|Mathematics"
  "random_matrices|random matrix theory spectral statistics|Mathematics"
  "percolation_theory|percolation theory random graphs phase transitions|Mathematics"
  "statistics_theory|statistical theory asymptotic inference decision theory|Mathematics"

  # Pure mathematics
  "number_theory|number theory arithmetic geometry modular forms|Mathematics"
  "analytic_number_theory|analytic number theory L-functions prime numbers|Mathematics"
  "algebraic_number_theory|algebraic number theory Galois representations|Mathematics"
  "algebraic_geometry|algebraic geometry schemes moduli spaces|Mathematics"
  "arithmetic_geometry|arithmetic geometry Diophantine geometry|Mathematics"
  "representation_theory|representation theory Lie algebras algebraic groups|Mathematics"
  "lie_theory|Lie theory Lie algebras Lie groups|Mathematics"
  "algebra|commutative algebra homological algebra|Mathematics"
  "commutative_algebra|commutative algebra local rings ideals|Mathematics"
  "homological_algebra|homological algebra derived categories|Mathematics"
  "noncommutative_algebra|noncommutative algebra rings algebras|Mathematics"
  "quantum_algebra|quantum algebra quantum groups Hopf algebras|Mathematics"
  "category_theory|category theory higher categories topos theory|Mathematics"
  "topology|algebraic topology homotopy theory|Mathematics"
  "algebraic_topology|algebraic topology homotopy theory cohomology|Mathematics"
  "geometric_topology|geometric topology low dimensional topology|Mathematics"
  "differential_topology|differential topology manifolds cobordism|Mathematics"
  "differential_geometry|differential geometry Riemannian geometry|Mathematics"
  "symplectic_geometry|symplectic geometry Hamiltonian dynamics|Mathematics"
  "metric_geometry|metric geometry geometric group theory|Mathematics"
  "geometric_group_theory|geometric group theory hyperbolic groups|Mathematics"
  "operator_algebras|operator algebras C*-algebras von Neumann algebras|Mathematics"
  "functional_analysis|functional analysis Banach spaces operator theory|Mathematics"
  "harmonic_analysis|harmonic analysis Fourier analysis|Mathematics"
  "complex_analysis|complex analysis several complex variables|Mathematics"
  "real_analysis|real analysis measure theory calculus of variations|Mathematics"
  "pde_theory|partial differential equations existence regularity theory|Mathematics"
  "dynamical_systems|dynamical systems ergodic theory chaos|Mathematics"
  "ergodic_theory|ergodic theory measure preserving systems|Mathematics"
  "mathematical_logic|mathematical logic proof theory model theory|Mathematics"
  "set_theory|set theory forcing large cardinals|Mathematics"
  "model_theory|model theory stability o-minimality|Mathematics"
  "proof_theory|proof theory ordinal analysis|Mathematics"
  "k_theory_homology|K-theory homology cohomology|Mathematics"

  # Economic theory and mathematical finance
  "game_theory|game theory equilibrium theory|Economics"
  "mechanism_design|mechanism design auction theory|Economics"
  "auction_theory|auction theory mechanism design|Economics"
  "social_choice|social choice theory voting theory|Economics"
  "microeconomic_theory|microeconomic theory general equilibrium|Economics"
  "general_equilibrium|general equilibrium theory welfare theorems|Economics"
  "decision_theory|decision theory utility theory|Economics"
  "matching_market_design|matching theory market design stable matching|Economics"
  "econ_theory|economic theory equilibrium mechanism design|Economics"
  "mathematical_finance|mathematical finance stochastic calculus option pricing|Economics"
  "stochastic_finance|stochastic finance martingales mathematical finance|Economics"
  "portfolio_theory|portfolio theory risk measures mathematical finance|Economics"

  # Theoretical and mathematical physics
  "quantum_information|quantum information theory entanglement|Physics"
  "quantum_computing_theory|quantum computing complexity algorithms|Physics"
  "quantum_error_correction|quantum error correction stabilizer codes|Physics"
  "quantum_field_theory|quantum field theory conformal field theory|Physics"
  "conformal_field_theory|conformal field theory vertex operator algebras|Physics"
  "string_theory|string theory holography AdS/CFT|Physics"
  "holography|holography AdS/CFT gauge gravity duality|Physics"
  "quantum_gravity|quantum gravity general relativity|Physics"
  "general_relativity_theory|general relativity mathematical relativity black holes|Physics"
  "high_energy_theory|high energy theory supersymmetry gauge theory|Physics"
  "statistical_mechanics|statistical mechanics phase transitions|Physics"
  "nonlinear_dynamics|nonlinear dynamics chaos solitons|Physics"
  "integrable_systems|integrable systems solitons exactly solvable models|Physics"
  "mathematical_physics|mathematical physics integrable systems|Physics"
  "condensed_matter_theory|condensed matter theory topological phases|Physics"
  "topological_phases|topological phases topological order condensed matter theory|Physics"
  "many_body_theory|many body theory quantum many body systems|Physics"
  "strongly_correlated_theory|strongly correlated systems condensed matter theory|Physics"
  "cosmology_theory|theoretical cosmology inflation dark energy|Physics"
  "black_hole_theory|black hole thermodynamics quantum gravity|Physics"
)

SELECTED_DOMAINS=()
for domain in "${THEORY_DOMAINS[@]}"; do
  IFS="|" read -r slug _query _fields <<< "$domain"
  if [[ -n "$DOMAIN_REGEX" && ! "$slug" =~ $DOMAIN_REGEX ]]; then
    continue
  fi
  SELECTED_DOMAINS+=("$domain")
  if (( DOMAIN_LIMIT > 0 && ${#SELECTED_DOMAINS[@]} >= DOMAIN_LIMIT )); then
    break
  fi
done

domain_count="${#SELECTED_DOMAINS[@]}"
if (( domain_count == 0 )); then
  echo "No domains selected. Check DOMAIN_REGEX=$DOMAIN_REGEX or DOMAIN_LIMIT=$DOMAIN_LIMIT." >&2
  exit 1
fi

base_quota="$((TARGET_TOTAL / domain_count))"
remainder="$((TARGET_TOTAL % domain_count))"

mkdir -p "$OUTPUT_ROOT"

echo "Writing theoretical S2-search citation graphs under: $OUTPUT_ROOT"
echo "Target seed papers: $TARGET_TOTAL across $domain_count domains"
echo "Minimum citations per seed paper: $MIN_CITATIONS"
echo "Max candidates per domain: $MAX_CANDIDATES"
echo "S2 search: batch_size=$S2_SEARCH_BATCH_SIZE max_pages=$S2_SEARCH_MAX_PAGES sort=$S2_SEARCH_SORT"
echo "S2 batch: batch_size=$S2_BATCH_SIZE concurrency=$S2_CONCURRENCY sleep=${S2_SLEEP}s"
echo "Skip existing: $SKIP_EXISTING; dry run: $DRY_RUN"
echo "Extra builder args: $*"

for idx in "${!SELECTED_DOMAINS[@]}"; do
  IFS="|" read -r slug query fields_of_study <<< "${SELECTED_DOMAINS[$idx]}"
  quota="$base_quota"
  if (( idx < remainder )); then
    quota="$((quota + 1))"
  fi
  if (( quota > MAX_CANDIDATES )); then
    quota="$MAX_CANDIDATES"
  fi

  output_dir="$OUTPUT_ROOT/$slug"
  if [[ "$SKIP_EXISTING" == "1" && -f "$output_dir/manifest.json" ]]; then
    echo "Skipping $slug; manifest already exists at $output_dir/manifest.json"
    continue
  fi

  cmd=(
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
  if [[ -n "$MIN_YEAR" ]]; then
    cmd+=(--min-year "$MIN_YEAR")
  fi
  if [[ -n "$MAX_YEAR" ]]; then
    cmd+=(--max-year "$MAX_YEAR")
  fi
  if [[ "$ALLOW_MISSING_S2_KEY" == "1" ]]; then
    cmd+=(--allow-missing-s2-key)
  fi

  echo
  echo "[$((idx + 1))/$domain_count] $slug -> quota $quota"
  echo "Query: $query"
  echo "Fields: $fields_of_study"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf "Command:"
    printf " %q" "${cmd[@]}" "$@"
    printf "\n"
  else
    "${cmd[@]}" "$@"
  fi
done

echo
echo "Done. Per-domain manifests are in $OUTPUT_ROOT/*/manifest.json"
