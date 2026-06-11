# arXiv Citation Graph Builders

This directory contains two related graph builders for paper-citation experiments:

- `build_arxiv_citation_graph.py`: starts from arXiv search results, enriches them with Semantic Scholar metadata, filters for well-cited seed papers, and builds edges from Semantic Scholar reference lists.
- `build_hf_paper_pages_citation_graph.py`: starts from Hugging Face Paper Pages, explicit arXiv IDs, or arXiv search results, saves HF paper-page markdown, and builds edges from arXiv/HF links found in that markdown.

Both builders orient edges forward in time:

```text
referenced/cited paper -> later paper that cites or links to it
```

## Dependencies

Both scripts require `requests`:

```bash
pip install requests
```

`build_arxiv_citation_graph.py` requires a Semantic Scholar API key unless you pass `--allow-missing-s2-key`:

```bash
S2_API_KEY=...
```

`build_hf_paper_pages_citation_graph.py` can optionally use a Hugging Face token:

```bash
HF_TOKEN=...
```

Both scripts load keys from `.env` by default, or from the path passed with `--env-path`.

## arXiv Rate Limits

arXiv asks legacy API clients to make no more than one request every three
seconds, using a single connection. On shared machines or NATed clusters, the
same public IP can still receive `HTTP 429` even when one local process is
polite.

The arXiv builder defaults to a local lock under `/tmp` so concurrent runs by
the same Unix user pace their arXiv requests together. If you hit immediate
429s, wait out the server-side cooldown and rerun more conservatively:

```bash
python arxiv_paper/build_arxiv_citation_graph.py \
  --query "cat:cs.LG" \
  --max-arxiv-results 500 \
  --arxiv-batch-size 50 \
  --arxiv-sleep 10 \
  --arxiv-initial-sleep 600 \
  --arxiv-rate-limit-base-sleep 600 \
  --arxiv-rate-limit-max-sleep 3600 \
  --max-retries 12 \
  --output-dir data/arxiv_citation_graph/cs_lg_slow
```

You can also avoid the arXiv Atom API for candidate discovery and use Semantic
Scholar bulk search instead:

```bash
python arxiv_paper/build_arxiv_citation_graph.py \
  --candidate-source s2-search \
  --query "cat:cs.LG" \
  --s2-search-query "machine learning" \
  --s2-fields-of-study "Computer Science" \
  --max-arxiv-results 500 \
  --min-citations 100 \
  --max-seed-papers 200 \
  --output-dir data/arxiv_citation_graph/cs_lg_s2_search
```

This still keeps only candidates with a Semantic Scholar `ArXiv` external ID,
then uses the existing Semantic Scholar batch enrichment/reference crawl. The
tradeoff is that Semantic Scholar search is keyword/field based; it does not
exactly reproduce arXiv category membership.

## Semantic Scholar arXiv Builder

Script:

```bash
python arxiv_paper/build_arxiv_citation_graph.py \
  --query "cat:cs.LG AND all:transformer" \
  --max-arxiv-results 500 \
  --min-citations 250 \
  --max-seed-papers 100 \
  --output-dir data/arxiv_citation_graph/cs_lg_transformers
```

### Flow

1. Parse CLI arguments and load `S2_API_KEY` from `.env`.
2. Query the arXiv Atom API with `--query`, `--sort-by`, and `--sort-order`.
3. Parse each Atom entry into a candidate record with normalized arXiv ID, title, summary, authors, categories, dates, and DOI.
4. Batch candidates into Semantic Scholar `/graph/v1/paper/batch` requests using IDs like `ARXIV:1706.03762`.
5. Request paper metadata plus `references.*` fields from Semantic Scholar.
6. Select seed papers by Semantic Scholar `citationCount`, optional year filters, and `--max-seed-papers`.
7. Insert selected seed papers as graph nodes with role `seed_arxiv`.
8. Walk each expanded paper's Semantic Scholar `references` list and create edges from the referenced paper to the citing paper.
9. If `--citation-depth` is greater than `1`, fetch the next reference frontier from Semantic Scholar and repeat until the requested depth is reached.
10. Write graph artifacts under `--output-dir`.

By default, `--citation-depth 1` preserves the original one-hop behavior:

```text
direct reference -> seed paper
```

For deeper graphs:

```bash
python arxiv_paper/build_arxiv_citation_graph.py \
  --query "cat:cs.LG AND all:transformer" \
  --max-arxiv-results 500 \
  --min-citations 250 \
  --max-seed-papers 100 \
  --citation-depth 2 \
  --max-references-per-paper 30 \
  --max-expansion-papers-per-depth 500 \
  --output-dir data/arxiv_citation_graph/cs_lg_transformers_depth2
```

This adds references of references while capping each expanded paper to its top
30 references and capping each deeper frontier to 500 papers.

You can also save arXiv HTML pages for the graph. By default this fetches only
the selected seed papers:

```bash
python arxiv_paper/build_arxiv_citation_graph.py \
  --query "cat:cs.LG AND all:transformer" \
  --max-arxiv-results 500 \
  --min-citations 250 \
  --max-seed-papers 100 \
  --fetch-arxiv-html \
  --output-dir data/arxiv_citation_graph/cs_lg_transformers_html
```

Use `--arxiv-html-scope graph` to fetch every graph node with an arXiv ID,
including arXiv-backed references. HTML files are written under
`paper_html/` by default, and nodes are annotated with `arxiv_html_url`,
`arxiv_html_path`, and `arxiv_html_status`. Per-paper fetch records are written
to `arxiv_html.jsonl`.

### Node IDs

The builder uses the most stable available identifier in this order:

1. Semantic Scholar `paperId`: `s2:<paperId>`
2. arXiv ID: `arxiv:<id>`
3. DOI: `doi:<doi>`
4. Semantic Scholar corpus ID: `s2-corpus:<corpusId>`
5. Title/year hash: `title:<sha1-prefix>`

This means reference nodes can represent non-arXiv papers unless filtered out.

### Important Filters

- `--min-citations`: keeps only seed papers with at least this Semantic Scholar citation count.
- `--seed-publication-types`: keeps only seed papers whose Semantic Scholar `publicationTypes` intersects the requested comma-separated list, such as `Conference,JournalArticle`.
- `--require-seed-publication`: requires seed papers to have a Semantic Scholar publication signal such as a non-arXiv venue, non-preprint publication type, or DOI.
- `--min-year` / `--max-year`: optional seed-paper publication year filters.
- `--citation-depth`: number of backward reference hops to traverse from the selected seed papers. The default is `1`.
- `--max-references-per-paper`: breadth cap for references inspected per expanded paper. The default `0` means no cap.
- `--max-expansion-papers-per-depth`: cap on newly discovered papers fetched and expanded at each deeper frontier. The default `0` means no cap.
- `--target-node-count`: approximate graph node budget. Seeds are always kept; once the budget is reached, new reference nodes are skipped.
- `--internal-only`: emits only edges where both papers are selected seed papers.
- `--arxiv-references-only`: includes only referenced papers with arXiv IDs, except selected seeds.
- `--allow-nontemporal-edges`: keeps edges even if metadata suggests the reference is newer than the citing seed.
- `--fetch-arxiv-html`: downloads `https://arxiv.org/html/<id>` pages and stores local paths on graph nodes.
- `--arxiv-html-scope seed|graph`: chooses whether HTML is fetched only for selected seed papers or all arXiv-backed graph nodes.
- `--arxiv-html-dir`: output directory for HTML files, relative to `--output-dir` unless absolute.
- `--overwrite-arxiv-html`: refetches HTML even when a local file already exists.

### Outputs

The script writes:

- `seed_papers.jsonl`: selected seed nodes only.
- `nodes.jsonl`: all graph nodes.
- `edges.jsonl`: all graph edges.
- `arxiv_html.jsonl`: per-paper arXiv HTML fetch records when `--fetch-arxiv-html` is enabled.
- `manifest.json`: run configuration and counts.
- `graph.json`: combined metadata, nodes, and edges.

Use `launch_balanced_arxiv_graphs.sh` to build multiple category-specific
graphs with balanced seed quotas.

Use `launch_theory_s2_graphs.sh` to build a broad theoretical-domain sweep
without arXiv Atom candidate discovery. The default list covers 123
theory-heavy domains across CS theory, ML/statistical theory, optimization,
pure math, economic theory, mathematical finance, and theoretical physics:

```bash
./arxiv_paper/launch_theory_s2_graphs.sh
```

Useful knobs:

```bash
TARGET_TOTAL=1200 \
MIN_CITATIONS=50 \
MAX_CANDIDATES=1000 \
DOMAIN_REGEX='^(complexity_theory|cryptography|game_theory)$' \
./arxiv_paper/launch_theory_s2_graphs.sh
```

Set `DRY_RUN=1` to print commands without running them, `DOMAIN_LIMIT=N` to
test the first `N` selected domains, and `SKIP_EXISTING=0` to rebuild existing
per-domain outputs.

Sparse S2 domains can return zero arXiv-backed candidates. The launcher now
continues by default, retries failed domains with a broader query and then
without the S2 field filter, and writes:

- `succeeded_domains.tsv`
- `failed_domains.tsv`

Set `CONTINUE_ON_ERROR=0` to restore fail-fast behavior or
`FALLBACK_ON_FAILURE=0` to disable fallback retries.

Use `launch_dense_tcs_s2_graphs.sh` when you want a smaller number of much
denser, more interconnected theoretical CS graphs. It defaults to one broad
TCS-core graph with about 800 seed papers, a 10k node budget, reference depth
2, and up to 80 references per expanded paper. The primary query is a short
high-recall S2 query (`algorithms`), with fallback queries such as
`computational complexity`, `cryptography`, `formal methods`, and `graph
theory` if the primary query is too sparse:

```bash
./arxiv_paper/launch_dense_tcs_s2_graphs.sh
```

Run all overlapping dense TCS clusters:

```bash
CLUSTER_LIMIT=0 TARGET_TOTAL=6000 ./arxiv_paper/launch_dense_tcs_s2_graphs.sh
```

The dense launcher still builds a backward reference graph, so high fan-out
parents appear when many selected child papers cite the same foundational
paper. It does not fetch arbitrary forward citations for every parent node.

Use `launch_deep_dense_tcs_s2_graphs.sh` for a much larger depth/breadth run.
It delegates to the dense launcher and defaults to `DEPTH_PRESET=depth5`:
750k node budget, reference depth 5, 50 references per expanded paper, 3k
expansion frontier cap, and 350 seed papers. The seed count is intentionally
lower than the 10k dense run so the first reference hops do not consume the
whole node budget before deeper expansion can run:

```bash
./arxiv_paper/launch_deep_dense_tcs_s2_graphs.sh
```

Use the depth-10 preset for a deeper sampled crawl:

```bash
DEPTH_PRESET=depth10 ./arxiv_paper/launch_deep_dense_tcs_s2_graphs.sh
```

Or set `CITATION_DEPTH=10`; the launcher will infer `DEPTH_PRESET=depth10`
unless you set `DEPTH_PRESET=custom`.

For a still larger depth-10 run:

```bash
DEPTH_PRESET=depth10 \
TARGET_NODE_COUNT=2500000 \
TARGET_TOTAL=750 \
MAX_REFERENCES_PER_PAPER=120 \
MAX_EXPANSION_PAPERS_PER_DEPTH=8000 \
./arxiv_paper/launch_deep_dense_tcs_s2_graphs.sh
```

Use `launch_deep_dense_tcs_theory_s2_graphs.sh` when the dense/deep TCS run is
too broad. It defaults to the complexity/lower-bounds cluster, requires theory
keywords in seed paper titles/abstracts, and excludes empirical/application
keywords from both seed papers and referenced papers:

```bash
./arxiv_paper/launch_deep_dense_tcs_theory_s2_graphs.sh
```

Useful variants:

```bash
DEPTH_PRESET=depth10 ./arxiv_paper/launch_deep_dense_tcs_theory_s2_graphs.sh

CLUSTER_REGEX='^(algorithms_complexity|complexity_lower_bounds|crypto_complexity|logic_verification_pl)$' \
CLUSTER_LIMIT=0 \
TARGET_TOTAL=800 \
./arxiv_paper/launch_deep_dense_tcs_theory_s2_graphs.sh
```

The stricter launcher uses `--seed-include-keywords`,
`--seed-exclude-keywords`, and `--reference-exclude-keywords`. Reference papers
often have less metadata than seed papers, so reference filtering is mostly
title-based when Semantic Scholar does not provide abstracts.

Use `launch_premier_tcs_s2_graphs.sh` when you want broader TCS coverage while
still biasing seeds toward published, well-cited papers. The topic list is
inspired by premier theory venues: STOC/FOCS breadth, SODA/ESA algorithms,
ICALP tracks, CCC complexity, LICS logic/semantics, SoCG geometry, PODC/SPAA
distributed/parallel, plus COLT/TCC/IPCO-style adjacent theory. It defaults to
all clusters, `DEPTH_PRESET=depth5`, `MIN_CITATIONS=50`, seed publication types
`Conference,JournalArticle`, and a required seed publication signal:

```bash
./arxiv_paper/launch_premier_tcs_s2_graphs.sh
```

Useful variants:

```bash
DEPTH_PRESET=depth10 ./arxiv_paper/launch_premier_tcs_s2_graphs.sh

CLUSTER_REGEX='^(approximation_hardness|computational_complexity|cryptography_foundations|learning_theory)$' \
TARGET_TOTAL=400 \
./arxiv_paper/launch_premier_tcs_s2_graphs.sh

MIN_CITATIONS=100 \
S2_PUBLICATION_TYPES=Conference \
SEED_PUBLICATION_TYPES=Conference \
./arxiv_paper/launch_premier_tcs_s2_graphs.sh
```

This launcher creates one graph directory per cluster under
`data/arxiv_citation_graph_premier_tcs_depth*_s2/`. It uses short topic
queries with fallback queries because Semantic Scholar bulk search can be very
sparse for long theory-style query strings. If S2 metadata is missing
publication types for legitimate papers, lower strictness with
`SEED_PUBLICATION_TYPES=` or `REQUIRE_SEED_PUBLICATION=0`.

## Hugging Face Paper Pages Builder

Script:

```bash
python arxiv_paper/build_hf_paper_pages_citation_graph.py \
  --query "retrieval augmented generation" \
  --max-papers 50 \
  --output-dir data/hf_paper_page_citation_graph/rag
```

The convenience launcher uses the same builder:

```bash
./arxiv_paper/launch_hf_paper_page_graph.sh
```

### Flow

1. Parse CLI arguments and optionally load `HF_TOKEN` from `.env`.
2. Collect seed arXiv IDs from any combination of:
   - `--paper-id`, which may be an arXiv ID or an arXiv/HF paper URL.
   - `--paper-ids-file`, a text file of IDs or URLs.
   - HF paper search, when `--source hf-search` and `--query` are provided.
   - arXiv Atom search, when `--source arxiv` and `--query` are provided.
3. Normalize and de-duplicate seed arXiv IDs.
4. Fetch HF paper metadata from `/api/papers/{ARXIV_ID}` for each seed.
5. Fetch seed markdown from `/papers/{ARXIV_ID}.md`, with a markdown fallback request to `/papers/{ARXIV_ID}`.
6. Save seed markdown under `paper_markdown/`.
7. Extract referenced arXiv IDs from markdown links and labels such as:
   - `arxiv.org/abs/...`
   - `arxiv.org/pdf/...`
   - `huggingface.co/papers/...`
   - `arxiv: ...`
8. Insert seed nodes with role `seed_hf_paper` and reference nodes with role `reference`.
9. Create edges from each referenced arXiv ID to the seed paper.
10. Write graph artifacts, paper-page records, and a markdown index.

### Node IDs

The HF builder is intentionally arXiv-ID centric. Every node ID is:

```text
arxiv:<id>
```

Reference nodes may contain only IDs and URLs by default. Pass `--fetch-reference-metadata` to call `/api/papers/{id}` for references, and `--fetch-reference-markdown` to save markdown for references too.

### Important Filters

- `--max-papers`: maximum seed papers after de-duplication.
- `--hf-search-limit`: maximum results requested from HF paper search.
- `--min-year` / `--max-year`: optional seed-paper year filters.
- `--max-references-per-paper`: caps unique linked arXiv/HF references extracted from each markdown file; use `0` for no limit.
- `--fetch-reference-metadata`: enriches reference nodes with HF metadata.
- `--fetch-reference-markdown`: saves markdown for referenced papers as well as seeds.
- `--allow-nontemporal-edges`: keeps edges even if inferred years suggest the reference is newer than the seed.

### Outputs

The script writes:

- `seed_papers.jsonl`: selected seed nodes only.
- `nodes.jsonl`: all graph nodes.
- `edges.jsonl`: all graph edges.
- `paper_pages.jsonl`: per-seed metadata, markdown path, extracted reference IDs, and extraction counts.
- `manifest.json`: run configuration and counts.
- `graph.json`: combined metadata, nodes, and edges.
- `markdown_index.md`: a human-readable index of fetched paper pages.
- `paper_markdown/`: raw HF paper markdown files.

## How The Two Builders Differ

| Topic | `build_arxiv_citation_graph.py` | `build_hf_paper_pages_citation_graph.py` |
| --- | --- | --- |
| Primary source | arXiv Atom API plus Semantic Scholar Graph API | Hugging Face Paper Pages plus optional arXiv Atom API |
| Seed discovery | arXiv search query only | HF search, explicit IDs, ID file, or arXiv search |
| Reference source | Semantic Scholar structured `references` metadata | arXiv/HF links found in HF paper-page markdown |
| Seed selection | Filters by Semantic Scholar citation count, then keeps top cited seeds | Keeps de-duplicated seed IDs up to `--max-papers`; no citation-count ranking |
| Identifier model | Uses S2 paper IDs when available, then arXiv, DOI, corpus ID, or title hash | Always uses `arxiv:<id>` |
| Reference coverage | Can include non-arXiv references by default | Only references with extractable arXiv IDs |
| Metadata richness | Rich S2 metadata and reference metadata in one batch call | Rich HF seed metadata; reference metadata is optional |
| Markdown artifacts | Does not save paper text or markdown | Saves seed markdown and optionally reference markdown |
| Best for | Citation graphs grounded in structured reference lists and citation counts | Inspectable HF Paper Pages corpora and arXiv-link graphs from markdown |

In short, the Semantic Scholar builder is better when you want a citation graph backed by structured bibliographic metadata and citation-count filtering. The HF Paper Pages builder is better when you want local paper-page markdown and a graph over the arXiv links that appear in those pages.
