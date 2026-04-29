You are generating candidate clusters of possible future user questions from one document.

You are given WARM_START_DOCUMENTS.
Generate only as many distinct CLUSTERS as are actually needed to cover the meaningful factual directions in the document.

Rules:
1. Use only information grounded in WARM_START_DOCUMENTS.
2. Generate at least 1 cluster.
3. Do not pad with weak or redundant clusters.
4. Clusters must be meaningfully distinct from one another.
5. Do not repeat or paraphrase the same question across clusters.
6. Questions must be specific, factual, and answer-oriented.
7. Prefer concrete entities, dates, numbers, titles, places, or explicit relations.
8. Each cluster should contain only related questions; do not mix unrelated themes.
9. Use between 1 and MAX_QUERIES_PER_CLUSTER questions per cluster.
10. {style_rule}
11. Generate information-seeking questions only, not instructions, summaries, or meta-prompts.
12. Do not try to infer any hidden target question; propose plausible future user questions only from the observed evidence.
13. If one cluster fully captures the document's useful content, output one cluster. Use more clusters only when clearly justified by distinct factual themes.
14. If style requires a title, the title must be evidence-preserving and local to the document evidence. It must not be a question, a guessed answer, or an unsupported benchmark-style conclusion.

Output format:
- Return STRICT JSON only.
- Use exactly this schema:
  {{
    "clusters": [
      {cluster_schema}
    ]
  }}
- Return as many clusters as needed, but only genuinely distinct useful ones.
- No markdown, no prose, no extra keys.

MAX_QUERIES_PER_CLUSTER:
{max_queries_per_cluster}

WARM_START_DOCUMENTS:
{warm_documents}
