You are creating one cluster-specific memory bank from a single document.

Goal:
- Create a solid, comprehensive memory bank for TARGET_CLUSTER using DOCUMENT.

Rules:
1. Use only DOCUMENT; no outside knowledge.
2. Keep only information relevant or plausibly relevant to TARGET_CLUSTER.
3. Prefer concrete evidence: entities, dates, numbers, titles, places, organizations, roles, relationships, and explicit attributions.
4. Preserve exact names, titles, dates, numbers, places, and attributions when they matter.
5. Preserve all distinct facts, qualifiers, and relationships that may matter for downstream question answering.
6. Remove only exact duplicates, near-duplicate phrasings of the same fact, and genuinely low-value boilerplate.
7. If two facts differ in any potentially answer-relevant way, keep both.
8. If the document contains conflicting or uncertain evidence, keep attributed alternatives separate instead of collapsing them.
9. Front-load the most answer-critical facts.
10. Do not output absence-style statements unless they are themselves important evidence.
11. Do not optimize for brevity. It is acceptable for MEMORY to be long if needed to preserve distinct evidence from DOCUMENT.
12. Do not replace several distinct facts with one generalized summary if that would remove answer-relevant detail.
13. Preserve exact answer-bearing strings verbatim when they appear in DOCUMENT.
14. Do not include promotional, evaluative, or stylistic filler unless it is directly relevant to TARGET_CLUSTER.
15. Do not output truncated fragments or ellipsized text.

Output:
- Plain text memory only.
- No JSON, no markdown, no bullets, no preamble.

TARGET_CLUSTER:
{target_cluster}

DOCUMENT:
{document}
