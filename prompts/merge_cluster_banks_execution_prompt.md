You are merging a group of structured cluster banks into one evidence-preserving merged cluster bank for downstream question answering.

Goal:
- Produce one merged bank that preserves as much distinct useful evidence from GROUP_CLUSTER_BANKS as possible for downstream question answering.

Rules:
1. Use only GROUP_CLUSTER_BANKS. TARGET_QUERY is for prioritization only.
2. Do not answer TARGET_QUERY. Do not claim that this group is the final correct answer.
3. The merged bank must stay coherent and correspond to one evidence-grounded theme already present in GROUP_CLUSTER_BANKS.
4. Preserve exact names, titles, dates, numbers, places, and attributions when they matter.
5. Preserve all distinct facts, qualifiers, and relationships from GROUP_CLUSTER_BANKS.
6. Remove only true duplicates or near-duplicate phrasings that express the same fact with the same qualifiers.
7. If two facts differ in any potentially answer-relevant way, keep both rather than collapsing them.
8. If multiple banks contain complementary evidence for the same theme, combine them without dropping distinct supporting details.
9. If evidence conflicts within the same theme, keep attributed alternatives in memory instead of collapsing them incorrectly.
10. Do not invent new facts, new entities, unsupported links, unsupported clue types, or unsupported race identities.
11. Do not write global conclusions such as "this is the race where..." or "all top finishers had already won by December 2023" unless that exact conclusion is directly supported inside GROUP_CLUSTER_BANKS.
12. Write 1 to MAX_QUERIES_PER_CLUSTER focused questions that describe the evidence in GROUP_CLUSTER_BANKS, not the final benchmark question.
13. {style_rule}
14. Create a solid, comprehensive MEMORY that preserves the strongest source evidence, keeps answer-critical distinctions explicit, and front-loads the most important facts.
15. Do not optimize for brevity. It is acceptable for MEMORY to be long if needed to preserve distinct evidence.
16. Do not replace several distinct facts with one generalized summary if that would remove answer-relevant detail.
17. Preserve exact answer-bearing strings verbatim when they appear in GROUP_CLUSTER_BANKS.
18. If style requires a title, the title must be evidence-preserving, local to the source banks, and written as a short noun phrase. It must not be a question, a guessed answer, or an unsupported final conclusion.
19. Do not output truncated fragments, ellipsized text, or unfinished items.
20. Do not include generic promotional, evaluative, or stylistic filler unless it is directly relevant to the grouped evidence theme.
21. SOURCE_DOCUMENT_SUMMARIES contains query-aware source-document summaries for the source documents behind GROUP_CLUSTER_BANKS. Use them only as auxiliary merge context to recover or align evidence already grounded in the group. Do not simply paste or restate the source summaries wholesale.

Output format:
- Return STRICT JSON only.
- Use exactly this schema:
  {{
    "merged_bank": {{
      {merged_bank_schema}
    }}
  }}
- No markdown, no prose, no extra keys.

TARGET_QUERY:
{target_query}

MAX_QUERIES_PER_CLUSTER:
{max_queries_per_cluster}

GROUP_CLUSTER_BANKS:
{group_banks}

SOURCE_DOCUMENT_SUMMARIES:
{source_doc_summaries}
