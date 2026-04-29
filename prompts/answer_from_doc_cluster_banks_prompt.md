You are answering a question using only concatenated per-document cluster banks and their memories.

Rules:
1. Use only DOCUMENT_CLUSTER_BANKS.
2. Do not use outside knowledge.
3. Give the single best-supported answer.
4. Return only one short final answer string.
5. Do not include explanation, reasoning, uncertainty notes, or extra context.
6. Satisfy all explicit constraints in TARGET_QUERY jointly; do not choose a candidate that only partially matches.
7. If memories conflict, choose the most direct and specific evidence.
8. Prefer evidence that links the race identity, incident clues, penalty clues, and participant/background clues into one coherent answer.
9. Prefer exact answer-bearing strings and explicit grounded relations over broad narrative or generic descriptive text.
10. If no single candidate satisfies all explicit constraints jointly, choose the candidate with the strongest fully grounded cross-clue support, not the most topically similar partial match.

TARGET_QUERY:
{target_query}

DOCUMENT_CLUSTER_BANKS:
{memory_text}

FINAL_ANSWER:
