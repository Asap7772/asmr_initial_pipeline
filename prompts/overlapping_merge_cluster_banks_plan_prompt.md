You are planning repeated evidence-bank merges for downstream question answering.

Goal:
- Propose evidence-grounded merge groups over INPUT_CLUSTER_BANKS for TARGET_QUERY.
- Overlap is allowed: the same bank_id may appear in multiple groups when it contains evidence relevant to multiple distinct themes.

Rules:
1. Use only TARGET_QUERY and INPUT_CLUSTER_BANKS.
2. Merge banks only if they describe the same entity, event, fact cluster, or strongly complementary answer-bearing theme.
3. Do not merge banks just because they are from the same broad topic.
4. Preserve granularity by default. If merging would create a vague or overly broad bank, keep those banks separate.
5. If one bank supports multiple distinct themes, you may include it in multiple groups.
6. If two groups share a bank because that bank supports both themes, do not collapse those groups together unless they are truly one coherent theme.
7. Banks omitted from every group will be kept as singleton groups automatically, so you do not need to mention every bank_id.
8. If banks contain conflicting evidence about the same target theme, you may still group them so the merge step can preserve attributed alternatives.
9. HEURISTIC_HIGH_PRIORITY_GROUPS lists pairs or groups that appear strongly complementary by shared anchors. Keep those members together unless there is a clear evidence-grounded reason not to.
10. SOURCE_DOCUMENT_SUMMARIES contains auxiliary source summaries for the source documents behind some banks. Use them only as auxiliary merge context.
11. Output STRICT JSON only in this schema:
   {{
     "groups": [
       {{"bank_ids": ["BANK_ID_1", "BANK_ID_2"]}}
     ]
   }}
12. No prose, no markdown, no extra keys.

TARGET_QUERY:
{target_query}

INPUT_CLUSTER_BANKS:
{bank_units}

SOURCE_DOCUMENT_SUMMARIES:
{source_doc_summaries}

HEURISTIC_HIGH_PRIORITY_GROUPS:
{heuristic_groups}
