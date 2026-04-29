You are controlling an oracle experiment that repeatedly regenerates layer-1 memory banks.

Goal:
- Decide whether another full layer-1 generation attempt is worth running.
- You may use the reported oracle embedding scores against the gold answer to decide if more exploration is worthwhile.

Rules:
1. Prefer stopping when the best score looks strong and recent attempts are no longer improving materially.
2. Prefer continuing when improvements are still arriving or current scores remain weak.
3. Respect the hard cap: if CURRENT_ATTEMPT equals MAX_ATTEMPTS, you must stop.
4. Output STRICT JSON only:
   {{
     "continue_search": 0 or 1,
     "reason": "short string <= 30 words"
   }}

QUESTION:
{question}

CURRENT_ATTEMPT:
{current_attempt}

MIN_ATTEMPTS:
{min_attempts}

MAX_ATTEMPTS:
{max_attempts}

ATTEMPT_SUMMARIES:
{attempt_summaries}
