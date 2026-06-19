"""
retrieval/params.py
───────────────────
Locked retrieval parameters from SK-06 tuning.

Do NOT change these values without re-running SK-06 and updating
sk06/reports/best_params.json. Recall@5 = 1.0, MRR = 0.8086 at these params.
"""

from __future__ import annotations

BEST_PARAMS: dict = {
    "match_count":    3,
    "min_similarity": 0.0,
    "ef_search":      40,
    "recall_at_5":    1.0,
    "recall_at_10":   1.0,
    "mrr":            0.8086,
    "p95_ms":         318.6,
}

MATCH_COUNT:    int   = BEST_PARAMS["match_count"]
MIN_SIMILARITY: float = BEST_PARAMS["min_similarity"]
EF_SEARCH:      int   = BEST_PARAMS["ef_search"]

# The p_doc_type filter MUST always be passed — without it recall drops below gate.
# This is a locked decision from SK-06: never call the RPC without this filter.
REQUIRE_DOC_TYPE_FILTER: bool = True
