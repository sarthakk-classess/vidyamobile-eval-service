"""
difficulty/features.py
──────────────────────
Aggregate chunk-level mastery records into one feature vector per (student, topic).

Single feature contract shared by trainer and predictor — if it changes, retrain.

Input record (one per (user, chunk), joined with its topic by RT-08/RT-12):
    {
        "user_id", "chunk_id", "topic", "doc_type",
        "card_state":     "new"|"learning"|"review"|"relearning",
        "stability":      float (days, > 0),
        "difficulty":     float [1, 10],
        "retrievability": float [0, 1],
        "review_count":   int,
        "lapse_count":    int,
        "last_rating":    int|None (1..4),
    }
"""

from __future__ import annotations
import math
from collections import defaultdict

FEATURE_NAMES: list[str] = [
    "n_chunks",
    "frac_seen",
    "frac_new",
    "frac_relearning",
    "frac_mature",
    "mean_difficulty",
    "max_difficulty",
    "mean_log_stability",
    "min_log_stability",
    "mean_retrievability",
    "min_retrievability",
    "mean_reviews",
    "total_lapses",
    "lapse_rate",
    "mean_last_rating",
    "frac_low_rating",
]

_NEW              = "new"
_RELEARNING       = "relearning"
_REVIEW           = "review"
_MATURE_STABILITY = 21.0


def _safe(v, default=0.0):
    return default if v is None else v


def extract_topic_features(records: list[dict]) -> list[float]:
    """Build one feature vector from all chunk records belonging to one (student, topic)."""
    if not records:
        raise ValueError("extract_topic_features called with no records")

    n = len(records)
    difficulties  = [_safe(r.get("difficulty"), 5.0) for r in records]
    stabilities   = [max(_safe(r.get("stability"), 1.0), 1e-6) for r in records]
    retrievabils  = [_safe(r.get("retrievability"), 1.0) for r in records]
    reviews       = [int(_safe(r.get("review_count"), 0)) for r in records]
    lapses        = [int(_safe(r.get("lapse_count"), 0)) for r in records]
    states        = [r.get("card_state", _NEW) for r in records]

    last_ratings  = [r.get("last_rating") for r in records]
    rated         = [lr for lr in last_ratings if lr is not None]
    mean_last_rating = (sum(rated) / len(rated)) if rated else 3.0
    frac_low_rating  = (sum(1 for lr in rated if lr <= 2) / len(rated)) if rated else 0.0

    n_new        = sum(1 for s in states if s == _NEW)
    n_relearning = sum(1 for s in states if s == _RELEARNING)
    n_mature     = sum(
        1 for r, s in zip(records, states)
        if s == _REVIEW and _safe(r.get("stability"), 0.0) > _MATURE_STABILITY
    )

    total_reviews = sum(reviews)
    total_lapses  = sum(lapses)
    log_stab      = [math.log1p(s) for s in stabilities]

    feats = {
        "n_chunks":            float(n),
        "frac_seen":           (n - n_new) / n,
        "frac_new":            n_new / n,
        "frac_relearning":     n_relearning / n,
        "frac_mature":         n_mature / n,
        "mean_difficulty":     sum(difficulties) / n,
        "max_difficulty":      max(difficulties),
        "mean_log_stability":  sum(log_stab) / n,
        "min_log_stability":   min(log_stab),
        "mean_retrievability": sum(retrievabils) / n,
        "min_retrievability":  min(retrievabils),
        "mean_reviews":        total_reviews / n,
        "total_lapses":        float(total_lapses),
        "lapse_rate":          total_lapses / max(total_reviews, 1),
        "mean_last_rating":    mean_last_rating,
        "frac_low_rating":     frac_low_rating,
    }
    return [feats[name] for name in FEATURE_NAMES]


def confidence(records: "list[dict] | int") -> float:
    """
    Confidence [0,1] in difficulty prediction based on review history depth.
    Accepts either a list of mastery records or a raw review count integer.
    """
    if isinstance(records, int):
        total_reviews = records
    else:
        total_reviews = sum(int(_safe(r.get("review_count"), 0)) for r in records)
    return round(1.0 - math.exp(-total_reviews / 6.0), 4)


def group_by_student_topic(records: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Group flat mastery records by (user_id, topic)."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["user_id"], r["topic"])].append(r)
    return groups
