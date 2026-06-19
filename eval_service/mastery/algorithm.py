"""
mastery/algorithm.py
────────────────────
FSRS-4.5 core equations — stability, difficulty, retrievability.

Source: Jarrett Ye's FSRS-4.5 algorithm paper (2023)
        https://github.com/open-spaced-repetition/fsrs4anki/wiki/The-Algorithm

Pure mathematics — no database access, no datetime logic.
SchedulingEngine (scheduler.py) wraps these into scheduling decisions.

Do NOT change the W parameters (w0–w19) without re-training on student data
or consulting the FSRS paper. Changing them invalidates SK-04 recall baselines.
"""

from __future__ import annotations
import math

# ── FSRS-4.5 default weights (w0 – w19) ───────────────────────────────────────
# Published defaults fitted on a large flashcard review dataset.
# Retrain on Vidya student data once ≥10,000 reviews are collected.

W: list[float] = [
    0.4072,   # w0  : initial stability for rating Again
    1.1829,   # w1  : initial stability for rating Hard
    3.1262,   # w2  : initial stability for rating Good
    7.2102,   # w3  : initial stability for rating Easy
    0.5316,   # w4  : difficulty contribution from initial rating
    1.0651,   # w5  : difficulty dampening factor
    0.8006,   # w6  : recall stability multiplier
    0.0589,   # w7  : recall stability — difficulty exponent
    1.4832,   # w8  : recall stability — retrievability factor
    0.1342,   # w9  : lapse stability base
    1.0166,   # w10 : lapse stability — difficulty exponent
    0.8145,   # w11 : lapse stability — stability exponent
    0.1191,   # w12 : lapse stability — retrievability factor
    0.3370,   # w13 : lapse stability — forgetting rate dampener
    2.0855,   # w14 : difficulty update — rating-to-delta multiplier
    0.1544,   # w15 : difficulty update — mean reversion weight
    4.6497,   # w16 : difficulty update — mean reversion target
    1.4481,   # w17 : hard penalty multiplier (G=2)
    0.5580,   # w18 : easy bonus multiplier (G=4)
    0.1834,   # w19 : short-term stability increase factor
]

DECAY  = -0.5
FACTOR = 0.9 ** (1.0 / DECAY) - 1   # ≈ 0.2346


def retrievability(t: float, S: float) -> float:
    """Probability [0,1] of recall t days after reviewing with stability S."""
    if t <= 0:
        return 1.0
    if S <= 0:
        raise ValueError(f"Stability must be positive, got S={S}")
    return (1.0 + FACTOR * t / S) ** DECAY


def initial_stability(rating: int) -> float:
    """Stability for a brand-new card based on first rating (1=Again…4=Easy)."""
    if rating < 1 or rating > 4:
        raise ValueError(f"Rating must be 1–4, got {rating}")
    return max(W[rating - 1], 0.1)


def initial_difficulty(rating: int) -> float:
    """Difficulty for a brand-new card. Formula: D₀(G) = w4 - (G-3)*w5."""
    d = W[4] - (rating - 3) * W[5]
    return _clamp_difficulty(d)


def next_difficulty(D: float, rating: int) -> float:
    """Update difficulty after a review with mean reversion toward w16."""
    delta = -W[14] * (rating - 3)
    d_prime = D + delta * ((10.0 - D) / 9.0)
    d_double_prime = W[15] * W[16] + (1.0 - W[15]) * d_prime
    return _clamp_difficulty(d_double_prime)


def next_stability_recall(S: float, D: float, R: float, rating: int) -> float:
    """Updated stability after a successful review (rating ≥ 2)."""
    hard_penalty = W[17] if rating == 2 else 1.0
    easy_bonus   = W[18] if rating == 4 else 1.0
    inner = (
        math.exp(W[6])
        * (11.0 - D)
        * S ** (-W[7])
        * (math.exp(W[8] * (1.0 - R)) - 1.0)
        * hard_penalty
        * easy_bonus
        + 1.0
    )
    return max(S * inner, 0.1)


def next_stability_lapse(S: float, D: float, R: float) -> float:
    """Updated stability after a failed review (rating=1, Again)."""
    s = (
        W[9]
        * D ** (-W[10])
        * ((S + 1.0) ** W[11] - 1.0)
        * math.exp(W[12] * (1.0 - R))
        * W[13]
    )
    return max(s, 0.1)


def next_interval(S: float, desired_retention: float = 0.9) -> float:
    """Days until next review targeting a specific retention probability."""
    if not 0.0 < desired_retention < 1.0:
        raise ValueError(f"desired_retention must be in (0,1), got {desired_retention}")
    interval = (S / FACTOR) * (desired_retention ** (1.0 / DECAY) - 1.0)
    return max(round(interval), 1.0)


def short_term_stability(S: float, rating: int) -> float:
    """Stability multiplier for same-day re-study within learning steps."""
    multiplier = math.exp(W[19] * (rating - 3 + 0.2))
    return max(S * multiplier, 0.1)


def _clamp_difficulty(d: float) -> float:
    return max(1.0, min(10.0, d))
