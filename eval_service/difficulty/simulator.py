"""
difficulty/simulator.py — Student review history simulator.

Drives the real SchedulingEngine to generate realistic (student, topic) mastery
records with a known latent struggle label. The difficulty model learns to
recover struggle from observable FSRS state alone.

Generative model:
  ability_s       ~ Normal(0, 1)
  d_z             = (intrinsic_difficulty - 5.5) / 2.0
  struggle(s, t)  = sigmoid(SLOPE * (d_z - ability_s))  in (0, 1)  <- label
"""

from __future__ import annotations
import math
import random
from datetime import datetime, timezone, timedelta

from eval_service.mastery.scheduler import SchedulingEngine
from eval_service.mastery.state     import MasteryState
from eval_service.mastery.algorithm import retrievability
from eval_service.difficulty.topics import TOPIC_CATALOG, Topic

SLOPE                = 1.15
SIM_HORIZON_DAYS     = 120
MAX_REVIEWS_PER_CHUNK = 12

_SCHEDULER_DOC_TYPE = {
    "syllabus":         "syllabus",
    "academic_reading": "academic_reading",
    "lecture_slides":   "lecture_slides",
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _struggle(intrinsic_difficulty: float, ability: float) -> float:
    d_z = (intrinsic_difficulty - 5.5) / 2.0
    return _sigmoid(SLOPE * (d_z - ability))


def _draw_rating(struggle: float, R: float, rng: random.Random) -> int:
    p_fail = min(0.95, max(0.02, struggle * 0.55 + (1.0 - R) * 0.45))
    if rng.random() < p_fail:
        return 1  # Again
    if struggle < 0.33:
        weights = (0.50, 0.40, 0.10)
    elif struggle < 0.66:
        weights = (0.20, 0.55, 0.25)
    else:
        weights = (0.05, 0.45, 0.50)
    r = rng.random()
    if r < weights[0]:          return 4  # Easy
    if r < weights[0] + weights[1]: return 3  # Good
    return 2                                   # Hard


def _simulate_chunk(
    engine: SchedulingEngine,
    chunk_id: str,
    user_id: str,
    doc_type: str,
    struggle: float,
    enrolled_at: datetime,
    horizon_end: datetime,
    rng: random.Random,
) -> MasteryState:
    state   = engine.schedule_new(chunk_id, user_id, doc_type, now=enrolled_at)
    reviews = 0
    while reviews < MAX_REVIEWS_PER_CHUNK:
        review_time = state.next_review_at
        if review_time is None or review_time > horizon_end:
            break
        elapsed = 0.0
        if state.last_review_at is not None:
            elapsed = max(0.0, (review_time - state.last_review_at).total_seconds() / 86400.0)
        R      = retrievability(elapsed, state.stability)
        rating = _draw_rating(struggle, R, rng)
        state  = engine.update(state, rating, now=review_time)
        reviews += 1
    return state


def _state_to_record(state: MasteryState, topic_key: str) -> dict:
    return {
        "user_id":        state.user_id,
        "chunk_id":       state.chunk_id,
        "topic":          topic_key,
        "doc_type":       state.doc_type,
        "card_state":     state.card_state,
        "stability":      state.stability,
        "difficulty":     state.difficulty,
        "retrievability": state.retrievability,
        "review_count":   state.review_count,
        "lapse_count":    state.lapse_count,
        "last_rating":    state.last_rating,
    }


def simulate(
    n_students: int = 400,
    seed: int = 7,
    topics: list[Topic] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Simulate n_students studying every topic in the catalog.

    Returns
    -------
    records : per-chunk mastery records (feature source for the model)
    labels  : per-(student, topic) struggle labels (regression targets)
    """
    rng         = random.Random(seed)
    engine      = SchedulingEngine()
    topics      = topics or TOPIC_CATALOG
    base_time   = datetime(2026, 1, 1, tzinfo=timezone.utc)
    horizon_end = base_time + timedelta(days=SIM_HORIZON_DAYS)

    records: list[dict] = []
    labels:  list[dict] = []

    for s in range(n_students):
        user_id     = f"sim_user_{s:04d}"
        ability     = rng.gauss(0.0, 1.0)
        enrolled_at = base_time + timedelta(days=rng.randint(0, 20))

        for topic in topics:
            struggle       = _struggle(topic.intrinsic_difficulty, ability)
            sched_doc_type = _SCHEDULER_DOC_TYPE.get(topic.doc_type, "academic_reading")

            for c in range(topic.n_chunks):
                chunk_id = f"{topic.key.replace(':', '_').replace(' ', '')}_{s:04d}_{c:02d}"
                state    = _simulate_chunk(
                    engine, chunk_id, user_id, sched_doc_type,
                    struggle, enrolled_at, horizon_end, rng,
                )
                records.append(_state_to_record(state, topic.key))

            labels.append({
                "user_id":  user_id,
                "topic":    topic.key,
                "struggle": round(struggle, 6),
                "ability":  round(ability, 4),
                "n_chunks": topic.n_chunks,
            })

    return records, labels
