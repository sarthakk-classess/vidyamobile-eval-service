"""
mastery/scheduler.py
────────────────────
SchedulingEngine: wraps FSRS-4.5 algorithm into scheduling decisions.

Single entry point Himanshu calls from HK-02 orchestrator to update a
student's mastery state after a review session.

Usage:
    from eval_service.mastery.scheduler import SchedulingEngine
    from eval_service.mastery.state import MasteryState, RATING_GOOD

    engine = SchedulingEngine()
    state  = engine.schedule_new("syl_abc123_0004", "user_xyz", "syllabus")
    state  = engine.update(state, rating=RATING_GOOD)
    due    = engine.due_chunks("user_xyz", all_states)
"""

from __future__ import annotations
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

from eval_service.mastery.state import (
    MasteryState,
    RATING_AGAIN, RATING_HARD, RATING_GOOD, RATING_EASY,
    CardState, DocType, RatingValue,
)
from eval_service.mastery.algorithm import (
    retrievability,
    initial_stability, initial_difficulty,
    next_difficulty,
    next_stability_recall, next_stability_lapse,
    next_interval, short_term_stability,
)

DESIRED_RETENTION: float     = 0.90
LEARNING_STEPS_MINUTES: list[int]   = [1, 10]
RELEARNING_STEPS_MINUTES: list[int] = [10]
MIN_REVIEW_INTERVAL: int = 1
MAX_REVIEW_INTERVAL: int = 36500


class SchedulingEngine:
    """Stateless FSRS-4.5 scheduler. Create once and reuse."""

    def __init__(self, desired_retention: float = DESIRED_RETENTION):
        if not 0.5 <= desired_retention < 1.0:
            raise ValueError(f"desired_retention must be in [0.5, 1.0), got {desired_retention}")
        self.desired_retention = desired_retention

    def schedule_new(
        self,
        chunk_id: str,
        user_id:  str,
        doc_type: DocType,
        now:      Optional[datetime] = None,
    ) -> MasteryState:
        """Create initial MasteryState for a (user, chunk) never reviewed before."""
        ts = now or datetime.now(timezone.utc)
        return MasteryState(
            chunk_id       = chunk_id,
            user_id        = user_id,
            doc_type       = doc_type,
            card_state     = "new",
            stability      = initial_stability(RATING_GOOD),
            difficulty     = initial_difficulty(RATING_GOOD),
            retrievability = 1.0,
            review_count   = 0,
            lapse_count    = 0,
            last_rating    = None,
            last_review_at = None,
            next_review_at = ts,
            created_at     = ts,
        )

    def update(
        self,
        state:  MasteryState,
        rating: RatingValue,
        now:    Optional[datetime] = None,
    ) -> MasteryState:
        """Update MasteryState after the student rates their recall. Returns new object."""
        if rating not in (1, 2, 3, 4):
            raise ValueError(f"Rating must be 1–4, got {rating}")
        ts = now or datetime.now(timezone.utc)
        elapsed_days = self._elapsed_days(state.last_review_at, ts)
        R = retrievability(elapsed_days, state.stability)

        if state.card_state == "new":
            return self._handle_new_card(state, rating, R, ts)
        elif state.card_state == "learning":
            return self._handle_learning_card(state, rating, R, ts)
        elif state.card_state == "review":
            return self._handle_review_card(state, rating, R, ts)
        elif state.card_state == "relearning":
            return self._handle_relearning_card(state, rating, R, ts)
        raise ValueError(f"Unknown card_state: {state.card_state!r}")

    def due_chunks(
        self,
        user_id:    str,
        all_states: list[MasteryState],
        now:        Optional[datetime] = None,
        limit:      int = 50,
    ) -> list[MasteryState]:
        """Return up to limit due chunks sorted by urgency (most overdue first)."""
        ts = now or datetime.now(timezone.utc)
        due = []
        for s in all_states:
            if s.user_id != user_id:
                continue
            if s.card_state == "new":
                due.append((2, s))
            elif s.next_review_at and ts >= s.next_review_at:
                due.append((1, s))
        due.sort(key=lambda x: (x[0], -(x[1].days_overdue)))
        return [s for _, s in due[:limit]]

    def forecast(self, state: MasteryState, days_ahead: int = 30) -> list[dict]:
        """Project review schedule for the next days_ahead days (assuming rating=Good)."""
        schedule = []
        ts = datetime.now(timezone.utc)
        sim = state
        while True:
            if sim.next_review_at is None:
                break
            delta = (sim.next_review_at - ts).days
            if delta > days_ahead:
                break
            R_at_review = retrievability(
                self._elapsed_days(sim.last_review_at, sim.next_review_at),
                sim.stability,
            )
            schedule.append({
                "day":           delta,
                "date":          sim.next_review_at.strftime("%Y-%m-%d"),
                "retrievability": round(R_at_review, 3),
                "stability":      round(sim.stability, 2),
            })
            sim = self.update(sim, RATING_GOOD, now=sim.next_review_at)
        return schedule

    # ── Card state handlers ───────────────────────────────────────────────────

    def _handle_new_card(self, state, rating, R, ts) -> MasteryState:
        S = initial_stability(rating)
        D = initial_difficulty(rating)
        if rating == RATING_AGAIN:
            next_dt, new_state = ts + timedelta(minutes=LEARNING_STEPS_MINUTES[0]), "learning"
        elif rating == RATING_HARD:
            next_dt, new_state = ts + timedelta(minutes=LEARNING_STEPS_MINUTES[0]), "learning"
        elif rating == RATING_GOOD:
            next_dt, new_state = ts + timedelta(minutes=LEARNING_STEPS_MINUTES[-1]), "learning"
        else:
            interval = next_interval(S, self.desired_retention)
            next_dt, new_state = ts + timedelta(days=interval), "review"
        return self._build(state, new_state, S, D, R, rating, ts, next_dt,
                           lapses=state.lapse_count + (1 if rating == RATING_AGAIN else 0))

    def _handle_learning_card(self, state, rating, R, ts) -> MasteryState:
        S, D = state.stability, state.difficulty
        if rating == RATING_AGAIN:
            S = short_term_stability(S, rating)
            D = next_difficulty(D, rating)
            next_dt, new_state = ts + timedelta(minutes=LEARNING_STEPS_MINUTES[0]), "learning"
        elif rating in (RATING_HARD, RATING_GOOD):
            S = short_term_stability(S, rating)
            D = next_difficulty(D, rating)
            next_dt, new_state = ts + timedelta(minutes=LEARNING_STEPS_MINUTES[-1]), "learning"
        else:
            S = next_stability_recall(S, D, R, rating)
            D = next_difficulty(D, rating)
            interval = next_interval(S, self.desired_retention)
            next_dt, new_state = ts + timedelta(days=interval), "review"
        days_to_next = (next_dt - ts).total_seconds() / 86400.0
        if days_to_next >= 1.0 and new_state == "learning":
            new_state = "review"
        return self._build(state, new_state, S, D, R, rating, ts, next_dt,
                           lapses=state.lapse_count + (1 if rating == RATING_AGAIN else 0))

    def _handle_review_card(self, state, rating, R, ts) -> MasteryState:
        D = next_difficulty(state.difficulty, rating)
        if rating == RATING_AGAIN:
            S = next_stability_lapse(state.stability, D, R)
            next_dt, new_state = ts + timedelta(minutes=RELEARNING_STEPS_MINUTES[0]), "relearning"
            lapses = state.lapse_count + 1
        else:
            S = next_stability_recall(state.stability, D, R, rating)
            interval = min(max(next_interval(S, self.desired_retention), MIN_REVIEW_INTERVAL), MAX_REVIEW_INTERVAL)
            next_dt, new_state = ts + timedelta(days=interval), "review"
            lapses = state.lapse_count
        return self._build(state, new_state, S, D, R, rating, ts, next_dt, lapses=lapses)

    def _handle_relearning_card(self, state, rating, R, ts) -> MasteryState:
        D = next_difficulty(state.difficulty, rating)
        if rating == RATING_AGAIN:
            S = short_term_stability(state.stability, rating)
            next_dt, new_state = ts + timedelta(minutes=RELEARNING_STEPS_MINUTES[0]), "relearning"
            lapses = state.lapse_count + 1
        else:
            S = next_stability_recall(state.stability, D, R, rating)
            interval = max(next_interval(S, self.desired_retention), MIN_REVIEW_INTERVAL)
            next_dt, new_state = ts + timedelta(days=interval), "review"
            lapses = state.lapse_count
        return self._build(state, new_state, S, D, R, rating, ts, next_dt, lapses=lapses)

    @staticmethod
    def _build(state, card_state, S, D, R, rating, ts, next_dt, lapses) -> MasteryState:
        return MasteryState(
            chunk_id       = state.chunk_id,
            user_id        = state.user_id,
            doc_type       = state.doc_type,
            card_state     = card_state,
            stability      = S,
            difficulty     = D,
            retrievability = R,
            review_count   = state.review_count + 1,
            lapse_count    = lapses,
            last_rating    = rating,
            last_review_at = ts,
            next_review_at = next_dt,
            created_at     = state.created_at,
        )

    @staticmethod
    def _elapsed_days(last_review_at: Optional[datetime], now: datetime) -> float:
        if last_review_at is None:
            return 0.0
        return max(0.0, (now - last_review_at).total_seconds() / 86400.0)


def infer_rating_from_quiz(score_pct: float) -> int:
    """Map a quiz percentage (0–100) to FSRS rating 1–4."""
    if score_pct < 1.0:   return 1
    if score_pct < 50.0:  return 2
    if score_pct < 85.0:  return 3
    return 4


def infer_rating_from_practice(correct: bool, time_taken_s: float, expected_time_s: float = 60.0) -> int:
    """Map practice result (correctness + time) to FSRS rating 1–4."""
    if not correct:
        return 1
    if expected_time_s <= 0:
        expected_time_s = 60.0
    ratio = time_taken_s / expected_time_s
    if ratio > 3.0:   return 2
    if ratio > 1.5:   return 3
    return 4
