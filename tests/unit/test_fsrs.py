"""
Unit tests for FSRS-4.5 algorithm and SchedulingEngine.
"""
import pytest
from datetime import datetime, timezone
from eval_service.mastery.algorithm import (
    retrievability, initial_stability, initial_difficulty, next_interval,
)
from eval_service.mastery.scheduler import (
    SchedulingEngine, infer_rating_from_quiz, infer_rating_from_practice,
)


@pytest.fixture
def engine():
    return SchedulingEngine(desired_retention=0.90)


def test_initial_stability_rating3():
    assert initial_stability(3) > 0


def test_initial_stability_increases_with_rating():
    stabilities = [initial_stability(r) for r in (1, 2, 3, 4)]
    assert stabilities == sorted(stabilities)


def test_initial_difficulty_in_range():
    for rating in (1, 2, 3, 4):
        d = initial_difficulty(rating)
        assert 1.0 <= d <= 10.0


def test_retrievability_at_zero():
    assert retrievability(0, 10.0) == pytest.approx(1.0, abs=0.01)


def test_retrievability_decays():
    r5  = retrievability(5,  10.0)
    r10 = retrievability(10, 10.0)
    r20 = retrievability(20, 10.0)
    assert r5 > r10 > r20


def test_next_interval_positive():
    # positional args — first param is named S, not stability
    assert next_interval(8.0, 0.90) >= 1


def test_schedule_new_creates_state(engine):
    state = engine.schedule_new("chunk_001", "user_abc", "syllabus")
    assert state.chunk_id  == "chunk_001"
    assert state.user_id   == "user_abc"
    assert state.card_state == "new"


def test_update_after_good_rating(engine):
    state   = engine.schedule_new("c1", "u1", "lecture_slides")
    updated = engine.update(state, rating=3)
    assert updated.review_count == 1
    assert updated.card_state in ("learning", "review")


def test_lapse_on_again(engine):
    state  = engine.schedule_new("c1", "u1", "academic_reading")
    state  = engine.update(state, rating=3)
    state2 = engine.update(state, rating=3)
    lapsed = engine.update(state2, rating=1)
    assert lapsed.lapse_count >= 1


def test_due_chunks_returns_list(engine):
    state   = engine.schedule_new("c1", "u1", "syllabus")
    updated = engine.update(state, rating=4)
    due     = engine.due_chunks("u1", [updated], limit=10)
    assert isinstance(due, list)


def test_forecast_returns_schedule(engine):
    state    = engine.schedule_new("c1", "u1", "syllabus")
    schedule = engine.forecast(state, days_ahead=30)
    assert isinstance(schedule, list) and len(schedule) > 0


@pytest.mark.parametrize("score,expected", [
    (1.0, 4), (0.8, 3), (0.6, 2), (0.3, 1),
])
def test_infer_rating_from_quiz(score, expected):
    assert infer_rating_from_quiz(score) == expected


def test_infer_rating_from_practice_correct_fast():
    assert infer_rating_from_practice(correct=True, time_taken_s=20, expected_time_s=60) == 4


def test_infer_rating_from_practice_incorrect():
    assert infer_rating_from_practice(correct=False, time_taken_s=90, expected_time_s=60) == 1
