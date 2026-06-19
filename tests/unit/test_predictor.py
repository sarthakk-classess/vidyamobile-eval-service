"""
Unit/integration tests for the difficulty model predictor.

Skips automatically if the model artifact hasn't been trained yet,
so the suite stays green on a fresh checkout.
Run `python eval_service/difficulty/train.py` first to exercise these.
"""
import pytest
import numpy as np

from eval_service.difficulty.predictor import DifficultyPredictor, MODEL_PATH

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.is_file(),
    reason="model artifact not trained yet — run: python eval_service/difficulty/train.py",
)


@pytest.fixture(scope="module")
def predictor():
    return DifficultyPredictor()


@pytest.fixture(scope="module")
def one_student_records():
    from eval_service.difficulty.simulator import simulate
    records, _ = simulate(n_students=1, seed=42)
    return records


def test_score_student_returns_one_row_per_topic(predictor, one_student_records):
    from eval_service.difficulty.topics import TOPIC_CATALOG
    scored = predictor.score_student(one_student_records)
    assert len(scored) == len(TOPIC_CATALOG)


def test_score_student_row_keys(predictor, one_student_records):
    scored = predictor.score_student(one_student_records)
    for row in scored:
        for key in ("user_id", "topic", "difficulty_score", "difficulty_10",
                    "confidence", "n_chunks", "should_nudge"):
            assert key in row


def test_score_student_value_ranges(predictor, one_student_records):
    for row in predictor.score_student(one_student_records):
        assert 0.0 <= row["difficulty_score"] <= 1.0
        assert 1.0 <= row["difficulty_10"] <= 10.0
        assert 0.0 <= row["confidence"] <= 1.0
        assert isinstance(row["should_nudge"], bool)


def test_score_student_sorted_descending(predictor, one_student_records):
    scored = predictor.score_student(one_student_records)
    scores = [r["difficulty_score"] for r in scored]
    assert scores == sorted(scores, reverse=True)


def test_empty_records_returns_empty(predictor):
    assert predictor.score_student([]) == []


def test_predictions_correlate_with_latent_struggle(predictor):
    """Across many students, predicted difficulty should correlate with truth."""
    from eval_service.difficulty.simulator import simulate
    records, labels = simulate(n_students=40, seed=321)
    truth = {(l["user_id"], l["topic"]): l["struggle"] for l in labels}

    by_user: dict = {}
    for r in records:
        by_user.setdefault(r["user_id"], []).append(r)

    preds, trues = [], []
    for uid, recs in by_user.items():
        for row in predictor.score_student(recs):
            preds.append(row["difficulty_score"])
            trues.append(truth[(uid, row["topic"])])

    corr = np.corrcoef(preds, trues)[0, 1]
    assert corr > 0.7, f"Low correlation {corr:.3f} — model may need retraining"


def test_to_source_events_only_emits_nudge_topics(predictor, one_student_records):
    scored = predictor.score_student(one_student_records)
    events = predictor.to_source_events(scored, tenant_id="t1")
    nudge_count = sum(1 for r in scored if r["should_nudge"])
    assert len(events) == nudge_count


def test_to_source_events_payload_structure(predictor, one_student_records):
    scored = predictor.score_student(one_student_records)
    events = predictor.to_source_events(scored, tenant_id="tenant-x")
    for e in events:
        assert e["kind"] == "topic_difficulty"
        assert e["tenant_id"] == "tenant-x"
        payload = e["payload"]
        assert payload["source"] == "SK-07"
        assert "difficulty_score" in payload
        assert "difficulty_10" in payload
