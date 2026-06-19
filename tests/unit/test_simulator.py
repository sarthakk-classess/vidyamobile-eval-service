"""
Unit tests for the student simulator used to generate difficulty training data.
No model artifact required — pure algorithmic tests.
"""
import pytest

from eval_service.difficulty.simulator import simulate, _struggle
from eval_service.difficulty.topics import TOPIC_CATALOG


def test_struggle_harder_topic_means_more_struggle():
    assert _struggle(9.0, 0.0) > _struggle(3.0, 0.0)


def test_struggle_stronger_student_means_less_struggle():
    assert _struggle(6.0, 2.0) < _struggle(6.0, -2.0)


def test_struggle_bounded():
    assert 0.0 < _struggle(10.0, -3.0) < 1.0
    assert 0.0 < _struggle(1.0, 3.0) < 1.0


def test_simulate_label_count():
    records, labels = simulate(n_students=5, seed=1)
    # One label per (student, topic)
    assert len(labels) == 5 * len(TOPIC_CATALOG)


def test_simulate_record_keys():
    records, _ = simulate(n_students=2, seed=1)
    required = {
        "user_id", "chunk_id", "topic", "doc_type", "card_state",
        "stability", "difficulty", "retrievability",
        "review_count", "lapse_count", "last_rating",
    }
    assert required.issubset(set(records[0].keys()))


def test_simulate_total_records():
    records, _ = simulate(n_students=5, seed=1)
    expected = 5 * sum(t.n_chunks for t in TOPIC_CATALOG)
    assert len(records) == expected


def test_simulate_deterministic():
    r1, l1 = simulate(n_students=3, seed=123)
    r2, l2 = simulate(n_students=3, seed=123)
    assert [r["stability"] for r in r1] == [r["stability"] for r in r2]
    assert [l["struggle"] for l in l1] == [l["struggle"] for l in l2]


def test_simulate_different_seed_differs():
    r1, _ = simulate(n_students=3, seed=1)
    r2, _ = simulate(n_students=3, seed=2)
    assert [r["stability"] for r in r1] != [r["stability"] for r in r2]


def test_harder_topics_produce_more_lapses():
    records, _ = simulate(n_students=60, seed=5)
    by_topic: dict = {}
    for r in records:
        by_topic.setdefault(r["topic"], []).append(r["lapse_count"])

    easy_avg = sum(by_topic["syllabus:Unit I"])  / len(by_topic["syllabus:Unit I"])
    hard_avg = sum(by_topic["syllabus:Unit V"])  / len(by_topic["syllabus:Unit V"])
    assert hard_avg > easy_avg


def test_label_struggle_in_range():
    _, labels = simulate(n_students=5, seed=7)
    for l in labels:
        assert 0.0 <= l["struggle"] <= 1.0


def test_label_keys():
    _, labels = simulate(n_students=2, seed=1)
    for l in labels:
        for k in ("user_id", "topic", "struggle"):
            assert k in l
