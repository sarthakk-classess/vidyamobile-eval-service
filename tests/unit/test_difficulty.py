"""
Unit tests for difficulty model features (offline — no model artifact required).
"""
import pytest
from eval_service.difficulty.features import FEATURE_NAMES, extract_topic_features, confidence


def test_feature_names_count():
    assert len(FEATURE_NAMES) == 16


def test_extract_topic_features_returns_all(topic_features_raw):
    vec = extract_topic_features([topic_features_raw])
    assert len(vec) == len(FEATURE_NAMES)


def test_extract_topic_features_numeric(topic_features_raw):
    vec = extract_topic_features([topic_features_raw])
    for v in vec:
        assert isinstance(v, (int, float))


def test_confidence_zero_reviews():
    assert confidence(0) == 0.0


def test_confidence_with_records():
    c = confidence([{"review_count": 3}, {"review_count": 3}])
    assert 0 < c <= 1.0


def test_confidence_increases_with_reviews():
    c3  = confidence(3)
    c10 = confidence(10)
    assert 0 < c3 < c10 <= 1.0


def test_confidence_caps_at_one():
    assert confidence(1000) <= 1.0
