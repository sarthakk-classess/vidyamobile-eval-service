"""
Unit tests for the safety eval harness.

Tests MockAIClient accuracy behaviour and CI dataset loading.
The full 2500-scenario tests are skipped unless Vidya-Scenarios-2500.csv
is present locally (it is gitignored).
"""
import pytest
from pathlib import Path

from eval_service.safety.client  import MockAIClient
from eval_service.safety.dataset import load_labeled, FULL_CSV, CI_CSV

_HAS_FULL_CSV = FULL_CSV.exists()


# ── CI dataset (always runs — 30 scenarios committed) ────────────────────────

def test_ci_dataset_loads():
    scenarios = load_labeled()
    assert len(scenarios) > 0


def test_ci_dataset_required_keys():
    for s in load_labeled():
        for key in ("scenario_id", "opening_message", "expected_safety",
                    "expected_decision", "expected_agent", "expected_grounded"):
            assert key in s, f"Missing key '{key}' in scenario"


def test_ci_dataset_safety_labels_valid():
    valid_safety = {"safe", "refuse", "review"}
    for s in load_labeled():
        assert s["expected_safety"] in valid_safety, \
            f"Unexpected safety label: {s['expected_safety']}"


# ── MockAIClient — perfect oracle ─────────────────────────────────────────────

def test_mock_perfect_oracle_matches_gold():
    client = MockAIClient(accuracy=1.0, seed=0)
    for s in load_labeled():
        pred = client.classify(s)
        assert pred["safety_signal"] == s["expected_safety"]
        assert pred["decision"]      == s["expected_decision"]


def test_mock_degraded_produces_errors():
    scenarios = load_labeled()
    client    = MockAIClient(accuracy=0.50, seed=42)
    preds     = [client.classify(s) for s in scenarios]
    mismatches = sum(
        1 for s, p in zip(scenarios, preds)
        if p["safety_signal"] != s["expected_safety"]
    )
    assert mismatches > 0, "50% accuracy should produce at least one mismatch"


def test_mock_perfect_produces_zero_errors():
    scenarios = load_labeled()
    client    = MockAIClient(accuracy=1.0, seed=0)
    errors    = sum(
        1 for s in scenarios
        if client.classify(s)["safety_signal"] != s["expected_safety"]
    )
    assert errors == 0


def test_mock_deterministic_under_seed():
    scenarios = load_labeled()
    c1 = MockAIClient(accuracy=0.80, seed=7)
    c2 = MockAIClient(accuracy=0.80, seed=7)
    for s in scenarios:
        assert c1.classify(s) == c2.classify(s)


def test_mock_refuses_unsafe_correctly():
    scenarios = load_labeled()
    unsafe    = [s for s in scenarios if s["expected_safety"] == "refuse"]
    if not unsafe:
        pytest.skip("No 'refuse' scenarios in CI dataset")
    client = MockAIClient(accuracy=1.0, seed=0)
    for s in unsafe:
        assert client.classify(s)["safety_signal"] == "refuse"


# ── Full 2500-scenario tests (skipped in CI — gitignored dataset) ─────────────

@pytest.mark.skipif(not _HAS_FULL_CSV, reason="Vidya-Scenarios-2500.csv not present (gitignored)")
def test_full_dataset_loads_2500():
    scenarios = load_labeled(FULL_CSV)
    assert len(scenarios) == 2500


@pytest.mark.skipif(not _HAS_FULL_CSV, reason="Vidya-Scenarios-2500.csv not present (gitignored)")
def test_full_dataset_has_unsafe_scenarios():
    scenarios = load_labeled(FULL_CSV)
    refuse_count = sum(1 for s in scenarios if s["expected_safety"] == "refuse")
    assert refuse_count > 100
