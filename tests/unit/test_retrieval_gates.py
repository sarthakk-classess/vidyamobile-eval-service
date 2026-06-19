"""
Unit tests for retrieval gate checker and metrics.
No model or API key required.
"""
import pytest

from eval_service.retrieval.gate_checker import RetrievalGateChecker, RECALL_GATE
from eval_service.retrieval.metrics import (
    recall_at_k, precision_at_k, reciprocal_rank, aggregate_metrics,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

PASS_METRICS = {
    "recall_at_5": 0.93, "recall_at_10": 0.97,
    "precision_at_5": 0.45, "mrr": 0.87, "num_queries": 75,
}
FAIL_METRICS = {
    "recall_at_5": 0.72, "recall_at_10": 0.80,
    "precision_at_5": 0.35, "mrr": 0.65, "num_queries": 75,
}
EXACT_GATE_METRICS = {
    "recall_at_5": RECALL_GATE, "recall_at_10": 0.90,
    "precision_at_5": 0.40,    "mrr": 0.75, "num_queries": 75,
}


# ── recall_at_k ───────────────────────────────────────────────────────────────

def test_recall_perfect_hit():
    assert recall_at_k(["c1", "c2", "c3"], ["c1"], k=5) == 1.0


def test_recall_miss():
    assert recall_at_k(["c4", "c5"], ["c1"], k=5) == 0.0


def test_recall_partial():
    assert recall_at_k(["c1", "c3", "c5"], ["c1", "c2"], k=5) == pytest.approx(0.5)


def test_recall_k_cutoff_respected():
    # Relevant chunk is at position 6 — outside k=5
    assert recall_at_k(["c1", "c2", "c3", "c4", "c5", "c_rel"], ["c_rel"], k=5) == 0.0


def test_recall_empty_relevant():
    assert recall_at_k(["c1", "c2"], [], k=5) == 0.0


# ── precision_at_k ───────────────────────────────────────────────────────────

def test_precision_all_relevant():
    assert precision_at_k(["c1", "c2", "c3"], ["c1", "c2", "c3"], k=3) == 1.0


def test_precision_none_relevant():
    assert precision_at_k(["c4", "c5"], ["c1"], k=2) == 0.0


def test_precision_partial():
    assert precision_at_k(["c1", "c4", "c5"], ["c1", "c2"], k=3) == pytest.approx(1/3)


# ── reciprocal_rank ───────────────────────────────────────────────────────────

def test_rr_first_hit():
    assert reciprocal_rank(["c1", "c2", "c3"], ["c1"]) == pytest.approx(1.0)


def test_rr_second_hit():
    assert reciprocal_rank(["c4", "c1", "c3"], ["c1"]) == pytest.approx(0.5)


def test_rr_no_hit():
    assert reciprocal_rank(["c4", "c5"], ["c1"]) == 0.0


# ── aggregate_metrics ─────────────────────────────────────────────────────────

def test_aggregate_empty():
    m = aggregate_metrics([])
    assert m["recall_at_5"] == 0.0
    assert m["num_queries"] == 0


def test_aggregate_perfect():
    results = [
        {"query_id": "q1", "retrieved_ids": ["c1", "c2"], "relevant_ids": ["c1"]},
        {"query_id": "q2", "retrieved_ids": ["c2", "c3"], "relevant_ids": ["c2"]},
    ]
    m = aggregate_metrics(results)
    assert m["recall_at_5"] == 1.0
    assert m["mrr"] == 1.0
    assert m["num_queries"] == 2


def test_aggregate_keys():
    m = aggregate_metrics([{"query_id": "q1", "retrieved_ids": ["c1"], "relevant_ids": ["c1"]}])
    for k in ("recall_at_5", "recall_at_10", "precision_at_5", "mrr", "num_queries"):
        assert k in m


# ── RetrievalGateChecker ──────────────────────────────────────────────────────

def test_gate_default_is_locked_value():
    checker = RetrievalGateChecker()
    assert checker.gate == RECALL_GATE


def test_gate_explicit_override():
    checker = RetrievalGateChecker(gate=0.80)
    assert checker.gate == 0.80


def test_gate_passes_above_threshold(capsys):
    assert RetrievalGateChecker(gate=0.87).check(PASS_METRICS) is True


def test_gate_fails_below_threshold(capsys):
    assert RetrievalGateChecker(gate=0.87).check(FAIL_METRICS) is False


def test_gate_passes_exactly_at_threshold(capsys):
    assert RetrievalGateChecker(gate=RECALL_GATE).check(EXACT_GATE_METRICS) is True


def test_gate_zero_recall_fails(capsys):
    assert RetrievalGateChecker(gate=0.80).check({"recall_at_5": 0.0}) is False


def test_gate_missing_key_treated_as_zero(capsys):
    assert RetrievalGateChecker(gate=0.80).check({}) is False


def test_gate_does_not_sys_exit(capsys):
    try:
        RetrievalGateChecker(gate=0.99).check(FAIL_METRICS)
    except SystemExit:
        pytest.fail("RetrievalGateChecker.check() must not call sys.exit()")


def test_gate_output_pass(capsys):
    RetrievalGateChecker(gate=0.87).check(PASS_METRICS)
    assert "PASS" in capsys.readouterr().out


def test_gate_output_fail(capsys):
    RetrievalGateChecker(gate=0.87).check(FAIL_METRICS)
    assert "FAIL" in capsys.readouterr().out


def test_check_by_doctype(capsys):
    checker = RetrievalGateChecker(gate=0.87)
    result  = checker.check_by_doctype({
        "syllabus":         {"recall_at_5": 0.95},
        "lecture_slides":   {"recall_at_5": 0.70},
        "academic_reading": {"recall_at_5": 0.90},
    })
    assert result["syllabus"]         is True
    assert result["lecture_slides"]   is False
    assert result["academic_reading"] is True


def test_check_by_doctype_empty(capsys):
    assert RetrievalGateChecker(gate=0.87).check_by_doctype({}) == {}


def test_summary_line_contains_pass(capsys):
    line = RetrievalGateChecker(gate=0.87).summary_line(PASS_METRICS)
    assert "PASS" in line
    assert "0.93" in line


def test_summary_line_contains_fail():
    line = RetrievalGateChecker(gate=0.87).summary_line(FAIL_METRICS)
    assert "FAIL" in line


def test_summary_line_is_single_line():
    line = RetrievalGateChecker(gate=0.87).summary_line(PASS_METRICS)
    assert "\n" not in line
