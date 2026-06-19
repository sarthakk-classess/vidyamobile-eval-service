"""
Integration tests for SK-10 tenant isolation (mock client only — no live Supabase).

These tests verify that the MockTenantKBClient (isolated mode) never returns
chunks from another tenant, and that the leaky mock produces detectable leakage.
"""
import pytest
from eval_service.tenant.client import MockTenantKBClient
from eval_service.tenant.leakage_eval import evaluate_leakage, build_known_chunks
from eval_service.tenant.tenant_eval import evaluate_tenant


TENANT_A_QUERIES = [
    {"query": "What is photosynthesis?",      "chunk_ids": ["ta_c1", "ta_c2"]},
    {"query": "Explain the water cycle.",     "chunk_ids": ["ta_c3"]},
    {"query": "What is Newton's first law?",  "chunk_ids": ["ta_c4", "ta_c5"]},
]
TENANT_B_QUERIES = [
    {"query": "Describe mitosis.",            "chunk_ids": ["tb_c1", "tb_c2"]},
    {"query": "What is an ecosystem?",        "chunk_ids": ["tb_c3"]},
]
LEAKAGE_PROBES = [
    {"query": "What is photosynthesis?", "tenant_id": "state-univ",   "forbidden_tenant_id": "city-college"},
    {"query": "Describe mitosis.",       "tenant_id": "city-college",  "forbidden_tenant_id": "state-univ"},
]


@pytest.fixture
def isolated_client():
    client = MockTenantKBClient(mode="isolated")
    client._seed({
        "state-univ":   TENANT_A_QUERIES,
        "city-college": TENANT_B_QUERIES,
    })
    return client


@pytest.fixture
def leaky_client():
    client = MockTenantKBClient(mode="leaky")
    client._seed({
        "state-univ":   TENANT_A_QUERIES,
        "city-college": TENANT_B_QUERIES,
    })
    return client


def test_isolated_tenant_recall(isolated_client):
    result = evaluate_tenant("state-univ", TENANT_A_QUERIES, isolated_client, top_k=5)
    assert result["recall_at_5"] >= 0.87


def test_isolated_no_leakage(isolated_client):
    known = build_known_chunks({
        "state-univ":   TENANT_A_QUERIES,
        "city-college": TENANT_B_QUERIES,
    })
    leakage = evaluate_leakage(LEAKAGE_PROBES, isolated_client, known, top_k=5)
    assert leakage["leakage_rate"] == 0.0, f"Expected 0 leakage, got {leakage['leakage_rate']}"


def test_leaky_produces_leakage(leaky_client):
    known = build_known_chunks({
        "state-univ":   TENANT_A_QUERIES,
        "city-college": TENANT_B_QUERIES,
    })
    leakage = evaluate_leakage(LEAKAGE_PROBES, leaky_client, known, top_k=5)
    assert leakage["leakage_rate"] > 0.0, "Leaky mock should produce cross-tenant leakage"
