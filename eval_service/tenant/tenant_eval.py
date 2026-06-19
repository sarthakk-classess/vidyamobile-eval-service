"""tenant/tenant_eval.py — Per-tenant retrieval quality metrics (SK-10)."""

from __future__ import annotations
from eval_service.tenant.client import TenantKBClient


def evaluate_tenant(
    tenant_id: str,
    queries:   list[dict],
    client:    TenantKBClient,
    top_k:     int = 5,
) -> dict:
    """
    Run all queries for one tenant and compute recall@5, MRR, precision@5.

    Returns:
        {tenant_id, n_queries, recall_at_5, mrr, precision_at_5, failures}
    """
    hits = rr_sum = precision_sum = 0.0
    failures = []

    for entry in queries:
        qid      = entry["query_id"]
        query    = entry["query"]
        relevant = set(entry["relevant_chunk_ids"])

        results       = client.query(tenant_id, query, top_k=top_k)
        retrieved_ids = [r["chunk_id"] for r in results]

        hit = any(rid in relevant for rid in retrieved_ids)
        hits += int(hit)

        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant:
                rr = 1.0 / rank
                break
        rr_sum += rr

        n_relevant_retrieved = sum(1 for rid in retrieved_ids if rid in relevant)
        precision_sum += n_relevant_retrieved / top_k

        if not hit:
            failures.append({
                "query_id":      qid,
                "query":         query,
                "retrieved_ids": retrieved_ids,
                "relevant_ids":  list(relevant),
            })

    n = len(queries)
    return {
        "tenant_id":      tenant_id,
        "n_queries":      n,
        "recall_at_5":    hits / n if n else 0.0,
        "mrr":            rr_sum / n if n else 0.0,
        "precision_at_5": precision_sum / n if n else 0.0,
        "failures":       failures,
    }
