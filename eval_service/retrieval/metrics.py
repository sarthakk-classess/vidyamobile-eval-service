"""
retrieval/metrics.py
────────────────────
Recall@K, Precision@K, MRR — self-contained, no external deps.
"""

from __future__ import annotations
import numpy as np


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    if not retrieved_ids or k == 0:
        return 0.0
    hits = sum(1 for cid in retrieved_ids[:k] if cid in set(relevant_ids))
    return hits / k


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant_set = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def aggregate_metrics(query_results: list[dict]) -> dict:
    """
    Parameters
    ----------
    query_results : list of {"query_id", "retrieved_ids", "relevant_ids"}

    Returns
    -------
    dict : recall_at_5, recall_at_10, precision_at_5, mrr, num_queries
    """
    if not query_results:
        return {"recall_at_5": 0.0, "recall_at_10": 0.0,
                "precision_at_5": 0.0, "mrr": 0.0, "num_queries": 0}

    r5  = [recall_at_k(r["retrieved_ids"], r["relevant_ids"], 5)  for r in query_results]
    r10 = [recall_at_k(r["retrieved_ids"], r["relevant_ids"], 10) for r in query_results]
    p5  = [precision_at_k(r["retrieved_ids"], r["relevant_ids"], 5) for r in query_results]
    rrs = [reciprocal_rank(r["retrieved_ids"], r["relevant_ids"]) for r in query_results]

    return {
        "recall_at_5":    round(float(np.mean(r5)),  4),
        "recall_at_10":   round(float(np.mean(r10)), 4),
        "precision_at_5": round(float(np.mean(p5)),  4),
        "mrr":            round(float(np.mean(rrs)), 4),
        "num_queries":    len(query_results),
    }
