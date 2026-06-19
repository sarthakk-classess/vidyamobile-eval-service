"""tenant/leakage_eval.py — Cross-tenant leakage detection (SK-10)."""

from __future__ import annotations
from eval_service.tenant.client import TenantKBClient


def evaluate_leakage(
    probes:                 list[dict],
    client:                 TenantKBClient,
    known_chunks_by_tenant: dict[str, set[str]],
    top_k:                  int = 5,
) -> dict:
    """
    Run leakage probes and report any cross-tenant hits.

    Returns:
        {n_probes, n_leaked, leakage_rate, leaked_cases}
    """
    n_leaked:     int        = 0
    leaked_cases: list[dict] = []

    for probe in probes:
        probe_id      = probe["probe_id"]
        probe_tenant  = probe["probe_tenant_id"]
        forbidden     = probe["forbidden_tenant_id"]
        query         = probe["query"]
        forbidden_set = known_chunks_by_tenant.get(forbidden, set())

        results       = client.query(probe_tenant, query, top_k=top_k)
        retrieved_ids = [r["chunk_id"] for r in results]

        leaked_ids = [rid for rid in retrieved_ids if rid in forbidden_set]
        if leaked_ids:
            n_leaked += 1
            leaked_cases.append({
                "probe_id":             probe_id,
                "query":                query,
                "probe_tenant_id":      probe_tenant,
                "forbidden_tenant_id":  forbidden,
                "leaked_chunk_ids":     leaked_ids,
            })

    n = len(probes)
    return {
        "n_probes":     n,
        "n_leaked":     n_leaked,
        "leakage_rate": n_leaked / n if n else 0.0,
        "leaked_cases": leaked_cases,
    }


def build_known_chunks(datasets: dict[str, list[dict]]) -> dict[str, set[str]]:
    """Build {tenant_id: set of chunk_ids} for leakage detection."""
    known: dict[str, set[str]] = {}
    for tenant_id, entries in datasets.items():
        ids: set[str] = set()
        for entry in entries:
            ids.update(entry["relevant_chunk_ids"])
        known[tenant_id] = ids
    return known
