# Datasets

## Retrieval eval datasets (`datasets/retrieval/`)

Used by `eval/retrieval_eval.py` — offline, no API key required.

| File | Size | Source |
|------|------|--------|
| `benchmark_dataset.json` | 75 QA pairs | Embedding selection benchmark (SK-02) |
| `retrieval_tests.json` | 15 structural checks | Chunker validation tests (SK-01) |

Format — benchmark_dataset.json:
```json
[
  {
    "query_id": "q001",
    "doc_type": "syllabus",
    "query": "What topics are covered in Unit II?",
    "relevant_chunk_ids": ["syl_abc123_0001"],
    "document_text": "Unit II covers data structures..."
  }
]
```

## Safety eval datasets (`datasets/safety/`)

Used by `eval/safety_eval.py`.

| File | Committed | Notes |
|------|-----------|-------|
| `ci_scenarios.csv` | Yes | 30-scenario CI subset |
| `Vidya-Scenarios-2500.csv` | **No (gitignored)** | Full 2500-scenario set, local only |

## Tenant isolation datasets (`datasets/tenant/`)

Used by `eval/tenant_eval.py` and `eval_service/tenant/seed_kb.py`.

| File | Tenant | Description |
|------|--------|-------------|
| `tenant_a_queries.json` | `state-univ` | Query + ground-truth chunk IDs |
| `tenant_b_queries.json` | `city-college` | Query + ground-truth chunk IDs |
| `leakage_probes.json` | cross-tenant | Probes that must NOT leak across tenants |
