# Datasets

## Safety eval datasets

Copy from the original VidyaMobile repo:

```
datasets/safety/ci_scenarios.csv   ← copy from sk08/datasets/ci_scenarios.csv
```

The full 2500-scenario CSV is gitignored — it stays local only.

## Tenant isolation datasets

These files need to be created (or copied from the original repo's sk10/datasets/ if they exist):

```
datasets/tenant/tenant_a_queries.json   # queries + ground-truth chunk IDs for "state-univ"
datasets/tenant/tenant_b_queries.json   # queries + ground-truth chunk IDs for "city-college"
datasets/tenant/leakage_probes.json     # cross-tenant probes to test isolation
```

### Format — tenant query files

```json
[
  {
    "query": "What is photosynthesis?",
    "chunk_ids": ["ta_c1", "ta_c2"]
  }
]
```

### Format — leakage probes

```json
[
  {
    "query": "What is photosynthesis?",
    "tenant_id": "state-univ",
    "forbidden_tenant_id": "city-college"
  }
]
```
