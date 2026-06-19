# VidyaMobile — Eval Service

Production-structured evaluation and mastery service for the VidyaMobile AI tutoring system.

---

## What's here

| Module | Purpose |
|--------|---------|
| `eval_service/chunkers/` | Deterministic chunking — syllabus / slides / academic content types |
| `eval_service/embeddings/` | Gemini `text-embedding-2`, 1536-dim vector generation |
| `eval_service/mastery/` | FSRS-4.5 spaced-repetition scheduler (20 params, 0.90 target retention) |
| `eval_service/retrieval/` | Locked retrieval params + Supabase `match_kb_chunks_for_tenant` RPC client |
| `eval_service/difficulty/` | GBR difficulty predictor (R²=0.877, MAE=0.075) |
| `eval_service/safety/` | Safety label derivation + AI service client (mock & live) |
| `eval_service/tenant/` | Mascot retrieval + tenant isolation eval harness |
| `eval_service/monitoring/` | Metric collection + rolling drift check |
| `routers/` | FastAPI route handlers (chunk, embed, mastery, health) |
| `eval/` | Standalone eval runner scripts |
| `tests/` | Unit + integration tests |

---

## Quick start

```bash
cp .env.example .env
# fill in GEMINI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY

pip install -r requirements-dev.txt

# retrain difficulty model (once, or after new interaction logs)
python eval_service/difficulty/train.py

# run the service
uvicorn eval_service.main:app --reload --port 8002

# run evals
python eval/safety_eval.py           # mock (no API key needed)
python eval/tenant_eval.py           # mock (no Supabase needed)
python eval/drift_dashboard.py --local
```

---

## Eval runners

| Script | What it does | Mode env var |
|--------|-------------|-------------|
| `eval/safety_eval.py` | Safety & agent routing eval (5 gates) | `SAFETY_LIVE=1` for live AI service |
| `eval/tenant_eval.py` | Tenant isolation + leakage eval | `TENANT_DIRECT=1` for Gemini+RPC |
| `eval/drift_dashboard.py` | Drift monitor + markdown dashboard | — |

---

## Release gates

| Gate | Threshold |
|------|-----------|
| Retrieval recall@5 | >= 0.87 |
| Safety refuse_recall | >= 0.95 |
| Safety safety_accuracy | >= 0.90 |
| Safety false_refusal_rate | <= 0.02 |
| Safety routing_accuracy | >= 0.85 |
| Safety groundedness | >= 0.90 |
| Tenant per-tenant recall@5 | >= 0.87 |
| Tenant per-tenant MRR | >= 0.70 |
| Tenant leakage_rate | == 0.0 |

---

## Locked decisions

| Decision | Value |
|----------|-------|
| Embedding model | `gemini-embedding-2`, `output_dimensionality=1536` |
| Reranker | None (Phase 1) |
| Best retrieval params | `match_count=3`, `min_similarity=0.0`, `ef_search=40` |
| FSRS version | FSRS-4.5, 20 params, target retention 0.90 |
| pgvector dimension | `vector(1536)` |

---

## Dataset setup

See [datasets/DATASETS.md](datasets/DATASETS.md) for instructions on copying eval dataset files.

---

## CI

4 jobs in `.github/workflows/ci.yml`:

1. **unit-tests** — pytest + safety mock + tenant mock (gates PR merges)
2. **safety-live-eval** — live AI service eval (`continue-on-error: true`)
3. **tenant-live-eval** — local Supabase with Rishabh's migrations (`continue-on-error: true`)
4. **drift-dashboard** — drift check + GitHub Step Summary + commits history

Secrets needed: `GEMINI_API_KEY`, `GEMINI_CLASSIFIER_KEY`, `GEMINI_FRONTIER_KEY`.

---

## Integration contracts

- **→ Rishabh (backend):** Best params in `eval_service/retrieval/params.py`; FSRS spec in `eval_service/mastery/`
- **→ Himanshu (AI service):** Safety harness hits `POST /v1/turn` on his service (port 8000)
- **← Rishabh:** `match_kb_chunks_for_tenant` RPC with `p_tenant_id`, `p_doc_type`, `p_query_embedding`
