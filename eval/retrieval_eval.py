"""
eval/retrieval_eval.py
──────────────────────
Offline retrieval evaluation runner.

Loads benchmark_dataset.json (75 queries) and retrieval_tests.json
(15 structural checks), runs cosine-similarity retrieval using a local
sentence-transformers model, checks the recall@5 gate, and writes a
JSON report to reports/retrieval/.

No API keys required. Runs in ~30s on CPU.

Usage
─────
    python eval/retrieval_eval.py              # offline, default gate (0.87)
    python eval/retrieval_eval.py --gate 0.80  # override gate
    python eval/retrieval_eval.py --quiet      # suppress per-query output

Exit codes
──────────
    0 : recall@5 >= gate (PASS)
    1 : recall@5 < gate  (FAIL)
    2 : dataset missing or error
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).parent.parent

DATASETS_DIR  = _ROOT / "datasets" / "retrieval"
BENCHMARK     = DATASETS_DIR / "benchmark_dataset.json"
STRUCT_TESTS  = DATASETS_DIR / "retrieval_tests.json"
REPORTS_DIR   = _ROOT / "reports" / "retrieval"

sys.path.insert(0, str(_ROOT))

from eval_service.retrieval.offline_retrieval import OfflineRetrieval
from eval_service.retrieval.gate_checker      import RetrievalGateChecker
from eval_service.retrieval.metrics           import aggregate_metrics, recall_at_k


def _load_benchmark() -> list[dict]:
    if not BENCHMARK.exists():
        raise FileNotFoundError(f"Benchmark dataset not found: {BENCHMARK}")
    data = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    print(f"  Benchmark dataset: {len(data)} queries")
    return data


def _load_struct_tests() -> list[dict]:
    if not STRUCT_TESTS.exists():
        print(f"  [WARN] Structural tests not found: {STRUCT_TESTS}")
        return []
    data = json.loads(STRUCT_TESTS.read_text(encoding="utf-8"))
    print(f"  Structural tests:  {len(data)} queries")
    return data


def _run_struct_checks(tests: list[dict], engine: OfflineRetrieval, verbose: bool) -> list[dict]:
    results = []
    for t in tests:
        expected = t.get("expected_chunk_contains", "")
        try:
            candidates = engine.retrieve(t["query"], top_k=5)
        except Exception as e:
            results.append({"test_id": t["test_id"], "passed": False, "error": str(e)})
            continue

        found = any(expected.lower() in c["text"].lower() for c in candidates if expected)
        passed = found or not expected

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {t['test_id']}: {t['query'][:55]}")

        results.append({
            "test_id":   t["test_id"],
            "doc_type":  t.get("doc_type", "unknown"),
            "query":     t["query"],
            "passed":    passed,
            "top_result": candidates[0]["text"][:200] if candidates else "",
        })

    n_pass = sum(1 for r in results if r["passed"])
    print(f"\n  Structural checks: {n_pass}/{len(results)} passed")
    return results


def _metrics_by_doctype(query_results: list[dict], dataset: list[dict]) -> dict[str, dict]:
    doc_type_map = {e["query_id"]: e.get("doc_type", "unknown") for e in dataset}
    by_type: dict[str, list] = {}
    for r in query_results:
        dt = doc_type_map.get(r["query_id"], "unknown")
        by_type.setdefault(dt, []).append(r)
    return {dt: aggregate_metrics(rs) for dt, rs in by_type.items()}


def _write_report(
    metrics: dict,
    metrics_by_type: dict,
    struct_results: list[dict],
    gate: float,
    passed: bool,
    elapsed_s: float,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"retrieval_eval_{ts}.json"

    report = {
        "timestamp":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode":            "OFFLINE",
        "gate":            gate,
        "passed_gate":     passed,
        "metrics": {**metrics, "elapsed_s": elapsed_s},
        "metrics_by_type": metrics_by_type,
        "structural_checks": {
            "total":      len(struct_results),
            "passed":     sum(1 for r in struct_results if r["passed"]),
            "failed":     sum(1 for r in struct_results if not r["passed"]),
            "failed_ids": [r["test_id"] for r in struct_results if not r["passed"]],
        },
    }

    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Report written: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Retrieval quality eval (offline)")
    parser.add_argument("--gate",  type=float, default=0.87, help="Recall@5 gate (default 0.87)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress per-query output")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Retrieval Evaluation — Offline Mode")
    print(f"Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    try:
        benchmark    = _load_benchmark()
        struct_tests = _load_struct_tests()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(2)

    engine = OfflineRetrieval()

    # Build corpus from benchmark + structural test texts
    combined = list(benchmark)
    for t in struct_tests:
        if t.get("expected_chunk_contains"):
            combined.append({
                "query_id":           t["test_id"],
                "query":              t["query"],
                "relevant_chunk_ids": [t["test_id"]],
                "document_text":      t["expected_chunk_contains"],
            })
    print(f"\n  Building corpus ({len(combined)} documents) ...")
    engine.build_corpus(combined)

    # Structural checks
    print(f"\n{'─'*60}")
    print(f"STRUCTURAL CHECKS ({len(struct_tests)} queries)")
    print("─" * 60)
    struct_results = _run_struct_checks(struct_tests, engine, verbose=not args.quiet)

    # Retrieval quality eval
    print(f"\n{'─'*60}")
    print(f"RETRIEVAL QUALITY ({len(benchmark)} queries)")
    print("─" * 60)

    t0 = time.time()
    query_results = engine.retrieve_batch(benchmark, top_k=20)
    elapsed = time.time() - t0
    print(f"  Retrieval complete in {elapsed:.1f}s")

    if not args.quiet:
        for r in query_results:
            r5     = recall_at_k(r["retrieved_ids"], r["relevant_ids"], 5)
            status = "PASS" if r5 > 0 else "FAIL"
            print(f"  [{status}] {r['query_id']}: recall@5={r5:.1f} — {r.get('query','')[:50]}")

    metrics         = aggregate_metrics(query_results)
    metrics_by_type = _metrics_by_doctype(query_results, benchmark)

    checker = RetrievalGateChecker(gate=args.gate)
    passed  = checker.check(metrics)
    checker.check_by_doctype(metrics_by_type)

    _write_report(metrics, metrics_by_type, struct_results, args.gate, passed, elapsed)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
