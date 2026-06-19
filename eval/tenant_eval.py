"""
eval/tenant_eval.py — Tenant Isolation Evaluation runner.

Usage:
    python eval/tenant_eval.py                      # offline mock (isolated)
    python eval/tenant_eval.py --mock-mode leaky    # leaky mock (gate must FAIL)
    TENANT_LIVE=1 python eval/tenant_eval.py        # live RT-18 edge function
    TENANT_DIRECT=1 python eval/tenant_eval.py      # Gemini embed + direct RPC (CI)

Exits non-zero if any gate fails.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_service.tenant.client       import get_client
from eval_service.tenant.tenant_eval  import evaluate_tenant
from eval_service.tenant.leakage_eval import evaluate_leakage, build_known_chunks
from eval_service.tenant.gates        import check_gates

_ROOT        = Path(__file__).parent.parent
DATASETS_DIR = _ROOT / "datasets" / "tenant"
REPORTS_DIR  = _ROOT / "reports"  / "tenant"

TENANT_DATASET_FILES = {
    "state-univ":   "tenant_a_queries.json",
    "city-college": "tenant_b_queries.json",
}


def load_datasets() -> dict[str, list[dict]]:
    datasets = {}
    for tenant_id, filename in TENANT_DATASET_FILES.items():
        path = DATASETS_DIR / filename
        datasets[tenant_id] = json.loads(path.read_text(encoding="utf-8"))
    return datasets


def load_probes() -> list[dict]:
    return json.loads((DATASETS_DIR / "leakage_probes.json").read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="Mascot retrieval — tenant isolation evaluation")
    ap.add_argument("--mock-mode", choices=["isolated", "leaky"], default=None)
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    if args.mock_mode:
        os.environ["TENANT_MOCK_MODE"] = args.mock_mode

    live      = os.environ.get("TENANT_LIVE") == "1"
    direct    = os.environ.get("TENANT_DIRECT") == "1"
    mock_mode = os.environ.get("TENANT_MOCK_MODE", "isolated")
    mode_label = "LIVE (RT-18)" if live else ("DIRECT (Gemini+RPC)" if direct else f"MOCK ({mock_mode})")

    print("=" * 60)
    print("Mascot Retrieval — Tenant Isolation Evaluation")
    print(f"Mode: {mode_label}")
    print("=" * 60)

    datasets     = load_datasets()
    probes       = load_probes()
    known_chunks = build_known_chunks(datasets)
    client       = get_client(dataset_dir=DATASETS_DIR if not live and not direct else None)

    tenant_results = []
    for tenant_id, queries in datasets.items():
        print(f"\n[{tenant_id}] evaluating {len(queries)} queries ...")
        result = evaluate_tenant(tenant_id, queries, client, top_k=args.top_k)
        tenant_results.append(result)
        print(f"  recall@{args.top_k}={result['recall_at_5']:.4f}  MRR={result['mrr']:.4f}  "
              f"({result['n_queries']} queries, {len(result['failures'])} misses)")

    print(f"\n[leakage] running {len(probes)} cross-tenant probes ...")
    leakage = evaluate_leakage(probes, client, known_chunks, top_k=args.top_k)
    print(f"  leakage_rate={leakage['leakage_rate']:.4f}  "
          f"({leakage['n_leaked']} leaked / {leakage['n_probes']} probes)")

    verdict = check_gates(tenant_results, leakage)

    print("\n-- Release gates --")
    for g in verdict["gates"]:
        tenant_tag = f" [{g['tenant_id']}]" if g["tenant_id"] != "all" else ""
        print(f"  [{'PASS' if g['passed'] else 'FAIL'}] {g['gate']:30s}{tenant_tag:16s} "
              f"{g['value']:.4f} {g['op']} {g['threshold']}")

    print("\n" + "=" * 60)
    print("RESULT:", "PASS — safe to ship Mascot" if verdict["passed"] else "FAIL — regression blocks Mascot release")
    print("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts          = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"tenant_eval_{ts}.json"
    report_path.write_text(
        json.dumps({"mode": mode_label, "tenant_results": tenant_results, "leakage": leakage, "verdict": verdict}, indent=2),
        encoding="utf-8",
    )
    print(f"Report: {report_path}")
    sys.exit(0 if verdict["passed"] else 1)


if __name__ == "__main__":
    main()
