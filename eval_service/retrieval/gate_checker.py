"""
retrieval/gate_checker.py
─────────────────────────
Release gate for retrieval recall@5. Threshold is locked in params.py.
"""

from __future__ import annotations

RECALL_GATE = 0.87


class RetrievalGateChecker:
    def __init__(self, gate: float = RECALL_GATE):
        self.gate = gate

    def check(self, metrics: dict) -> bool:
        r5     = metrics.get("recall_at_5", 0.0)
        passed = r5 >= self.gate
        self._print(metrics, r5, passed)
        return passed

    def check_by_doctype(self, metrics_by_doctype: dict[str, dict]) -> dict[str, bool]:
        results = {}
        for doc_type, m in metrics_by_doctype.items():
            r5 = m.get("recall_at_5", 0.0)
            ok = r5 >= self.gate
            results[doc_type] = ok
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {doc_type:<20} recall@5={r5:.4f}  gate={self.gate:.2f}")
        return results

    def summary_line(self, metrics: dict) -> str:
        r5     = metrics.get("recall_at_5", 0.0)
        status = "PASS" if r5 >= self.gate else "FAIL"
        return (
            f"{status} | recall@5={r5:.4f} | gate={self.gate:.2f} | "
            f"precision@5={metrics.get('precision_at_5', 0.0):.4f} | "
            f"MRR={metrics.get('mrr', 0.0):.4f}"
        )

    def _print(self, metrics: dict, r5: float, passed: bool) -> None:
        w = 60
        print("\n" + "=" * w)
        print("RETRIEVAL RECALL GATE CHECK")
        print("=" * w)
        print(f"  Recall@5        : {r5:.4f}")
        print(f"  Recall@10       : {metrics.get('recall_at_10', 0.0):.4f}")
        print(f"  Precision@5     : {metrics.get('precision_at_5', 0.0):.4f}")
        print(f"  MRR             : {metrics.get('mrr', 0.0):.4f}")
        print(f"  Num queries     : {metrics.get('num_queries', '?')}")
        print(f"  Gate threshold  : {self.gate:.2f}")
        print("-" * w)
        if passed:
            print(f"  PASS — recall@5 {r5:.4f} >= gate {self.gate:.2f}")
        else:
            print(f"  FAIL — recall@5 {r5:.4f} is {self.gate - r5:.4f} below gate {self.gate:.2f}")
            print("  Do NOT ship until recall is restored.")
        print("=" * w + "\n")
