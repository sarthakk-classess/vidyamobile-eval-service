"""
monitoring/collector.py — Metrics collector for the eval dashboard.

Finds the latest eval report from each module (retrieval, safety, tenant) and
extracts a flat metrics snapshot for the history store and drift checker.

Returns None for any module whose report directory is missing or empty
so the dashboard degrades gracefully rather than crashing.
"""

from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).parent.parent.parent


def _latest_json(reports_dir: Path, glob: str) -> Path | None:
    files = sorted(reports_dir.glob(glob), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def collect_retrieval() -> dict | None:
    """Collect metrics from the latest retrieval quality eval report."""
    report = _latest_json(_REPO / "reports" / "retrieval", "retrieval_eval_*.json")
    if not report:
        return None
    data = json.loads(report.read_text(encoding="utf-8"))
    m = data.get("metrics", {})
    return {
        "recall_at_5":    m.get("recall_at_5"),
        "recall_at_10":   m.get("recall_at_10"),
        "precision_at_5": m.get("precision_at_5"),
        "mrr":            m.get("mrr"),
        "elapsed_s":      m.get("elapsed_s"),
        "gate":           data.get("gate"),
        "passed_gate":    data.get("passed_gate"),
        "report_file":    report.name,
    }


def collect_safety() -> dict | None:
    """Collect metrics from the latest safety & agent eval report."""
    report = _latest_json(_REPO / "reports" / "safety", "safety_eval_*.json")
    if not report:
        return None
    data    = json.loads(report.read_text(encoding="utf-8"))
    safety  = data.get("safety", {})
    agents  = data.get("agents", {})
    verdict = data.get("verdict", {})
    passed  = verdict.get("passed", False) if isinstance(verdict, dict) else all(
        g.get("passed", False) for g in verdict
    )
    return {
        "mode":               data.get("mode", "unknown"),
        "refuse_recall":      safety.get("refuse_recall"),
        "safety_accuracy":    safety.get("safety_accuracy"),
        "false_refusal_rate": safety.get("false_refusal_rate"),
        "routing_accuracy":   agents.get("routing_accuracy"),
        "groundedness_rate":  agents.get("groundedness_rate"),
        "passed":             passed,
        "report_file":        report.name,
    }


def collect_tenant() -> dict | None:
    """Collect metrics from the latest tenant isolation eval report."""
    report = _latest_json(_REPO / "reports" / "tenant", "tenant_eval_*.json")
    if not report:
        return None
    data       = json.loads(report.read_text(encoding="utf-8"))
    leakage    = data.get("leakage", {})
    verdict    = data.get("verdict", {})
    per_tenant = {
        r["tenant_id"]: {"recall_at_5": r.get("recall_at_5"), "mrr": r.get("mrr")}
        for r in data.get("tenant_results", [])
    }
    return {
        "mode":         data.get("mode", "unknown"),
        "leakage_rate": leakage.get("leakage_rate"),
        "n_leaked":     leakage.get("n_leaked"),
        "per_tenant":   per_tenant,
        "passed":       verdict.get("passed", False),
        "report_file":  report.name,
    }


def collect_all(run_id: str = "", commit: str = "", branch: str = "") -> dict:
    """Collect a full snapshot from all eval modules."""
    return {
        "run_id":    run_id  or os.environ.get("GITHUB_RUN_ID", "local"),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "branch":    branch  or os.environ.get("GITHUB_REF_NAME", "local"),
        "commit":    commit  or os.environ.get("GITHUB_SHA", "")[:8],
        "retrieval": collect_retrieval(),
        "safety":    collect_safety(),
        "tenant":    collect_tenant(),
    }
