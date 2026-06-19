"""
monitoring/drift.py — Drift detector for eval metrics.

Compares the current metrics snapshot against a rolling baseline and
returns a list of alerts for any metric that has regressed beyond threshold.

Direction matters: recall/accuracy metrics alert on drops; error/leakage
rate metrics alert on rises.
"""

from __future__ import annotations
import json
from pathlib import Path

_HISTORY_FILE = Path(__file__).parent.parent.parent / "history" / "metrics.jsonl"

# (metric_path, direction, threshold)
_DRIFT_RULES: list[tuple[str, str, float]] = [
    ("retrieval.recall_at_5",       "down", 0.05),
    ("retrieval.mrr",               "down", 0.05),
    ("safety.refuse_recall",        "down", 0.03),
    ("safety.safety_accuracy",      "down", 0.03),
    ("safety.false_refusal_rate",   "up",   0.02),
    ("safety.routing_accuracy",     "down", 0.05),
    ("safety.groundedness_rate",    "down", 0.05),
    ("tenant.leakage_rate",         "up",   0.0),   # any leak is critical
]


def _get(snapshot: dict, path: str):
    node = snapshot
    for p in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(p)
    return node


def load_history(n: int = 10) -> list[dict]:
    """Return the last N snapshots from the history file."""
    if not _HISTORY_FILE.exists():
        return []
    lines   = [l.strip() for l in _HISTORY_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records[-n:]


def check_drift(current: dict, history: list[dict], window: int = 5) -> list[dict]:
    """
    Compare current snapshot against rolling mean of last window records.

    Returns list of alert dicts:
        {metric, current, baseline, delta, direction, threshold, severity}
    severity: "critical" if delta > 2x threshold, else "warning"
    """
    baseline_records = history[-window:] if history else []
    if not baseline_records:
        return []

    alerts = []

    for metric_path, direction, threshold in _DRIFT_RULES:
        current_val = _get(current, metric_path)
        if current_val is None:
            continue
        baseline_vals = [v for r in baseline_records if (v := _get(r, metric_path)) is not None]
        if not baseline_vals:
            continue
        baseline_mean = sum(baseline_vals) / len(baseline_vals)
        delta         = current_val - baseline_mean
        triggered = (
            (direction == "down" and delta < -threshold) or
            (direction == "up"   and delta >  threshold)
        )
        if triggered:
            alerts.append({
                "metric":    metric_path,
                "current":   round(current_val, 4),
                "baseline":  round(baseline_mean, 4),
                "delta":     round(delta, 4),
                "direction": direction,
                "threshold": threshold,
                "severity":  "critical" if abs(delta) > 2 * threshold else "warning",
            })

    # Per-tenant recall drift
    for tenant_id, tenant_data in (_get(current, "tenant.per_tenant") or {}).items():
        current_r5 = tenant_data.get("recall_at_5") if isinstance(tenant_data, dict) else None
        if current_r5 is None:
            continue
        baseline_vals = [
            v for r in baseline_records
            if (v := _get(r, f"tenant.per_tenant.{tenant_id}.recall_at_5")) is not None
        ]
        if not baseline_vals:
            continue
        baseline_mean = sum(baseline_vals) / len(baseline_vals)
        delta         = current_r5 - baseline_mean
        if delta < -0.05:
            alerts.append({
                "metric":    f"tenant.per_tenant.{tenant_id}.recall_at_5",
                "current":   round(current_r5, 4),
                "baseline":  round(baseline_mean, 4),
                "delta":     round(delta, 4),
                "direction": "down",
                "threshold": 0.05,
                "severity":  "critical" if abs(delta) > 0.10 else "warning",
            })

    return alerts


def append_history(snapshot: dict) -> None:
    """Append snapshot as a new line in history/metrics.jsonl."""
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, separators=(",", ":")) + "\n")
