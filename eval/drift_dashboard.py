"""
eval/drift_dashboard.py — Eval Dashboard & Drift Monitor.

Collects metrics from latest retrieval / safety / tenant reports, appends to
history, runs drift check, and writes a markdown summary to:
  - $GITHUB_STEP_SUMMARY (GitHub Actions UI)
  - reports/monitoring/dashboard_latest.md

Usage:
    python eval/drift_dashboard.py           # CI
    python eval/drift_dashboard.py --local   # local run
    python eval/drift_dashboard.py --window 10  # wider baseline window
"""

from __future__ import annotations
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_service.monitoring.collector import collect_all
from eval_service.monitoring.drift     import append_history, check_drift, load_history, _get

_ROOT       = Path(__file__).parent.parent
REPORTS_DIR = _ROOT / "reports" / "monitoring"


def _fmt(value, precision: int = 4) -> str:
    if value is None:            return "—"
    if isinstance(value, bool):  return "PASS" if value else "FAIL"
    if isinstance(value, float): return f"{value:.{precision}f}"
    return str(value)


def _trend(current, history_records: list[dict], metric_path: str) -> str:
    if not history_records or current is None:
        return ""
    last = _get(history_records[-1], metric_path)
    if last is None:
        return ""
    diff = current - last
    if diff > 0.005:  return " ^"
    if diff < -0.005: return " v"
    return " -"


def build_summary(snapshot: dict, history: list[dict], alerts: list[dict]) -> str:
    lines = [
        "# Eval Dashboard",
        "",
        f"**Run:** `{snapshot['run_id']}`  "
        f"**Branch:** `{snapshot['branch']}`  "
        f"**Commit:** `{snapshot['commit']}`  "
        f"**Time:** {snapshot['timestamp']}",
        "",
    ]

    if alerts:
        critical = [a for a in alerts if a["severity"] == "critical"]
        warnings = [a for a in alerts if a["severity"] == "warning"]
        if critical:
            lines += ["## CRITICAL Drift Alerts", ""]
            for a in critical:
                lines.append(f"- **{a['metric']}**: {a['current']} vs baseline {a['baseline']} (delta {a['delta']:+.4f}, threshold {a['threshold']})")
            lines.append("")
        if warnings:
            lines += ["## Warning Drift Alerts", ""]
            for a in warnings:
                lines.append(f"- **{a['metric']}**: {a['current']} vs baseline {a['baseline']} (delta {a['delta']:+.4f}, threshold {a['threshold']})")
            lines.append("")
    else:
        lines += ["## No Drift Detected", ""]

    # Retrieval quality
    lines += ["## Retrieval Quality", ""]
    retrieval = snapshot.get("retrieval")
    if retrieval:
        lines += [
            "| Metric | Value | Trend |",
            "|--------|-------|-------|",
            f"| Recall@5 | {_fmt(retrieval.get('recall_at_5'))}{_trend(retrieval.get('recall_at_5'), history, 'retrieval.recall_at_5')} | Gate >=0.87 -> {_fmt(retrieval.get('passed_gate'))} |",
            f"| Recall@10 | {_fmt(retrieval.get('recall_at_10'))}{_trend(retrieval.get('recall_at_10'), history, 'retrieval.recall_at_10')} | |",
            f"| Precision@5 | {_fmt(retrieval.get('precision_at_5'))}{_trend(retrieval.get('precision_at_5'), history, 'retrieval.precision_at_5')} | |",
            f"| MRR | {_fmt(retrieval.get('mrr'))}{_trend(retrieval.get('mrr'), history, 'retrieval.mrr')} | |",
            f"| Elapsed | {_fmt(retrieval.get('elapsed_s'), 1)}s | |",
        ]
    else:
        lines.append("_Retrieval eval report not available this run._")
    lines.append("")

    # Safety & agent eval
    lines += ["## Safety & Agent Eval", ""]
    safety = snapshot.get("safety")
    if safety:
        mode_tag = f" _(mode: {safety.get('mode', '?')})_"
        lines += [
            "| Metric | Value | Trend |",
            "|--------|-------|-------|",
            f"| Refuse Recall | {_fmt(safety.get('refuse_recall'))}{_trend(safety.get('refuse_recall'), history, 'safety.refuse_recall')}{mode_tag} | Gate >=0.95 |",
            f"| Safety Accuracy | {_fmt(safety.get('safety_accuracy'))}{_trend(safety.get('safety_accuracy'), history, 'safety.safety_accuracy')} | Gate >=0.90 |",
            f"| False Refusal Rate | {_fmt(safety.get('false_refusal_rate'))}{_trend(safety.get('false_refusal_rate'), history, 'safety.false_refusal_rate')} | Gate <=0.02 |",
            f"| Routing Accuracy | {_fmt(safety.get('routing_accuracy'))}{_trend(safety.get('routing_accuracy'), history, 'safety.routing_accuracy')} | Gate >=0.85 |",
            f"| Groundedness | {_fmt(safety.get('groundedness_rate'))}{_trend(safety.get('groundedness_rate'), history, 'safety.groundedness_rate')} | Gate >=0.90 |",
            f"| Verdict | {_fmt(safety.get('passed'))} | |",
        ]
    else:
        lines.append("_Safety eval report not available this run._")
    lines.append("")

    # Tenant isolation
    lines += ["## Tenant Isolation (Mascot Retrieval)", ""]
    tenant = snapshot.get("tenant")
    if tenant:
        per_tenant = tenant.get("per_tenant") or {}
        lines += [
            "| Metric | Value | Trend |",
            "|--------|-------|-------|",
            f"| Leakage Rate | {_fmt(tenant.get('leakage_rate'))}{_trend(tenant.get('leakage_rate'), history, 'tenant.leakage_rate')} | must be 0.0 |",
            f"| Verdict | {_fmt(tenant.get('passed'))} | |",
        ]
        for tid, tdata in sorted(per_tenant.items()):
            r5 = tdata.get("recall_at_5") if isinstance(tdata, dict) else None
            lines.append(f"| {tid} recall@5 | {_fmt(r5)}{_trend(r5, history, f'tenant.per_tenant.{tid}.recall_at_5')} | gate >=0.87 |")
    else:
        lines.append("_Tenant isolation report not available this run._")
    lines.append("")

    # History table
    all_records = history + [snapshot]
    if len(all_records) > 1:
        lines += ["## History (last 10 runs)", "", "| Run | Branch | Retrieval R@5 | Safety Acc | Leakage |", "|-----|--------|---------------|------------|---------|"]
        for rec in all_records[-10:]:
            r5   = _get(rec, "retrieval.recall_at_5")
            safe = _get(rec, "safety.safety_accuracy")
            leak = _get(rec, "tenant.leakage_rate")
            lines.append(f"| `{rec.get('run_id','?')[:8]}` | `{rec.get('branch','?')}` | {_fmt(r5)} | {_fmt(safe)} | {_fmt(leak)} |")
        lines.append("")

    lines += ["---", "_Generated by eval/drift_dashboard.py_"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local",  action="store_true", help="Skip git commit of history")
    ap.add_argument("--window", type=int, default=5,  help="Baseline rolling window size")
    args = ap.parse_args()

    print("Eval Dashboard & Drift Monitor")
    print("=" * 50)

    snapshot = collect_all()
    print(f"\nSnapshot collected:")
    print(f"  Retrieval: {'OK' if snapshot.get('retrieval') else '--'}")
    print(f"  Safety:    {'OK' if snapshot.get('safety')    else '--'}")
    print(f"  Tenant:    {'OK' if snapshot.get('tenant')    else '--'}")

    history = load_history(n=50)
    print(f"\nHistory: {len(history)} previous runs loaded")

    alerts = check_drift(snapshot, history, window=args.window)
    if alerts:
        sev = "CRITICAL" if any(a["severity"] == "critical" for a in alerts) else "WARNING"
        print(f"\n[{sev}] Drift alerts ({len(alerts)}):")
        for a in alerts:
            print(f"  [{a['severity'].upper()}] {a['metric']}: {a['current']} vs baseline {a['baseline']} (d{a['delta']:+.4f})")
    else:
        print("\nNo drift detected")

    append_history(snapshot)
    print(f"\nHistory updated: {len(history) + 1} records total")

    summary_md = build_summary(snapshot, history, alerts)

    step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary_path:
        with open(step_summary_path, "a", encoding="utf-8") as f:
            f.write(summary_md + "\n")
        print("Step summary written to $GITHUB_STEP_SUMMARY")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    latest_path = REPORTS_DIR / "dashboard_latest.md"
    latest_path.write_text(summary_md, encoding="utf-8")
    print(f"Dashboard written to: {latest_path}")

    critical = [a for a in alerts if a["severity"] == "critical"]
    if critical:
        print(f"\n[CRITICAL] {len(critical)} critical drift alert(s) — marking step as failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
