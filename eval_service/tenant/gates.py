"""tenant/gates.py — SK-10 release gate thresholds and verdict."""

from __future__ import annotations

GATES = {
    "per_tenant_recall_at_5": {"threshold": 0.87, "op": ">="},
    "per_tenant_mrr":         {"threshold": 0.70, "op": ">="},
    "leakage_rate":           {"threshold": 0.0,  "op": "=="},
}


def check_gates(tenant_results: list[dict], leakage_result: dict) -> dict:
    """
    Evaluate all release gates.

    Returns:
        {passed: bool, gates: [{gate, tenant_id, value, op, threshold, passed}]}
    """
    gate_results = []

    for tr in tenant_results:
        tid = tr["tenant_id"]
        for gate_name in ("per_tenant_recall_at_5", "per_tenant_mrr"):
            cfg        = GATES[gate_name]
            metric_key = gate_name.replace("per_tenant_", "")
            value      = tr[metric_key]
            gate_results.append({
                "gate":      gate_name,
                "tenant_id": tid,
                "value":     value,
                "op":        cfg["op"],
                "threshold": cfg["threshold"],
                "passed":    _check(value, cfg["op"], cfg["threshold"]),
            })

    leakage_cfg = GATES["leakage_rate"]
    leakage_val = leakage_result["leakage_rate"]
    gate_results.append({
        "gate":      "leakage_rate",
        "tenant_id": "all",
        "value":     leakage_val,
        "op":        leakage_cfg["op"],
        "threshold": leakage_cfg["threshold"],
        "passed":    _check(leakage_val, leakage_cfg["op"], leakage_cfg["threshold"]),
    })

    return {
        "passed": all(g["passed"] for g in gate_results),
        "gates":  gate_results,
    }


def _check(value: float, op: str, threshold: float) -> bool:
    if op == ">=": return value >= threshold
    if op == "==": return value == threshold
    if op == "<=": return value <= threshold
    raise ValueError(f"Unknown op: {op}")
