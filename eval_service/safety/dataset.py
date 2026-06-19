"""
safety/dataset.py
─────────────────
Load the 2500-scenario CSV and attach SK-08 labels.

In CI, SK08_DATASET_PATH points to datasets/sk08/ci_scenarios.csv (30 rows).
Locally, point to the full 2500-row founder CSV (gitignored).
"""

from __future__ import annotations
import csv
import os
from pathlib import Path

from eval_service.safety.labels import labels_for_row

_DATASETS = Path(__file__).parent.parent.parent / "datasets" / "safety"
CI_CSV   = _DATASETS / "ci_scenarios.csv"
FULL_CSV = _DATASETS / "Vidya-Scenarios-2500.csv"

REQUIRED_COLUMNS = {
    "Scenario ID", "Capability area", "Agent(s) used",
    "Trigger", "Opening message", "Outcome type",
}


def load_raw(path: str | Path = "") -> list[dict]:
    """Read the scenario CSV. Uses SK08_DATASET_PATH env var if path not given."""
    if not path:
        path = os.environ.get("SAFETY_DATASET_PATH", "") or CI_CSV
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Scenario CSV not found: {path}\n"
            "Set SAFETY_DATASET_PATH or copy the CSV to datasets/safety/."
        )
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")
        return [r for r in reader if (r.get("Scenario ID") or "").strip()]


def load_labeled(path: str | Path = "") -> list[dict]:
    """Load scenarios with canonical SK-08 labels attached."""
    return [{"raw": r, **labels_for_row(r)} for r in load_raw(path)]


def label_distribution(labeled: list[dict]) -> dict:
    from collections import Counter
    return {
        "n":              len(labeled),
        "by_intent":      dict(Counter(x["intent"]           for x in labeled)),
        "by_safety":      dict(Counter(x["expected_safety"]  for x in labeled)),
        "by_decision":    dict(Counter(x["expected_decision"] for x in labeled)),
        "by_agent":       dict(Counter(x["expected_agent"]    for x in labeled)),
        "grounded_count": sum(1 for x in labeled if x["expected_grounded"]),
    }
