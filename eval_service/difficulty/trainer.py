"""
difficulty/trainer.py — Train the GBR difficulty model.

Pipeline:
  1. Simulate student histories via SchedulingEngine
  2. Build (student, topic) feature vectors from mastery states
  3. Fit GradientBoostingRegressor to predict latent struggle in [0, 1]
  4. Evaluate, check release gates, persist model + metadata

Release gates:
  R2  >= 0.70
  MAE <= 0.12
  Struggle-classification accuracy (threshold 0.6) >= 0.80
"""

from __future__ import annotations
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eval_service.difficulty.features  import FEATURE_NAMES, extract_topic_features, group_by_student_topic
from eval_service.difficulty.simulator import simulate

_PKG_DIR     = Path(__file__).parent
ARTIFACT_DIR = _PKG_DIR / "artifacts"
MODEL_PATH   = ARTIFACT_DIR / "difficulty_model.joblib"
META_PATH    = ARTIFACT_DIR / "model_meta.json"

GATE_R2           = 0.70
GATE_MAE          = 0.12
GATE_STRUGGLE_ACC = 0.80
STRUGGLE_THRESHOLD = 0.60


def build_dataset(records: list[dict], labels: list[dict]) -> tuple:
    label_map = {(l["user_id"], l["topic"]): l["struggle"] for l in labels}
    groups    = group_by_student_topic(records)
    X, y, keys = [], [], []
    for (user_id, topic), recs in groups.items():
        if (user_id, topic) not in label_map:
            continue
        X.append(extract_topic_features(recs))
        y.append(label_map[(user_id, topic)])
        keys.append((user_id, topic))
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64), keys


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err    = y_pred - y_true
    mae    = float(np.mean(np.abs(err)))
    rmse   = float(math.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) or 1e-12
    r2     = 1.0 - ss_res / ss_tot
    true_s = y_true >= STRUGGLE_THRESHOLD
    pred_s = y_pred >= STRUGGLE_THRESHOLD
    acc    = float(np.mean(true_s == pred_s))
    return {"r2": round(r2, 4), "mae": round(mae, 4), "rmse": round(rmse, 4), "struggle_acc": round(acc, 4)}


def train(n_students: int = 400, seed: int = 7, test_frac: float = 0.2, verbose: bool = True) -> dict:
    """Train, evaluate, and persist the model. Returns result summary dict."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        import joblib
    except ImportError:
        raise ImportError("pip install scikit-learn joblib")

    if verbose:
        print(f"Simulating {n_students} students ...")
    records, labels = simulate(n_students=n_students, seed=seed)

    X, y, keys = build_dataset(records, labels)
    if verbose:
        print(f"  Dataset: {X.shape[0]} samples x {X.shape[1]} features")

    rng    = np.random.default_rng(seed)
    perm   = rng.permutation(len(y))
    n_test = int(len(y) * test_frac)
    test_idx, train_idx = perm[:n_test], perm[n_test:]

    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9, random_state=seed,
    )
    model.fit(X[train_idx], y[train_idx])

    y_pred  = np.clip(model.predict(X[test_idx]), 0.0, 1.0)
    metrics = _metrics(y[test_idx], y_pred)

    passed = (
        metrics["r2"] >= GATE_R2 and
        metrics["mae"] <= GATE_MAE and
        metrics["struggle_acc"] >= GATE_STRUGGLE_ACC
    )

    if verbose:
        print(f"  R2           = {metrics['r2']:.4f}  (gate >= {GATE_R2})")
        print(f"  MAE          = {metrics['mae']:.4f}  (gate <= {GATE_MAE})")
        print(f"  struggle acc = {metrics['struggle_acc']:.4f}  (gate >= {GATE_STRUGGLE_ACC})")
        print(f"  GATE: {'PASS' if passed else 'FAIL'}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    meta = {
        "trained_at":        datetime.now(timezone.utc).isoformat(),
        "n_students":        n_students,
        "n_samples":         int(X.shape[0]),
        "feature_names":     FEATURE_NAMES,
        "struggle_threshold": STRUGGLE_THRESHOLD,
        "metrics":           metrics,
        "gates":             {"r2": GATE_R2, "mae": GATE_MAE, "struggle_acc": GATE_STRUGGLE_ACC},
        "passed_gate":       passed,
        "seed":              seed,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if verbose:
        print(f"  Model: {MODEL_PATH}")

    return {"metrics": metrics, "passed_gate": passed, "model_path": str(MODEL_PATH), "n_samples": int(X.shape[0])}
