"""
difficulty/predictor.py
───────────────────────
DifficultyPredictor — loads the trained SK-07 GBR model and scores topics.

This is what RT-12 calls. Input is the student's mastery rows joined with
their topic by RT-08; output is one difficulty row per topic, ready to write
into source_events for the proactivity watcher.

Usage:
    from eval_service.difficulty.predictor import DifficultyPredictor

    predictor = DifficultyPredictor()              # loads artifacts once
    rows      = predictor.score_student(mastery_records)
"""

from __future__ import annotations
import json
import os
from pathlib import Path

from eval_service.difficulty.features import (
    FEATURE_NAMES, extract_topic_features, confidence, group_by_student_topic,
)

_ARTIFACTS = Path(__file__).parent / "artifacts"
MODEL_PATH = _ARTIFACTS / "difficulty_model.joblib"
META_PATH  = _ARTIFACTS / "model_meta.json"

DEFAULT_NUDGE_THRESHOLD = 0.60


class DifficultyPredictor:
    """Wraps the trained GBR regressor for per-topic difficulty scoring."""

    def __init__(
        self,
        model_path: Path | str = MODEL_PATH,
        meta_path:  Path | str = META_PATH,
    ):
        try:
            import joblib
        except ImportError:
            raise ImportError("pip install joblib scikit-learn")

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"No trained model at {model_path}. "
                "Run: python -m eval_service.difficulty.train"
            )
        self._model = joblib.load(model_path)

        self.meta: dict = {}
        meta_path = Path(meta_path)
        if meta_path.is_file():
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))

        trained_features = self.meta.get("feature_names")
        if trained_features and trained_features != FEATURE_NAMES:
            raise RuntimeError(
                "Feature contract mismatch between model_meta.json and features.py. "
                "Retrain: python -m eval_service.difficulty.train"
            )

    def score_student(
        self,
        records: list[dict],
        nudge_threshold: float = DEFAULT_NUDGE_THRESHOLD,
    ) -> list[dict]:
        """
        Score every topic the student has a record for.

        Returns one dict per topic:
            {user_id, topic, topic_label, difficulty_score [0,1],
             difficulty_10 [1,10], confidence [0,1], n_chunks, should_nudge}
        """
        import numpy as np

        groups = group_by_student_topic(records)
        if not groups:
            return []

        keys = list(groups.keys())
        X    = np.asarray([extract_topic_features(groups[k]) for k in keys], dtype=np.float64)
        preds = np.clip(self._model.predict(X), 0.0, 1.0)

        out = []
        for (user_id, topic), score in zip(keys, preds):
            recs = groups[(user_id, topic)]
            conf = confidence(recs)
            out.append({
                "user_id":          user_id,
                "topic":            topic,
                "difficulty_score": round(float(score), 4),
                "difficulty_10":    round(float(1.0 + score * 9.0), 2),
                "confidence":       conf,
                "n_chunks":         len(recs),
                "should_nudge":     bool(score >= nudge_threshold and conf >= 0.5),
            })

        out.sort(key=lambda r: r["difficulty_score"], reverse=True)
        return out

    def to_source_events(self, scored_rows: list[dict], tenant_id: str | None = None) -> list[dict]:
        """Convert scored rows to source_events insert payloads for RT-12 (should_nudge only)."""
        return [
            {
                "user_id":   r["user_id"],
                "tenant_id": tenant_id,
                "kind":      "topic_difficulty",
                "payload": {
                    "topic":            r["topic"],
                    "difficulty_score": r["difficulty_score"],
                    "difficulty_10":    r["difficulty_10"],
                    "confidence":       r["confidence"],
                    "n_chunks":         r["n_chunks"],
                    "source":           "SK-07",
                },
            }
            for r in scored_rows
            if r["should_nudge"]
        ]
