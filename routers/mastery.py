"""
routers/mastery.py — Mastery scheduling endpoints (SK-03 FSRS-4.5).

POST /v1/mastery/update  — update state after a review rating
POST /v1/mastery/new     — create initial state for a new (user, chunk) pair
POST /v1/mastery/due     — filter a list of states to only due ones
GET  /v1/mastery/forecast — project review schedule for one state
"""

from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from eval_service.config import settings
from eval_service.mastery.scheduler import SchedulingEngine, infer_rating_from_quiz, infer_rating_from_practice
from eval_service.mastery.state import MasteryState, DocType

router = APIRouter(tags=["mastery"])
_engine: SchedulingEngine | None = None


def _get_engine() -> SchedulingEngine:
    global _engine
    if _engine is None:
        _engine = SchedulingEngine(desired_retention=settings.mastery_desired_retention)
    return _engine


class NewStateRequest(BaseModel):
    chunk_id: str
    user_id:  str
    doc_type: DocType


class UpdateRequest(BaseModel):
    state:  dict
    rating: int   # 1=Again, 2=Hard, 3=Good, 4=Easy


class DueRequest(BaseModel):
    user_id:    str
    all_states: list[dict]
    limit:      int = 50


class ForecastRequest(BaseModel):
    state:      dict
    days_ahead: int = 30


@router.post("/mastery/new")
def mastery_new(req: NewStateRequest):
    """Create initial MasteryState for a (user, chunk) pair first seen."""
    state = _get_engine().schedule_new(req.chunk_id, req.user_id, req.doc_type)
    return state.to_dict()


@router.post("/mastery/update")
def mastery_update(req: UpdateRequest):
    """Update MasteryState after a student review."""
    if req.rating not in (1, 2, 3, 4):
        raise HTTPException(status_code=422, detail="Rating must be 1–4")
    try:
        state   = MasteryState.from_dict(req.state)
        updated = _get_engine().update(state, rating=req.rating)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return updated.to_dict()


@router.post("/mastery/due")
def mastery_due(req: DueRequest):
    """Return due chunks for a user from a list of states."""
    all_states = [MasteryState.from_dict(s) for s in req.all_states]
    due        = _get_engine().due_chunks(req.user_id, all_states, limit=req.limit)
    return {"user_id": req.user_id, "due_count": len(due), "due": [s.to_dict() for s in due]}


@router.post("/mastery/forecast")
def mastery_forecast(req: ForecastRequest):
    """Project review schedule for the next N days (assuming rating=Good)."""
    state    = MasteryState.from_dict(req.state)
    schedule = _get_engine().forecast(state, days_ahead=req.days_ahead)
    return {"schedule": schedule}


@router.post("/mastery/infer-rating/quiz")
def infer_quiz(score_pct: float):
    """Map a quiz percentage score to FSRS rating 1–4."""
    return {"score_pct": score_pct, "rating": infer_rating_from_quiz(score_pct)}


@router.post("/mastery/infer-rating/practice")
def infer_practice(correct: bool, time_taken_s: float, expected_time_s: float = 60.0):
    """Map a practice result (correctness + time) to FSRS rating 1–4."""
    return {
        "correct":          correct,
        "rating":           infer_rating_from_practice(correct, time_taken_s, expected_time_s),
    }
