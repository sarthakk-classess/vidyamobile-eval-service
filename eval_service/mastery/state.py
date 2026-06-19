"""
mastery/state.py
────────────────
MasteryState: persistent record for one (user, chunk) pair.

Rishabh stores this in the mastery table (RT-08).
Himanshu reads it to decide whether to show a chunk in review.
The SchedulingEngine reads and writes it to compute next_review_at.

Do NOT add or remove fields without updating:
  - rishabh_integration_spec.md  (database schema)
  - himanshu_integration_spec.md (orchestrator field access)
  - scheduler.py                 (SchedulingEngine builds these)
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Literal

RATING_AGAIN = 1
RATING_HARD  = 2
RATING_GOOD  = 3
RATING_EASY  = 4

RatingValue = Literal[1, 2, 3, 4]
CardState   = Literal["new", "learning", "review", "relearning"]
DocType     = Literal["syllabus", "lecture_slides", "academic_reading"]


@dataclass
class MasteryState:
    """Complete spaced repetition state for one student on one chunk."""

    chunk_id:  str
    user_id:   str
    doc_type:  DocType

    card_state:     CardState = "new"

    stability:      float = 1.0
    difficulty:     float = 5.0
    retrievability: float = 1.0

    review_count: int           = 0
    lapse_count:  int           = 0
    last_rating:  Optional[int] = None

    last_review_at: Optional[datetime] = None
    next_review_at: Optional[datetime] = None
    created_at:     datetime           = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def is_new(self) -> bool:
        return self.card_state == "new"

    @property
    def days_overdue(self) -> float:
        if self.next_review_at is None:
            return 0.0
        delta = datetime.now(timezone.utc) - self.next_review_at
        return delta.total_seconds() / 86400.0

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("last_review_at", "next_review_at", "created_at"):
            val = d.get(key)
            if isinstance(val, datetime):
                d[key] = val.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "MasteryState":
        d = dict(data)
        for key in ("last_review_at", "next_review_at", "created_at"):
            val = d.get(key)
            if isinstance(val, str):
                d[key] = datetime.fromisoformat(val)
        return cls(**d)

    def __repr__(self) -> str:
        return (
            f"MasteryState(chunk_id={self.chunk_id!r}, user_id={self.user_id!r}, "
            f"state={self.card_state!r}, S={self.stability:.2f}, D={self.difficulty:.2f}, "
            f"R={self.retrievability:.3f}, reviews={self.review_count})"
        )
