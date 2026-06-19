"""
difficulty/topics.py — Topic catalog and chunk→topic derivation.

A "topic" is coarser than a chunk and is what the proactivity engine nudges
on — e.g. "you're slipping on Trees", not "you're slipping on chunk syl_abc_0007".
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Topic:
    key:                  str
    label:                str
    doc_type:             str    # syllabus | lecture_slides | academic_reading
    intrinsic_difficulty: float  # [1, 10] — latent "hard for everyone" level
    n_chunks:             int


TOPIC_CATALOG: list[Topic] = [
    Topic("syllabus:Unit I",    "Unit I — Intro & Complexity",     "syllabus",         3.0, 3),
    Topic("syllabus:Unit II",   "Unit II — Linked Lists",          "syllabus",         5.0, 4),
    Topic("syllabus:Unit III",  "Unit III — Stacks & Queues",      "syllabus",         4.5, 4),
    Topic("syllabus:Unit IV",   "Unit IV — Trees",                 "syllabus",         7.5, 5),
    Topic("syllabus:Unit V",    "Unit V — Graphs & Hashing",       "syllabus",         8.5, 5),
    Topic("academic:Sorting",   "Sorting Algorithms",              "academic_reading", 6.0, 4),
    Topic("academic:Recursion", "Recursion & Divide-and-Conquer",  "academic_reading", 7.0, 3),
    Topic("slides:BST Ops",     "BST Operations (lecture)",        "lecture_slides",   8.0, 4),
]

TOPIC_BY_KEY: dict[str, Topic] = {t.key: t for t in TOPIC_CATALOG}

_DOC_TYPE_PREFIX = {
    "syllabus":         "syllabus",
    "lecture_slides":   "slides",
    "slides":           "slides",
    "academic_reading": "academic",
    "academic":         "academic",
}


def derive_topic(metadata: dict, doc_type: str) -> str:
    """
    Derive a topic key from a chunk's metadata + doc_type.
    Single source of truth used by every difficulty component.
    """
    prefix = _DOC_TYPE_PREFIX.get(doc_type, doc_type or "unknown")
    units  = metadata.get("unit_numbers") if metadata else None
    if units:
        return f"syllabus:Unit {units[0]}"
    for k in ("topic", "section"):
        v = metadata.get(k) if metadata else None
        if v:
            return f"{prefix}:{v}"
    title = (metadata or {}).get("document_title")
    if title:
        return f"{prefix}:{title}"
    return f"{prefix}:unknown"
