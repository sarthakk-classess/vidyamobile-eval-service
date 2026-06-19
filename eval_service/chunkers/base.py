"""
chunkers/base.py
────────────────
Shared types, dataclasses, and base class used by all three chunkers.
Rishabh imports ChunkDict and ChunkError from here — do not rename these.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional


class ChunkError(Exception):
    """
    Raised when a document cannot be chunked.

    Attributes
    ----------
    reason : str
        Machine-readable reason code:
        "no_text_layer"      — scanned PDF, OCR required (out of scope Phase 1)
        "too_short"          — document has < 100 characters after cleaning
        "unsupported_format" — file extension not in Phase 1 scope
        "parse_error"        — pdfplumber / python-pptx raised an exception
    """

    def __init__(self, reason: str, message: str):
        self.reason = reason
        self.message = message
        super().__init__(f"[{reason}] {message}")


@dataclass
class SyllabusChunk:
    chunk_id:       str
    doc_type:       str = "syllabus"
    document_title: str = ""
    university:     Optional[str] = None
    semester:       Optional[int] = None
    course_code:    Optional[str] = None
    course_title:   Optional[str] = None
    # course_block | assessment | schedule | preamble | references | prose_block
    section_type:   str = "prose_block"
    unit_numbers:   list = field(default_factory=list)
    char_count:     int = 0
    chunk_index:    int = 0
    text:           str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SlideChunk:
    chunk_id:       str
    doc_type:       str = "lecture_slides"
    document_title: str = ""
    lecture_name:   str = ""
    week_number:    Optional[int] = None
    slide_range:    str = ""          # e.g. "7-14"
    slide_start:    int = 0
    slide_end:      int = 0
    slide_titles:   list = field(default_factory=list)
    topic:          str = ""
    char_count:     int = 0
    chunk_index:    int = 0
    text:           str = ""          # "Slide N: title\nbody\n\nSlide M: ..."

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AcademicChunk:
    chunk_id:        str
    doc_type:        str = "academic_reading"
    document_title:  str = ""
    author:          Optional[str] = None
    section_title:   str = ""
    section_number:  Optional[str] = None
    paragraph_index: int = 0
    page_number:     Optional[int] = None
    has_code:        bool = False
    char_count:      int = 0
    chunk_index:     int = 0
    text:            str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class BaseChunker:
    """Base class enforcing SK-01 hard limits for all chunker types."""

    MIN_CHARS = 100
    MAX_CHARS = 2000

    def validate_chunk_text(self, text: str) -> bool:
        return self.MIN_CHARS <= len(text) <= self.MAX_CHARS

    def enforce_limits(self, chunks: list[str], overlap: int, target_max: int) -> list[str]:
        """Merge chunks below MIN_CHARS, split chunks above MAX_CHARS."""
        from eval_service.chunkers.utils.text_utils import split_at_sentence_boundary, merge_short_texts

        merged = merge_short_texts(chunks, min_chars=self.MIN_CHARS)
        result = []
        for chunk in merged:
            if len(chunk) > self.MAX_CHARS:
                result.extend(
                    split_at_sentence_boundary(chunk, max_chars=target_max, overlap=overlap)
                )
            else:
                result.append(chunk)
        return result
