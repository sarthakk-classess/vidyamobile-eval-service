"""
Deterministic chunk ID generation.

Format: {prefix}_{doc_hash}_{chunk_index:04d}

IDs are path-hashed (not content-hashed) so editing one sentence doesn't force
a full re-index. If you need to force re-index, delete chunks by doc_hash prefix.
"""

import hashlib
from typing import Literal

DocType = Literal["syllabus", "lecture_slides", "academic_reading"]

PREFIX_MAP: dict[str, str] = {
    "syllabus":         "syl",
    "lecture_slides":   "sld",
    "academic_reading": "acad",
}


def generate_chunk_id(doc_type: str, document_path: str, chunk_index: int) -> str:
    """
    Generate a deterministic chunk ID.

    Parameters
    ----------
    doc_type      : "syllabus" | "lecture_slides" | "academic_reading"
    document_path : stable path or identifier of the source document
    chunk_index   : 0-based position of this chunk within the document
    """
    prefix   = PREFIX_MAP.get(doc_type, doc_type[:4])
    doc_hash = hashlib.sha256(document_path.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{doc_hash}_{chunk_index:04d}"


def extract_doc_hash(chunk_id: str) -> str:
    """Extract the doc_hash portion from a chunk_id for bulk deletion."""
    parts = chunk_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid chunk_id format: '{chunk_id}'")
    return parts[1]
