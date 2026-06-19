"""
routers/chunk.py — POST /v1/chunk

Accepts raw document text + doc_type and returns SK-01 chunks.
Called by Rishabh's RT-05 ingest worker.
"""

from __future__ import annotations
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["chunking"])

DocType = Literal["syllabus", "lecture_slides", "academic_reading"]


class ChunkRequest(BaseModel):
    doc_type:       DocType
    text:           str
    document_title: str = ""
    metadata:       dict = {}


class ChunkResponse(BaseModel):
    doc_type:   str
    n_chunks:   int
    chunks:     list[dict]


@router.post("/chunk", response_model=ChunkResponse)
def chunk_document(req: ChunkRequest):
    """
    Chunk a document using the appropriate SK-01 chunker.

    NOTE: The actual chunker implementations (syllabus_chunker.py,
    slide_chunker.py, academic_chunker.py) must be copied from
    sk01/chunkers/ into eval_service/chunkers/ to activate this endpoint.
    """
    try:
        chunks = _dispatch_chunker(req)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return ChunkResponse(
        doc_type = req.doc_type,
        n_chunks = len(chunks),
        chunks   = chunks,
    )


def _dispatch_chunker(req: ChunkRequest) -> list[dict]:
    """Route to the correct SK-01 chunker based on doc_type."""
    if req.doc_type == "syllabus":
        from eval_service.chunkers.syllabus import SyllabusChunker
        chunker = SyllabusChunker()
        return [c.to_dict() for c in chunker.chunk(req.text, document_title=req.document_title, **req.metadata)]
    elif req.doc_type == "lecture_slides":
        from eval_service.chunkers.slides import SlideChunker
        chunker = SlideChunker()
        return [c.to_dict() for c in chunker.chunk(req.text, document_title=req.document_title, **req.metadata)]
    elif req.doc_type == "academic_reading":
        from eval_service.chunkers.academic import AcademicChunker
        chunker = AcademicChunker()
        return [c.to_dict() for c in chunker.chunk(req.text, document_title=req.document_title, **req.metadata)]
    raise ValueError(f"Unknown doc_type: {req.doc_type}")
