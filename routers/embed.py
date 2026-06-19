"""
routers/embed.py — POST /v1/embed

Embeds text using Gemini embedding-2 (1536-dim, SK-02 locked decision).
Called by Rishabh's ingest worker and Himanshu's retrieval path.
"""

from __future__ import annotations
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from eval_service.config import settings
from eval_service.embeddings.gemini import GeminiEmbedder

router   = APIRouter(tags=["embeddings"])
_embedder: GeminiEmbedder | None = None


def _get_embedder() -> GeminiEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = GeminiEmbedder(api_key=settings.gemini_api_key)
    return _embedder


TaskType = Literal["RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY"]


class EmbedRequest(BaseModel):
    text:      str
    task_type: TaskType = "RETRIEVAL_DOCUMENT"


class EmbedResponse(BaseModel):
    model:     str
    dimension: int
    embedding: list[float]


@router.post("/embed", response_model=EmbedResponse)
def embed_text(req: EmbedRequest):
    """Embed text using gemini-embedding-2 (1536-dim)."""
    try:
        embedder  = _get_embedder()
        embedding = embedder._embed(req.text, task_type=req.task_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return EmbedResponse(
        model     = "gemini-embedding-2",
        dimension = len(embedding),
        embedding = embedding,
    )
