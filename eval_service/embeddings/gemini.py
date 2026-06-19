"""
embeddings/gemini.py
────────────────────
GeminiEmbedder — production embedding client using gemini-embedding-2.

Locked decision (SK-02/SK-06): gemini-embedding-2, output_dimensionality=1536.
Do not change the model or dimension without re-running SK-06 tuning and
re-seeding the pgvector index in Supabase.
"""

from __future__ import annotations
import os
from typing import Literal

EMBED_MODEL = "gemini-embedding-2"
EMBED_DIM   = 1536

TaskType = Literal[
    "RETRIEVAL_QUERY",
    "RETRIEVAL_DOCUMENT",
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
]


class GeminiEmbedder:
    """
    Thin wrapper around google-genai for Vidya's embedding pipeline.

    Usage:
        embedder = GeminiEmbedder(api_key="...")
        vector   = embedder.embed_query("What is a stack?")
        vectors  = embedder.embed_documents(["text1", "text2"])
    """

    def __init__(self, api_key: str = ""):
        from google import genai
        self._client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY", ""))

    def embed_query(self, text: str) -> list[float]:
        """Embed a retrieval query (RETRIEVAL_QUERY task type)."""
        return self._embed(text, task_type="RETRIEVAL_QUERY")

    def embed_document(self, text: str) -> list[float]:
        """Embed a document chunk for storage (RETRIEVAL_DOCUMENT task type)."""
        return self._embed(text, task_type="RETRIEVAL_DOCUMENT")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed document chunks. Calls the API once per text (no batching in SDK v1)."""
        return [self.embed_document(t) for t in texts]

    def _embed(self, text: str, task_type: TaskType) -> list[float]:
        from google.genai import types as genai_types

        res = self._client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
            config=genai_types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBED_DIM,
            ),
        )
        return res.embeddings[0].values
