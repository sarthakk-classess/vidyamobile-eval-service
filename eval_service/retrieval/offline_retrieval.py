"""
retrieval/offline_retrieval.py
──────────────────────────────
Local sentence-transformers retrieval for offline CI evaluation.
No API key required — uses all-MiniLM-L6-v2 (CPU-friendly).
"""

from __future__ import annotations
import numpy as np
from typing import Optional

OFFLINE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class OfflineRetrieval:
    """
    Corpus-based retrieval using local sentence-transformers embeddings.

    Parameters
    ----------
    model_name : sentence-transformers model (default: all-MiniLM-L6-v2)
    """

    def __init__(self, model_name: str = OFFLINE_MODEL):
        self.model_name = model_name
        self._model = None
        self._corpus: dict[str, str] = {}
        self._matrix: Optional[np.ndarray] = None
        self._ids: list[str] = []

    def build_corpus(self, dataset: list[dict]) -> None:
        """
        Build retrieval corpus from benchmark dataset entries.

        Each entry contributes its document_text keyed by relevant_chunk_ids[0].
        """
        corpus: dict[str, str] = {}
        for entry in dataset:
            doc_text = entry.get("document_text", "")
            if doc_text and entry.get("relevant_chunk_ids"):
                corpus[entry["relevant_chunk_ids"][0]] = doc_text
        self._corpus = corpus
        self._ids    = list(corpus.keys())
        self._matrix = self._encode_batch([corpus[cid] for cid in self._ids])

    def build_corpus_from_chunks(self, chunks: list[dict]) -> None:
        """Alternative: build from raw chunk dicts with 'chunk_id' and 'text'."""
        self._corpus = {c["chunk_id"]: c["text"] for c in chunks if c.get("text")}
        self._ids    = list(self._corpus.keys())
        self._matrix = self._encode_batch([self._corpus[cid] for cid in self._ids])

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        """Return top_k chunks for a query sorted by cosine similarity."""
        if self._matrix is None or not self._ids:
            raise RuntimeError("Call build_corpus() first.")
        q_vec  = self._encode_single(query)
        scores = np.asarray(self._matrix) @ np.asarray(q_vec)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {"chunk_id": self._ids[i], "text": self._corpus[self._ids[i]], "score": float(scores[i])}
            for i in top_idx
        ]

    def retrieve_batch(self, queries: list[dict], top_k: int = 20) -> list[dict]:
        """Run retrieve() for a list of {"query_id", "query", "relevant_chunk_ids"} entries."""
        return [
            {
                "query_id":      e["query_id"],
                "query":         e["query"],
                "retrieved_ids": [c["chunk_id"] for c in self.retrieve(e["query"], top_k)],
                "relevant_ids":  e["relevant_chunk_ids"],
            }
            for e in queries
        ]

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError("pip install sentence-transformers")
            print(f"  Loading model: {self.model_name} ...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode_single(self, text: str) -> np.ndarray:
        return self._load_model().encode([text], normalize_embeddings=True)[0]

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1))
        return np.array(
            self._load_model().encode(
                texts, normalize_embeddings=True, batch_size=64,
                show_progress_bar=len(texts) > 20,
            )
        )
