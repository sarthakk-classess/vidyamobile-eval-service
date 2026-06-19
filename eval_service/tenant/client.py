"""
tenant/client.py
────────────────
TenantKBClient interface with three implementations:

  MockTenantKBClient(mode="isolated") — correct isolation, all gates pass
  MockTenantKBClient(mode="leaky")   — deliberately leaks cross-tenant data
  LiveTenantKBClient                 — calls RT-18 kb-query edge function
  DirectTenantKBClient               — embeds via Gemini + calls RPC (CI mode)

Switch via environment:
  TENANT_LIVE=1     → LiveTenantKBClient
  TENANT_DIRECT=1   → DirectTenantKBClient
  (default)         → MockTenantKBClient (mode=TENANT_MOCK_MODE or "isolated")
"""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

import numpy as np


class TenantKBClient:
    def query(self, tenant_id: str, query: str, top_k: int = 5) -> list[dict]:
        raise NotImplementedError


class MockTenantKBClient(TenantKBClient):
    """Offline retrieval over synthetic tenant datasets using MiniLM."""

    def __init__(self, mode: str = "isolated"):
        assert mode in ("isolated", "leaky"), f"unknown mock mode: {mode}"
        self.mode    = mode
        self._model  = None
        self._corpus: dict[str, dict] = {}
        self._ids:    list[str]       = []
        self._matrix: np.ndarray | None = None

    def load_datasets(self, dataset_dir: str | Path) -> None:
        dataset_dir = Path(dataset_dir)
        corpus: dict[str, dict] = {}
        for json_file in dataset_dir.glob("tenant_*.json"):
            for entry in json.loads(json_file.read_text(encoding="utf-8")):
                cid = entry["relevant_chunk_ids"][0]
                corpus[cid] = {
                    "content":   entry["document_text"],
                    "tenant_id": entry["tenant_id"],
                    "title":     f"{entry['tenant_id']}:{cid}",
                    "chunk_id":  cid,
                }
        self._corpus = corpus
        self._ids    = list(corpus.keys())
        self._matrix = self._encode_batch([corpus[cid]["content"] for cid in self._ids])

    def query(self, tenant_id: str, query: str, top_k: int = 5) -> list[dict]:
        if self._matrix is None:
            raise RuntimeError("Call load_datasets() before querying.")
        q_vec  = self._encode_single(query)
        scores = np.asarray(self._matrix) @ np.asarray(q_vec)
        ranked = sorted(zip(scores.tolist(), self._ids), key=lambda x: x[0], reverse=True)
        results = []
        for score, cid in ranked:
            chunk = self._corpus[cid]
            if self.mode == "isolated" and chunk["tenant_id"] != tenant_id:
                continue
            results.append({
                "chunk_id":  cid,
                "content":   chunk["content"],
                "title":     chunk["title"],
                "score":     float(score),
                "tenant_id": chunk["tenant_id"],
            })
            if len(results) == top_k:
                break
        return results

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self._model

    def _encode_single(self, text: str) -> np.ndarray:
        return self._load_model().encode([text], normalize_embeddings=True)[0]

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1))
        return np.array(self._load_model().encode(texts, normalize_embeddings=True, batch_size=64))


class LiveTenantKBClient(TenantKBClient):
    """Calls RT-18 kb-query Supabase edge function."""

    def __init__(self):
        base          = os.environ.get("KB_SERVICE_URL", "http://localhost:54321")
        self.endpoint = f"{base.rstrip('/')}/functions/v1/kb-query"
        self.token    = os.environ.get("KB_SERVICE_TOKEN", "")

    def query(self, tenant_id: str, query: str, top_k: int = 5) -> list[dict]:
        import httpx
        resp = httpx.post(
            self.endpoint,
            json={"query": query, "top_k": top_k, "tenant_id": tenant_id},
            headers={"X-Vidya-Service-Token": self.token},
            timeout=15.0,
        )
        resp.raise_for_status()
        return [
            {
                "chunk_id":  hashlib.md5(item.get("content", "").encode()).hexdigest(),
                "content":   item.get("content", ""),
                "title":     item.get("title", f"{tenant_id} KB"),
                "url":       item.get("url", ""),
                "score":     float(item.get("score", 0.0)),
                "tenant_id": tenant_id,
            }
            for item in resp.json().get("chunks", [])
        ]


class DirectTenantKBClient(TenantKBClient):
    """
    CI mode — embeds via Gemini SDK, calls match_kb_chunks_for_tenant RPC directly.
    Tests the SQL isolation guarantee without needing the kb-query edge function.

    Required env vars: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, GEMINI_API_KEY
    """

    _EMBED_MODEL = "gemini-embedding-2"
    _EMBED_DIM   = 1536

    def __init__(self):
        self.supabase_url = os.environ.get("SUPABASE_URL", "http://127.0.0.1:54321")
        self.srk          = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        self._gemini      = None

    def _get_gemini(self):
        if self._gemini is None:
            from google import genai
            self._gemini = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
        return self._gemini

    def query(self, tenant_id: str, query: str, top_k: int = 5) -> list[dict]:
        import httpx
        from google.genai import types as genai_types

        client = self._get_gemini()
        res    = client.models.embed_content(
            model=self._EMBED_MODEL,
            contents=query,
            config=genai_types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self._EMBED_DIM,
            ),
        )
        embedding = res.embeddings[0].values

        rpc_resp = httpx.post(
            f"{self.supabase_url}/rest/v1/rpc/match_kb_chunks_for_tenant",
            json={
                "p_tenant_id":     tenant_id,
                "query_embedding": embedding,
                "match_count":     top_k,
                "min_similarity":  0.0,
            },
            headers={
                "apikey":        self.srk,
                "Authorization": f"Bearer {self.srk}",
                "Content-Type":  "application/json",
            },
            timeout=15.0,
        )
        rpc_resp.raise_for_status()
        return [
            {
                "chunk_id":  row["chunk_id"],
                "content":   row["content"],
                "title":     row.get("title", ""),
                "url":       row.get("url", ""),
                "score":     float(row["relevance_score"]),
                "tenant_id": tenant_id,
            }
            for row in rpc_resp.json()
        ]


def get_client(dataset_dir: str | Path | None = None) -> TenantKBClient:
    if os.environ.get("TENANT_LIVE") == "1":
        return LiveTenantKBClient()
    if os.environ.get("TENANT_DIRECT") == "1":
        return DirectTenantKBClient()
    mode   = os.environ.get("TENANT_MOCK_MODE", "isolated")
    client = MockTenantKBClient(mode=mode)
    if dataset_dir is not None:
        client.load_datasets(dataset_dir)
    return client
