"""
retrieval/client.py
───────────────────
SupabaseRetrievalClient — calls match_kb_chunks_for_tenant RPC.

This is what Sarthak's eval harnesses (SK-04, SK-10) use when testing
against a live or local Supabase instance.

Best params are locked in params.py (SK-06 output):
  match_count=3, min_similarity=0.0, ef_search=40
"""

from __future__ import annotations
import os

from eval_service.retrieval.params import MATCH_COUNT, MIN_SIMILARITY, EF_SEARCH
from eval_service.embeddings.gemini import GeminiEmbedder


class SupabaseRetrievalClient:
    """
    Embeds a query via Gemini then calls match_kb_chunks_for_tenant RPC.

    Required env vars (or pass explicitly):
        SUPABASE_URL              — e.g. http://127.0.0.1:54321
        SUPABASE_SERVICE_ROLE_KEY — local or remote service-role key
        GEMINI_API_KEY            — Gemini embed key
    """

    def __init__(
        self,
        supabase_url: str = "",
        service_role_key: str = "",
        gemini_api_key: str = "",
    ):
        self.supabase_url      = (supabase_url or os.environ.get("SUPABASE_URL", "")).rstrip("/")
        self.service_role_key  = service_role_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        self._embedder         = GeminiEmbedder(api_key=gemini_api_key)

    def query(
        self,
        query: str,
        tenant_id: str,
        doc_type: str | None = None,
        match_count: int = MATCH_COUNT,
        min_similarity: float = MIN_SIMILARITY,
    ) -> list[dict]:
        """
        Embed query and call match_kb_chunks_for_tenant RPC.

        Parameters
        ----------
        query          : natural-language query string
        tenant_id      : tenant to search within (isolation enforced by SQL)
        doc_type       : "syllabus" | "lecture_slides" | "academic_reading" | None
                         Always pass — without it recall drops below gate (SK-06 finding)
        match_count    : number of results (default from SK-06 best params)
        min_similarity : minimum cosine similarity threshold

        Returns
        -------
        list of {chunk_id, content, title, url, relevance_score, tenant_id}
        """
        import httpx

        embedding = self._embedder.embed_query(query)

        body: dict = {
            "p_tenant_id":     tenant_id,
            "query_embedding": embedding,
            "match_count":     match_count,
            "min_similarity":  min_similarity,
        }
        if doc_type:
            body["p_doc_type"] = doc_type

        resp = httpx.post(
            f"{self.supabase_url}/rest/v1/rpc/match_kb_chunks_for_tenant",
            json=body,
            headers={
                "apikey":        self.service_role_key,
                "Authorization": f"Bearer {self.service_role_key}",
                "Content-Type":  "application/json",
            },
            timeout=15.0,
        )
        resp.raise_for_status()
        return resp.json()
