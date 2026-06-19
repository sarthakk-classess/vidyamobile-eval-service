"""
tenant/seed_kb.py
─────────────────
Inserts synthetic tenant KB content into a local Supabase instance
so the tenant isolation live eval has real chunks to retrieve.

Run once after `supabase start` and migrations are applied.

Usage
─────
    python eval_service/tenant/seed_kb.py

Required env vars
─────────────────
    SUPABASE_URL              — e.g. http://localhost:54321
    SUPABASE_SERVICE_ROLE_KEY — local service-role key
    GEMINI_API_KEY            — for chunk embeddings (gemini-embedding-2)
"""

from __future__ import annotations
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
from google import genai
from google.genai import types as genai_types

_ROOT        = Path(__file__).parent.parent.parent
DATASETS_DIR = _ROOT / "datasets" / "tenant"

TENANT_FILES = {
    "state-univ":   ("tenant_a_queries.json", "State University"),
    "city-college": ("tenant_b_queries.json", "City College"),
}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
SUPABASE_URL   = os.environ.get("SUPABASE_URL", "")
SUPABASE_SRK   = os.environ.get("SUPABASE_SERVICE_ROLE_KEY",
                  os.environ.get("SUPABASE_SERVICE_KEY", ""))

_EMBED_MODEL = "gemini-embedding-2"
_EMBED_DIM   = 1536


def _headers() -> dict:
    return {
        "apikey":        SUPABASE_SRK,
        "Authorization": f"Bearer {SUPABASE_SRK}",
        "Content-Type":  "application/json",
        "Prefer":        "resolution=merge-duplicates",
    }


async def _embed(texts: list[str]) -> list[list[float]]:
    client = genai.Client(api_key=GEMINI_API_KEY)
    loop   = asyncio.get_event_loop()

    def _batch() -> list[list[float]]:
        out = []
        for text in texts:
            res = client.models.embed_content(
                model=_EMBED_MODEL,
                contents=text,
                config=genai_types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=_EMBED_DIM,
                ),
            )
            out.append(res.embeddings[0].values)
        return out

    return await loop.run_in_executor(None, _batch)


async def _upsert_tenants(client: httpx.AsyncClient) -> None:
    rows = [
        {"id": "state-univ",   "name": "State University", "domain": "state-univ.edu",  "plan": "partner"},
        {"id": "city-college", "name": "City College",     "domain": "city-college.edu", "plan": "partner"},
    ]
    resp = await client.post(f"{SUPABASE_URL}/rest/v1/tenants", headers=_headers(), json=rows)
    resp.raise_for_status()
    print(f"  tenants upserted: {[r['id'] for r in rows]}")


async def _seed_tenant(client: httpx.AsyncClient, tenant_id: str, filename: str) -> int:
    entries = json.loads((DATASETS_DIR / filename).read_text(encoding="utf-8"))
    texts   = [e["document_text"] for e in entries]
    cids    = [e["relevant_chunk_ids"][0] for e in entries]

    print(f"  [{tenant_id}] embedding {len(texts)} chunks ...")
    embeddings = await _embed(texts)

    rows = [
        {
            "id":        cid,
            "tenant_id": tenant_id,
            "url":       f"https://{tenant_id}.edu/kb/{cid}",
            "title":     cid,
            "content":   text,
            "embedding": emb,
            "char_count": len(text),
            "metadata":  {"source": "seed_kb"},
        }
        for cid, text, emb in zip(cids, texts, embeddings)
    ]

    resp = await client.post(f"{SUPABASE_URL}/rest/v1/kb_chunks", headers=_headers(), json=rows)
    resp.raise_for_status()
    print(f"  [{tenant_id}] upserted {len(rows)} chunks into kb_chunks")
    return len(rows)


async def main() -> None:
    if not SUPABASE_URL or not SUPABASE_SRK:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        sys.exit(1)
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY must be set")
        sys.exit(1)

    print("Tenant KB seeder")
    print(f"  Supabase: {SUPABASE_URL}")
    print(f"  Embed:    {_EMBED_MODEL} (Gemini SDK direct)")

    async with httpx.AsyncClient(timeout=30) as client:
        print("\n[1/2] Upserting tenant rows ...")
        await _upsert_tenants(client)

        print("\n[2/2] Seeding KB chunks ...")
        total = 0
        for tenant_id, (filename, _) in TENANT_FILES.items():
            total += await _seed_tenant(client, tenant_id, filename)

    print(f"\nDone — {total} chunks seeded across {len(TENANT_FILES)} tenants.")
    print("Run: TENANT_DIRECT=1 python eval/tenant_eval.py")


if __name__ == "__main__":
    asyncio.run(main())
