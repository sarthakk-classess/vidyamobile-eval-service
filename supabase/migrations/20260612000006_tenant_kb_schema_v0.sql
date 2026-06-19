-- SK-10: Tenant KB schema — tenants + kb_chunks + match_kb_chunks_for_tenant RPC
-- Supports Mascot tenant isolation eval (DirectTenantKBClient in CI)

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- tenants: one row per institution using Vidya's Mascot feature
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tenants (
    id         text PRIMARY KEY,
    name       text NOT NULL,
    domain     text NOT NULL,
    plan       text NOT NULL DEFAULT 'free',
    created_at timestamptz NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- kb_chunks: Mascot KB content scoped to a tenant
-- embedding must match SK-02 model: gemini-embedding-2 / 1536-dim
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS kb_chunks (
    id         text PRIMARY KEY,
    tenant_id  text NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    url        text NOT NULL DEFAULT '',
    title      text NOT NULL DEFAULT '',
    content    text NOT NULL,
    embedding  vector(1536),
    char_count int  NOT NULL DEFAULT 0,
    metadata   jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS kb_chunks_tenant_id_idx ON kb_chunks(tenant_id);

CREATE INDEX IF NOT EXISTS kb_chunks_embedding_hnsw ON kb_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ---------------------------------------------------------------------------
-- match_kb_chunks_for_tenant: ANN search scoped to a single tenant
-- SECURITY DEFINER so service_role can bypass RLS during CI seeding + eval
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION match_kb_chunks_for_tenant(
    p_tenant_id     text,
    query_embedding vector(1536),
    match_count     int   DEFAULT 5,
    min_similarity  float DEFAULT 0.0
)
RETURNS TABLE (
    chunk_id        text,
    content         text,
    title           text,
    url             text,
    relevance_score float
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT
        c.id                                           AS chunk_id,
        c.content,
        c.title,
        c.url,
        1 - (c.embedding <=> query_embedding)         AS relevance_score
    FROM kb_chunks c
    WHERE c.tenant_id = p_tenant_id
      AND c.embedding IS NOT NULL
      AND 1 - (c.embedding <=> query_embedding) >= min_similarity
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
$$;

GRANT ALL ON TABLE tenants   TO service_role;
GRANT ALL ON TABLE kb_chunks TO service_role;
GRANT EXECUTE ON FUNCTION match_kb_chunks_for_tenant(text, vector, int, float) TO service_role;
