-- RT-06: Retrieval with match_chunks
-- Embeds the query vector, finds the nearest chunks for a user using the HNSW index,
-- and returns top candidates. Reranking is done by the caller (AI service /v1/rerank).

-- ---------------------------------------------------------------------------
-- match_chunks: ANN search scoped to the requesting user via RLS
--
-- Parameters:
--   query_embedding  — the pre-computed query vector (caller embeds first)
--   match_count      — how many candidates to return (before reranking)
--   min_similarity   — cosine similarity floor (0–1); rejects low-quality matches
--
-- Returns columns consumed by the AI service reranker:
--   id, document_id, doc_type, chunk_index, text, metadata, char_count, similarity
--
-- Security: RLS on chunks ensures a user only sees their own chunks.
-- The function runs with the INVOKER's rights, so the caller's JWT is active.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION match_chunks(
  query_embedding  vector(1536),
  match_count      int     DEFAULT 20,
  min_similarity   float   DEFAULT 0.0
)
RETURNS TABLE (
  id           text,
  document_id  uuid,
  doc_type     text,
  chunk_index  int,
  text         text,
  metadata     jsonb,
  char_count   int,
  similarity   float
)
LANGUAGE sql
STABLE
SECURITY INVOKER
AS $$
  SELECT
    c.id,
    c.document_id,
    c.doc_type,
    c.chunk_index,
    c.text,
    c.metadata,
    c.char_count,
    1 - (c.embedding <=> query_embedding) AS similarity
  FROM chunks c
  WHERE c.embedding IS NOT NULL
    AND 1 - (c.embedding <=> query_embedding) >= min_similarity
  ORDER BY c.embedding <=> query_embedding   -- HNSW index picks this up
  LIMIT match_count;
$$;

-- ---------------------------------------------------------------------------
-- match_chunks_for_document: same search restricted to a single document
-- Useful when the user is studying a specific file and wants context from it only.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION match_chunks_for_document(
  query_embedding  vector(1536),
  p_document_id    uuid,
  match_count      int     DEFAULT 20,
  min_similarity   float   DEFAULT 0.0
)
RETURNS TABLE (
  id           text,
  document_id  uuid,
  doc_type     text,
  chunk_index  int,
  text         text,
  metadata     jsonb,
  char_count   int,
  similarity   float
)
LANGUAGE sql
STABLE
SECURITY INVOKER
AS $$
  SELECT
    c.id,
    c.document_id,
    c.doc_type,
    c.chunk_index,
    c.text,
    c.metadata,
    c.char_count,
    1 - (c.embedding <=> query_embedding) AS similarity
  FROM chunks c
  WHERE c.document_id = p_document_id
    AND c.embedding IS NOT NULL
    AND 1 - (c.embedding <=> query_embedding) >= min_similarity
  ORDER BY c.embedding <=> query_embedding
  LIMIT match_count;
$$;

-- Allow authenticated users to call these functions (RLS on chunks still applies)
GRANT EXECUTE ON FUNCTION match_chunks(vector, int, float)             TO authenticated;
GRANT EXECUTE ON FUNCTION match_chunks_for_document(vector, uuid, int, float) TO authenticated;
