-- SK-06: Add optional doc_type filter to match_chunks
-- When p_doc_type is provided, restricts ANN search to that doc type only.
-- Fixes cross-doc-type retrieval (e.g. syllabus queries pulling academic chunks).

CREATE OR REPLACE FUNCTION match_chunks(
  query_embedding  vector(1536),
  match_count      int     DEFAULT 20,
  min_similarity   float   DEFAULT 0.0,
  p_doc_type       text    DEFAULT NULL
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
    AND (p_doc_type IS NULL OR c.doc_type = p_doc_type)
    AND 1 - (c.embedding <=> query_embedding) >= min_similarity
  ORDER BY c.embedding <=> query_embedding
  LIMIT match_count;
$$;

GRANT EXECUTE ON FUNCTION match_chunks(vector, int, float, text) TO authenticated;
