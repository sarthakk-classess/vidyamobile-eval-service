-- RT-07: Vector store and indexes
-- Adds pgvector extension, documents + chunks tables, and HNSW index for ANN retrieval.
-- Embedding dimension: 1536 (matches text-embedding-3-small / ada-002).
-- If SK-02 selects a different model, update the vector() size here AND drop/recreate the index.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- documents: tracks every ingested source file per user
-- ---------------------------------------------------------------------------
CREATE TABLE documents (
  id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       uuid        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  file_name     text        NOT NULL,
  -- file_path is the stable key for SK-01 deterministic chunk ID generation
  file_path     text        NOT NULL,
  doc_type      text        NOT NULL CHECK (doc_type IN ('syllabus', 'slides', 'academic')),
  -- SHA-256 of file content; used to detect unchanged re-uploads (idempotency)
  content_hash  text        NOT NULL,
  status        text        NOT NULL DEFAULT 'processing'
                            CHECK (status IN ('processing', 'complete', 'failed')),
  error         text,
  chunk_count   int,
  created_at    timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX documents_user_id_idx ON documents(user_id);
-- Prevents duplicate ingestion of the same file content for the same user
CREATE UNIQUE INDEX documents_user_content_hash_idx ON documents(user_id, content_hash);

-- ---------------------------------------------------------------------------
-- chunks: SK-01 chunks with embeddings
-- id is the deterministic SK-01 chunk ID: {prefix}_{sha256(file_path)[:12]}_{index:04d}
-- Using text PK for idempotent upsert without sequence conflicts
-- ---------------------------------------------------------------------------
CREATE TABLE chunks (
  id            text        PRIMARY KEY,
  document_id   uuid        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  user_id       uuid        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  doc_type      text        NOT NULL CHECK (doc_type IN ('syllabus', 'slides', 'academic')),
  chunk_index   int         NOT NULL,
  text          text        NOT NULL,
  -- vector(1536) — must match the embedding model selected by SK-02
  embedding     vector(1536),
  -- All SK-01 metadata fields (section_type, slide_range, page_number, etc.)
  metadata      jsonb       NOT NULL DEFAULT '{}'::jsonb,
  char_count    int         NOT NULL,
  created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX chunks_user_id_idx   ON chunks(user_id);
CREATE INDEX chunks_document_id_idx ON chunks(document_id);
-- GIN index for fast metadata queries (e.g., filter by section_type or slide_range)
CREATE INDEX chunks_metadata_idx  ON chunks USING gin(metadata);

-- HNSW index for approximate nearest-neighbour search using cosine similarity.
-- m=16, ef_construction=64 are pgvector defaults; tune against Sarthak's recall targets.
CREATE INDEX chunks_embedding_hnsw_idx ON chunks
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- ---------------------------------------------------------------------------
-- Row-level security
-- ---------------------------------------------------------------------------
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks    ENABLE ROW LEVEL SECURITY;

CREATE POLICY "documents: owner all"
  ON documents FOR ALL
  USING      (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "chunks: owner select"
  ON chunks FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "chunks: owner insert"
  ON chunks FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "chunks: owner delete"
  ON chunks FOR DELETE
  USING (auth.uid() = user_id);

-- ---------------------------------------------------------------------------
-- Helpers
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;

CREATE TRIGGER documents_updated_at
  BEFORE UPDATE ON documents
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ---------------------------------------------------------------------------
-- Storage bucket: 'documents'
-- Private bucket for raw uploaded files. Per-user path: {user_id}/{doc_id}/{filename}
-- Signed URL access will be enforced by RT-09 storage policies.
-- ---------------------------------------------------------------------------
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
  'documents',
  'documents',
  false,
  20971520,   -- 20 MB
  ARRAY[
    'application/pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'text/plain',
    'text/markdown'
  ]
)
ON CONFLICT (id) DO NOTHING;

-- RLS policies for storage.objects (scopes reads/writes to the file owner)
DROP POLICY IF EXISTS "documents storage: owner insert" ON storage.objects;
DROP POLICY IF EXISTS "documents storage: owner select" ON storage.objects;
DROP POLICY IF EXISTS "documents storage: owner delete" ON storage.objects;

CREATE POLICY "documents storage: owner insert"
  ON storage.objects FOR INSERT
  WITH CHECK (
    bucket_id = 'documents'
    AND auth.uid()::text = (string_to_array(name, '/'))[1]
  );

CREATE POLICY "documents storage: owner select"
  ON storage.objects FOR SELECT
  USING (
    bucket_id = 'documents'
    AND auth.uid()::text = (string_to_array(name, '/'))[1]
  );

CREATE POLICY "documents storage: owner delete"
  ON storage.objects FOR DELETE
  USING (
    bucket_id = 'documents'
    AND auth.uid()::text = (string_to_array(name, '/'))[1]
  );
