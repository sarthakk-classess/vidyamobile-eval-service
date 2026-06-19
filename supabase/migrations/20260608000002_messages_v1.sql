-- RT-04: Messages table v1 — additional columns for AI service turn metadata

ALTER TABLE public.messages
  ADD COLUMN status       text        NOT NULL DEFAULT 'complete'
                          CHECK (status IN ('pending', 'complete', 'failed')),
  ADD COLUMN ai_message_id text       UNIQUE,          -- message_id from the AI service (assistant turns only)
  ADD COLUMN turn_id      text,                        -- turn_id from the AI service start event
  ADD COLUMN model_tier   text,                        -- tier selected by the router
  ADD COLUMN finish_reason text,                       -- stop | length | refused
  ADD COLUMN grounded     boolean,                     -- true if reply used retrieved chunks
  ADD COLUMN sources      jsonb,                       -- source chunks used (from done event)
  ADD COLUMN usage        jsonb;                       -- token and retrieval usage (from done event)

CREATE INDEX idx_messages_ai_message_id ON public.messages (ai_message_id)
  WHERE ai_message_id IS NOT NULL;
