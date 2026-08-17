CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    chunk_index INTEGER NOT NULL
);

-- No approximate-nearest-neighbor index (ivfflat/hnsw): this project's expected
-- corpus (a few hundred HR policy documents, low thousands of chunks) is small
-- enough that an exact sequential scan via pgvector's `<=>` operator is fast
-- (milliseconds) and always correct. An ivfflat index was tried and reverted —
-- at this row count it caused `search_similar` to silently drop matching rows
-- (see project history). Revisit with a properly-tuned index (and `probes`
-- tuning) only if the corpus grows to tens of thousands of chunks.

CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    telegram_user_id BIGINT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversations_user_created_idx
    ON conversations (telegram_user_id, created_at);

CREATE TABLE IF NOT EXISTS whitelist (
    telegram_user_id BIGINT PRIMARY KEY,
    display_name TEXT NOT NULL,
    added_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
