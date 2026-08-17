# HR RAG Telegram Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Telegram bot that answers employee questions about HR policies (leave, benefits, regulations) using RAG over internally ingested PDF/Word documents, with source citations, multi-turn context per user, and a DM whitelist.

**Architecture:** 5 focused Python packages — `ingest/` (CLI to load/chunk/embed documents into Postgres), `storage/` (pgvector-backed persistence: documents, chunks, conversations, whitelist), `rag/` (provider-agnostic retrieval + generation core, no Telegram knowledge), `bot/` (Telegram wiring: whitelist check, rate limiting, formatting), `config.py` (env-driven settings). `rag/` and `storage/` are designed to be reused by a future Slack bot.

**Tech Stack:** Python 3.11+, LiteLLM (DeepSeek completion + OpenAI `text-embedding-3-small` embedding), Postgres + pgvector, `psycopg` v3, `python-telegram-bot` v21+, `pypdf` / `python-docx` for document loading, `tiktoken` for token-based chunking, `pytest` for tests, Docker Compose for local Postgres.

**Spec:** `docs/superpowers/specs/2026-08-17-hr-rag-telegram-bot-design.md`

---

## Before You Start

- Read the spec at the path above — it defines the data model, retrieve/generate flow, error handling, and testing strategy this plan implements.
- You'll need Docker running locally for integration tests (Postgres + pgvector).
- You'll need real API keys only for Task 18 (golden QA set) and manual end-to-end testing (Task 19) — everything else uses mocks and does not call real LLM/embedding APIs.
- Confirm the exact DeepSeek model string to pass to LiteLLM before Task 19 (spec leaves this as an open question — check LiteLLM's supported-providers docs for the current DeepSeek model id). Task 2 wires it as an env var (`LLM_MODEL`) so this can be changed without code changes.

---

### Task 1: Project scaffolding & dependencies

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `pytest.ini`
- Create: `storage/__init__.py`, `ingest/__init__.py`, `rag/__init__.py`, `bot/__init__.py` (empty)
- Create: `tests/__init__.py`, `tests/ingest/__init__.py`, `tests/storage/__init__.py`, `tests/rag/__init__.py`, `tests/bot/__init__.py`, `tests/integration/__init__.py` (empty)

No TDD here — this is inert scaffolding.

- [ ] **Step 1: Create `requirements.txt`**

```
litellm>=1.50
python-telegram-bot>=21.0
psycopg[binary]>=3.1
pgvector>=0.2
pypdf>=4.0
python-docx>=1.1
tiktoken>=0.7
python-dotenv>=1.0
pyyaml>=6.0
```

- [ ] **Step 2: Create `requirements-dev.txt`**

```
-r requirements.txt
pytest>=8.0
pytest-asyncio>=0.23
```

- [ ] **Step 3: Create `.env.example`**

```
DATABASE_URL=postgresql://hrbot:hrbot@localhost:5433/hrbot
TELEGRAM_BOT_TOKEN=
LLM_MODEL=deepseek/deepseek-chat
EMBEDDING_MODEL=text-embedding-3-small
DEEPSEEK_API_KEY=
OPENAI_API_KEY=
CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K=5
SIMILARITY_THRESHOLD=0.3
CONVERSATION_CONTEXT_SIZE=10
CONVERSATION_RETENTION_DAYS=30
RATE_LIMIT_PER_MINUTE=5
```

- [ ] **Step 4: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
.env
```

- [ ] **Step 5: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
markers =
    integration: requires local docker Postgres+pgvector (see docker-compose.yml)
    golden: requires real LLM/embedding API keys, run manually via `pytest -m golden`
```

- [ ] **Step 6: Create empty `__init__.py` files**

```bash
mkdir -p storage ingest rag bot tests/ingest tests/storage tests/rag tests/bot tests/integration
touch storage/__init__.py ingest/__init__.py rag/__init__.py bot/__init__.py
touch tests/__init__.py tests/ingest/__init__.py tests/storage/__init__.py tests/rag/__init__.py tests/bot/__init__.py tests/integration/__init__.py
```

- [ ] **Step 7: Set up virtualenv and install dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

- [ ] **Step 8: Commit**

```bash
git add requirements.txt requirements-dev.txt .env.example .gitignore pytest.ini storage ingest rag bot tests
git commit -m "chore: project scaffolding and dependencies"
```

---

### Task 2: Config module

**Files:**
- Create: `config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest

from config import load_settings


REQUIRED_ENV = {
    "DATABASE_URL": "postgresql://hrbot:hrbot@localhost:5433/hrbot",
    "TELEGRAM_BOT_TOKEN": "test-token",
}


def test_load_settings_uses_defaults(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("CHUNK_SIZE", raising=False)

    settings = load_settings()

    assert settings.database_url == REQUIRED_ENV["DATABASE_URL"]
    assert settings.telegram_bot_token == "test-token"
    assert settings.llm_model == "deepseek/deepseek-chat"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 100
    assert settings.top_k == 5
    assert settings.similarity_threshold == 0.3
    assert settings.conversation_context_size == 10
    assert settings.conversation_retention_days == 30
    assert settings.rate_limit_per_minute == 5


def test_load_settings_reads_overrides(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.5")

    settings = load_settings()

    assert settings.chunk_size == 500
    assert settings.similarity_threshold == 0.5


def test_load_settings_raises_on_missing_required(monkeypatch):
    # Prevent a real .env file (e.g. created by following the README) from
    # repopulating these vars via load_dotenv() and masking the failure.
    monkeypatch.setattr("config.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        load_settings()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: Write minimal implementation**

```python
# config.py
import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    database_url: str
    telegram_bot_token: str
    llm_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    similarity_threshold: float
    conversation_context_size: int
    conversation_retention_days: int
    rate_limit_per_minute: int


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_settings() -> Settings:
    load_dotenv()  # no-op if no .env file exists; never overrides already-set env vars
    return Settings(
        database_url=_require("DATABASE_URL"),
        telegram_bot_token=_require("TELEGRAM_BOT_TOKEN"),
        llm_model=os.getenv("LLM_MODEL", "deepseek/deepseek-chat"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        top_k=int(os.getenv("TOP_K", "5")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.3")),
        conversation_context_size=int(os.getenv("CONVERSATION_CONTEXT_SIZE", "10")),
        conversation_retention_days=int(os.getenv("CONVERSATION_RETENTION_DAYS", "30")),
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "5")),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: add env-driven settings module"
```

---

### Task 3: Database schema, connection helper, migrate script

**Files:**
- Create: `storage/schema.sql`
- Create: `storage/db.py`
- Create: `storage/migrate.py`

No TDD — this is verified by Task 4's integration fixture and Task 10/11's integration tests, which will fail loudly if the schema is wrong.

- [ ] **Step 1: Create `storage/schema.sql`**

```sql
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

CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
    ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

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
```

`documents.filename` is `UNIQUE` — ingest treats filename (not full path) as the document identity, so two files with the same name in different subfolders of the ingest root will collide. Document this constraint in the ingest CLI's `--help` text in Task 13.

- [ ] **Step 2: Create `storage/db.py`**

```python
# storage/db.py
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from config import Settings


def get_connection(settings: Settings) -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url, row_factory=dict_row)
    register_vector(conn)
    return conn
```

`register_vector` lets `vector` columns accept plain Python lists as query parameters and return them as numpy arrays — every other `storage/` module relies on this being called before use.

- [ ] **Step 3: Create `storage/migrate.py`**

```python
# storage/migrate.py
from pathlib import Path

from config import load_settings
from storage.db import get_connection

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def main() -> None:
    settings = load_settings()
    with get_connection(settings) as conn:
        conn.execute(SCHEMA_PATH.read_text())
        conn.commit()
    print("Schema applied.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add storage/schema.sql storage/db.py storage/migrate.py
git commit -m "feat: add pgvector schema, connection helper, migrate script"
```

---

### Task 4: Docker Compose + integration test fixture

**Files:**
- Create: `docker-compose.yml`
- Create: `tests/integration/conftest.py`

- [ ] **Step 1: Create `docker-compose.yml`**

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: hrbot
      POSTGRES_PASSWORD: hrbot
      POSTGRES_DB: hrbot
    ports:
      - "5433:5432"
    volumes:
      - hrbot_pgdata:/var/lib/postgresql/data

volumes:
  hrbot_pgdata:
```

Port `5433` (not the default `5432`) avoids clashing with any Postgres already running on the host.

- [ ] **Step 2: Start the database**

```bash
docker compose up -d
```

- [ ] **Step 3: Create `tests/integration/conftest.py`**

```python
# tests/integration/conftest.py
import os
from pathlib import Path

import psycopg
import pytest
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "storage" / "schema.sql"
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql://hrbot:hrbot@localhost:5433/hrbot"
)


@pytest.fixture
def conn():
    connection = psycopg.connect(TEST_DATABASE_URL, row_factory=dict_row)
    register_vector(connection)
    connection.execute(SCHEMA_PATH.read_text())
    connection.commit()
    yield connection
    connection.execute(
        "TRUNCATE documents, document_chunks, conversations, whitelist RESTART IDENTITY CASCADE"
    )
    connection.commit()
    connection.close()
```

- [ ] **Step 4: Verify the fixture works with a throwaway smoke test**

```bash
cat > /tmp/test_smoke.py <<'EOF'
def test_conn_fixture_applies_schema(conn):
    row = conn.execute("SELECT to_regclass('documents') AS reg").fetchone()
    assert row["reg"] == "documents"
EOF
cp /tmp/test_smoke.py tests/integration/test_smoke.py
pytest tests/integration/test_smoke.py -v
```

Expected: PASS. Then delete the smoke test — it was only to prove the fixture works, real integration tests come in Task 10-11.

```bash
rm tests/integration/test_smoke.py
```

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml tests/integration/conftest.py
git commit -m "feat: add docker-compose Postgres and integration test fixture"
```

---

### Task 5: Chunking utility

**Files:**
- Create: `ingest/chunker.py`
- Test: `tests/ingest/test_chunker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_chunker.py
from ingest.chunker import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("", chunk_size=10, overlap=2) == []


def test_short_text_returns_single_chunk():
    chunks = chunk_text("một hai ba", chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == "một hai ba"


def test_long_text_produces_overlapping_chunks():
    # 50 distinct word-tokens, easy to reason about with a small chunk_size
    text = " ".join(f"tu{i}" for i in range(50))
    chunks = chunk_text(text, chunk_size=10, overlap=3)

    assert len(chunks) > 1
    # every chunk except the last respects the configured size (in tokens)
    for chunk in chunks[:-1]:
        assert len(chunk) > 0
    # consecutive chunks share overlapping content
    assert chunks[0].split()[-1] in chunks[1]


def test_rejects_invalid_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=10, overlap=-1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingest/test_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingest.chunker'`

- [ ] **Step 3: Write minimal implementation**

```python
# ingest/chunker.py
import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    tokens = _ENCODING.encode(text)
    if not tokens:
        return []

    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(_ENCODING.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += step
    return chunks
```

Token count is measured with `tiktoken`'s `cl100k_base` encoding (the encoding used by recent OpenAI models) — this is the tokenizer the 800/100 defaults in `config.py` were chosen against.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingest/test_chunker.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add ingest/chunker.py tests/ingest/test_chunker.py
git commit -m "feat: add token-based text chunker with overlap"
```

---

### Task 6: Document loader

**Files:**
- Create: `ingest/loader.py`
- Test: `tests/ingest/test_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_loader.py
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from docx import Document as DocxDocument

from ingest.loader import load_text


def test_load_docx_joins_paragraphs(tmp_path):
    path = tmp_path / "sample.docx"
    doc = DocxDocument()
    doc.add_paragraph("Chính sách nghỉ phép.")
    doc.add_paragraph("Nhân viên được nghỉ 12 ngày mỗi năm.")
    doc.save(path)

    text = load_text(path)

    assert "Chính sách nghỉ phép." in text
    assert "Nhân viên được nghỉ 12 ngày mỗi năm." in text


def test_load_pdf_joins_page_text(tmp_path):
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF-1.4 fake")  # content irrelevant, PdfReader is mocked

    fake_page_1 = MagicMock()
    fake_page_1.extract_text.return_value = "Trang 1"
    fake_page_2 = MagicMock()
    fake_page_2.extract_text.return_value = "Trang 2"

    with patch("ingest.loader.PdfReader") as mock_reader:
        mock_reader.return_value.pages = [fake_page_1, fake_page_2]
        text = load_text(path)

    assert text == "Trang 1\nTrang 2"


def test_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("hello")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_text(path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingest/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingest.loader'`

- [ ] **Step 3: Write minimal implementation**

```python
# ingest/loader.py
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".docx"}


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _load_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingest/test_loader.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add ingest/loader.py tests/ingest/test_loader.py
git commit -m "feat: add PDF/DOCX text loader"
```

---

### Task 7: Storage — documents & chunks modules

**Files:**
- Create: `storage/documents.py`
- Create: `storage/chunks.py`
- Test: `tests/storage/test_chunks.py` (pure-function part only; full CRUD is covered by Task 10's integration test)

- [ ] **Step 1: Write the failing test (pure conversion function only)**

```python
# tests/storage/test_chunks.py
from storage.chunks import distance_to_similarity


def test_distance_to_similarity_identical_vectors():
    assert distance_to_similarity(0.0) == 1.0


def test_distance_to_similarity_orthogonal_vectors():
    assert distance_to_similarity(1.0) == 0.0


def test_distance_to_similarity_opposite_vectors():
    assert distance_to_similarity(2.0) == -1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/storage/test_chunks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'storage.chunks'`

- [ ] **Step 3: Write `storage/documents.py`**

```python
# storage/documents.py
import hashlib

from psycopg import Connection


def compute_content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def get_document_by_filename(conn: Connection, filename: str) -> dict | None:
    return conn.execute(
        "SELECT id, filename, content_hash FROM documents WHERE filename = %s",
        (filename,),
    ).fetchone()


def upsert_document(conn: Connection, filename: str, content_hash: str) -> tuple[int, bool]:
    """Insert or update a document row.

    Returns (document_id, changed). changed=False means the file already
    exists with identical content — caller should skip re-chunking/embedding.
    """
    existing = get_document_by_filename(conn, filename)
    if existing is None:
        row = conn.execute(
            "INSERT INTO documents (filename, content_hash) VALUES (%s, %s) RETURNING id",
            (filename, content_hash),
        ).fetchone()
        conn.commit()
        return row["id"], True

    if existing["content_hash"] == content_hash:
        return existing["id"], False

    conn.execute(
        "UPDATE documents SET content_hash = %s, ingested_at = now() WHERE id = %s",
        (content_hash, existing["id"]),
    )
    conn.execute("DELETE FROM document_chunks WHERE document_id = %s", (existing["id"],))
    conn.commit()
    return existing["id"], True
```

- [ ] **Step 4: Write `storage/chunks.py`**

```python
# storage/chunks.py
from psycopg import Connection


def distance_to_similarity(distance: float) -> float:
    """pgvector's <=> operator returns cosine *distance* (1 - cosine similarity).
    Callers must use this conversion, never the raw distance, when comparing
    against a similarity threshold."""
    return 1 - distance


def insert_chunk(
    conn: Connection,
    document_id: int,
    chunk_text: str,
    embedding: list[float],
    chunk_index: int,
) -> None:
    conn.execute(
        """
        INSERT INTO document_chunks (document_id, chunk_text, embedding, chunk_index)
        VALUES (%s, %s, %s, %s)
        """,
        (document_id, chunk_text, embedding, chunk_index),
    )
    conn.commit()


def search_similar(conn: Connection, query_embedding: list[float], top_k: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT dc.chunk_text, d.filename, dc.embedding <=> %s AS distance
        FROM document_chunks dc
        JOIN documents d ON d.id = dc.document_id
        ORDER BY dc.embedding <=> %s
        LIMIT %s
        """,
        (query_embedding, query_embedding, top_k),
    ).fetchall()
    return [
        {
            "chunk_text": row["chunk_text"],
            "filename": row["filename"],
            "similarity": distance_to_similarity(row["distance"]),
        }
        for row in rows
    ]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/storage/test_chunks.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add storage/documents.py storage/chunks.py tests/storage/test_chunks.py
git commit -m "feat: add document/chunk storage with idempotent ingest and similarity search"
```

---

### Task 8: Storage — conversations module + cleanup script

**Files:**
- Create: `storage/conversations.py`
- Create: `storage/cleanup.py`

No dedicated unit test — pure DB round-trip logic, covered by Task 11's integration test.

- [ ] **Step 1: Create `storage/conversations.py`**

```python
# storage/conversations.py
from psycopg import Connection


def save_turn(conn: Connection, telegram_user_id: int, role: str, content: str) -> None:
    conn.execute(
        "INSERT INTO conversations (telegram_user_id, role, content) VALUES (%s, %s, %s)",
        (telegram_user_id, role, content),
    )
    conn.commit()


def get_recent_turns(conn: Connection, telegram_user_id: int, limit: int) -> list[dict]:
    """Returns up to `limit` most recent turns, oldest first (ready to feed into a prompt)."""
    rows = conn.execute(
        """
        SELECT role, content FROM conversations
        WHERE telegram_user_id = %s
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (telegram_user_id, limit),
    ).fetchall()
    return list(reversed(rows))


def cleanup_old_turns(conn: Connection, retention_days: int) -> int:
    """Hard-deletes conversation rows older than retention_days, regardless of
    the per-user context window used by get_recent_turns. Returns rows deleted."""
    cursor = conn.execute(
        "DELETE FROM conversations WHERE created_at < now() - %s::interval",
        (f"{retention_days} days",),
    )
    conn.commit()
    return cursor.rowcount
```

- [ ] **Step 2: Create `storage/cleanup.py`**

```python
# storage/cleanup.py
from config import load_settings
from storage.conversations import cleanup_old_turns
from storage.db import get_connection


def main() -> None:
    settings = load_settings()
    with get_connection(settings) as conn:
        deleted = cleanup_old_turns(conn, settings.conversation_retention_days)
    print(f"Deleted {deleted} conversation rows older than {settings.conversation_retention_days} days.")


if __name__ == "__main__":
    main()
```

Run this periodically (cron/systemd timer) once deployed: `python -m storage.cleanup`. Deployment/scheduling mechanism is out of scope for this plan (spec: "Auto-deploy/hosting — quyết định sau").

- [ ] **Step 3: Commit**

```bash
git add storage/conversations.py storage/cleanup.py
git commit -m "feat: add conversation history storage and retention cleanup script"
```

---

### Task 9: Storage — whitelist module + CLI

**Files:**
- Create: `storage/whitelist.py`
- Test: `tests/storage/test_whitelist.py`

- [ ] **Step 1: Write the failing test (pure logic, mocked connection)**

```python
# tests/storage/test_whitelist.py
from unittest.mock import MagicMock

from storage.whitelist import is_whitelisted


def test_is_whitelisted_true_when_row_found():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {"telegram_user_id": 42}

    assert is_whitelisted(conn, 42) is True
    conn.execute.assert_called_once()


def test_is_whitelisted_false_when_no_row():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None

    assert is_whitelisted(conn, 999) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/storage/test_whitelist.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'storage.whitelist'`

- [ ] **Step 3: Write minimal implementation**

```python
# storage/whitelist.py
import argparse

from psycopg import Connection

from config import load_settings
from storage.db import get_connection


def is_whitelisted(conn: Connection, telegram_user_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM whitelist WHERE telegram_user_id = %s", (telegram_user_id,)
    ).fetchone()
    return row is not None


def add_user(conn: Connection, telegram_user_id: int, display_name: str) -> None:
    conn.execute(
        """
        INSERT INTO whitelist (telegram_user_id, display_name)
        VALUES (%s, %s)
        ON CONFLICT (telegram_user_id) DO UPDATE SET display_name = EXCLUDED.display_name
        """,
        (telegram_user_id, display_name),
    )
    conn.commit()


def remove_user(conn: Connection, telegram_user_id: int) -> None:
    conn.execute("DELETE FROM whitelist WHERE telegram_user_id = %s", (telegram_user_id,))
    conn.commit()


def list_users(conn: Connection) -> list[dict]:
    return conn.execute(
        "SELECT telegram_user_id, display_name, added_at FROM whitelist ORDER BY added_at"
    ).fetchall()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the Telegram bot whitelist")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Whitelist a Telegram user")
    add_parser.add_argument("telegram_user_id", type=int)
    add_parser.add_argument("display_name")

    remove_parser = subparsers.add_parser("remove", help="Remove a user from the whitelist")
    remove_parser.add_argument("telegram_user_id", type=int)

    subparsers.add_parser("list", help="List whitelisted users")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    settings = load_settings()
    with get_connection(settings) as conn:
        if args.command == "add":
            add_user(conn, args.telegram_user_id, args.display_name)
            print(f"Added {args.telegram_user_id} ({args.display_name})")
        elif args.command == "remove":
            remove_user(conn, args.telegram_user_id)
            print(f"Removed {args.telegram_user_id}")
        elif args.command == "list":
            for row in list_users(conn):
                print(f"{row['telegram_user_id']}\t{row['display_name']}\t{row['added_at']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/storage/test_whitelist.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add storage/whitelist.py tests/storage/test_whitelist.py
git commit -m "feat: add whitelist storage and admin CLI (add/remove/list)"
```

---

### Task 10: Integration test — ingest → retrieve

**Files:**
- Test: `tests/integration/test_ingest_retrieve.py`

This exercises `storage/documents.py` + `storage/chunks.py` end-to-end against real Postgres+pgvector, with `embed()` faked (deterministic vectors) so no real API calls happen.

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_ingest_retrieve.py
import pytest

from storage.chunks import insert_chunk, search_similar
from storage.documents import compute_content_hash, upsert_document

pytestmark = pytest.mark.integration


def _fake_embedding(seed: float) -> list[float]:
    # 1536-dim vector, mostly zeros with one distinctive component —
    # deterministic and cheap, good enough to test ordering/idempotency.
    vec = [0.0] * 1536
    vec[0] = seed
    return vec


def test_upsert_document_is_idempotent_on_unchanged_content(conn):
    content = b"chinh sach nghi phep"
    content_hash = compute_content_hash(content)

    doc_id_1, changed_1 = upsert_document(conn, "policy.pdf", content_hash)
    doc_id_2, changed_2 = upsert_document(conn, "policy.pdf", content_hash)

    assert doc_id_1 == doc_id_2
    assert changed_1 is True
    assert changed_2 is False


def test_upsert_document_replaces_chunks_when_content_changes(conn):
    content_hash_v1 = compute_content_hash(b"version 1")
    doc_id, _ = upsert_document(conn, "policy.pdf", content_hash_v1)
    insert_chunk(conn, doc_id, "old chunk", _fake_embedding(1.0), 0)

    content_hash_v2 = compute_content_hash(b"version 2")
    doc_id_again, changed = upsert_document(conn, "policy.pdf", content_hash_v2)

    remaining = conn.execute(
        "SELECT count(*) AS n FROM document_chunks WHERE document_id = %s", (doc_id,)
    ).fetchone()

    assert doc_id_again == doc_id
    assert changed is True
    assert remaining["n"] == 0


def test_search_similar_returns_closest_chunk_first(conn):
    doc_id, _ = upsert_document(conn, "policy.pdf", compute_content_hash(b"content"))
    insert_chunk(conn, doc_id, "chunk far", _fake_embedding(10.0), 0)
    insert_chunk(conn, doc_id, "chunk close", _fake_embedding(1.01), 1)

    results = search_similar(conn, _fake_embedding(1.0), top_k=2)

    assert results[0]["chunk_text"] == "chunk close"
    assert results[0]["filename"] == "policy.pdf"
    assert results[0]["similarity"] > results[1]["similarity"]
```

- [ ] **Step 2: Run test to verify it fails first (before this task, if run against a clean checkout it should pass since Task 7/9 already exist — instead verify by temporarily breaking `search_similar`'s ORDER BY and confirming the test catches it)**

Run: `pytest tests/integration/test_ingest_retrieve.py -v`
Expected: PASS (3 tests) — Task 7 already implemented the code under test, so this task is pure test-writing. If any test fails, fix `storage/documents.py` or `storage/chunks.py`, not the test.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_ingest_retrieve.py
git commit -m "test: add integration coverage for idempotent ingest and similarity search"
```

---

### Task 11: Integration test — conversations + whitelist

**Files:**
- Test: `tests/integration/test_conversations_whitelist.py`

- [ ] **Step 1: Write the test**

```python
# tests/integration/test_conversations_whitelist.py
import pytest

from storage.conversations import cleanup_old_turns, get_recent_turns, save_turn
from storage.whitelist import add_user, is_whitelisted, list_users, remove_user

pytestmark = pytest.mark.integration


def test_conversation_round_trip_preserves_order(conn):
    save_turn(conn, 111, "user", "cau hoi 1")
    save_turn(conn, 111, "assistant", "tra loi 1")
    save_turn(conn, 111, "user", "cau hoi 2")

    turns = get_recent_turns(conn, 111, limit=10)

    assert [t["content"] for t in turns] == ["cau hoi 1", "tra loi 1", "cau hoi 2"]


def test_get_recent_turns_respects_limit(conn):
    for i in range(5):
        save_turn(conn, 222, "user", f"msg{i}")

    turns = get_recent_turns(conn, 222, limit=2)

    assert [t["content"] for t in turns] == ["msg3", "msg4"]


def test_cleanup_old_turns_deletes_past_retention_window(conn):
    save_turn(conn, 333, "user", "old enough to be backdated")
    conn.execute(
        "UPDATE conversations SET created_at = now() - interval '31 days' WHERE telegram_user_id = 333"
    )
    save_turn(conn, 333, "user", "recent")
    conn.commit()

    deleted = cleanup_old_turns(conn, retention_days=30)
    remaining = get_recent_turns(conn, 333, limit=10)

    assert deleted == 1
    assert [t["content"] for t in remaining] == ["recent"]


def test_whitelist_add_remove_list_round_trip(conn):
    add_user(conn, 444, "Nguyen Van A")

    assert is_whitelisted(conn, 444) is True
    users = list_users(conn)
    assert any(u["telegram_user_id"] == 444 for u in users)

    remove_user(conn, 444)

    assert is_whitelisted(conn, 444) is False
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/integration/test_conversations_whitelist.py -v`
Expected: PASS (4 tests)

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_conversations_whitelist.py
git commit -m "test: add integration coverage for conversation history and whitelist CRUD"
```

---

### Task 12: LLM wrapper (LiteLLM completion + embedding, with retry)

**Files:**
- Create: `rag/llm.py`
- Test: `tests/rag/test_llm.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/rag/test_llm.py
from unittest.mock import patch

import pytest

from rag.llm import LLMError, complete, embed


def test_embed_returns_vector_from_litellm_response():
    fake_response = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    with patch("rag.llm.litellm.embedding", return_value=fake_response) as mock_embed:
        result = embed("cau hoi", model="text-embedding-3-small")

    assert result == [0.1, 0.2, 0.3]
    mock_embed.assert_called_once_with(model="text-embedding-3-small", input=["cau hoi"])


def test_embed_wraps_exceptions_as_llm_error():
    with patch("rag.llm.litellm.embedding", side_effect=RuntimeError("boom")):
        with pytest.raises(LLMError):
            embed("cau hoi", model="text-embedding-3-small")


def test_complete_returns_text_on_first_success():
    fake_response = {"choices": [{"message": {"content": "cau tra loi"}}]}
    with patch("rag.llm.litellm.completion", return_value=fake_response) as mock_complete:
        result = complete([{"role": "user", "content": "hi"}], model="deepseek/deepseek-chat")

    assert result == "cau tra loi"
    assert mock_complete.call_count == 1


def test_complete_retries_once_then_succeeds():
    fake_response = {"choices": [{"message": {"content": "ok after retry"}}]}
    with patch(
        "rag.llm.litellm.completion",
        side_effect=[RuntimeError("transient"), fake_response],
    ) as mock_complete:
        result = complete([{"role": "user", "content": "hi"}], model="deepseek/deepseek-chat")

    assert result == "ok after retry"
    assert mock_complete.call_count == 2


def test_complete_raises_llm_error_after_exhausting_retries():
    with patch("rag.llm.litellm.completion", side_effect=RuntimeError("down")):
        with pytest.raises(LLMError):
            complete([{"role": "user", "content": "hi"}], model="deepseek/deepseek-chat")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.llm'`

- [ ] **Step 3: Write minimal implementation**

```python
# rag/llm.py
import litellm


class LLMError(Exception):
    """Raised when an embedding or completion call fails after retries."""


def embed(text: str, model: str) -> list[float]:
    try:
        response = litellm.embedding(model=model, input=[text])
    except Exception as exc:
        raise LLMError(f"Embedding call failed: {exc}") from exc
    return response["data"][0]["embedding"]


def complete(messages: list[dict], model: str, max_retries: int = 1) -> str:
    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            response = litellm.completion(model=model, messages=messages)
            return response["choices"][0]["message"]["content"]
        except Exception as exc:
            last_error = exc
    raise LLMError(f"Completion call failed after {max_retries + 1} attempts: {last_error}") from last_error
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag/test_llm.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add rag/llm.py tests/rag/test_llm.py
git commit -m "feat: add LiteLLM wrapper with single-retry completion"
```

---

### Task 13: Ingest CLI

**Files:**
- Create: `ingest/cli.py`
- Test: `tests/ingest/test_cli.py`

Wires together `ingest/loader.py` (Task 6), `ingest/chunker.py` (Task 5), `rag/llm.py`'s `embed()` (Task 12), and `storage/documents.py` / `storage/chunks.py` (Task 7) into the directory-walking pipeline the spec describes as `ingest/`'s primary deliverable. This is the module the README (Task 19) and manual end-to-end verification invoke as `python -m ingest.cli --path docs_source/`.

- [ ] **Step 1: Write the failing test**

```python
# tests/ingest/test_cli.py
from unittest.mock import MagicMock, patch

from docx import Document as DocxDocument

from config import Settings
from ingest.cli import ingest_file


def _settings() -> Settings:
    return Settings(
        database_url="postgresql://x",
        telegram_bot_token="token",
        llm_model="deepseek/deepseek-chat",
        embedding_model="text-embedding-3-small",
        chunk_size=800,
        chunk_overlap=100,
        top_k=5,
        similarity_threshold=0.3,
        conversation_context_size=10,
        conversation_retention_days=30,
        rate_limit_per_minute=5,
    )


@patch("ingest.cli.insert_chunk")
@patch("ingest.cli.embed", return_value=[0.1] * 1536)
@patch("ingest.cli.upsert_document", return_value=(1, True))
def test_ingest_file_chunks_and_embeds_new_document(mock_upsert, mock_embed, mock_insert, tmp_path):
    path = tmp_path / "policy.docx"
    doc = DocxDocument()
    doc.add_paragraph("Noi dung chinh sach nghi phep cua cong ty.")
    doc.save(path)
    conn = MagicMock()

    ingest_file(conn, _settings(), path)

    mock_upsert.assert_called_once()
    assert mock_embed.call_count >= 1
    assert mock_insert.call_count == mock_embed.call_count


@patch("ingest.cli.insert_chunk")
@patch("ingest.cli.embed")
@patch("ingest.cli.upsert_document", return_value=(1, False))
def test_ingest_file_skips_unchanged_document(mock_upsert, mock_embed, mock_insert, tmp_path):
    path = tmp_path / "policy.docx"
    path.write_bytes(b"irrelevant, upsert_document is mocked so content is never parsed")
    conn = MagicMock()

    ingest_file(conn, _settings(), path)

    mock_embed.assert_not_called()
    mock_insert.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingest/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingest.cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# ingest/cli.py
import argparse
import logging
from pathlib import Path

from psycopg import Connection

from config import Settings, load_settings
from ingest.chunker import chunk_text
from ingest.loader import load_text
from rag.llm import embed
from storage.chunks import insert_chunk
from storage.db import get_connection
from storage.documents import compute_content_hash, upsert_document

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".pdf", ".docx"}


def ingest_file(conn: Connection, settings: Settings, path: Path) -> None:
    content = path.read_bytes()
    content_hash = compute_content_hash(content)
    document_id, changed = upsert_document(conn, path.name, content_hash)
    if not changed:
        logger.info("Skip unchanged file: %s", path.name)
        return

    text = load_text(path)
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    for index, chunk in enumerate(chunks):
        embedding = embed(chunk, settings.embedding_model)
        insert_chunk(conn, document_id, chunk, embedding, index)
    logger.info("Ingested %s: %d chunks", path.name, len(chunks))


def ingest_path(root: Path) -> None:
    settings = load_settings()
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    with get_connection(settings) as conn:
        for path in files:
            ingest_file(conn, settings, path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest HR policy PDF/DOCX documents into Postgres. "
            "Filenames must be unique across the ingest root (documents.filename has a "
            "UNIQUE constraint) — two files with the same name in different subfolders "
            "will collide."
        )
    )
    parser.add_argument(
        "--path", required=True, type=Path, help="Directory to scan recursively for .pdf/.docx files"
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args()
    ingest_path(args.path)


if __name__ == "__main__":
    main()
```

`ingest_file` is factored out from `ingest_path` specifically so the test above can exercise the chunk/embed/store orchestration without touching a real filesystem walk or database connection.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingest/test_cli.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add ingest/cli.py tests/ingest/test_cli.py
git commit -m "feat: add ingest CLI wiring loader, chunker, embedding, and storage"
```

---

### Task 14: Prompt builder

**Files:**
- Create: `rag/prompt.py`
- Test: `tests/rag/test_prompt.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/rag/test_prompt.py
from rag.prompt import SYSTEM_PROMPT, build_messages


def test_first_message_is_the_guardrail_system_prompt():
    messages = build_messages("cau hoi", history=[], chunks=[])

    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "chỉ trả lời" in SYSTEM_PROMPT.lower()


def test_history_is_included_in_order_between_system_and_final_user_turn():
    history = [
        {"role": "user", "content": "cau hoi cu"},
        {"role": "assistant", "content": "tra loi cu"},
    ]
    messages = build_messages("cau hoi moi", history=history, chunks=[])

    assert messages[1] == {"role": "user", "content": "cau hoi cu"}
    assert messages[2] == {"role": "assistant", "content": "tra loi cu"}


def test_final_user_message_includes_context_and_question():
    chunks = [
        {"filename": "policy.pdf", "chunk_text": "12 ngay nghi phep moi nam"},
    ]
    messages = build_messages("bao nhieu ngay nghi phep?", history=[], chunks=chunks)

    final = messages[-1]
    assert final["role"] == "user"
    assert "policy.pdf" in final["content"]
    assert "12 ngay nghi phep moi nam" in final["content"]
    assert "bao nhieu ngay nghi phep?" in final["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_prompt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.prompt'`

- [ ] **Step 3: Write minimal implementation**

```python
# rag/prompt.py
SYSTEM_PROMPT = (
    "Bạn là trợ lý trả lời câu hỏi về chính sách nhân sự (HR) nội bộ công ty. "
    "Chỉ trả lời dựa trên các đoạn trích trong phần NGỮ CẢNH được cung cấp bên dưới. "
    "Nếu ngữ cảnh không đủ để trả lời, hãy nói rõ là bạn không biết — "
    "không suy diễn, không bịa thông tin, không trả lời ngoài chủ đề chính sách HR."
)


def build_messages(question: str, history: list[dict], chunks: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    context = "\n\n".join(f"[Nguồn: {c['filename']}]\n{c['chunk_text']}" for c in chunks)
    messages.append(
        {
            "role": "user",
            "content": f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI: {question}",
        }
    )
    return messages
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag/test_prompt.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add rag/prompt.py tests/rag/test_prompt.py
git commit -m "feat: add prompt builder with scope-limiting guardrail"
```

---

### Task 15: RAG engine (orchestration)

**Files:**
- Create: `rag/engine.py`
- Test: `tests/rag/test_engine.py`

This is the core module described in the spec as "không biết gì về Telegram" — it takes a generic `telegram_user_id` (the whitelist/conversation key) and returns a plain result object; `bot/` is the only caller.

- [ ] **Step 1: Write the failing test**

```python
# tests/rag/test_engine.py
from unittest.mock import MagicMock, patch

from config import Settings
from rag.engine import FALLBACK_MESSAGE, answer


def _settings(**overrides) -> Settings:
    base = dict(
        database_url="postgresql://x",
        telegram_bot_token="token",
        llm_model="deepseek/deepseek-chat",
        embedding_model="text-embedding-3-small",
        chunk_size=800,
        chunk_overlap=100,
        top_k=5,
        similarity_threshold=0.3,
        conversation_context_size=10,
        conversation_retention_days=30,
        rate_limit_per_minute=5,
    )
    base.update(overrides)
    return Settings(**base)


@patch("rag.engine.save_turn")
@patch("rag.engine.get_recent_turns", return_value=[])
@patch("rag.engine.complete", return_value="cau tra loi that")
@patch("rag.engine.search_similar")
@patch("rag.engine.embed", return_value=[0.1] * 1536)
def test_answer_returns_completion_and_sources_when_chunks_found(
    mock_embed, mock_search, mock_complete, mock_history, mock_save
):
    mock_search.return_value = [
        {"chunk_text": "12 ngay phep", "filename": "policy.pdf", "similarity": 0.8},
    ]
    conn = MagicMock()

    result = answer(conn, _settings(), telegram_user_id=1, question="bao nhieu ngay phep?")

    assert result.text == "cau tra loi that"
    assert result.sources == ["policy.pdf"]
    mock_complete.assert_called_once()
    assert mock_save.call_count == 2  # user turn + assistant turn


@patch("rag.engine.save_turn")
@patch("rag.engine.complete")
@patch("rag.engine.search_similar", return_value=[])
@patch("rag.engine.embed", return_value=[0.1] * 1536)
def test_answer_falls_back_when_no_chunks_found(mock_embed, mock_search, mock_complete, mock_save):
    conn = MagicMock()

    result = answer(conn, _settings(), telegram_user_id=1, question="cau hoi la")

    assert result.text == FALLBACK_MESSAGE
    assert result.sources == []
    mock_complete.assert_not_called()
    assert mock_save.call_count == 2  # fallback is still saved as a valid turn


@patch("rag.engine.save_turn")
@patch("rag.engine.complete")
@patch("rag.engine.search_similar")
@patch("rag.engine.embed", return_value=[0.1] * 1536)
def test_answer_falls_back_when_best_similarity_below_threshold(
    mock_embed, mock_search, mock_complete, mock_save
):
    mock_search.return_value = [
        {"chunk_text": "khong lien quan", "filename": "other.pdf", "similarity": 0.1},
    ]
    conn = MagicMock()

    result = answer(conn, _settings(similarity_threshold=0.3), telegram_user_id=1, question="?")

    assert result.text == FALLBACK_MESSAGE
    mock_complete.assert_not_called()


@patch("rag.engine.save_turn")
@patch("rag.engine.get_recent_turns", return_value=[])
@patch("rag.engine.complete", side_effect=Exception("llm down"))
@patch("rag.engine.search_similar")
@patch("rag.engine.embed", return_value=[0.1] * 1536)
def test_answer_does_not_save_turn_when_completion_fails(
    mock_embed, mock_search, mock_complete, mock_history, mock_save
):
    mock_search.return_value = [
        {"chunk_text": "12 ngay phep", "filename": "policy.pdf", "similarity": 0.8},
    ]
    conn = MagicMock()

    with pytest.raises(Exception):
        answer(conn, _settings(), telegram_user_id=1, question="?")

    mock_save.assert_not_called()
```

Note: this test file uses `pytest.raises`, so add `import pytest` to the top of the file alongside the existing imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/rag/test_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.engine'`

- [ ] **Step 3: Write minimal implementation**

```python
# rag/engine.py
from dataclasses import dataclass

from psycopg import Connection

from config import Settings
from rag.llm import complete, embed
from rag.prompt import build_messages
from storage.chunks import search_similar
from storage.conversations import get_recent_turns, save_turn

FALLBACK_MESSAGE = (
    "Không tìm thấy thông tin này trong chính sách hiện có, vui lòng liên hệ HR trực tiếp."
)


@dataclass
class AnswerResult:
    text: str
    sources: list[str]


def answer(conn: Connection, settings: Settings, telegram_user_id: int, question: str) -> AnswerResult:
    query_embedding = embed(question, settings.embedding_model)
    chunks = search_similar(conn, query_embedding, settings.top_k)

    best_similarity = chunks[0]["similarity"] if chunks else 0.0
    if not chunks or best_similarity < settings.similarity_threshold:
        save_turn(conn, telegram_user_id, "user", question)
        save_turn(conn, telegram_user_id, "assistant", FALLBACK_MESSAGE)
        return AnswerResult(text=FALLBACK_MESSAGE, sources=[])

    history = get_recent_turns(conn, telegram_user_id, settings.conversation_context_size)
    messages = build_messages(question, history, chunks)
    reply_text = complete(messages, settings.llm_model)  # raises LLMError -> propagates, no save

    sources = sorted({c["filename"] for c in chunks})
    save_turn(conn, telegram_user_id, "user", question)
    save_turn(conn, telegram_user_id, "assistant", reply_text)
    return AnswerResult(text=reply_text, sources=sources)
```

If `complete()` raises, no `save_turn` call has happened yet for this turn — matching the spec's "không lưu khi gặp lỗi kỹ thuật" rule. `embed()` failing raises before any DB read/write for the same reason.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/rag/test_engine.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add rag/engine.py tests/rag/test_engine.py
git commit -m "feat: add RAG engine orchestrating retrieval, guardrail, and generation"
```

---

### Task 16: Rate limiter

**Files:**
- Create: `bot/ratelimit.py`
- Test: `tests/bot/test_ratelimit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/bot/test_ratelimit.py
from bot.ratelimit import RateLimiter


def test_allows_up_to_the_configured_limit():
    limiter = RateLimiter(max_per_minute=3)

    assert limiter.allow(user_id=1, now=0.0) is True
    assert limiter.allow(user_id=1, now=1.0) is True
    assert limiter.allow(user_id=1, now=2.0) is True
    assert limiter.allow(user_id=1, now=3.0) is False


def test_limits_are_independent_per_user():
    limiter = RateLimiter(max_per_minute=1)

    assert limiter.allow(user_id=1, now=0.0) is True
    assert limiter.allow(user_id=2, now=0.0) is True
    assert limiter.allow(user_id=1, now=0.5) is False


def test_window_slides_after_sixty_seconds():
    limiter = RateLimiter(max_per_minute=1)

    assert limiter.allow(user_id=1, now=0.0) is True
    assert limiter.allow(user_id=1, now=30.0) is False
    assert limiter.allow(user_id=1, now=61.0) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/bot/test_ratelimit.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bot.ratelimit'`

- [ ] **Step 3: Write minimal implementation**

```python
# bot/ratelimit.py
import time
from collections import defaultdict, deque


class RateLimiter:
    """In-memory sliding-window rate limiter, per Telegram user.

    In-memory means limits reset on process restart and don't share state
    across multiple bot instances — acceptable for a single-process MVP
    deployment (see spec's open deployment question)."""

    def __init__(self, max_per_minute: int):
        self._max_per_minute = max_per_minute
        self._timestamps: dict[int, deque] = defaultdict(deque)

    def allow(self, user_id: int, now: float | None = None) -> bool:
        current = now if now is not None else time.monotonic()
        window_start = current - 60
        timestamps = self._timestamps[user_id]
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()
        if len(timestamps) >= self._max_per_minute:
            return False
        timestamps.append(current)
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/bot/test_ratelimit.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add bot/ratelimit.py tests/bot/test_ratelimit.py
git commit -m "feat: add per-user sliding-window rate limiter"
```

---

### Task 17: Telegram bot

**Files:**
- Create: `bot/telegram_bot.py`
- Test: `tests/bot/test_telegram_bot.py`

`python-telegram-bot`'s `Application` processes updates sequentially by default (`concurrent_updates=False`), which is what serializes access to `conversations` per spec's "xử lý tuần tự" requirement — no extra locking code needed here.

- [ ] **Step 1: Write the failing test**

```python
# tests/bot/test_telegram_bot.py
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.ratelimit import RateLimiter
from bot.telegram_bot import (
    GENERIC_ERROR_MESSAGE,
    RATE_LIMIT_MESSAGE,
    WHITELIST_REJECTION,
    handle_message,
)
from config import Settings
from rag.engine import AnswerResult
from rag.llm import LLMError


def _settings() -> Settings:
    return Settings(
        database_url="postgresql://x",
        telegram_bot_token="token",
        llm_model="deepseek/deepseek-chat",
        embedding_model="text-embedding-3-small",
        chunk_size=800,
        chunk_overlap=100,
        top_k=5,
        similarity_threshold=0.3,
        conversation_context_size=10,
        conversation_retention_days=30,
        rate_limit_per_minute=5,
    )


def _mock_update_and_context(user_id: int, text: str, settings: Settings, limiter: RateLimiter):
    update = MagicMock()
    update.effective_user.id = user_id
    update.message.text = text
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.bot_data = {"settings": settings, "rate_limiter": limiter}
    return update, context


@pytest.mark.asyncio
@patch("bot.telegram_bot.get_connection")
@patch("bot.telegram_bot.is_whitelisted", return_value=False)
async def test_handle_message_rejects_non_whitelisted_user(mock_whitelisted, mock_get_conn):
    settings = _settings()
    update, context = _mock_update_and_context(1, "hi", settings, RateLimiter(5))

    await handle_message(update, context)

    update.message.reply_text.assert_awaited_once_with(WHITELIST_REJECTION)


@pytest.mark.asyncio
@patch("bot.telegram_bot.get_connection")
@patch("bot.telegram_bot.is_whitelisted", return_value=True)
async def test_handle_message_rejects_when_rate_limited(mock_whitelisted, mock_get_conn):
    settings = _settings()
    limiter = RateLimiter(max_per_minute=0)
    update, context = _mock_update_and_context(1, "hi", settings, limiter)

    await handle_message(update, context)

    update.message.reply_text.assert_awaited_once_with(RATE_LIMIT_MESSAGE)


@pytest.mark.asyncio
@patch("bot.telegram_bot.answer")
@patch("bot.telegram_bot.get_connection")
@patch("bot.telegram_bot.is_whitelisted", return_value=True)
async def test_handle_message_appends_sources_on_success(mock_whitelisted, mock_get_conn, mock_answer):
    mock_answer.return_value = AnswerResult(text="Ban duoc nghi 12 ngay", sources=["policy.pdf"])
    settings = _settings()
    update, context = _mock_update_and_context(1, "bao nhieu ngay phep?", settings, RateLimiter(5))

    await handle_message(update, context)

    sent = update.message.reply_text.await_args.args[0]
    assert "Ban duoc nghi 12 ngay" in sent
    assert "policy.pdf" in sent


@pytest.mark.asyncio
@patch("bot.telegram_bot.answer", side_effect=LLMError("down"))
@patch("bot.telegram_bot.get_connection")
@patch("bot.telegram_bot.is_whitelisted", return_value=True)
async def test_handle_message_sends_generic_error_on_llm_failure(mock_whitelisted, mock_get_conn, mock_answer):
    settings = _settings()
    update, context = _mock_update_and_context(1, "hi", settings, RateLimiter(5))

    await handle_message(update, context)

    update.message.reply_text.assert_awaited_once_with(GENERIC_ERROR_MESSAGE)
```

This requires `pytest-asyncio` (already in `requirements-dev.txt`). Add `asyncio_mode = auto` to `pytest.ini` so `@pytest.mark.asyncio` isn't required on every test — or keep the explicit marker as written above, either works; the plan uses the explicit marker to avoid an extra config change.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/bot/test_telegram_bot.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'bot.telegram_bot'`

- [ ] **Step 3: Write minimal implementation**

```python
# bot/telegram_bot.py
import logging

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from bot.ratelimit import RateLimiter
from config import load_settings
from rag.engine import answer
from rag.llm import LLMError
from storage.db import get_connection
from storage.whitelist import is_whitelisted

logger = logging.getLogger(__name__)

WHITELIST_REJECTION = "Xin lỗi, bạn chưa được cấp quyền sử dụng bot này. Vui lòng liên hệ HR/admin."
RATE_LIMIT_MESSAGE = "Bạn đang gửi câu hỏi quá nhanh, vui lòng chờ một chút rồi thử lại."
GENERIC_ERROR_MESSAGE = "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Xin chào! Hãy hỏi tôi về các chính sách HR của công ty.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    settings = context.bot_data["settings"]
    limiter: RateLimiter = context.bot_data["rate_limiter"]
    user_id = update.effective_user.id
    question = update.message.text

    with get_connection(settings) as conn:
        if not is_whitelisted(conn, user_id):
            await update.message.reply_text(WHITELIST_REJECTION)
            return

    if not limiter.allow(user_id):
        await update.message.reply_text(RATE_LIMIT_MESSAGE)
        return

    try:
        with get_connection(settings) as conn:
            result = answer(conn, settings, user_id, question)
    except LLMError:
        logger.exception("LLM call failed for user %s", user_id)
        await update.message.reply_text(GENERIC_ERROR_MESSAGE)
        return

    reply = result.text
    if result.sources:
        sources_block = "\n".join(f"📄 Nguồn: {s}" for s in result.sources)
        reply = f"{reply}\n\n{sources_block}"
    await update.message.reply_text(reply)


def build_application() -> Application:
    settings = load_settings()
    application = Application.builder().token(settings.telegram_bot_token).build()
    application.bot_data["settings"] = settings
    application.bot_data["rate_limiter"] = RateLimiter(settings.rate_limit_per_minute)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    return application


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    build_application().run_polling()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/bot/test_telegram_bot.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add bot/telegram_bot.py tests/bot/test_telegram_bot.py
git commit -m "feat: add Telegram bot wiring whitelist, rate limit, and RAG engine"
```

---

### Task 18: Golden QA test set

**Files:**
- Create: `tests/golden_qa.yaml`
- Create: `tests/test_golden.py`

This is a regression harness against **real** ingested documents and **real** API keys — it is not run in normal CI/dev loops, only manually once real HR policy files have been ingested (see Task 19).

- [ ] **Step 1: Create `tests/golden_qa.yaml`**

```yaml
# Điền câu hỏi + đáp án kỳ vọng sau khi đã ingest tài liệu chính sách thật (Task 19).
# Mỗi mục:
#   question: câu hỏi thực tế nhân viên có thể hỏi
#   expected_sources: tên file phải xuất hiện trong danh sách nguồn trả về
#   expected_keywords: các từ khóa bắt buộc phải có trong nội dung câu trả lời (không phân biệt hoa/thường)
- question: "Nhân viên chính thức được nghỉ phép năm bao nhiêu ngày?"
  expected_sources: []
  expected_keywords: []
- question: "Quy trình xin nghỉ ốm như thế nào?"
  expected_sources: []
  expected_keywords: []
```

Placeholder file — fill in `expected_sources`/`expected_keywords` once real policy documents exist (see Task 19, Step 4).

- [ ] **Step 2: Create `tests/test_golden.py`**

```python
# tests/test_golden.py
from pathlib import Path

import pytest
import yaml

from config import load_settings
from rag.engine import answer
from storage.db import get_connection

GOLDEN_PATH = Path(__file__).parent / "golden_qa.yaml"


def _load_cases() -> list[dict]:
    if not GOLDEN_PATH.exists():
        return []
    return yaml.safe_load(GOLDEN_PATH.read_text()) or []


@pytest.mark.golden
@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["question"])
def test_golden_qa(case):
    settings = load_settings()
    with get_connection(settings) as conn:
        result = answer(conn, settings, telegram_user_id=0, question=case["question"])

    for source in case.get("expected_sources", []):
        assert source in result.sources, f"missing source {source!r} in {result.sources!r}"
    for keyword in case.get("expected_keywords", []):
        assert keyword.lower() in result.text.lower(), f"missing keyword {keyword!r} in answer"
```

- [ ] **Step 3: Verify it collects cleanly without executing**

`pytest.ini` (Task 1) defines the `golden` marker but does not auto-deselect it — a plain `pytest tests/test_golden.py` would try to run these tests for real and fail on missing API keys/DB. Use `--collect-only` to check wiring without executing anything:

Run: `pytest tests/test_golden.py --collect-only`
Expected: 2 tests collected, 0 executed.

From here on, always run the full suite as `pytest -m "not golden"` (see Task 19, Step 2) unless you specifically intend to hit real APIs.

- [ ] **Step 4: Commit**

```bash
git add tests/golden_qa.yaml tests/test_golden.py
git commit -m "test: add golden QA regression harness (run manually against real documents)"
```

---

### Task 19: README & manual end-to-end verification

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# HR RAG Telegram Bot

Trả lời câu hỏi nhân viên về chính sách HR qua Telegram, dựa trên tài liệu PDF/Word nội bộ. Xem thiết kế đầy đủ tại `docs/superpowers/specs/2026-08-17-hr-rag-telegram-bot-design.md`.

## Setup

1. `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements-dev.txt`
3. `cp .env.example .env` và điền:
   - `TELEGRAM_BOT_TOKEN` — lấy từ [@BotFather](https://t.me/BotFather)
   - `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`
   - Xác nhận `LLM_MODEL` đúng model id DeepSeek hiện có trên LiteLLM
4. `docker compose up -d` — khởi động Postgres + pgvector cục bộ
5. `python -m storage.migrate` — tạo schema
6. `python -m storage.whitelist add <telegram_user_id> "<Tên nhân viên>"` — whitelist người dùng test

## Ingest tài liệu

Đặt file PDF/Word vào một thư mục (ví dụ `docs_source/`), rồi:

```bash
python -m ingest.cli --path docs_source/
```

Chạy lại an toàn — file không đổi nội dung sẽ được bỏ qua (idempotent theo `content_hash`).

## Chạy bot

```bash
python -m bot.telegram_bot
```

## Test

```bash
pytest                        # unit tests, không cần Docker/API key
docker compose up -d && pytest -m integration   # cần Postgres chạy local
pytest -m golden               # cần API key thật + tài liệu đã ingest (xem tests/golden_qa.yaml)
```

## Dọn lịch sử hội thoại cũ

```bash
python -m storage.cleanup   # xóa conversations cũ hơn CONVERSATION_RETENTION_DAYS
```

Chạy định kỳ (cron/systemd timer) sau khi deploy — cơ chế lập lịch chưa nằm trong phạm vi dự án này.
```

- [ ] **Step 2: Run the full unit + integration suite**

```bash
docker compose up -d
pytest -m "not golden" -v
```

Expected: all tests pass (unit tests from Tasks 2, 5, 6, 7, 9, 12, 13, 14, 15, 16, 17 + integration tests from Tasks 10-11).

- [ ] **Step 3: Manual end-to-end verification checklist**

Requires: real `TELEGRAM_BOT_TOKEN`, real `DEEPSEEK_API_KEY`/`OPENAI_API_KEY`, at least 2-3 real HR policy files.

```bash
python -m storage.migrate
python -m ingest.cli --path docs_source/
python -m storage.whitelist add <your_telegram_user_id> "Test User"
python -m bot.telegram_bot
```

Then in Telegram, DM the bot and verify:
- [ ] A question with a clear answer in the ingested docs returns a correct answer **with the right source filename cited**.
- [ ] A follow-up question referring back to the previous answer (multi-turn) is answered coherently using conversation context.
- [ ] A question unrelated to any ingested policy triggers the fallback "Không tìm thấy thông tin này..." message, not a hallucinated answer.
- [ ] Messaging from a non-whitelisted Telegram account gets the rejection message and does **not** trigger an LLM call (check logs/API usage).
- [ ] Sending 6+ messages within a minute triggers the rate-limit message.

- [ ] **Step 4: Fill in `tests/golden_qa.yaml` with real questions/answers from the documents just ingested, then run**

```bash
pytest -m golden -v
```

Expected: PASS. This becomes the regression suite for future prompt/chunking/model changes.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/golden_qa.yaml
git commit -m "docs: add setup/usage README and fill in golden QA set"
```

---

## Deferred / Explicitly Out of Scope

Per the spec's "Ngoài phạm vi" section — do not implement in this plan:
- Slack integration (reuse `rag/` + `storage/` when it happens)
- Automated document sync from Drive/Notion
- Web admin UI
- Department/role-based permissions
- Hosting/deployment automation (manual `python -m bot.telegram_bot` for now)
