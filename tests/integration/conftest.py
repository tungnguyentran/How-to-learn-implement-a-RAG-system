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
    try:
        register_vector(connection)
        registered = True
    except psycopg.ProgrammingError:
        # pgvector extension not yet installed (e.g. a genuinely fresh database,
        # before schema.sql's `CREATE EXTENSION IF NOT EXISTS vector` has run).
        connection.rollback()
        registered = False

    connection.execute(SCHEMA_PATH.read_text())
    connection.commit()

    if not registered:
        # Extension now exists (schema.sql created it) — register for real.
        # Unlike storage.db.get_connection, this fixture cannot defer
        # registration to a later call: tests insert/query vector data
        # against this same connection.
        register_vector(connection)

    yield connection
    connection.execute(
        "TRUNCATE documents, document_chunks, conversations, whitelist RESTART IDENTITY CASCADE"
    )
    connection.commit()
    connection.close()
