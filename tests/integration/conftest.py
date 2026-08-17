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
