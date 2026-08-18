# storage/db.py
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from config import Settings


def get_connection(settings: Settings) -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url, row_factory=dict_row)
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        # pgvector extension not yet installed (e.g. the very first storage.migrate
        # run against a brand-new database, before schema.sql's
        # `CREATE EXTENSION IF NOT EXISTS vector` has executed). DDL-only callers
        # like migrate don't need vector-type adaptation; callers that do will get
        # it on their next get_connection() call, once the extension exists.
        conn.rollback()
    return conn
