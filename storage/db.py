# storage/db.py
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from config import Settings


def get_connection(settings: Settings) -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url, row_factory=dict_row)
    register_vector(conn)
    return conn
