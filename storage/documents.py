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
