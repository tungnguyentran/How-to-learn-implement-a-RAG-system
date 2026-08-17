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
