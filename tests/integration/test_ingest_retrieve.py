# tests/integration/test_ingest_retrieve.py
import pytest

from storage.chunks import insert_chunk, search_similar
from storage.documents import compute_content_hash, upsert_document

pytestmark = pytest.mark.integration


def _fake_embedding(seed: float) -> list[float]:
    # 1536-dim vector, mostly zeros with two distinctive components —
    # deterministic and cheap, good enough to test ordering/idempotency.
    # A single nonzero component would make every embedding collinear
    # (cosine distance 0 between any two), so a second fixed component
    # is needed to give seeds a real angular (cosine) separation.
    vec = [0.0] * 1536
    vec[0] = 1.0
    vec[1] = seed
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
