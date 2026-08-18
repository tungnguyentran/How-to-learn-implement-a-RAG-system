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
