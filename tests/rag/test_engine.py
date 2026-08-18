# tests/rag/test_engine.py
from unittest.mock import MagicMock, patch

import pytest

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
