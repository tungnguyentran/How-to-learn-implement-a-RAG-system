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
