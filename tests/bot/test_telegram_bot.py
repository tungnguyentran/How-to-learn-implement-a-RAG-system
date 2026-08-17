# tests/bot/test_telegram_bot.py
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Update

from bot.ratelimit import RateLimiter
from bot.telegram_bot import (
    GENERIC_ERROR_MESSAGE,
    RATE_LIMIT_MESSAGE,
    WHITELIST_REJECTION,
    handle_error,
    handle_message,
)
from config import Settings
from rag.engine import AnswerResult
from rag.llm import LLMError


def _settings() -> Settings:
    return Settings(
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


def _mock_update_and_context(user_id: int, text: str, settings: Settings, limiter: RateLimiter):
    update = MagicMock()
    update.effective_user.id = user_id
    update.message.text = text
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.bot_data = {"settings": settings, "rate_limiter": limiter}
    return update, context


@pytest.mark.asyncio
@patch("bot.telegram_bot.get_connection")
@patch("bot.telegram_bot.is_whitelisted", return_value=False)
async def test_handle_message_rejects_non_whitelisted_user(mock_whitelisted, mock_get_conn):
    settings = _settings()
    update, context = _mock_update_and_context(1, "hi", settings, RateLimiter(5))

    await handle_message(update, context)

    update.message.reply_text.assert_awaited_once_with(WHITELIST_REJECTION)


@pytest.mark.asyncio
@patch("bot.telegram_bot.get_connection")
@patch("bot.telegram_bot.is_whitelisted", return_value=True)
async def test_handle_message_rejects_when_rate_limited(mock_whitelisted, mock_get_conn):
    settings = _settings()
    limiter = RateLimiter(max_per_minute=0)
    update, context = _mock_update_and_context(1, "hi", settings, limiter)

    await handle_message(update, context)

    update.message.reply_text.assert_awaited_once_with(RATE_LIMIT_MESSAGE)


@pytest.mark.asyncio
@patch("bot.telegram_bot.answer")
@patch("bot.telegram_bot.get_connection")
@patch("bot.telegram_bot.is_whitelisted", return_value=True)
async def test_handle_message_appends_sources_on_success(mock_whitelisted, mock_get_conn, mock_answer):
    mock_answer.return_value = AnswerResult(text="Ban duoc nghi 12 ngay", sources=["policy.pdf"])
    settings = _settings()
    update, context = _mock_update_and_context(1, "bao nhieu ngay phep?", settings, RateLimiter(5))

    await handle_message(update, context)

    sent = update.message.reply_text.await_args.args[0]
    assert "Ban duoc nghi 12 ngay" in sent
    assert "policy.pdf" in sent


@pytest.mark.asyncio
@patch("bot.telegram_bot.answer", side_effect=LLMError("down"))
@patch("bot.telegram_bot.get_connection")
@patch("bot.telegram_bot.is_whitelisted", return_value=True)
async def test_handle_message_sends_generic_error_on_llm_failure(mock_whitelisted, mock_get_conn, mock_answer):
    settings = _settings()
    update, context = _mock_update_and_context(1, "hi", settings, RateLimiter(5))

    await handle_message(update, context)

    update.message.reply_text.assert_awaited_once_with(GENERIC_ERROR_MESSAGE)


@pytest.mark.asyncio
async def test_handle_error_replies_with_generic_error_message():
    update = MagicMock(spec=Update)
    update.message.reply_text = AsyncMock()

    context = MagicMock()
    context.error = RuntimeError("db connection refused")

    await handle_error(update, context)

    update.message.reply_text.assert_awaited_once_with(GENERIC_ERROR_MESSAGE)
