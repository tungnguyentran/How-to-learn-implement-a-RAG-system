# tests/test_config.py
import pytest

from config import load_settings


REQUIRED_ENV = {
    "DATABASE_URL": "postgresql://hrbot:hrbot@localhost:5433/hrbot",
    "TELEGRAM_BOT_TOKEN": "test-token",
}


def test_load_settings_uses_defaults(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("CHUNK_SIZE", raising=False)

    settings = load_settings()

    assert settings.database_url == REQUIRED_ENV["DATABASE_URL"]
    assert settings.telegram_bot_token == "test-token"
    assert settings.llm_model == "deepseek/deepseek-chat"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 100
    assert settings.top_k == 5
    assert settings.similarity_threshold == 0.3
    assert settings.conversation_context_size == 10
    assert settings.conversation_retention_days == 30
    assert settings.rate_limit_per_minute == 5


def test_load_settings_reads_overrides(monkeypatch):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.5")

    settings = load_settings()

    assert settings.chunk_size == 500
    assert settings.similarity_threshold == 0.5


def test_load_settings_raises_on_missing_required(monkeypatch):
    # Prevent a real .env file (e.g. created by following the README) from
    # repopulating these vars via load_dotenv() and masking the failure.
    monkeypatch.setattr("config.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        load_settings()
