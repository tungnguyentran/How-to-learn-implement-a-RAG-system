# config.py
import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    database_url: str
    telegram_bot_token: str
    llm_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    similarity_threshold: float
    conversation_context_size: int
    conversation_retention_days: int
    rate_limit_per_minute: int


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_settings() -> Settings:
    load_dotenv()  # no-op if no .env file exists; never overrides already-set env vars
    return Settings(
        database_url=_require("DATABASE_URL"),
        telegram_bot_token=_require("TELEGRAM_BOT_TOKEN"),
        llm_model=os.getenv("LLM_MODEL", "deepseek/deepseek-chat"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        top_k=int(os.getenv("TOP_K", "5")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.3")),
        conversation_context_size=int(os.getenv("CONVERSATION_CONTEXT_SIZE", "10")),
        conversation_retention_days=int(os.getenv("CONVERSATION_RETENTION_DAYS", "30")),
        rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "5")),
    )
