# rag/engine.py
from dataclasses import dataclass

from psycopg import Connection

from config import Settings
from rag.llm import complete, embed
from rag.prompt import build_messages
from storage.chunks import search_similar
from storage.conversations import get_recent_turns, save_turn

FALLBACK_MESSAGE = (
    "Không tìm thấy thông tin này trong chính sách hiện có, vui lòng liên hệ HR trực tiếp."
)


@dataclass
class AnswerResult:
    text: str
    sources: list[str]


def answer(conn: Connection, settings: Settings, telegram_user_id: int, question: str) -> AnswerResult:
    query_embedding = embed(question, settings.embedding_model)
    chunks = search_similar(conn, query_embedding, settings.top_k)

    best_similarity = chunks[0]["similarity"] if chunks else 0.0
    if not chunks or best_similarity < settings.similarity_threshold:
        save_turn(conn, telegram_user_id, "user", question)
        save_turn(conn, telegram_user_id, "assistant", FALLBACK_MESSAGE)
        return AnswerResult(text=FALLBACK_MESSAGE, sources=[])

    history = get_recent_turns(conn, telegram_user_id, settings.conversation_context_size)
    messages = build_messages(question, history, chunks)
    reply_text = complete(messages, settings.llm_model)  # raises LLMError -> propagates, no save

    sources = sorted({c["filename"] for c in chunks})
    save_turn(conn, telegram_user_id, "user", question)
    save_turn(conn, telegram_user_id, "assistant", reply_text)
    return AnswerResult(text=reply_text, sources=sources)
