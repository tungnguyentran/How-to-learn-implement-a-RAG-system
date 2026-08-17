# rag/llm.py
import litellm


class LLMError(Exception):
    """Raised when an embedding or completion call fails after retries."""


def embed(text: str, model: str, max_retries: int = 1) -> list[float]:
    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            response = litellm.embedding(model=model, input=[text])
            return response["data"][0]["embedding"]
        except Exception as exc:
            last_error = exc
    raise LLMError(f"Embedding call failed after {max_retries + 1} attempts: {last_error}") from last_error


def complete(messages: list[dict], model: str, max_retries: int = 1) -> str:
    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            response = litellm.completion(model=model, messages=messages)
            return response["choices"][0]["message"]["content"]
        except Exception as exc:
            last_error = exc
    raise LLMError(f"Completion call failed after {max_retries + 1} attempts: {last_error}") from last_error
