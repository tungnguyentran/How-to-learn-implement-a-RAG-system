# tests/rag/test_prompt.py
from rag.prompt import SYSTEM_PROMPT, build_messages


def test_first_message_is_the_guardrail_system_prompt():
    messages = build_messages("cau hoi", history=[], chunks=[])

    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "chỉ trả lời" in SYSTEM_PROMPT.lower()


def test_history_is_included_in_order_between_system_and_final_user_turn():
    history = [
        {"role": "user", "content": "cau hoi cu"},
        {"role": "assistant", "content": "tra loi cu"},
    ]
    messages = build_messages("cau hoi moi", history=history, chunks=[])

    assert messages[1] == {"role": "user", "content": "cau hoi cu"}
    assert messages[2] == {"role": "assistant", "content": "tra loi cu"}


def test_final_user_message_includes_context_and_question():
    chunks = [
        {"filename": "policy.pdf", "chunk_text": "12 ngay nghi phep moi nam"},
    ]
    messages = build_messages("bao nhieu ngay nghi phep?", history=[], chunks=chunks)

    final = messages[-1]
    assert final["role"] == "user"
    assert "policy.pdf" in final["content"]
    assert "12 ngay nghi phep moi nam" in final["content"]
    assert "bao nhieu ngay nghi phep?" in final["content"]
