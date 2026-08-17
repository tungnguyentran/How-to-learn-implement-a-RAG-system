# rag/prompt.py
SYSTEM_PROMPT = (
    "Bạn là trợ lý trả lời câu hỏi về chính sách nhân sự (HR) nội bộ công ty. "
    "Chỉ trả lời dựa trên các đoạn trích trong phần NGỮ CẢNH được cung cấp bên dưới. "
    "Nếu ngữ cảnh không đủ để trả lời, hãy nói rõ là bạn không biết — "
    "không suy diễn, không bịa thông tin, không trả lời ngoài chủ đề chính sách HR."
)


def build_messages(question: str, history: list[dict], chunks: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    context = "\n\n".join(f"[Nguồn: {c['filename']}]\n{c['chunk_text']}" for c in chunks)
    messages.append(
        {
            "role": "user",
            "content": f"NGỮ CẢNH:\n{context}\n\nCÂU HỎI: {question}",
        }
    )
    return messages
