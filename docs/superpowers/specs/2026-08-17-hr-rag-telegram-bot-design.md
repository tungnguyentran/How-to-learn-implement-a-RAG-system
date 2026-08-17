# Thiết kế: RAG Chatbot HR Policy qua Telegram

**Ngày:** 2026-08-17
**Trạng thái:** Approved (chờ review spec)

## Mục tiêu

Nhân viên công ty có thể hỏi trực tiếp trên Telegram về các chính sách HR (nghỉ phép, quy chế, phúc lợi...) và nhận câu trả lời chính xác, có trích dẫn nguồn tài liệu, dựa trên các file chính sách nội bộ (PDF/Word). Slack sẽ được thêm sau, dùng lại engine RAG.

## Ngoài phạm vi (giai đoạn này)

- Slack integration (giai đoạn 2, tái sử dụng module `rag/` và `storage/`).
- Tự động quét/đồng bộ tài liệu từ Drive/Notion — ingest là CLI chạy thủ công.
- Giao diện web quản trị — HR dùng CLI.
- Phân quyền theo phòng ban/cấp bậc — chỉ có whitelist user Telegram cho toàn bộ nội dung.
- Auto-deploy/hosting — quyết định sau khi có code chạy được local.

## Kiến trúc tổng quan

```
docs/ (PDF/Word)  →  [CLI ingest]  →  Postgres + pgvector
                                              ↑ (retrieve)
Telegram user → [Bot handler] → whitelist check → RAG engine → LiteLLM (DeepSeek) → trả lời + trích nguồn
```

5 module Python, mỗi module một trách nhiệm rõ ràng, không phụ thuộc chéo ngoài hướng phụ thuộc bên dưới:

- **`ingest/`** — CLI đọc file PDF/Word trong một thư mục, tách text, chunk, gọi LiteLLM embedding (`text-embedding-3-small` qua OpenAI), upsert vào Postgres. Idempotent theo `content_hash`.
- **`storage/`** — schema Postgres (pgvector) + các hàm truy vấn: insert/update chunk, similarity search, đọc/ghi lịch sử hội thoại, kiểm tra whitelist.
- **`rag/`** — logic lõi, không biết gì về Telegram: nhận câu hỏi + lịch sử hội thoại → embed câu hỏi → tìm top-k chunk liên quan → dựng prompt (kèm nguồn) → gọi LiteLLM completion (DeepSeek) → trả về câu trả lời + danh sách nguồn.
- **`bot/`** — `python-telegram-bot`: nhận tin nhắn, kiểm tra whitelist, gọi `rag/`, format và gửi trả lời. Slack sau này chỉ cần thêm `bot/slack_bot.py` mới dùng lại `rag/` + `storage/`.
- **`config.py`** — đọc API keys, tên model (DeepSeek qua LiteLLM, embedding OpenAI), danh sách whitelist, chunk size, top-k từ biến môi trường.

## Lựa chọn công nghệ

| Thành phần | Lựa chọn | Lý do |
|---|---|---|
| LLM completion | DeepSeek (qua LiteLLM) | Chi phí thấp, LiteLLM giúp đổi provider sau này không cần sửa code |
| Embedding | OpenAI `text-embedding-3-small` (qua LiteLLM) | DeepSeek không có embedding API; text-embedding-3-small rẻ, chất lượng tốt cho tiếng Việt |
| Vector store | Postgres + pgvector | Dùng chung 1 DB cho vector, lịch sử hội thoại, whitelist — không thêm service |
| RAG pipeline | Tự viết (không dùng LangChain/LlamaIndex) | Quy mô nhỏ (vài trăm file policy), ưu tiên đơn giản, dễ debug, ít dependency |
| Bot framework | `python-telegram-bot` | Chuẩn phổ biến, hỗ trợ tốt async |
| Ngôn ngữ | Python | Hệ sinh thái RAG/PDF-parsing phong phú, LiteLLM native Python |

## Data model (Postgres + pgvector)

```sql
documents(
  id, filename, content_hash, ingested_at
)

document_chunks(
  id, document_id REFERENCES documents(id),
  chunk_text, embedding vector(1536), chunk_index
)

conversations(
  id, telegram_user_id, role, content, created_at
)  -- giữ N tin nhắn gần nhất mỗi user cho ngữ cảnh multi-turn

allowlist(
  telegram_user_id, display_name, added_at
)
```

- `content_hash` trên `documents` đảm bảo **ingest idempotent**: chạy lại script không tạo trùng chunk nếu nội dung file không đổi; nếu file đổi, xóa chunk cũ của document và insert lại.
- `conversations` chỉ giữ N tin nhắn gần nhất mỗi user (mặc định 10) — dọn định kỳ, không cần cơ chế TTL phức tạp.

## Luồng retrieve + generate

1. Kiểm tra `telegram_user_id` có trong `allowlist` — nếu không, trả lời từ chối lịch sự và dừng (không gọi LLM).
2. Lấy N tin nhắn gần nhất của user từ `conversations` làm ngữ cảnh.
3. Embed câu hỏi (OpenAI qua LiteLLM) → similarity search top-k (mặc định k=5) trên `document_chunks`.
4. Dựng prompt gồm: câu hỏi hiện tại, lịch sử hội thoại, các chunk liên quan kèm tên file nguồn.
5. Gọi LiteLLM completion (DeepSeek) → nhận câu trả lời.
6. Gửi lại Telegram: câu trả lời kèm dòng trích nguồn (ví dụ `📄 Nguồn: Quy chế nghỉ phép 2024.pdf`).
7. Lưu lượt hỏi-đáp vào `conversations`.

## Xử lý lỗi / edge case

- **Không tìm thấy chunk liên quan** (similarity dưới ngưỡng, mặc định cosine < 0.3): trả lời "Không tìm thấy thông tin này trong chính sách hiện có, vui lòng liên hệ HR trực tiếp" — không để LLM tự bịa câu trả lời.
- **User không trong whitelist**: từ chối ngay, không gọi LLM (tiết kiệm chi phí, tránh lộ thông tin nội bộ).
- **Lỗi gọi LiteLLM** (timeout/lỗi API): retry 1 lần, nếu vẫn lỗi thì trả lời thân thiện cho user và log lỗi chi tiết.
- **Ingest lại file đã đổi nội dung**: phát hiện qua `content_hash` khác giá trị đã lưu, xóa toàn bộ chunk cũ của document đó rồi insert lại từ đầu.

## Testing

- **Unit test** (không gọi API thật): hàm chunk text, hàm dựng prompt, logic kiểm tra whitelist, logic ngưỡng similarity.
- **Integration test**: Postgres+pgvector chạy qua Docker, mock các lời gọi LiteLLM, kiểm tra luồng ingest → retrieve trên 1-2 file mẫu.
- **Test thủ công**: chạy bot local với vài file policy thật — hỏi 1 câu có trong tài liệu (kiểm tra câu trả lời + nguồn đúng), 1 câu không có trong tài liệu (kiểm tra fallback "không tìm thấy"), thử nhắn tin từ user không nằm trong whitelist.

## Câu hỏi mở / quyết định sau

- Hosting/deployment: chưa quyết định, sẽ chọn sau khi có code chạy local ổn định.
- Tên model DeepSeek cụ thể qua LiteLLM sẽ được xác nhận và cấu hình qua biến môi trường khi triển khai.
