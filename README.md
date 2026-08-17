# HR RAG Telegram Bot

Trả lời câu hỏi nhân viên về chính sách HR qua Telegram, dựa trên tài liệu PDF/Word nội bộ. Xem thiết kế đầy đủ tại `docs/superpowers/specs/2026-08-17-hr-rag-telegram-bot-design.md`.

## Setup

1. `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements-dev.txt`
3. `cp .env.example .env` và điền:
   - `TELEGRAM_BOT_TOKEN` — lấy từ [@BotFather](https://t.me/BotFather)
   - `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`
   - Xác nhận `LLM_MODEL` đúng model id DeepSeek hiện có trên LiteLLM
4. `podman compose up -d` (hoặc `docker compose up -d` nếu máy có Docker) — khởi động Postgres + pgvector cục bộ
5. `python -m storage.migrate` — tạo schema
6. `python -m storage.whitelist add <telegram_user_id> "<Tên nhân viên>"` — whitelist người dùng test

## Ingest tài liệu

Đặt file PDF/Word vào một thư mục (ví dụ `docs_source/`), rồi:

```bash
python -m ingest.cli --path docs_source/
```

Chạy lại an toàn — file không đổi nội dung sẽ được bỏ qua (idempotent theo `content_hash`). Nếu một file lỗi (PDF/Word hỏng), file đó sẽ bị bỏ qua và log lỗi, các file khác trong batch vẫn được ingest bình thường — chạy lại sau khi sửa file lỗi sẽ tự động ingest lại nó (không bị kẹt ở trạng thái dở dang).

## Chạy bot

```bash
python -m bot.telegram_bot
```

## Test

```bash
pytest -m "not golden"                          # unit + integration tests, cần Postgres chạy local (xem bước 4)
pytest tests/test_golden.py --collect-only       # xem golden set mà không chạy (cần API key thật để chạy thật)
pytest -m golden                                 # cần API key thật + tài liệu đã ingest (xem tests/golden_qa.yaml)
```

Lưu ý về `pytest -m golden`: mỗi lần chạy sẽ ghi lịch sử hội thoại thật vào bảng `conversations` (dùng `telegram_user_id=0` làm sentinel) và không tự dọn dẹp — chạy lặp lại nhiều lần có thể làm ngữ cảnh hội thoại tích lũy giữa các lần chạy ảnh hưởng đến câu trả lời. Nếu cần, dọn thủ công: `DELETE FROM conversations WHERE telegram_user_id = 0;`

## Dọn lịch sử hội thoại cũ

```bash
python -m storage.cleanup   # xóa conversations cũ hơn CONVERSATION_RETENTION_DAYS
```

Chạy định kỳ (cron/systemd timer) sau khi deploy — cơ chế lập lịch chưa nằm trong phạm vi dự án này.

## Quản lý whitelist

```bash
python -m storage.whitelist add <telegram_user_id> "<Tên>"
python -m storage.whitelist remove <telegram_user_id>
python -m storage.whitelist list
```
