# tests/ingest/test_cli.py
from unittest.mock import MagicMock, patch

from docx import Document as DocxDocument

from config import Settings
from ingest.cli import ingest_file, ingest_path


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


@patch("ingest.cli.insert_chunk")
@patch("ingest.cli.embed", return_value=[0.1] * 1536)
@patch("ingest.cli.upsert_document", return_value=(1, True))
def test_ingest_file_chunks_and_embeds_new_document(mock_upsert, mock_embed, mock_insert, tmp_path):
    path = tmp_path / "policy.docx"
    doc = DocxDocument()
    doc.add_paragraph("Noi dung chinh sach nghi phep cua cong ty.")
    doc.save(path)
    conn = MagicMock()

    ingest_file(conn, _settings(), path)

    mock_upsert.assert_called_once()
    assert mock_embed.call_count >= 1
    assert mock_insert.call_count == mock_embed.call_count


@patch("ingest.cli.insert_chunk")
@patch("ingest.cli.embed")
@patch("ingest.cli.upsert_document", return_value=(1, False))
def test_ingest_file_skips_unchanged_document(mock_upsert, mock_embed, mock_insert, tmp_path):
    path = tmp_path / "policy.docx"
    path.write_bytes(b"irrelevant, upsert_document is mocked so content is never parsed")
    conn = MagicMock()

    ingest_file(conn, _settings(), path)

    mock_embed.assert_not_called()
    mock_insert.assert_not_called()


@patch("ingest.cli.ingest_file", side_effect=[RuntimeError("corrupt file"), None])
@patch("ingest.cli.get_connection")
@patch("ingest.cli.load_settings")
def test_ingest_path_skips_file_that_fails_and_continues(mock_load_settings, mock_get_conn, mock_ingest_file, tmp_path):
    (tmp_path / "bad.pdf").write_bytes(b"not a real pdf")
    (tmp_path / "good.docx").write_bytes(b"not a real docx either, ingest_file is mocked")
    mock_get_conn.return_value.__enter__.return_value = MagicMock()

    ingest_path(tmp_path)  # should not raise, despite the first call raising

    assert mock_ingest_file.call_count == 2
