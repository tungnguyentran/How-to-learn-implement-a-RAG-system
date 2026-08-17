# ingest/cli.py
import argparse
import logging
from pathlib import Path

from psycopg import Connection

from config import Settings, load_settings
from ingest.chunker import chunk_text
from ingest.loader import load_text
from rag.llm import embed
from storage.chunks import insert_chunk
from storage.db import get_connection
from storage.documents import compute_content_hash, upsert_document

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".pdf", ".docx"}


def ingest_file(conn: Connection, settings: Settings, path: Path) -> None:
    content = path.read_bytes()
    content_hash = compute_content_hash(content)
    document_id, changed = upsert_document(conn, path.name, content_hash)
    if not changed:
        logger.info("Skip unchanged file: %s", path.name)
        return

    text = load_text(path)
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    for index, chunk in enumerate(chunks):
        embedding = embed(chunk, settings.embedding_model)
        insert_chunk(conn, document_id, chunk, embedding, index)
    logger.info("Ingested %s: %d chunks", path.name, len(chunks))


def ingest_path(root: Path) -> None:
    settings = load_settings()
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    with get_connection(settings) as conn:
        for path in files:
            ingest_file(conn, settings, path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest HR policy PDF/DOCX documents into Postgres. "
            "Filenames must be unique across the ingest root (documents.filename has a "
            "UNIQUE constraint) — two files with the same name in different subfolders "
            "will collide."
        )
    )
    parser.add_argument(
        "--path", required=True, type=Path, help="Directory to scan recursively for .pdf/.docx files"
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args()
    ingest_path(args.path)


if __name__ == "__main__":
    main()
