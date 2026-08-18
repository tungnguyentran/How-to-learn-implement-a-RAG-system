# storage/migrate.py
from pathlib import Path

from config import load_settings
from storage.db import get_connection

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def main() -> None:
    settings = load_settings()
    with get_connection(settings) as conn:
        conn.execute(SCHEMA_PATH.read_text())
        conn.commit()
    print("Schema applied.")


if __name__ == "__main__":
    main()
