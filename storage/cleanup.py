# storage/cleanup.py
from config import load_settings
from storage.conversations import cleanup_old_turns
from storage.db import get_connection


def main() -> None:
    settings = load_settings()
    with get_connection(settings) as conn:
        deleted = cleanup_old_turns(conn, settings.conversation_retention_days)
    print(f"Deleted {deleted} conversation rows older than {settings.conversation_retention_days} days.")


if __name__ == "__main__":
    main()
