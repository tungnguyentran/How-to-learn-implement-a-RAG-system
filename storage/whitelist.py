# storage/whitelist.py
import argparse

from psycopg import Connection

from config import load_settings
from storage.db import get_connection


def is_whitelisted(conn: Connection, telegram_user_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM whitelist WHERE telegram_user_id = %s", (telegram_user_id,)
    ).fetchone()
    return row is not None


def add_user(conn: Connection, telegram_user_id: int, display_name: str) -> None:
    conn.execute(
        """
        INSERT INTO whitelist (telegram_user_id, display_name)
        VALUES (%s, %s)
        ON CONFLICT (telegram_user_id) DO UPDATE SET display_name = EXCLUDED.display_name
        """,
        (telegram_user_id, display_name),
    )
    conn.commit()


def remove_user(conn: Connection, telegram_user_id: int) -> None:
    conn.execute("DELETE FROM whitelist WHERE telegram_user_id = %s", (telegram_user_id,))
    conn.commit()


def list_users(conn: Connection) -> list[dict]:
    return conn.execute(
        "SELECT telegram_user_id, display_name, added_at FROM whitelist ORDER BY added_at"
    ).fetchall()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the Telegram bot whitelist")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Whitelist a Telegram user")
    add_parser.add_argument("telegram_user_id", type=int)
    add_parser.add_argument("display_name")

    remove_parser = subparsers.add_parser("remove", help="Remove a user from the whitelist")
    remove_parser.add_argument("telegram_user_id", type=int)

    subparsers.add_parser("list", help="List whitelisted users")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    settings = load_settings()
    with get_connection(settings) as conn:
        if args.command == "add":
            add_user(conn, args.telegram_user_id, args.display_name)
            print(f"Added {args.telegram_user_id} ({args.display_name})")
        elif args.command == "remove":
            remove_user(conn, args.telegram_user_id)
            print(f"Removed {args.telegram_user_id}")
        elif args.command == "list":
            for row in list_users(conn):
                print(f"{row['telegram_user_id']}\t{row['display_name']}\t{row['added_at']}")


if __name__ == "__main__":
    main()
