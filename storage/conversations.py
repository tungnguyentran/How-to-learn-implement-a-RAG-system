# storage/conversations.py
from psycopg import Connection


def save_turn(conn: Connection, telegram_user_id: int, role: str, content: str) -> None:
    conn.execute(
        "INSERT INTO conversations (telegram_user_id, role, content) VALUES (%s, %s, %s)",
        (telegram_user_id, role, content),
    )
    conn.commit()


def get_recent_turns(conn: Connection, telegram_user_id: int, limit: int) -> list[dict]:
    """Returns up to `limit` most recent turns, oldest first (ready to feed into a prompt)."""
    rows = conn.execute(
        """
        SELECT role, content FROM conversations
        WHERE telegram_user_id = %s
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (telegram_user_id, limit),
    ).fetchall()
    return list(reversed(rows))


def cleanup_old_turns(conn: Connection, retention_days: int) -> int:
    """Hard-deletes conversation rows older than retention_days, regardless of
    the per-user context window used by get_recent_turns. Returns rows deleted."""
    cursor = conn.execute(
        "DELETE FROM conversations WHERE created_at < now() - %s::interval",
        (f"{retention_days} days",),
    )
    conn.commit()
    return cursor.rowcount
