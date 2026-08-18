# tests/integration/test_conversations_whitelist.py
import pytest

from storage.conversations import cleanup_old_turns, get_recent_turns, save_turn
from storage.whitelist import add_user, is_whitelisted, list_users, remove_user

pytestmark = pytest.mark.integration


def test_conversation_round_trip_preserves_order(conn):
    save_turn(conn, 111, "user", "cau hoi 1")
    save_turn(conn, 111, "assistant", "tra loi 1")
    save_turn(conn, 111, "user", "cau hoi 2")

    turns = get_recent_turns(conn, 111, limit=10)

    assert [t["content"] for t in turns] == ["cau hoi 1", "tra loi 1", "cau hoi 2"]


def test_get_recent_turns_respects_limit(conn):
    for i in range(5):
        save_turn(conn, 222, "user", f"msg{i}")

    turns = get_recent_turns(conn, 222, limit=2)

    assert [t["content"] for t in turns] == ["msg3", "msg4"]


def test_cleanup_old_turns_deletes_past_retention_window(conn):
    save_turn(conn, 333, "user", "old enough to be backdated")
    conn.execute(
        "UPDATE conversations SET created_at = now() - interval '31 days' WHERE telegram_user_id = 333"
    )
    save_turn(conn, 333, "user", "recent")
    conn.commit()

    deleted = cleanup_old_turns(conn, retention_days=30)
    remaining = get_recent_turns(conn, 333, limit=10)

    assert deleted == 1
    assert [t["content"] for t in remaining] == ["recent"]


def test_whitelist_add_remove_list_round_trip(conn):
    add_user(conn, 444, "Nguyen Van A")

    assert is_whitelisted(conn, 444) is True
    users = list_users(conn)
    assert any(u["telegram_user_id"] == 444 for u in users)

    remove_user(conn, 444)

    assert is_whitelisted(conn, 444) is False
