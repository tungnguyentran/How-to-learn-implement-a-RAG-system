# tests/storage/test_whitelist.py
from unittest.mock import MagicMock

from storage.whitelist import is_whitelisted


def test_is_whitelisted_true_when_row_found():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = {"telegram_user_id": 42}

    assert is_whitelisted(conn, 42) is True
    conn.execute.assert_called_once()


def test_is_whitelisted_false_when_no_row():
    conn = MagicMock()
    conn.execute.return_value.fetchone.return_value = None

    assert is_whitelisted(conn, 999) is False
