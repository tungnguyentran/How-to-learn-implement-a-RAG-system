# tests/test_golden.py
from pathlib import Path

import pytest
import yaml

from config import load_settings
from rag.engine import answer
from storage.db import get_connection

GOLDEN_PATH = Path(__file__).parent / "golden_qa.yaml"


def _load_cases() -> list[dict]:
    if not GOLDEN_PATH.exists():
        return []
    return yaml.safe_load(GOLDEN_PATH.read_text()) or []


@pytest.mark.golden
@pytest.mark.parametrize("case", _load_cases(), ids=lambda c: c["question"])
def test_golden_qa(case):
    settings = load_settings()
    with get_connection(settings) as conn:
        result = answer(conn, settings, telegram_user_id=0, question=case["question"])

    for source in case.get("expected_sources", []):
        assert source in result.sources, f"missing source {source!r} in {result.sources!r}"
    for keyword in case.get("expected_keywords", []):
        assert keyword.lower() in result.text.lower(), f"missing keyword {keyword!r} in answer"
