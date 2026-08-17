# bot/ratelimit.py
import time
from collections import defaultdict, deque


class RateLimiter:
    """In-memory sliding-window rate limiter, per Telegram user.

    In-memory means limits reset on process restart and don't share state
    across multiple bot instances — acceptable for a single-process MVP
    deployment (see spec's open deployment question)."""

    def __init__(self, max_per_minute: int):
        self._max_per_minute = max_per_minute
        self._timestamps: dict[int, deque] = defaultdict(deque)

    def allow(self, user_id: int, now: float | None = None) -> bool:
        current = now if now is not None else time.monotonic()
        window_start = current - 60
        timestamps = self._timestamps[user_id]
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()
        if len(timestamps) >= self._max_per_minute:
            return False
        timestamps.append(current)
        return True
