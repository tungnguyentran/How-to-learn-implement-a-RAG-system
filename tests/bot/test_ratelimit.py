# tests/bot/test_ratelimit.py
from bot.ratelimit import RateLimiter


def test_allows_up_to_the_configured_limit():
    limiter = RateLimiter(max_per_minute=3)

    assert limiter.allow(user_id=1, now=0.0) is True
    assert limiter.allow(user_id=1, now=1.0) is True
    assert limiter.allow(user_id=1, now=2.0) is True
    assert limiter.allow(user_id=1, now=3.0) is False


def test_limits_are_independent_per_user():
    limiter = RateLimiter(max_per_minute=1)

    assert limiter.allow(user_id=1, now=0.0) is True
    assert limiter.allow(user_id=2, now=0.0) is True
    assert limiter.allow(user_id=1, now=0.5) is False


def test_window_slides_after_sixty_seconds():
    limiter = RateLimiter(max_per_minute=1)

    assert limiter.allow(user_id=1, now=0.0) is True
    assert limiter.allow(user_id=1, now=30.0) is False
    assert limiter.allow(user_id=1, now=61.0) is True
