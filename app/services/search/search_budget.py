"""Daily paid-search budgets shared by every worker process (`search-daily-budget-v1`, 2026-10-11).

The per-paper ceilings (`SEARCH_ESCALATION_MAX_CALLS`, `WEB_SEARCH_HARD_MAX_CALLS`)
bound one run; nothing bounded a day. `WEB_SEARCH_DAILY_MAX_CALLS` caps each
paid provider for the whole deployment and `WEB_SEARCH_SCOPE_DAILY_MAX_CALLS`
for each checking scope (a Personal user, an institution's course or
assessment scope), per UTC day. Both are empty (no cap) unless an operator
sets them, e.g. `brave:5000,exa:3000`.

A call is reserved before it is made and released if it is not made, in one
atomic Redis step for both counters, so concurrent papers cannot overshoot.
A refused call is a skipped search (`budget_skipped`), which leaves the
reference's search incomplete and retryable, never a finding of absence. When
a cap is set and Redis cannot be reached, calls to that provider are refused
(`daily_budget_unavailable`): a spending cap fails closed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import logging

import redis

from app.config import settings

logger = logging.getLogger(__name__)

POLICY = "search-daily-budget-v1"
_PREFIX = "sourcefidelity:search-budget:"
_TTL_SECONDS = 2 * 86_400
_client = None

# KEYS: the counters to check; ARGV: their limits (0 = none), then the TTL.
_RESERVE = """
for i, key in ipairs(KEYS) do
  local limit = tonumber(ARGV[i])
  if limit > 0 and tonumber(redis.call('GET', key) or '0') >= limit then return i end
end
for i, key in ipairs(KEYS) do
  redis.call('INCR', key)
  redis.call('EXPIRE', key, tonumber(ARGV[#KEYS + 1]))
end
return 0
"""


def _limits(value: str) -> dict[str, int]:
    limits = {}
    for item in str(value or "").split(","):
        name, separator, raw = item.strip().partition(":")
        if separator and name:
            try:
                limits[name.strip().lower()] = max(0, int(raw))
            except ValueError:
                logger.warning("Ignoring invalid daily search limit")
    return limits


def _redis():
    global _client
    if _client is None:
        _client = redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=0.25, socket_timeout=1.0)
    return _client


def _keys(provider: str, scope: tuple[str, str] | None, day: str) -> list[tuple[str, int]]:
    keys = []
    deployment = _limits(getattr(settings, "WEB_SEARCH_DAILY_MAX_CALLS", "")).get(provider, 0)
    if deployment:
        keys.append((f"{_PREFIX}{day}:{provider}", deployment))
    per_scope = _limits(getattr(settings, "WEB_SEARCH_SCOPE_DAILY_MAX_CALLS", "")).get(provider, 0)
    if per_scope and scope and scope[0] and scope[1]:
        # Scope identifiers are hashed: Redis keys carry no user or course names.
        digest = hashlib.sha256(f"{scope[0]}\x00{scope[1]}".encode()).hexdigest()[:32]
        keys.append((f"{_PREFIX}{day}:{provider}:scope:{digest}", per_scope))
    return keys


def _day(now: datetime | None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")


def reserve(provider: str, scope: tuple[str, str] | None, *, now: datetime | None = None) -> tuple[str | None, list[str]]:
    """Reserve one call. Returns (refusal reason or None, keys to release if unused)."""
    keys = _keys(provider.lower(), scope, _day(now))
    if not keys:
        return None, []
    try:
        refused = int(_redis().eval(_RESERVE, len(keys), *[key for key, _ in keys],
                                    *[limit for _, limit in keys], _TTL_SECONDS))
    except redis.RedisError as exc:
        logger.warning("Daily search budget unavailable; refusing capped calls (%s)", type(exc).__name__)
        return "daily_budget_unavailable", []
    if refused:
        return ("scope_daily_call_ceiling" if ":scope:" in keys[refused - 1][0] else "daily_call_ceiling"), []
    return None, [key for key, _ in keys]


def release(keys: list[str]) -> None:
    """Give back a reserved call that was not made."""
    if not keys:
        return
    try:
        pipeline = _redis().pipeline()
        for key in keys:
            pipeline.decr(key)
        pipeline.execute()
    except redis.RedisError as exc:
        logger.warning("Daily search budget not released (%s)", type(exc).__name__)
