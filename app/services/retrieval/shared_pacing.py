"""Provider pacing shared by every worker process (2026-09-25).

A provider's rate limit belongs to the API key, but each adapter paced only
its own process. The worker runs four processes, so when papers overlapped the
key saw up to four times its allowance. Measured over stored runs: Semantic
Scholar failed 6% of calls with no other paper running and 75% with three or
more, and CORE, which stalls a second in-flight request on one key, degraded
the same way. The shared state lives in the Redis instance the task queue
already uses.

Every function degrades to "not available" (None) when Redis is unreachable
or shared pacing is disabled, and callers then fall back to their existing
process-local pacing. Nothing here sleeps: callers wait with their own clock,
so their pacing stays testable.
"""
from __future__ import annotations

from contextlib import contextmanager
import logging
import time

import redis

from app.config import settings

logger = logging.getLogger(__name__)

_PREFIX = "sourcefidelity:provider-pacing:"
_RETRY_AFTER_SECONDS = 30.0
_client = None
_unavailable_until = 0.0

# Reserve the next start slot atomically: a caller starts at the later of now
# and the reserved next start, and pushes the next start one interval on.
_RESERVE = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local nxt = tonumber(redis.call('GET', KEYS[1]) or '0')
local start = math.max(now, nxt)
redis.call('SET', KEYS[1], start + tonumber(ARGV[1]), 'PX', tonumber(ARGV[2]))
return start - now
"""
# Push the next allowed start to at least now + delay (a provider instruction).
_DEFER = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local nxt = tonumber(redis.call('GET', KEYS[1]) or '0')
local target = math.max(nxt, now + tonumber(ARGV[1]))
redis.call('SET', KEYS[1], target, 'PX', tonumber(ARGV[2]))
return target - now
"""


# As _RESERVE, but books nothing when the slot is more than ARGV[3] ms away,
# so a caller that will not wait never pushes later callers back.
_RESERVE_WITHIN = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local nxt = tonumber(redis.call('GET', KEYS[1]) or '0')
local start = math.max(now, nxt)
if start - now > tonumber(ARGV[3]) then return -1 end
redis.call('SET', KEYS[1], start + tonumber(ARGV[1]), 'PX', tonumber(ARGV[2]))
return start - now
"""
# Take one of ARGV[1] in-flight places for ARGV[2] ms; expired places free themselves.
_ACQUIRE_PLACE = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[1]) then return 0 end
redis.call('ZADD', KEYS[1], now + tonumber(ARGV[2]), ARGV[3])
redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[2]) + 60000)
return 1
"""


def _redis():
    global _client
    if not getattr(settings, "RETRIEVAL_SHARED_PACING_ENABLED", True):
        return None
    if time.monotonic() < _unavailable_until:
        return None
    if _client is None:
        _client = redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=0.25, socket_timeout=1.0)
    return _client


def _disable(exc: Exception) -> None:
    global _unavailable_until
    _unavailable_until = time.monotonic() + _RETRY_AFTER_SECONDS
    logger.warning("Shared provider pacing unavailable; pacing per process (%s)", type(exc).__name__)


def reserve_start(provider: str, interval_seconds: float) -> float | None:
    """Seconds this caller must wait so starts stay `interval` apart across processes."""
    client = _redis()
    if client is None:
        return None
    interval_ms = max(0, int(interval_seconds * 1000))
    try:
        wait_ms = client.eval(_RESERVE, 1, _PREFIX + provider + ":next", interval_ms,
                              max(interval_ms * 4, 60_000))
    except redis.RedisError as exc:
        _disable(exc)
        return None
    return max(0.0, int(wait_ms) / 1000)


def defer(provider: str, seconds: float) -> None:
    """Hold every process back for `seconds` (a Retry-After the provider sent)."""
    client = _redis()
    if client is None or seconds <= 0:
        return
    try:
        client.eval(_DEFER, 1, _PREFIX + provider + ":next", int(seconds * 1000),
                    int(seconds * 1000) + 60_000)
    except redis.RedisError as exc:
        _disable(exc)


@contextmanager
def exclusive(provider: str, *, hold_seconds: float, wait_seconds: float):
    """One in-flight request per provider across processes.

    Yields True when held, False when another process held it for longer than
    `wait_seconds`, and None when shared pacing is unavailable (the caller then
    relies on its process-local lock alone). The lock expires after
    `hold_seconds`, so a crashed worker cannot block the provider.
    """
    client = _redis()
    if client is None:
        yield None
        return
    lock = client.lock(_PREFIX + provider + ":in-flight", timeout=hold_seconds,
                       blocking_timeout=wait_seconds)
    try:
        acquired = lock.acquire()
    except redis.RedisError as exc:
        _disable(exc)
        yield None
        return
    try:
        yield bool(acquired)
    finally:
        if acquired:
            try:
                lock.release()
            except redis.RedisError:
                pass   # expired while held; the timeout already freed it


def reserve_start_within(provider: str, interval_seconds: float, max_wait_seconds: float) -> float | None:
    """Like `reserve_start`, but returns -1.0 and books nothing when the slot
    is more than `max_wait_seconds` away."""
    client = _redis()
    if client is None:
        return None
    interval_ms = max(0, int(interval_seconds * 1000))
    try:
        wait_ms = client.eval(_RESERVE_WITHIN, 1, _PREFIX + provider + ":next", interval_ms,
                              max(interval_ms * 4, 60_000), max(0, int(max_wait_seconds * 1000)))
    except redis.RedisError as exc:
        _disable(exc)
        return None
    return -1.0 if int(wait_ms) < 0 else int(wait_ms) / 1000


def acquire_place(provider: str, limit: int, hold_seconds: float, token: str) -> bool | None:
    """Take one of `limit` in-flight places shared by every process; None when
    shared pacing is unavailable. A place expires after `hold_seconds`."""
    client = _redis()
    if client is None:
        return None
    try:
        return bool(client.eval(_ACQUIRE_PLACE, 1, _PREFIX + provider + ":places", int(limit),
                                max(1, int(hold_seconds * 1000)), token))
    except redis.RedisError as exc:
        _disable(exc)
        return None


def release_place(provider: str, token: str) -> None:
    client = _redis()
    if client is None:
        return
    try:
        client.zrem(_PREFIX + provider + ":places", token)
    except redis.RedisError as exc:
        _disable(exc)


def store_limits(provider: str, limits: dict, ttl_seconds: int = 86_400) -> None:
    """Remember the limits a provider reported (numbers only) for every process."""
    import json
    client = _redis()
    if client is None:
        return
    try:
        client.set(_PREFIX + provider + ":limits", json.dumps(limits, sort_keys=True), ex=ttl_seconds)
    except redis.RedisError as exc:
        _disable(exc)


def stored_limits(provider: str) -> dict | None:
    import json
    client = _redis()
    if client is None:
        return None
    try:
        raw = client.get(_PREFIX + provider + ":limits")
        value = json.loads(raw) if raw else None
    except (redis.RedisError, ValueError) as exc:
        if isinstance(exc, redis.RedisError):
            _disable(exc)
        return None
    return value if isinstance(value, dict) else None
