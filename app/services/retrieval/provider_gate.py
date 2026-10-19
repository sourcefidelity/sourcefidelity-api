"""One request gate per provider, shared by every worker process (2026-10-11).

`provider-gate-v1`. Before this, only CORE, Semantic Scholar and a self-hosted metasearch fallback were
paced across processes; Brave, Exa, OpenAlex and Crossref were not paced at
all, and Gutenberg, DataCite, Open Library, ERIC and Europe PMC declared an
interval that nothing enforced. One paper at a time hid that; many papers at
once (an Institutional submission deadline) multiply the request rate.

A caller waits for the provider's next shared start slot and, where the
provider limits requests in flight (Crossref), for a free place. It waits
only as long as the reference's elapsed budget allows (at most
`MAX_WAIT_SECONDS`); otherwise it raises `ProviderBusy` without booking a
slot, and the caller records a skipped call, never a provider failure.

Rates (requests per second) come from `PROVIDER_REQUESTS_PER_SECOND`, from an
adapter's own `min_interval_seconds`, or from limits a provider reported in
its response headers (`learn_limits`), which win once seen: they describe the
account's actual plan. Without Redis each process paces alone at a quarter of
the rate, as Semantic Scholar already does.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import threading
import time
import uuid

from app.config import settings
from app.services.retrieval_deadline import remaining_seconds
from app.services.retrieval import shared_pacing

POLICY = "provider-gate-v1"
MAX_WAIT_SECONDS = 60.0
# Worker processes sharing a key when Redis is unavailable (docker-compose).
_UNSHARED_MULTIPLIER = 4
_local_lock = threading.Lock()
_local_next: dict[str, float] = {}
_local_places: dict[str, threading.BoundedSemaphore] = {}
_clients: dict[tuple[str, int], object] = {}
# Crossref reports its pool's limits as `X-Rate-Limit-*` headers; its live
# polite pool said 3 per 1 s and 3 in flight (2026-10-11), not the documented
# 10 per second. Pacing stays this fraction under a reported rate, so network
# jitter cannot bunch four starts into one of its one-second windows.
_REPORTED_RATE_MARGIN = 0.9


class ProviderBusy(Exception):
    """The provider's next free slot is further away than this call may wait."""

    reason_code = "provider_call_skipped_paced_beyond_budget"


def _rate_in(value: str, provider: str) -> tuple[bool, float | None]:
    for item in str(value or "").split(","):
        name, separator, raw = item.strip().partition(":")
        if separator and name.strip().lower() == provider.lower():
            try:
                rate = float(raw)
            except ValueError:
                return True, None
            return True, rate if rate > 0 else None
    return False, None


def configured_rate(provider: str) -> float | None:
    """Requests per second set for `provider`, or None. A provider the
    configured value does not name keeps the code default's rate
    (owner decision 2026-10-19)."""
    from app.config import Settings
    named, rate = _rate_in(getattr(settings, "PROVIDER_REQUESTS_PER_SECOND", ""), provider)
    if named:
        return rate
    return _rate_in(Settings.model_fields["PROVIDER_REQUESTS_PER_SECOND"].default, provider)[1]


def interval_for(provider: str, default_interval: float = 0.0) -> float:
    """Seconds between request starts: a reported limit, else the setting, else the adapter's."""
    learned = shared_pacing.stored_limits(provider) or {}
    per_second = learned.get("per_second")
    if isinstance(per_second, (int, float)) and per_second > 0:
        return 1.0 / per_second
    rate = configured_rate(provider)
    if rate:
        return 1.0 / rate
    return max(0.0, float(default_interval or 0.0))


def places_for(provider: str, default_places: int | None) -> int | None:
    """In-flight places: the adapter's, lowered to a concurrency limit the provider reported."""
    learned = (shared_pacing.stored_limits(provider) or {}).get("in_flight")
    if isinstance(learned, int) and learned > 0:
        return min(default_places, learned) if default_places else learned
    return default_places


def _wait_budget(max_wait: float | None) -> float:
    budget = MAX_WAIT_SECONDS if max_wait is None else max(0.0, max_wait)
    left = remaining_seconds()
    return budget if left is None else max(0.0, min(budget, left))


def _local_start(provider: str, interval: float, max_wait: float) -> float:
    with _local_lock:
        now = time.monotonic()
        start = max(now, _local_next.get(provider, 0.0))
        if start - now > max_wait:
            return -1.0
        _local_next[provider] = start + interval * _UNSHARED_MULTIPLIER
        return start - now


def _take_place(provider: str, limit: int, hold: float, deadline: float) -> tuple[str, str]:
    token = uuid.uuid4().hex
    while True:
        taken = shared_pacing.acquire_place(provider, limit, hold, token)
        if taken is None:
            with _local_lock:
                place = _local_places.setdefault(
                    provider, threading.BoundedSemaphore(max(1, limit // _UNSHARED_MULTIPLIER)))
            if place.acquire(timeout=max(0.0, deadline - time.monotonic())):
                return "local", token
            raise ProviderBusy(ProviderBusy.reason_code)
        if taken:
            return "shared", token
        if time.monotonic() >= deadline:
            raise ProviderBusy(ProviderBusy.reason_code)
        time.sleep(0.05)


@contextmanager
def turn(provider: str, *, interval: float, places: int | None = None,
         hold_seconds: float = 60.0, max_wait: float | None = None):
    """Wait for `provider`'s next start slot (and an in-flight place when
    `places` is set), then run the request inside the block."""
    if not getattr(settings, "PROVIDER_GATE_ENABLED", True):
        yield
        return
    budget = _wait_budget(max_wait)
    deadline = time.monotonic() + budget
    held = None
    if places:
        held = _take_place(provider, places, hold_seconds, deadline)
    try:
        if interval > 0:
            left = max(0.0, deadline - time.monotonic())
            wait = shared_pacing.reserve_start_within(provider, interval, left)
            if wait is None:
                wait = _local_start(provider, interval, left)
            if wait < 0:
                raise ProviderBusy(ProviderBusy.reason_code)
            if wait:
                time.sleep(wait)
        yield
    finally:
        if held is not None:
            kind, token = held
            if kind == "shared":
                shared_pacing.release_place(provider, token)
            else:
                _local_places[provider].release()


def defer(provider: str, seconds: float | None) -> None:
    """Hold every process back after a provider refused a request (429)."""
    if seconds is not None and 0 < seconds <= MAX_WAIT_SECONDS:
        shared_pacing.defer(provider, seconds)


def _header_values(headers, name: str) -> list[str]:
    return [part.strip() for part in str(headers.get(name) or "").split(",") if part.strip()]


def learn_limits(provider: str, headers) -> dict | None:
    """Record the windows a provider reported in `X-RateLimit-*` headers.

    Brave sends matching comma-separated lists, one value per window, with
    window lengths in `X-RateLimit-Policy` (`1;w=1, 15000;w=2592000`). The
    one-second window becomes the pacing rate; the longest window's limit and
    remainder are kept for operators. Values are numbers only.
    """
    if headers.get("X-Rate-Limit-Limit") or headers.get("x-rate-limit-limit"):
        return _learn_interval_limits(provider, headers)
    limits = _header_values(headers, "X-RateLimit-Limit")
    policies = _header_values(headers, "X-RateLimit-Policy")
    left = _header_values(headers, "X-RateLimit-Remaining")
    if not limits or len(policies) != len(limits):
        return None
    windows = []
    for index, (limit, policy) in enumerate(zip(limits, policies)):
        try:
            seconds = int(policy.split("w=", 1)[1].split(";", 1)[0]) if "w=" in policy else None
            windows.append((seconds, int(limit), int(left[index]) if index < len(left) else None))
        except (ValueError, IndexError):
            return None
    learned = {"policy_version": POLICY, "observed_at": int(time.time())}
    for seconds, limit, remainder in windows:
        if seconds == 1 and limit > 0:
            learned["per_second"] = limit
    longest = max((w for w in windows if w[0]), key=lambda w: w[0], default=None)
    if longest is not None:
        learned.update(window_seconds=longest[0], window_limit=longest[1], window_remaining=longest[2])
    shared_pacing.store_limits(provider, learned)
    return learned


def _header(headers, name: str) -> str:
    return str(headers.get(name) or headers.get(name.lower()) or "").strip()


def _learn_interval_limits(provider: str, headers) -> dict | None:
    """Crossref's form: `X-Rate-Limit-Limit: 3`, `X-Rate-Limit-Interval: 1s`,
    `X-Concurrency-Limit: 3`. Stored only when it changes."""
    try:
        limit = int(_header(headers, "X-Rate-Limit-Limit"))
        interval = _header(headers, "X-Rate-Limit-Interval").lower()
        seconds = float(interval[:-1]) if interval.endswith("s") else float(interval)
        concurrency = _header(headers, "X-Concurrency-Limit")
        in_flight = int(concurrency) if concurrency else None
    except ValueError:
        return None
    if limit <= 0 or seconds <= 0:
        return None
    learned = {"policy_version": POLICY, "reported_limit": limit, "reported_interval_seconds": seconds,
               "per_second": round(limit / seconds * _REPORTED_RATE_MARGIN, 4)}
    if in_flight and in_flight > 0:
        learned["in_flight"] = in_flight
    stored = shared_pacing.stored_limits(provider) or {}
    if {k: stored.get(k) for k in learned if k != "policy_version"} != {
            k: v for k, v in learned.items() if k != "policy_version"}:
        shared_pacing.store_limits(provider, {**learned, "observed_at": int(time.time())})
    return learned


def reported_limits(provider: str) -> dict | None:
    """The limits `provider` last reported (for operators and tests)."""
    return shared_pacing.stored_limits(provider)


def retry_unopened_connection(request, *, delay: float = 1.0):
    """Run `request()`, and once more after `delay` when the connection never
    opened (`httpx.ConnectError`, including a TLS handshake cut off), while the
    reference's budget allows. Such a request never reached the provider, so a
    paid search is not billed twice; a timeout or a refused request is not
    retried here (2026-10-11: bursts of dropped connections failed Brave, Exa
    and OpenAlex calls)."""
    import httpx
    try:
        return request()
    except httpx.ConnectError:
        left = remaining_seconds()
        if left is not None and left <= delay + 5.0:
            raise
        time.sleep(delay)
        return request()


def _client(provider: str):
    """One keep-alive client per provider and process, so each request does
    not open a new TLS connection (created after the worker fork)."""
    import httpx
    key = (provider, os.getpid())
    with _local_lock:
        client = _clients.get(key)
        if client is None:
            client = _clients[key] = httpx.Client()
        return client


def paced_get(provider: str, default_interval: float, url: str, *, places: int | None = None,
              keep_alive: bool = False, **kwargs):
    """`httpx.get` through `provider`'s gate (raises ProviderBusy when held back).

    Limits the provider reports in its headers are learned for every process,
    and a 429 holds every process back for its Retry-After (default 2 s)."""
    import httpx
    with turn(provider, interval=interval_for(provider, default_interval),
              places=places_for(provider, places)):
        response = _client(provider).get(url, **kwargs) if keep_alive else httpx.get(url, **kwargs)
    headers = getattr(response, "headers", None)
    if headers is not None and (headers.get("X-Rate-Limit-Limit") or headers.get("x-rate-limit-limit")):
        learn_limits(provider, headers)
    if getattr(response, "status_code", None) == 429:
        try:
            retry_after = float(headers.get("Retry-After") or 2.0) if headers is not None else 2.0
        except ValueError:
            retry_after = 2.0
        defer(provider, retry_after)
    return response
