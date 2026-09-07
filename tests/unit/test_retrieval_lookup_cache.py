"""Canonical abstract lookup caching avoids redundant adapter calls."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from app.services.retrieval.base import RetrievalResult
from app.services.retrieval_lookup_cache import (
    LookupCacheRecord,
    RetrievalLookupCache,
)
from app.services.source_resolver import SourceResolver
from app.services.source_resolver import SourceResolutionError
from app.services.schemas import ParsedReference


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.values[key] = value
        self.ttls[key] = ttl

    def delete(self, key: str) -> None:
        self.values.pop(key, None)


def _abstract_result() -> RetrievalResult:
    return RetrievalResult(
        source_name="openalex",
        success=True,
        abstract="A reusable canonical abstract.",
        doi="10.1234/example",
        title="Example Work",
        year="2024",
        authors=["Rivera, Alex"],
        metadata={"raw_provider_payload": "must not be cached"},
    )


def test_abstract_cache_reuses_only_bounded_fields_and_tracks_freshness() -> None:
    client = _FakeRedis()
    cache = RetrievalLookupCache(client)
    cache.put_abstract(
        _abstract_result(),
        doi="10.1234/example",
        title="Example Work",
        author="Rivera, Alex",
        year="2024",
        student_url="https://example.org/item",
        policy_signature="policy-a",
    )

    record = cache.get(
        doi="https://doi.org/10.1234/EXAMPLE",
        title=None,
        author=None,
        year=None,
    )

    assert record is not None
    assert record.result is not None
    assert record.result.abstract == "A reusable canonical abstract."
    assert record.result.metadata["from_lookup_cache"] is True
    assert "raw_provider_payload" not in record.result.metadata
    assert record.is_fresh(
        policy_signature="policy-a",
        student_url="https://example.org/item",
    )
    assert not record.is_fresh(
        policy_signature="policy-a",
        student_url="https://example.org/a-different-item",
    )
    assert not record.is_fresh(
        policy_signature="policy-b",
        student_url="https://example.org/item",
    )


def test_negative_cache_uses_a_shorter_refresh_window() -> None:
    client = _FakeRedis()
    cache = RetrievalLookupCache(client)
    cache.put_miss(
        doi="10.1234/missing",
        title=None,
        author=None,
        year=None,
        student_url=None,
        policy_signature="policy-a",
    )

    record = cache.get(doi="10.1234/missing", title=None, author=None, year=None)

    assert record is not None
    assert record.outcome == "miss"
    assert record.result is None
    assert record.refresh_after > record.stored_at


def test_title_cache_isolated_by_bibliographic_work_type() -> None:
    client = _FakeRedis()
    cache = RetrievalLookupCache(client)
    cache.put_abstract(
        _abstract_result(),
        doi=None,
        title="The Master Switch",
        author="Review Author",
        year="2012",
        student_url=None,
        policy_signature="policy-a",
        source_kind="book_review",
    )

    assert cache.get(
        doi=None,
        title="The Master Switch",
        author="Review Author",
        year="2012",
        source_kind="book_review",
    ) is not None
    assert cache.get(
        doi=None,
        title="The Master Switch",
        author="Review Author",
        year="2012",
        source_kind="monograph",
    ) is None


def test_resolver_returns_fresh_cached_abstract_without_adapter_calls() -> None:
    now = datetime.now(timezone.utc)
    cached = LookupCacheRecord(
        outcome="abstract",
        stored_at=now,
        refresh_after=now + timedelta(days=1),
        policy_signature="policy-a",
        student_url_hash=None,
        result=_abstract_result(),
    )
    cache = Mock()
    cache.get.return_value = cached
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._backend = None
    resolver._retrieval_sources = []
    resolver._acquisition_capabilities = None
    resolver._lookup_cache = cache
    resolver._check_local_cache = Mock(
        return_value=RetrievalResult(source_name="local_cache", success=False)
    )
    resolver._lookup_policy_signature = Mock(return_value="policy-a")

    result = resolver.resolve(doi="10.1234/example", title="Example Work")

    assert result.abstract == "A reusable canonical abstract."
    cache.put_abstract.assert_not_called()


def test_resolve_reference_bypasses_unbound_fresh_cache_for_discovery_trace() -> None:
    now = datetime.now(timezone.utc)
    cached = LookupCacheRecord(
        outcome="abstract",
        stored_at=now,
        refresh_after=now + timedelta(days=1),
        policy_signature="policy-a",
        student_url_hash=None,
        result=_abstract_result(),
    )
    cache = Mock()
    cache.get.return_value = cached
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._backend = None
    resolver._retrieval_sources = []
    resolver._acquisition_capabilities = None
    resolver._lookup_cache = cache
    resolver._check_local_cache = Mock(
        return_value=RetrievalResult(source_name="local_cache", success=False)
    )
    resolver._lookup_policy_signature = Mock(return_value="policy-a")

    with pytest.raises(SourceResolutionError) as raised:
        resolver.resolve_reference(
            ParsedReference(
                reference_id="ref-cache-provenance",
                author="Rivera, Alex",
                year="2024",
                title="Example Work",
                doi="10.1234/example",
            )
        )

    trace = raised.value.reference_discovery_trace
    assert trace is not None
    assert any(
        "unbound lookup-cache result was bypassed" in limitation
        for limitation in trace["limitations"]
    )
    cache.put_miss.assert_called_once()


def test_resolver_uses_stale_abstract_if_refresh_finds_nothing() -> None:
    now = datetime.now(timezone.utc)
    cached = LookupCacheRecord(
        outcome="abstract",
        stored_at=now - timedelta(days=60),
        refresh_after=now - timedelta(days=30),
        policy_signature="policy-a",
        student_url_hash=None,
        result=_abstract_result(),
    )
    cache = Mock()
    cache.get.return_value = cached
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._backend = None
    resolver._retrieval_sources = []
    resolver._acquisition_capabilities = set()
    resolver._lookup_cache = cache
    resolver._check_local_cache = Mock(
        return_value=RetrievalResult(source_name="local_cache", success=False)
    )
    resolver._lookup_policy_signature = Mock(return_value="policy-a")

    result = resolver.resolve(doi="10.1234/example", title="Example Work")

    assert result.abstract == "A reusable canonical abstract."
    assert result.metadata["lookup_cache_stale"] is True
    cache.put_miss.assert_not_called()


def test_lookup_refresh_due_bypasses_cache_for_a_changed_student_url() -> None:
    client = _FakeRedis()
    cache = RetrievalLookupCache(client)
    cache.put_abstract(
        _abstract_result(),
        doi="10.1234/example",
        title="Example Work",
        author="Rivera, Alex",
        year="2024",
        student_url="https://example.org/original",
        policy_signature="policy-a",
    )
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._lookup_cache = cache
    resolver._lookup_policy_signature = Mock(return_value="policy-a")

    assert not resolver.lookup_refresh_due(
        doi="10.1234/example",
        student_url="https://example.org/original",
    )
    assert resolver.lookup_refresh_due(
        doi="10.1234/example",
        student_url="https://example.org/replacement",
    )
