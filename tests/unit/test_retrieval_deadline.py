"""Elapsed budgets preserve findings boundaries and do not leave background work."""
from unittest.mock import Mock

import httpx
import pytest

from app.services import retrieval_deadline as clock
from app.services.retrieval.base import RetrievalResult
from app.services.safe_fetch import safe_request
from app.services.source_resolver import SourceResolver, _timed_retrieval


@pytest.fixture
def ticks(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(clock.time, "monotonic", lambda: now[0])
    return now


def test_nested_scope_never_extends_budget_and_restores(ticks):
    with clock.deadline_scope(10):
        ticks[0] = 4
        with clock.deadline_scope(20):
            assert clock.remaining(30) == 6
        ticks[0] = 10
        with pytest.raises(httpx.TimeoutException):
            clock.remaining(30)
    assert not clock.expired()
    assert clock.remaining(30) == 30


def test_expired_route_does_not_execute_and_records_timing(ticks):
    operation = Mock()

    @_timed_retrieval
    def lookup():
        operation()

    with clock.deadline_scope(1):
        ticks[0] = 2
        result = lookup()
    operation.assert_not_called()
    assert not result.success
    assert "timeout" in result.error
    assert result.metadata["operation_timing"]["elapsed_seconds"] >= 0


def test_stream_elapsed_timeout_even_while_bytes_keep_arriving(ticks, monkeypatch):
    closed = []

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(10):
                ticks[0] += 4
                yield b"data"

        def close(self):
            closed.append(True)

    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, request=request, stream=Stream())))
    monkeypatch.setattr("app.services.safe_fetch.httpx.Client", lambda **kw: client)
    monkeypatch.setattr("app.services.safe_fetch._validate_url", lambda url: None)
    with pytest.raises(httpx.TimeoutException):
        safe_request("https://source.example/file", timeout=10)
    assert ticks[0] == 12
    assert closed and client.is_closed


def test_redirects_share_elapsed_allowance(ticks, monkeypatch):
    calls = []

    def request(req):
        calls.append(req)
        ticks[0] += 6
        return httpx.Response(302, headers={"location": "/next"}, request=req)

    client = httpx.Client(transport=httpx.MockTransport(request))
    monkeypatch.setattr("app.services.safe_fetch.httpx.Client", lambda **kw: client)
    monkeypatch.setattr("app.services.safe_fetch._validate_url", lambda url: None)
    with pytest.raises(httpx.TimeoutException):
        safe_request("https://source.example/start", timeout=10)
    assert len(calls) == 2
    assert calls[1].extensions["timeout"]["read"] == 4
    assert client.is_closed


def test_expired_reference_does_not_even_resolve_host(ticks, monkeypatch):
    check = Mock()
    monkeypatch.setattr("app.services.safe_fetch._validate_url", check)
    with clock.deadline_scope(1):
        ticks[0] = 2
        with pytest.raises(httpx.TimeoutException):
            safe_request("https://source.example/file")
    check.assert_not_called()


def test_completed_representation_not_erased_by_timing_wrapper(ticks):
    original = RetrievalResult(source_name="test", success=True, full_text=b"existing")

    @_timed_retrieval
    def _download_and_cache():
        return original

    with clock.deadline_scope(1):
        ticks[0] = 2
        result = _download_and_cache()
    assert result.full_text == original.full_text


def test_expired_identity_search_is_incomplete_not_fabricated(ticks, monkeypatch):
    from app.services.schemas import ParsedReference
    from app.services.reference_credibility import assess_reference_credibility

    obj = SourceResolver.__new__(SourceResolver)
    obj._acquisition_capabilities = None
    provider = Mock(name="adapter")
    provider.name = "crossref"
    provider.capabilities = {"metadata_only_search", "title_author"}
    obj._retrieval_sources = [provider]
    ref = ParsedReference(reference_id="frozen", title="A sufficiently specific work title",
        author="Writer, A.", year="2020", source_kind="journal_article",
        raw_ref="Writer, A. (2020). A sufficiently specific work title.")
    with clock.deadline_scope(1):
        ticks[0] = 2
        result = obj.resolve_reference(ref, identity_only=True)
    provider.search_by_title_author.assert_not_called()
    trace = result.metadata["reference_discovery_trace"]
    assert result.metadata["reference_discovery"]["outcome"] == "search_incomplete"
    assert any("reference-elapsed-budget-v1" in value for value in trace["limitations"])
    # 2026-09-23: this was `operational_failure`, which named the adapter as
    # having failed a call that never left the process. The outcome stays in
    # the family that forces `search_incomplete` above; only the attribution
    # changed. See tests/unit/test_reference_budget_attribution.py.
    assert trace["attempts"][0]["outcome"] == "unavailable"
    assert trace["attempts"][0]["reason_code"] == "route_elapsed_budget_exhausted"
    # The query must say the same thing. Left as `operational_failure`, it named
    # the provider as the cause, which is what the provider-recovery trigger
    # reads -- so a reference our own budget cut short could be re-run whenever
    # that provider was reported as recovering.
    query_ids = set(trace["attempts"][0]["query_ids"])
    assert query_ids
    assert {q["execution_outcome"] for q in trace["queries"]
            if q["query_id"] in query_ids} == {"budget_skipped"}
    assert not assess_reference_credibility(ref, result.metadata["reference_discovery"], trace)["findings"]
