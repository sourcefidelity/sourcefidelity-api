import pytest
from unittest.mock import Mock
import json

import httpx

from pydantic import SecretStr
from app.services.retrieval.core import CoreRetriever
from app.services.retrieval import core as core_module
from app.services.retrieval.openalex import OpenAlexRetriever
from app.services.retrieval.provider_runtime import ProviderHealthStore, ProviderPolicy
from app.services.retrieval.semantic_scholar import SemanticScholarRetriever
from app.services.search.tavily import TavilySearch
from app.services.search.brightdata import BrightDataSearch
from app.services.search.brave import BraveSearch
from app.config import settings


def _response(status_code: int) -> httpx.Response:
    request = httpx.Request("GET", "https://provider.example/query")
    return httpx.Response(status_code, request=request)


def test_tavily_quota_response_opens_run_level_circuit(monkeypatch) -> None:
    request = Mock(return_value=_response(432))
    monkeypatch.setattr("app.services.search.tavily.httpx.post", request)
    provider = TavilySearch("test-key")

    assert provider.search("first query") == []
    assert provider.last_status == "rate_limited"
    assert provider.search("second query") == []
    assert provider.last_status == "rate_limited"
    request.assert_called_once()


def test_brightdata_unparsed_body_is_not_reported_as_completed_no_results(
    monkeypatch,
) -> None:
    """A body without an organic block is a failure, not a certified empty.

    The distinction decides whether the application may treat the answer as
    evidence that a reference was not found anywhere.
    """
    response = _response(200)
    response._content = json.dumps({"status": "blocked"}).encode()
    monkeypatch.setattr(
        "app.services.search.brightdata.httpx.post", Mock(return_value=response)
    )
    provider = BrightDataSearch("token", "zone")

    assert provider.search("bounded query") == []
    assert provider.last_status != "completed"
    assert provider.last_reason_code == "brightdata_serp_v1_missing_organic"


def test_brightdata_explicit_empty_organic_is_a_completed_search(
    monkeypatch,
) -> None:
    response = _response(200)
    response._content = json.dumps({"organic": []}).encode()
    monkeypatch.setattr(
        "app.services.search.brightdata.httpx.post", Mock(return_value=response)
    )
    provider = BrightDataSearch("token", "zone")

    assert provider.search("bounded query") == []
    assert provider.last_status == "completed"
    assert provider.last_reason_code == "brightdata_serp_v1_empty"


def test_search_failure_log_hashes_query_and_omits_request_url(
    monkeypatch, caplog
) -> None:
    query = "private bibliographic query"
    request = httpx.Request(
        "GET", "https://provider.example/search", params={"q": query}
    )
    response = httpx.Response(422, request=request)
    monkeypatch.setattr(
        "app.services.search.brave.httpx.get", Mock(return_value=response)
    )

    assert BraveSearch("test-key").search(query) == []
    assert query not in caplog.text
    assert str(request.url) not in caplog.text
    assert "query_sha256=" in caplog.text
    assert "failure=HTTP_422" in caplog.text


def test_core_opens_run_level_circuit_after_429(monkeypatch) -> None:
    request = Mock(return_value=_response(429))
    monkeypatch.setattr("app.services.retrieval.core._core_request", request)
    retriever = CoreRetriever()

    first = retriever.search_by_doi("10.1234/example")
    second = retriever.search_by_doi("10.1234/other")

    assert first.success is False
    assert second.success is False
    request.assert_called_once()
    assert retriever.provider_metrics["calls"] == 1
    assert retriever.provider_metrics["rate_limited"] == 1
    assert retriever.provider_metrics["circuit_skips"] == 1
    assert retriever.provider_metrics["timeouts"] == 0
    assert request.call_args.kwargs["headers"]["Authorization"].startswith("Bearer ")


def test_openalex_401_opens_circuit_without_leaking_query_key(
    monkeypatch, caplog
) -> None:
    secret = "openalex-secret-test-key"
    monkeypatch.setattr(settings, "OPENALEX_API_KEY", SecretStr(secret))
    request = Mock(return_value=_response(401))
    monkeypatch.setattr("app.services.retrieval.openalex.httpx.get", request)
    retriever = OpenAlexRetriever()

    first = retriever.search_by_doi("10.1234/example")
    second = retriever.search_by_doi("10.1234/other")

    assert first.success is False
    assert second.success is False
    request.assert_called_once()
    assert secret not in (first.error or "")
    assert secret not in (second.error or "")
    assert secret not in caplog.text
    assert "401" in (first.error or "")
    assert "circuit" in (second.error or "").lower()


def test_openalex_preflight_checks_configured_key_once(monkeypatch) -> None:
    monkeypatch.setattr(settings, "OPENALEX_API_KEY", SecretStr("configured-test-key"))
    request = Mock(return_value=_response(200))
    monkeypatch.setattr("app.services.retrieval.openalex.httpx.get", request)

    okay, detail = OpenAlexRetriever().preflight(require_api_key=True)

    assert okay is True
    assert "configured key" in detail
    request.assert_called_once()


def test_openalex_grouped_doi_prefetch_populates_cache(monkeypatch) -> None:
    monkeypatch.setattr(settings, "OPENALEX_API_KEY", SecretStr("configured-test-key"))
    monkeypatch.setattr(settings, "RETRIEVAL_PROVIDER_CONFIG", "{}")
    payload = {
        "results": [
            {
                "doi": "https://doi.org/10.1234/example",
                "title": "Grouped OpenAlex paper",
                "publication_year": 2024,
                "authorships": [],
                "locations": [],
            }
        ]
    }
    response = _response(200)
    response._content = json.dumps(payload).encode()
    request = Mock(return_value=response)
    monkeypatch.setattr("app.services.retrieval.openalex.httpx.get", request)

    retriever = OpenAlexRetriever()
    count = retriever.prefetch_dois(
        ["10.1234/EXAMPLE", "https://doi.org/10.1234/missing"]
    )
    found = retriever.search_by_doi("doi:10.1234/example")
    missing = retriever.search_by_doi("10.1234/missing")

    assert count == 1
    assert found.success is True
    assert found.title == "Grouped OpenAlex paper"
    assert missing.success is False
    request.assert_called_once()
    assert request.call_args.kwargs["params"] == {
        "filter": (
            "doi:https://doi.org/10.1234/example|"
            "https://doi.org/10.1234/missing"
        ),
        "per_page": 2,
        "api_key": "configured-test-key",
    }
    assert retriever.provider_metrics["grouped_doi_calls"] == 1
    assert retriever.provider_metrics["grouped_doi_items"] == 2
    assert retriever.provider_metrics["grouped_doi_cache_hits"] == 2


def test_openalex_grouped_title_candidates_are_attributed_and_cached(monkeypatch) -> None:
    monkeypatch.setattr(settings, "OPENALEX_API_KEY", SecretStr("configured-test-key"))
    monkeypatch.setattr(settings, "OPENALEX_GROUPED_TITLE_PREFETCH_ENABLED", True)
    monkeypatch.setattr(settings, "RETRIEVAL_PROVIDER_CONFIG", "{}")
    payload = {
        "results": [
            {
                "doi": "https://doi.org/10.1234/first",
                "title": "First Distinctive Research Article",
                "publication_year": 2022,
                "authorships": [{"author": {"display_name": "Alex Rivera"}}],
                "locations": [],
            },
            {
                "doi": "https://doi.org/10.1234/second",
                "title": "Second Unusual Scholarly Study",
                "publication_year": 2023,
                "authorships": [{"author": {"display_name": "Morgan Chen"}}],
                "locations": [],
            },
        ]
    }
    response = _response(200)
    response._content = json.dumps(payload).encode()
    request = Mock(return_value=response)
    monkeypatch.setattr("app.services.retrieval.openalex.httpx.get", request)

    retriever = OpenAlexRetriever()
    matched = retriever.prefetch_titles([
        ("First Distinctive Research Article", "Rivera, Alex"),
        ("Second Unusual Scholarly Study", "Chen, Morgan"),
    ])
    first = retriever.search_by_title_author(
        "First Distinctive Research Article", "Rivera, Alex"
    )
    second = retriever.search_by_title_author(
        "Second Unusual Scholarly Study", "Chen, Morgan"
    )

    assert matched == 2
    assert first.doi == "10.1234/first"
    assert second.doi == "10.1234/second"
    request.assert_called_once()
    assert request.call_args.kwargs["params"] == {
        "search.exact": (
            '("First Distinctive Research Article" OR '
            '"Second Unusual Scholarly Study")'
        ),
        "per_page": 100,
        "api_key": "configured-test-key",
    }
    assert retriever.provider_metrics["grouped_title_calls"] == 1
    assert retriever.provider_metrics["grouped_title_items"] == 2
    assert retriever.provider_metrics["grouped_title_matches"] == 2
    assert retriever.provider_metrics["grouped_title_cache_hits"] == 2
    assert retriever.provider_metrics["individual_title_fallbacks"] == 0


def test_openalex_unresolved_grouped_title_uses_individual_fallback(monkeypatch) -> None:
    monkeypatch.setattr(settings, "OPENALEX_API_KEY", SecretStr("configured-test-key"))
    monkeypatch.setattr(settings, "OPENALEX_GROUPED_TITLE_PREFETCH_ENABLED", True)
    monkeypatch.setattr(settings, "RETRIEVAL_PROVIDER_CONFIG", "{}")
    grouped = _response(200)
    grouped._content = b'{"results": []}'
    individual = _response(200)
    individual._content = json.dumps({
        "results": [{
            "doi": "https://doi.org/10.1234/fallback",
            "title": "A Title Recovered Individually",
            "publication_year": 2024,
            "authorships": [{"author": {"display_name": "Jamie Patel"}}],
            "locations": [],
        }]
    }).encode()
    request = Mock(side_effect=[grouped, individual])
    monkeypatch.setattr("app.services.retrieval.openalex.httpx.get", request)

    retriever = OpenAlexRetriever()
    matched = retriever.prefetch_titles([
        ("A Title Recovered Individually", "Patel, Jamie")
    ])
    result = retriever.search_by_title_author(
        "A Title Recovered Individually", "Patel, Jamie"
    )

    assert matched == 0
    assert result.success is True
    assert result.doi == "10.1234/fallback"
    assert request.call_count == 2
    assert request.call_args_list[1].kwargs["params"] == {
        "search": "A Title Recovered Individually",
        "per_page": 5,
        "api_key": "configured-test-key",
    }
    assert retriever.provider_metrics["individual_title_fallbacks"] == 1


def test_core_grouped_doi_prefetch_uses_boolean_query_and_cache(monkeypatch) -> None:
    payload = {
        "results": [{
            "id": 1,
            "title": "CORE paper",
            "doi": "10.1234/example",
            "authors": [],
            "downloadUrl": "https://example.org/paper.pdf",
        }]
    }
    response = _response(200)
    response._content = json.dumps(payload).encode()
    response.headers["X-RateLimit-Limit"] = "150"
    response.headers["X-RateLimit-Remaining"] = "149"
    request = Mock(return_value=response)
    monkeypatch.setattr("app.services.retrieval.core._core_request", request)

    retriever = CoreRetriever()
    count = retriever.prefetch_dois(["10.1234/EXAMPLE", "10.1234/missing"])
    found = retriever.search_by_doi("https://doi.org/10.1234/example")
    missing = retriever.search_by_doi("10.1234/missing")

    assert count == 1
    assert found.success is True
    assert missing.success is False
    request.assert_called_once()
    assert request.call_args.args[1] == {
        "q": "doi:10.1234/example OR doi:10.1234/missing",
        "limit": 2,
    }
    assert retriever.provider_metrics["grouped_doi_calls"] == 1
    assert retriever.provider_metrics["grouped_doi_items"] == 2
    assert retriever.provider_metrics["grouped_doi_cache_hits"] == 2
    assert retriever.provider_metrics["rate_limit_limit"] == 150
    assert retriever.provider_metrics["rate_limit_remaining"] == 149


def test_core_retries_short_header_directed_429(monkeypatch) -> None:
    limited = _response(429)
    limited.headers["X-RateLimit-Retry-After"] = "2"
    recovered = _response(200)
    recovered._content = b'{"results": []}'
    request = Mock(side_effect=[limited, recovered])
    sleep = Mock()
    monkeypatch.setattr("app.services.retrieval.core._core_request", request)
    monkeypatch.setattr("app.services.retrieval.core.time.sleep", sleep)

    result = CoreRetriever().search_by_doi("10.1234/example")

    assert result.success is False
    assert result.error == "No results"
    assert request.call_count == 2
    sleep.assert_not_called()


def test_core_retry_after_paces_only_an_exhausted_quota(monkeypatch) -> None:
    """CORE stamps Retry-After on every response; it binds only at zero remaining."""
    clock = [100.0]
    sleeps: list[float] = []
    exhausted = _response(200)
    exhausted.headers["X-RateLimit-Retry-After"] = "7"
    exhausted.headers["X-RateLimit-Remaining"] = "0"
    responses = iter([exhausted, _response(200)])

    def advance(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(core_module, "_core_last_request", 0.0)
    monkeypatch.setattr(core_module, "_core_next_allowed_request", 0.0)
    monkeypatch.setattr(core_module, "_core_quota_interval", None)
    monkeypatch.setattr(core_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(core_module.time, "sleep", advance)
    monkeypatch.setattr(core_module.httpx, "get", lambda *args, **kwargs: next(responses))

    core_module._core_request("https://provider.example/query", {}, min_interval=1)
    core_module._core_request("https://provider.example/query", {}, min_interval=1)

    assert sleeps == [7.0]


def test_core_ignores_retry_after_while_quota_remains(monkeypatch) -> None:
    clock = [100.0]
    sleeps: list[float] = []
    first = _response(200)
    first.headers["X-RateLimit-Retry-After"] = "7"
    first.headers["X-RateLimit-Remaining"] = "9"
    responses = iter([first, _response(200)])

    def advance(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(core_module, "_core_last_request", 0.0)
    monkeypatch.setattr(core_module, "_core_next_allowed_request", 0.0)
    monkeypatch.setattr(core_module, "_core_quota_interval", None)
    monkeypatch.setattr(core_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(core_module.time, "sleep", advance)
    monkeypatch.setattr(core_module.httpx, "get", lambda *args, **kwargs: next(responses))

    core_module._core_request("https://provider.example/query", {}, min_interval=1)
    core_module._core_request("https://provider.example/query", {}, min_interval=1)

    assert sleeps == [1.0]


def test_core_paces_from_the_reported_per_minute_quota(monkeypatch) -> None:
    clock = [100.0]
    sleeps: list[float] = []
    first = _response(200)
    first.headers["X-RateLimit-Limit"] = "25"
    first.headers["X-RateLimit-Remaining"] = "24"
    responses = iter([first, _response(200)])

    def advance(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(core_module, "_core_last_request", 0.0)
    monkeypatch.setattr(core_module, "_core_next_allowed_request", 0.0)
    monkeypatch.setattr(core_module, "_core_quota_interval", None)
    monkeypatch.setattr(core_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(core_module.time, "sleep", advance)
    monkeypatch.setattr(core_module.httpx, "get", lambda *args, **kwargs: next(responses))

    core_module._core_request("https://provider.example/query", {})
    core_module._core_request("https://provider.example/query", {})

    assert sleeps == [pytest.approx(60 / 25 * 1.05)]   # not the 10s fallback


def test_core_uses_ten_second_fallback_without_valid_header(monkeypatch) -> None:
    clock = [100.0]
    sleeps: list[float] = []
    responses = iter([_response(200), _response(200)])

    def advance(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(core_module, "_core_last_request", 0.0)
    monkeypatch.setattr(core_module, "_core_next_allowed_request", 0.0)
    monkeypatch.setattr(core_module, "_core_quota_interval", None)
    monkeypatch.setattr(core_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(core_module.time, "sleep", advance)
    monkeypatch.setattr(core_module.httpx, "get", lambda *args, **kwargs: next(responses))

    core_module._core_request("https://provider.example/query", {})
    core_module._core_request("https://provider.example/query", {})

    assert sleeps == [10.0]


def test_core_parses_iso_retry_timestamp() -> None:
    response = _response(200)
    response.headers["X-RateLimit-Retry-After"] = "2026-08-15T08:00:10+00:00"

    wait = core_module._retry_wait_seconds(
        response,
        now=core_module.datetime(2026, 8, 15, 8, 0, 0, tzinfo=core_module.timezone.utc),
    )

    assert wait == 10.0


def test_core_ignores_unbounded_success_wait_and_uses_fallback(monkeypatch) -> None:
    clock = [100.0]
    sleeps: list[float] = []
    first = _response(200)
    first.headers["X-RateLimit-Retry-After"] = "3600"
    responses = iter([first, _response(200)])

    def advance(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(core_module, "_core_last_request", 0.0)
    monkeypatch.setattr(core_module, "_core_next_allowed_request", 0.0)
    monkeypatch.setattr(core_module, "_core_quota_interval", None)
    monkeypatch.setattr(core_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(core_module.time, "sleep", advance)
    monkeypatch.setattr(core_module.httpx, "get", lambda *args, **kwargs: next(responses))

    core_module._core_request("https://provider.example/query", {})
    core_module._core_request("https://provider.example/query", {})

    assert sleeps == [10.0]


def test_core_opens_bounded_timeout_circuit_and_cooldown(monkeypatch, tmp_path) -> None:
    request = Mock(side_effect=httpx.ReadTimeout("CORE stalled"))
    monkeypatch.setattr("app.services.retrieval.core._core_request", request)
    store = ProviderHealthStore(str(tmp_path / "health.json"))
    retriever = CoreRetriever(store)

    first = retriever.search_by_doi("10.1234/first")
    second = retriever.search_by_doi("10.1234/second")
    third = retriever.search_by_doi("10.1234/third")

    assert first.success is False
    assert second.success is False
    assert third.success is False
    assert request.call_count == 2
    assert retriever.provider_metrics["timeouts"] == 2
    assert retriever.provider_metrics["timeout_circuit_opens"] == 1
    assert retriever.provider_metrics["circuit_skips"] == 1
    assert retriever.provider_metrics["cooldown_seconds"] == 300
    assert store.cooldown_remaining("core") > 0


def test_provider_health_probe_lease_is_persistent_and_single_claim(tmp_path) -> None:
    path = tmp_path / "health.json"
    first = ProviderHealthStore(str(path))
    second = ProviderHealthStore(str(path))
    policy = ProviderPolicy(cooldown_seconds=0, max_cooldown_seconds=0)
    first.record_unavailable("searxng:google cse", policy, status="rate_limited")

    assert first.claim_recovery_probe("searxng:google cse", lease_seconds=60) is True
    assert second.claim_recovery_probe("searxng:google cse", lease_seconds=60) is False
    assert second.incident("searxng:google cse")["last_status"] == "rate_limited"


def test_provider_success_reports_recovery_once_and_clears_incident(tmp_path) -> None:
    store = ProviderHealthStore(str(tmp_path / "health.json"))
    policy = ProviderPolicy(cooldown_seconds=0, max_cooldown_seconds=0)
    store.record_timeout("searxng:google scholar", policy)

    assert store.record_success("searxng:google scholar") is True
    assert store.record_success("searxng:google scholar") is False
    assert store.incident("searxng:google scholar") is None


def test_core_opens_circuit_after_cumulative_intermittent_timeouts(
    monkeypatch, tmp_path
) -> None:
    success = _response(200)
    success._content = b'{"results": []}'
    request = Mock(
        side_effect=[
            httpx.ReadTimeout("first stall"),
            success,
            httpx.ReadTimeout("second stall"),
            success,
            httpx.ReadTimeout("third stall"),
        ]
    )
    monkeypatch.setattr("app.services.retrieval.core._core_request", request)
    retriever = CoreRetriever(ProviderHealthStore(str(tmp_path / "health.json")))

    for index in range(5):
        retriever.search_by_doi(f"10.1234/{index}")
    skipped = retriever.search_by_doi("10.1234/skipped")

    assert skipped.success is False
    assert request.call_count == 5
    assert retriever.provider_metrics["timeouts"] == 3
    assert retriever.provider_metrics["timeout_circuit_opens"] == 1
    assert retriever.provider_metrics["circuit_skips"] == 1


def test_semantic_scholar_opens_circuit_only_after_three_exhausted_calls(monkeypatch, tmp_path) -> None:
    # Random gateway 429s (measured 2026-09-25) are retried briefly; only three
    # calls in a row that stay refused after every retry open the circuit.
    request = Mock(return_value=_response(429))
    sleep = Mock()
    monkeypatch.setattr("app.services.retrieval.semantic_scholar.httpx.request", request)
    monkeypatch.setattr("app.services.retrieval.semantic_scholar._throttle", Mock())
    monkeypatch.setattr("app.services.retrieval.semantic_scholar.time.sleep", sleep)
    store = ProviderHealthStore(str(tmp_path / "health.json"))
    retriever = SemanticScholarRetriever(store)

    # A title search that stays refused is that reference's failed search only.
    first = retriever.search_by_title_author("A refused title")
    assert not first.success and "429" in first.error
    assert retriever.provider_metrics["rate_limited"] == 0
    assert store.cooldown_remaining("semantic_scholar") == 0
    # A batch gets a second retry round: two more exhausted calls, the third in a row.
    assert retriever.prefetch_dois(["10.1234/c"]) == 0
    assert retriever.provider_metrics["batch_retry_rounds"] == 1
    assert retriever.prefetch_dois(["10.1234/d"]) == 0

    assert request.call_count == 18                  # three calls of 1 + 5 retries; the last skipped
    assert [c.args[0] for c in sleep.call_args_list[:5]] == [1.5, 2.0, 3.0, 4.0, 5.0]
    assert retriever.provider_metrics["exhausted_calls"] == 3
    assert retriever.provider_metrics["rate_limited"] == 1
    assert retriever.provider_metrics["circuit_skips"] == 1
    assert retriever.provider_metrics["cooldown_seconds"] == 900
    assert store.cooldown_remaining("semantic_scholar") > 0
    failure = retriever.search_by_doi("10.1234/d")
    assert failure.metadata["prefetch_diagnostic"]["outcome"] == "rate_limited"
    assert not failure.success and "not prefetched" not in failure.error
    assert request.call_count == 18  # Reading the diagnostic never retries.


def test_semantic_scholar_success_resets_the_exhausted_count(monkeypatch, tmp_path) -> None:
    ok = _s2_batch_response([1])
    request = Mock(side_effect=[_response(429)] * 12 + [ok] + [_response(429)] * 12)
    monkeypatch.setattr("app.services.retrieval.semantic_scholar.httpx.request", request)
    monkeypatch.setattr("app.services.retrieval.semantic_scholar._throttle", Mock())
    monkeypatch.setattr("app.services.retrieval.semantic_scholar.time.sleep", Mock())
    retriever = SemanticScholarRetriever(ProviderHealthStore(str(tmp_path / "health.json")))
    retriever.prefetch_dois(["10.1/a"])                     # two exhausted rounds
    assert retriever.prefetch_dois(["10.1/1"]) == 1         # success resets the count
    retriever.prefetch_dois(["10.1/b"])                     # two more
    assert retriever.provider_metrics["exhausted_calls"] == 4
    assert retriever.provider_metrics["rate_limited"] == 0  # never three in a row


def test_semantic_scholar_batch_recovers_from_one_boundary_429(monkeypatch, tmp_path) -> None:
    payload = {
        "paperId": "abc123",
        "title": "Recovered paper",
        "year": 2024,
        "authors": [],
        "externalIds": {"DOI": "10.1234/example"},
        "openAccessPdf": None,
        "abstract": "Recovered abstract.",
    }
    recovered = _response(200)
    recovered._content = json.dumps([payload]).encode()
    request = Mock(side_effect=[_response(429), recovered])
    monkeypatch.setattr("app.services.retrieval.semantic_scholar.httpx.request", request)
    monkeypatch.setattr("app.services.retrieval.semantic_scholar._throttle", Mock())
    monkeypatch.setattr("app.services.retrieval.semantic_scholar.time.sleep", Mock())

    retriever = SemanticScholarRetriever(ProviderHealthStore(str(tmp_path / "health.json")))
    prefetched = retriever.prefetch_dois(["10.1234/example"])
    result = retriever.search_by_doi("10.1234/example")

    assert prefetched == 1
    assert result.success is True
    assert retriever.provider_metrics["calls"] == 2
    assert retriever.provider_metrics["rate_limit_retries"] == 1
    assert retriever.provider_metrics["rate_limited"] == 0


def _s2_search_response(rows, status_code: int = 200) -> httpx.Response:
    request = httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search")
    return httpx.Response(status_code, json={"data": rows, "total": len(rows)}, request=request)


def test_semantic_scholar_title_search_excludes_the_author_from_the_query(monkeypatch) -> None:
    """Adding the author collapses this API's recall, including on real works.

    Measured 2026-09-21: "The separation of platforms and commerce" returns 8
    results by title and 0 with "Khan" appended. The author is compared by the
    caller afterwards; it must never narrow the query.
    """
    request = Mock(return_value=_s2_search_response(
        [{"title": "Exact Example Source Title", "authors": [{"name": "A Author"}], "year": 2019}]
    ))
    monkeypatch.setattr("app.services.retrieval.semantic_scholar.httpx.request", request)
    monkeypatch.setattr("app.services.retrieval.semantic_scholar._throttle", Mock())

    result = SemanticScholarRetriever().search_by_title_author(
        "Exact Example Source Title", "Author"
    )

    assert result.success is True
    assert result.title == "Exact Example Source Title"
    sent_query = request.call_args.kwargs["params"]["query"]
    assert sent_query == "Exact Example Source Title"
    assert "Author" not in sent_query


def test_semantic_scholar_title_search_reports_an_empty_index_as_no_results(monkeypatch) -> None:
    monkeypatch.setattr("app.services.retrieval.semantic_scholar.httpx.request",
                        Mock(return_value=_s2_search_response([])))
    monkeypatch.setattr("app.services.retrieval.semantic_scholar._throttle", Mock())

    result = SemanticScholarRetriever().search_by_title_author("A missing work")

    assert result.success is False
    assert result.error == "No results"
    assert result.metadata["identity_search_result_count"] == 0


def test_semantic_scholar_declined_call_is_not_reported_as_a_failed_search() -> None:
    from app.services.source_resolver import _search_execution_outcome

    retriever = SemanticScholarRetriever()
    retriever._rate_limited = True
    result = retriever.search_by_title_author("Exact Example Source Title")

    assert result.success is False
    assert _search_execution_outcome(result) == "cooldown_skipped"


def test_semantic_scholar_prefetch_preserves_final_http_failure_after_retries(monkeypatch, tmp_path):
    request = Mock(side_effect=[_response(429), _response(429), _response(503)])
    monkeypatch.setattr('app.services.retrieval.semantic_scholar.httpx.request', request)
    monkeypatch.setattr('app.services.retrieval.semantic_scholar._throttle', Mock())
    monkeypatch.setattr('app.services.retrieval.semantic_scholar.time.sleep', Mock())
    adapter = SemanticScholarRetriever(ProviderHealthStore(str(tmp_path/'health.json')))
    assert adapter.prefetch_dois(['10.1234/test']) == 0
    result = adapter.search_by_doi('10.1234/test')
    assert result.metadata['prefetch_diagnostic']['outcome'] == 'http_503'
    assert not result.success and 'not prefetched' not in result.error
    assert adapter.provider_metrics['http_status:429'] == 2
    assert adapter.provider_metrics['http_status:503'] == 1
    assert adapter.provider_metrics['batch_failure:http_503'] == 1
    assert request.call_count == 3
    from app.services.source_resolver import _search_execution_outcome
    assert _search_execution_outcome(result) == 'operational_failure'
    assert 'provider.example' not in json.dumps(result.metadata)


def test_semantic_scholar_invalid_batch_is_not_a_successful_empty_search(monkeypatch, tmp_path):
    response = _response(200)
    response._content = b'{"private_message":"PRIVATE CONTENT"}'
    monkeypatch.setattr('app.services.retrieval.semantic_scholar.httpx.request', Mock(return_value=response))
    monkeypatch.setattr('app.services.retrieval.semantic_scholar._throttle', Mock())
    adapter = SemanticScholarRetriever(ProviderHealthStore(str(tmp_path/'health.json')))
    assert adapter.prefetch_dois(['10.1234/test']) == 0
    result = adapter.search_by_doi('10.1234/test')
    assert result.metadata['prefetch_diagnostic']['outcome'] == 'response_invalid'
    assert adapter.provider_metrics['http_status:200'] == 1
    assert adapter.provider_metrics['batch_failures'] == 1
    assert 'PRIVATE' not in str(result) and not result.success


def test_semantic_scholar_cooldown_failure_does_not_make_a_request(monkeypatch):
    store = Mock(); store.cooldown_remaining.return_value = 50
    request = Mock()
    monkeypatch.setattr('app.services.retrieval.semantic_scholar.httpx.request', request)
    adapter = SemanticScholarRetriever(store)
    assert adapter.prefetch_dois(['10.1234/test']) == 0
    assert adapter.search_by_doi('10.1234/test').metadata['prefetch_diagnostic']['outcome'] == 'cooldown_skipped'
    assert adapter.provider_metrics['calls'] == 0
    request.assert_not_called()


def test_semantic_scholar_batch_prefetch_populates_doi_cache(monkeypatch, tmp_path) -> None:
    payload = [{
        "paperId": "abc123", "title": "Batch paper", "year": 2024,
        "authors": [], "externalIds": {"DOI": "10.1234/example"},
        "openAccessPdf": None, "abstract": "Batch abstract.",
    }, None]
    response = _response(200)
    response._content = json.dumps(payload).encode()
    request = Mock(return_value=response)
    monkeypatch.setattr("app.services.retrieval.semantic_scholar.httpx.request", request)
    monkeypatch.setattr("app.services.retrieval.semantic_scholar._throttle", Mock())

    retriever = SemanticScholarRetriever(ProviderHealthStore(str(tmp_path / "health.json")))
    count = retriever.prefetch_dois(["10.1234/EXAMPLE", "10.1234/missing"])
    found = retriever.search_by_doi("https://doi.org/10.1234/example")
    missing = retriever.search_by_doi("10.1234/missing")

    assert count == 1
    assert found.success is True
    assert found.title == "Batch paper"
    assert missing.success is False
    assert request.call_count == 1
    assert request.call_args.args[:2] == (
        "POST", "https://api.semanticscholar.org/graph/v1/paper/batch"
    )
    assert request.call_args.kwargs["json"] == {
        "ids": ["DOI:10.1234/example", "DOI:10.1234/missing"]
    }
    assert retriever.provider_metrics["batch_calls"] == 1
    assert retriever.provider_metrics["batch_items"] == 2
    assert retriever.provider_metrics["batch_cache_hits"] == 2


def _s2_batch_response(ids):
    response = _response(200)
    response._content = json.dumps([
        {"paperId": str(i), "title": f"Paper {i}", "year": 2024, "authors": [],
         "externalIds": {"DOI": f"10.1/{i}"}, "openAccessPdf": None, "abstract": None}
        for i in ids
    ]).encode()
    return response


def test_semantic_scholar_batches_every_doi_in_chunks_of_one_hundred(monkeypatch, tmp_path) -> None:
    """Chunks of 5 capped at 5 batches left every DOI after the 25th unlooked-up."""
    request = Mock(side_effect=[_s2_batch_response(range(0, 100)), _s2_batch_response(range(100, 130))])
    monkeypatch.setattr("app.services.retrieval.semantic_scholar.httpx.request", request)
    monkeypatch.setattr("app.services.retrieval.semantic_scholar._throttle", Mock())

    retriever = SemanticScholarRetriever(ProviderHealthStore(str(tmp_path / "health.json")))
    assert retriever.prefetch_dois([f"10.1/{i}" for i in range(130)]) == 130
    assert [len(call.kwargs["json"]["ids"]) for call in request.call_args_list] == [100, 30]
    assert retriever.search_by_doi("10.1/129").success is True


def test_semantic_scholar_title_search_replaces_hyphens(monkeypatch) -> None:
    """The API documents that a hyphenated term matches nothing."""
    request = Mock(return_value=_s2_search_response(
        [{"title": "COVID-19 and self-efficacy", "authors": [], "year": 2021}]))
    monkeypatch.setattr("app.services.retrieval.semantic_scholar.httpx.request", request)
    monkeypatch.setattr("app.services.retrieval.semantic_scholar._throttle", Mock())
    SemanticScholarRetriever().search_by_title_author("COVID-19 and self-efficacy – a study", None)
    assert request.call_args.kwargs["params"]["query"] == "COVID 19 and self efficacy – a study"


def test_semantic_scholar_follows_a_bounded_retry_after(monkeypatch, tmp_path) -> None:
    limited = _response(429)
    limited.headers["Retry-After"] = "3"
    request = Mock(side_effect=[limited, _s2_batch_response([1])])
    sleep = Mock()
    monkeypatch.setattr("app.services.retrieval.semantic_scholar.httpx.request", request)
    monkeypatch.setattr("app.services.retrieval.semantic_scholar._throttle", Mock())
    monkeypatch.setattr("app.services.retrieval.semantic_scholar.time.sleep", sleep)
    retriever = SemanticScholarRetriever(ProviderHealthStore(str(tmp_path / "health.json")))
    assert retriever.prefetch_dois(["10.1/1"]) == 1
    sleep.assert_called_once_with(3.0)          # not the 15s policy delay


def test_semantic_scholar_cooldown_is_capped_at_an_hour() -> None:
    from app.services.retrieval.semantic_scholar import _DEFAULT_POLICY
    assert _DEFAULT_POLICY.max_cooldown_seconds == 3600 and _DEFAULT_POLICY.cooldown_seconds == 900


def test_semantic_scholar_paces_four_times_wider_without_shared_pacing(monkeypatch) -> None:
    from app.services.retrieval import semantic_scholar as s2
    clock = [100.0]; sleeps = []
    monkeypatch.setattr(s2, "_last_request_time", 99.0)
    monkeypatch.setattr(s2.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(s2.time, "sleep", lambda seconds: (sleeps.append(seconds), clock.__setitem__(0, clock[0] + seconds)))
    s2._throttle(1.1)                          # shared pacing is off in unit tests
    assert sleeps == [pytest.approx(1.1 * 4 - 1.0)]
    monkeypatch.setattr(s2.shared_pacing, "reserve_start", lambda provider, interval: 0.25)
    sleeps.clear(); clock[0] += 10
    s2._throttle(1.1)                          # shared reservation available: its wait, local interval 1.1
    assert sleeps == [0.25]


def test_semantic_scholar_batch_isolates_one_rejected_doi(monkeypatch, tmp_path) -> None:
    rejected = "DOI:10.1/bad"

    def response_for_request(*_args, **kwargs):
        ids = kwargs["json"]["ids"]
        if rejected in ids:
            response = _response(400)
            response._content = b'{"error":"bad paper id"}'
            return response
        response = _response(200)
        response._content = json.dumps(
            [
                {
                    "paperId": value,
                    "title": f"Paper {value}",
                    "year": 2024,
                    "authors": [],
                    "externalIds": {"DOI": value.removeprefix("DOI:")},
                    "openAccessPdf": None,
                }
                for value in ids
            ]
        ).encode()
        return response

    request = Mock(side_effect=response_for_request)
    monkeypatch.setattr("app.services.retrieval.semantic_scholar.httpx.request", request)
    monkeypatch.setattr("app.services.retrieval.semantic_scholar._throttle", Mock())

    retriever = SemanticScholarRetriever(
        ProviderHealthStore(str(tmp_path / "health.json"))
    )
    count = retriever.prefetch_dois(["10.1/good-a", "10.1/bad", "10.1/good-b"])

    assert count == 2
    assert retriever.search_by_doi("10.1/good-a").success is True
    assert retriever.search_by_doi("10.1/good-b").success is True
    assert retriever.search_by_doi("10.1/bad").success is False
    assert request.call_count == 5


def test_core_busy_in_another_worker_is_a_skip_not_a_request(monkeypatch) -> None:
    from contextlib import contextmanager
    from app.services.source_resolver import _search_execution_outcome

    @contextmanager
    def busy(*args, **kwargs):
        yield False

    get = Mock()
    monkeypatch.setattr(core_module.shared_pacing, "exclusive", busy)
    monkeypatch.setattr(core_module.httpx, "get", get)
    monkeypatch.setattr(core_module.settings, "CORE_API_KEY", SecretStr("test-key"))
    retriever = CoreRetriever()
    result = retriever.search_by_doi("10.1234/example")
    get.assert_not_called()
    assert retriever.provider_metrics["calls"] == 0 and retriever.provider_metrics["busy_skips"] == 1
    assert _search_execution_outcome(result) == "cooldown_skipped"
    assert "busy in another worker" in result.error


def test_core_shares_its_pacing_and_retry_after_with_other_workers(monkeypatch) -> None:
    clock = [100.0]; sleeps = []; deferred = []
    exhausted = _response(200)
    exhausted.headers["X-RateLimit-Remaining"] = "0"
    exhausted.headers["X-RateLimit-Retry-After"] = "5"
    monkeypatch.setattr(core_module, "_core_last_request", 0.0)
    monkeypatch.setattr(core_module, "_core_next_allowed_request", 0.0)
    monkeypatch.setattr(core_module, "_core_quota_interval", None)
    monkeypatch.setattr(core_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(core_module.time, "sleep", lambda seconds: (sleeps.append(seconds), clock.__setitem__(0, clock[0] + seconds)))
    monkeypatch.setattr(core_module.httpx, "get", lambda *args, **kwargs: exhausted)
    monkeypatch.setattr(core_module.shared_pacing, "reserve_start", lambda provider, interval: 4.0)
    monkeypatch.setattr(core_module.shared_pacing, "defer", lambda provider, seconds: deferred.append((provider, seconds)))
    core_module._core_request("https://provider.example/query", {}, min_interval=1)
    assert sleeps == [4.0]                      # another worker's reservation
    assert deferred == [("core", 5.0)]           # exhausted quota: every worker waits
