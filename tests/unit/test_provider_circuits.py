from unittest.mock import Mock
import json

import httpx

from app.services.retrieval.core import CoreRetriever
from app.services.retrieval import core as core_module
from app.services.retrieval.openalex import OpenAlexRetriever
from app.services.retrieval.provider_runtime import ProviderHealthStore, ProviderPolicy
from app.services.retrieval.semantic_scholar import SemanticScholarRetriever
from app.services.search.tavily import TavilySearch
from app.services.search.duckduckgo import DuckDuckGoSearch
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


def test_direct_duckduckgo_challenge_is_not_reported_as_completed_no_results(
    monkeypatch,
) -> None:
    response = _response(200)
    response._content = b'<html><form id="challenge-form">not a Robot</form></html>'
    monkeypatch.setattr(
        "app.services.search.duckduckgo.httpx.post", Mock(return_value=response)
    )
    provider = DuckDuckGoSearch()

    assert provider.search("bounded query") == []
    assert provider.last_status == "captcha"


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
    monkeypatch.setattr(settings, "OPENALEX_API_KEY", secret)
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
    monkeypatch.setattr(settings, "OPENALEX_API_KEY", "configured-test-key")
    request = Mock(return_value=_response(200))
    monkeypatch.setattr("app.services.retrieval.openalex.httpx.get", request)

    okay, detail = OpenAlexRetriever().preflight(require_api_key=True)

    assert okay is True
    assert "configured key" in detail
    request.assert_called_once()


def test_openalex_grouped_doi_prefetch_populates_cache(monkeypatch) -> None:
    monkeypatch.setattr(settings, "OPENALEX_API_KEY", "configured-test-key")
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
    monkeypatch.setattr(settings, "OPENALEX_API_KEY", "configured-test-key")
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
    monkeypatch.setattr(settings, "OPENALEX_API_KEY", "configured-test-key")
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


def test_core_success_header_paces_the_next_shared_request(monkeypatch) -> None:
    clock = [100.0]
    sleeps: list[float] = []
    first = _response(200)
    first.headers["X-RateLimit-Retry-After"] = "7"
    responses = iter([first, _response(200)])

    def advance(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(core_module, "_core_last_request", 0.0)
    monkeypatch.setattr(core_module, "_core_next_allowed_request", 0.0)
    monkeypatch.setattr(core_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(core_module.time, "sleep", advance)
    monkeypatch.setattr(core_module.httpx, "get", lambda *args, **kwargs: next(responses))

    core_module._core_request("https://provider.example/query", {}, min_interval=1)
    core_module._core_request("https://provider.example/query", {}, min_interval=1)

    assert sleeps == [7.0]


def test_core_uses_ten_second_fallback_without_valid_header(monkeypatch) -> None:
    clock = [100.0]
    sleeps: list[float] = []
    responses = iter([_response(200), _response(200)])

    def advance(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(core_module, "_core_last_request", 0.0)
    monkeypatch.setattr(core_module, "_core_next_allowed_request", 0.0)
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


def test_semantic_scholar_opens_circuit_and_persists_cooldown(monkeypatch, tmp_path) -> None:
    request = Mock(side_effect=[_response(429), _response(429), _response(429)])
    monkeypatch.setattr("app.services.retrieval.semantic_scholar.httpx.request", request)
    monkeypatch.setattr("app.services.retrieval.semantic_scholar._throttle", Mock())
    monkeypatch.setattr("app.services.retrieval.semantic_scholar.time.sleep", Mock())
    store = ProviderHealthStore(str(tmp_path / "health.json"))
    retriever = SemanticScholarRetriever(store)

    first = retriever.prefetch_dois(["10.1234/example"])
    second = retriever.prefetch_dois(["10.1234/other"])

    assert first == 0
    assert second == 0
    assert request.call_count == 3
    assert retriever.provider_metrics["rate_limit_retries"] == 2
    assert retriever.provider_metrics["rate_limited"] == 1
    assert retriever.provider_metrics["circuit_skips"] == 1
    assert retriever.provider_metrics["cooldown_seconds"] == 900
    assert store.cooldown_remaining("semantic_scholar") > 0


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


def test_semantic_scholar_title_lookup_is_disabled() -> None:
    result = SemanticScholarRetriever().search_by_title_author(
        "Exact Example Source Title", "Author"
    )

    assert result.success is False
    assert "DOI-only" in result.error


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


def test_semantic_scholar_batch_prefetch_chunks_conservatively(monkeypatch, tmp_path) -> None:
    responses = []
    for start in (0, 5):
        response = _response(200)
        response._content = json.dumps([
            {
                "paperId": str(i), "title": f"Paper {i}", "year": 2024,
                "authors": [], "externalIds": {"DOI": f"10.1/{i}"},
                "openAccessPdf": None, "abstract": None,
            }
            for i in range(start, min(start + 5, 7))
        ]).encode()
        responses.append(response)
    request = Mock(side_effect=responses)
    monkeypatch.setattr("app.services.retrieval.semantic_scholar.httpx.request", request)
    monkeypatch.setattr("app.services.retrieval.semantic_scholar._throttle", Mock())

    retriever = SemanticScholarRetriever(ProviderHealthStore(str(tmp_path / "health.json")))
    assert retriever.prefetch_dois([f"10.1/{i}" for i in range(7)]) == 7
    assert request.call_count == 2
    assert [len(call.kwargs["json"]["ids"]) for call in request.call_args_list] == [5, 2]


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
