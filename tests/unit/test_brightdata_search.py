"""Bright Data SERP adapter contract.

The adapter exists because web results become negative evidence about a
student's reference, so these tests fix the boundary between "the engine
found nothing" and "this request did not complete".
"""
import json
from unittest.mock import Mock

import httpx
import pytest

from pydantic import SecretStr
from app.services.search import get_search_provider
from app.services.search.brightdata import (
    _UNBILLED_RETRY_ATTEMPTS,
    BrightDataSearch,
)


def _response(status_code: int, body: object) -> httpx.Response:
    request = httpx.Request("POST", "https://api.brightdata.com/request")
    response = httpx.Response(status_code, request=request)
    response._content = json.dumps(body).encode()
    return response


@pytest.fixture(autouse=True)
def _no_real_backoff(monkeypatch):
    """Retry pauses are asserted by the tests that care, never waited out."""
    monkeypatch.setattr("app.services.search.brightdata.time.sleep", lambda _: None)


def _patched(monkeypatch, response):
    post = Mock(return_value=response)
    monkeypatch.setattr("app.services.search.brightdata.httpx.post", post)
    return post


def test_organic_results_are_returned_with_titles_and_urls(monkeypatch) -> None:
    _patched(monkeypatch, _response(200, {"organic": [
        {"title": "The Classical Hollywood Cinema", "link": "https://example.org/a",
         "description": "film style"},
        {"title": "A paper", "link": "https://example.org/b.pdf"},
    ]}))
    provider = BrightDataSearch("token", "zone")

    results = provider.search('"The Classical Hollywood Cinema" Bordwell')

    assert [r.url for r in results] == ["https://example.org/a", "https://example.org/b.pdf"]
    assert results[0].title == "The Classical Hollywood Cinema"
    assert results[1].is_pdf is True
    assert provider.last_status == "completed"


def test_alternate_results_key_is_accepted(monkeypatch) -> None:
    _patched(monkeypatch, _response(200, {"results": [
        {"title": "T", "url": "https://example.org/c"}]}))

    assert BrightDataSearch("token", "zone").search("q")[0].url == "https://example.org/c"


def test_query_and_zone_are_sent_without_engine_markup_parsing(monkeypatch) -> None:
    post = _patched(monkeypatch, _response(200, {"organic": []}))
    BrightDataSearch("token", "zone-name", engine="google").search("antitrust paradox")

    body = post.call_args.kwargs["json"]
    assert body["zone"] == "zone-name"
    # brd_json asks the provider to parse; the adapter never reads engine HTML.
    assert "brd_json=1" in body["url"]
    assert "antitrust+paradox" in body["url"]
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer token"


def test_provider_query_rewriting_is_explicitly_disabled(monkeypatch) -> None:
    """The query executed must be the query recorded.

    Bright Data can rewrite "some queries ... while others are sent
    unchanged". A completed empty search may become negative evidence about a
    student's reference, so the request disables rewriting rather than relying
    on a console default.
    """
    post = _patched(monkeypatch, _response(200, {"organic": []}))
    BrightDataSearch("token", "zone").search('"An Exact Title" Author 2006')

    body = post.call_args.kwargs["json"]
    assert body["search_rewrite"] is False
    assert "An+Exact+Title" in body["url"]


@pytest.mark.parametrize("body, reason", [
    ({"status": "blocked"}, "brightdata_serp_v1_missing_organic"),
    ({"error": "zone suspended"}, "brightdata_serp_v1_error_envelope"),
    ({"organic": [{"title": "no link"}]}, "brightdata_serp_v1_invalid_results"),
    ([], "brightdata_serp_v1_invalid_envelope"),
])
def test_malformed_bodies_are_failures_not_empty_searches(monkeypatch, body, reason) -> None:
    _patched(monkeypatch, _response(200, body))
    provider = BrightDataSearch("token", "zone")

    assert provider.search("q") == []
    assert provider.last_status != "completed"
    assert provider.last_reason_code == reason


def test_stated_zero_results_is_a_certified_empty_search(monkeypatch) -> None:
    """A genuinely empty result set omits `organic` and states the count.

    Observed live: a phrase with no matches returns `general.results_cnt == 0`
    and no organic block. Treating that as a failure would leave this provider
    unable to support any finding that rests on a completed empty search.
    """
    _patched(monkeypatch, _response(200, {
        "general": {"results_cnt": 0}, "input": {}, "spelling": {}}))
    provider = BrightDataSearch("token", "zone")

    assert provider.search("q") == []
    assert provider.last_status == "completed"
    assert provider.last_reason_code == "brightdata_serp_v1_zero_results"


def test_missing_organic_without_a_stated_count_is_still_a_failure(
    monkeypatch,
) -> None:
    _patched(monkeypatch, _response(200, {"general": {}, "input": {}}))
    provider = BrightDataSearch("token", "zone")

    assert provider.search("q") == []
    assert provider.last_status != "completed"
    assert provider.last_reason_code == "brightdata_serp_v1_missing_organic"


def test_nonzero_count_without_organic_is_a_failure(monkeypatch) -> None:
    """Results were reported to exist but none arrived: not an empty search."""
    _patched(monkeypatch, _response(200, {"general": {"results_cnt": 9920}}))
    provider = BrightDataSearch("token", "zone")

    assert provider.search("q") == []
    assert provider.last_reason_code == "brightdata_serp_v1_missing_organic"


def test_upstream_failure_behind_transport_200_is_not_an_empty_search(
    monkeypatch,
) -> None:
    """Bright Data returns transport 200 with the real outcome in a header.

    Observed live: HTTP 200, zero-length body, `x-brd-status-code: 502`.
    Read as a completed empty search that would become false evidence that a
    student's reference could not be found anywhere.
    """
    response = _response(200, {})
    response._content = b""
    response.headers["x-brd-status-code"] = "502"
    _patched(monkeypatch, response)
    provider = BrightDataSearch("token", "zone")

    assert provider.search("q") == []
    assert provider.last_status != "completed"
    assert provider.last_reason_code == "brightdata_serp_v1_upstream_status_502"


def test_empty_body_without_upstream_header_is_also_refused(monkeypatch) -> None:
    response = _response(200, {})
    response._content = b""
    _patched(monkeypatch, response)
    provider = BrightDataSearch("token", "zone")

    assert provider.search("q") == []
    assert provider.last_reason_code == "brightdata_serp_v1_empty_body"


def test_upstream_success_header_is_accepted(monkeypatch) -> None:
    response = _response(200, {"organic": []})
    response.headers["x-brd-status-code"] = "200"
    _patched(monkeypatch, response)
    provider = BrightDataSearch("token", "zone")

    assert provider.search("q") == []
    assert provider.last_status == "completed"
    assert provider.last_reason_code == "brightdata_serp_v1_empty"


def test_unsupported_page_size_parameter_is_not_sent(monkeypatch) -> None:
    """`num` is rejected by the SERP API and stripped with a warning."""
    post = _patched(monkeypatch, _response(200, {"organic": []}))
    BrightDataSearch("token", "zone").search("q", num_results=5)

    assert "num=" not in post.call_args.kwargs["json"]["url"]


def test_http_error_is_classified_and_query_is_not_logged(monkeypatch, caplog) -> None:
    _patched(monkeypatch, _response(429, {"organic": []}))
    provider = BrightDataSearch("token", "zone")

    with caplog.at_level("WARNING"):
        assert provider.search("private bibliographic query") == []

    assert provider.last_status == "rate_limited"
    assert "private bibliographic query" not in caplog.text


def test_factory_requires_both_token_and_zone(monkeypatch) -> None:
    from app.config import settings
    monkeypatch.setattr(settings, "BRIGHTDATA_API_TOKEN", SecretStr("token"))
    monkeypatch.setattr(settings, "BRIGHTDATA_SERP_ZONE", None)
    assert get_search_provider("brightdata") is None

    monkeypatch.setattr(settings, "BRIGHTDATA_SERP_ZONE", "zone")
    assert isinstance(get_search_provider("brightdata"), BrightDataSearch)


def test_duckduckgo_is_refused_by_name(monkeypatch) -> None:
    """A stale configuration must fail loudly, not resolve to a scraper."""
    assert get_search_provider("duckduckgo") is None
    with pytest.raises(ModuleNotFoundError):
        import app.services.search.duckduckgo  # noqa: F401


# --- Locale pinning and unbilled retry (September 23) -----------------------
#
# A full paper completed 13 of 42 queries before these. The engine and the
# zone each inferred a locale, and an upstream 502 arriving under transport
# 200 ended the request on the first try.


def test_locale_is_pinned_in_the_query_and_the_collection_region(monkeypatch) -> None:
    post = _patched(monkeypatch, _response(200, {"organic": []}))

    BrightDataSearch("token", "zone", language="en", region="us").search("q")

    body = post.call_args.kwargs["json"]
    assert "hl=en" in body["url"] and "gl=us" in body["url"]
    # Pinning the collection region too, so the zone cannot auto-select a
    # different one between two runs of the same query.
    assert body["country"] == "us"


def test_bing_receives_its_own_locale_parameter_names(monkeypatch) -> None:
    post = _patched(monkeypatch, _response(200, {"organic": []}))

    BrightDataSearch("token", "zone", engine="bing", language="de", region="de").search("q")

    url = post.call_args.kwargs["json"]["url"]
    assert "setlang=de" in url and "cc=de" in url


def test_upstream_gateway_failure_is_retried_and_can_succeed(monkeypatch) -> None:
    failed = _response(200, {})
    failed.headers["x-brd-status-code"] = "502"
    ok = _response(200, {"organic": [{"title": "T", "link": "https://example.org/a"}]})
    post = Mock(side_effect=[failed, ok])
    monkeypatch.setattr("app.services.search.brightdata.httpx.post", post)

    provider = BrightDataSearch("token", "zone")
    results = provider.search("q")

    assert [r.url for r in results] == ["https://example.org/a"]
    assert post.call_count == 2
    assert provider.last_status == "completed"


def test_retry_is_bounded_and_a_persistent_failure_is_not_an_empty_search(monkeypatch) -> None:
    failed = _response(200, {})
    failed.headers["x-brd-status-code"] = "502"
    post = _patched(monkeypatch, failed)

    provider = BrightDataSearch("token", "zone")
    assert provider.search("q") == []

    assert post.call_count == 3
    # The distinction the application depends on: this did not complete, so
    # it may not stand as evidence that the engine found nothing.
    assert provider.last_status != "completed"
    assert provider.last_reason_code == "brightdata_serp_v1_upstream_status_502"


def test_a_contract_failure_is_not_retried(monkeypatch) -> None:
    post = _patched(monkeypatch, _response(200, {"unexpected": 1}))

    provider = BrightDataSearch("token", "zone")
    assert provider.search("q") == []

    # A malformed body will not become well-formed; retrying only adds latency.
    assert post.call_count == 1
    assert provider.last_reason_code == "brightdata_serp_v1_missing_organic"


def test_a_certified_empty_search_is_not_retried(monkeypatch) -> None:
    post = _patched(monkeypatch, _response(200, {"general": {"results_cnt": 0}}))

    provider = BrightDataSearch("token", "zone")
    assert provider.search("q") == []

    assert post.call_count == 1
    assert provider.last_status == "completed"
    assert provider.last_reason_code == "brightdata_serp_v1_zero_results"


def test_a_rate_limit_is_not_retried_in_search(monkeypatch) -> None:
    """Measured: retrying 429s took completion from 88.1% to 78.6%.

    The limit is account-level, so a pause measured in seconds cannot clear
    it and the attempts only add to the rate that caused it. Surfacing it
    immediately lets the health store suspend the provider instead.
    """
    limited = _response(200, {})
    limited.headers["x-brd-status-code"] = "429"
    post = _patched(monkeypatch, limited)

    provider = BrightDataSearch("token", "zone")
    assert provider.search("q") == []

    assert post.call_count == 1
    assert provider.last_status == "rate_limited"


def test_a_gateway_fault_is_still_retried_after_a_pause(monkeypatch) -> None:
    failed = _response(200, {})
    failed.headers["x-brd-status-code"] = "502"
    ok = _response(200, {"organic": [{"title": "T", "link": "https://example.org/a"}]})
    slept: list[float] = []
    monkeypatch.setattr("app.services.search.brightdata.time.sleep", slept.append)
    monkeypatch.setattr("app.services.search.brightdata.httpx.post",
                        Mock(side_effect=[failed, ok]))

    assert BrightDataSearch("token", "zone").search("q")[0].url == "https://example.org/a"
    assert slept == [1.0]


def test_a_retry_that_would_outlive_the_deadline_is_not_attempted(monkeypatch) -> None:
    limited = _response(200, {})
    limited.headers["x-brd-status-code"] = "502"
    post = _patched(monkeypatch, limited)
    monkeypatch.setattr("app.services.search.brightdata.time.sleep",
                        Mock(side_effect=AssertionError("must not pause")))
    # Less time left than the first pause would consume.
    monkeypatch.setattr("app.services.search.brightdata.remaining", lambda _: 0.5)

    provider = BrightDataSearch("token", "zone")
    assert provider.search("q") == []

    assert post.call_count == 1
    # The observed failure, not a generic exhaustion code, and not "completed".
    assert provider.last_reason_code == "brightdata_serp_v1_upstream_status_502"
    assert provider.last_status != "completed"


def test_a_rate_limit_is_reported_as_one_not_as_a_malformed_body(monkeypatch) -> None:
    """The cooldown machinery keys on the status, so it has to be the truth.

    Bright Data answers a rate limit with transport 200 and the real code in a
    header. Classifying that by the body's shape gives `response_invalid` -- a
    malformed answer -- and `WebSearchRetriever` never suspends the provider.
    """
    limited = _response(200, {})
    limited.headers["x-brd-status-code"] = "429"
    _patched(monkeypatch, limited)

    provider = BrightDataSearch("token", "zone")
    assert provider.search("q") == []

    assert provider.last_status == "rate_limited"


def test_an_authorization_failure_is_reported_as_access_restricted(monkeypatch) -> None:
    denied = _response(200, {})
    denied.headers["x-brd-status-code"] = "403"
    _patched(monkeypatch, denied)

    provider = BrightDataSearch("token", "zone")
    assert provider.search("q") == []

    assert provider.last_status == "access_restricted"


def test_a_genuinely_malformed_body_is_still_response_invalid(monkeypatch) -> None:
    _patched(monkeypatch, _response(200, {"unexpected": 1}))

    provider = BrightDataSearch("token", "zone")
    assert provider.search("q") == []

    assert provider.last_status == "response_invalid"


def test_the_configured_timeout_bounds_the_whole_search_not_each_attempt(monkeypatch) -> None:
    """Three attempts must not cost three times the configured timeout."""
    clock = iter([0.0, 0.0, 25.0, 25.0, 25.0, 50.0, 50.0, 50.0, 75.0, 75.0])
    monkeypatch.setattr("app.services.search.brightdata.time.monotonic",
                        lambda: next(clock))
    failed = _response(200, {})
    failed.headers["x-brd-status-code"] = "502"
    post = _patched(monkeypatch, failed)

    provider = BrightDataSearch("token", "zone", timeout_seconds=60.0)
    assert provider.search("q") == []

    # Each attempt gets what is left of the 60s, never a fresh 60s.
    timeouts = [c.kwargs["timeout"] for c in post.call_args_list]
    assert timeouts[0] == 60.0
    assert all(t < 60.0 for t in timeouts[1:])
    assert sum(1 for _ in timeouts) <= _UNBILLED_RETRY_ATTEMPTS


def test_a_search_with_no_budget_left_reports_that_rather_than_no_results(monkeypatch) -> None:
    clock = iter([0.0, 100.0])
    monkeypatch.setattr("app.services.search.brightdata.time.monotonic",
                        lambda: next(clock))
    post = _patched(monkeypatch, _response(200, {"organic": []}))

    provider = BrightDataSearch("token", "zone", timeout_seconds=60.0)
    assert provider.search("q") == []

    assert post.call_count == 0
    assert provider.last_status != "completed"
    assert provider.last_reason_code == "brightdata_serp_v1_no_budget_remaining"


# ---- spend records (2026-09-25) ------------------------------------------------

@pytest.fixture
def priced(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "BRIGHTDATA_USD_PER_REQUEST", 0.0015)


def test_a_delivered_request_records_its_price(monkeypatch, priced) -> None:
    _patched(monkeypatch, _response(200, {"organic": [{"title": "T", "link": "https://example.org/a"}]}))
    provider = BrightDataSearch("token", "zone")
    provider.search("q")
    assert provider.last_cost_usd == pytest.approx(0.0015)


def test_retried_failures_are_free_and_one_delivery_is_billed_once(monkeypatch, priced) -> None:
    ok = _response(200, {"organic": []})
    post = Mock(side_effect=[_response(502, {}), ok])
    monkeypatch.setattr("app.services.search.brightdata.httpx.post", post)
    provider = BrightDataSearch("token", "zone")
    provider.search("q")
    assert post.call_count == 2 and provider.last_cost_usd == pytest.approx(0.0015)


def test_an_undelivered_search_costs_nothing_recorded(monkeypatch, priced) -> None:
    _patched(monkeypatch, _response(401, {}))
    provider = BrightDataSearch("token", "zone")
    provider.search("q")
    assert provider.last_status != "completed" and provider.last_cost_usd is None


def test_a_delivered_but_unparseable_body_is_still_billed(monkeypatch, priced) -> None:
    _patched(monkeypatch, _response(200, {"status": "blocked"}))
    provider = BrightDataSearch("token", "zone")
    provider.search("q")
    assert provider.last_status != "completed" and provider.last_cost_usd == pytest.approx(0.0015)


def test_without_a_price_the_cost_is_unknown_not_zero(monkeypatch) -> None:
    from app.config import settings
    monkeypatch.setattr(settings, "BRIGHTDATA_USD_PER_REQUEST", None)
    _patched(monkeypatch, _response(200, {"organic": []}))
    provider = BrightDataSearch("token", "zone")
    provider.search("q")
    assert provider.last_cost_usd is None
