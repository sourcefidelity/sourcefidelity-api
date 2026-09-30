"""Synthetic response-contract tests; no retained provider output."""
import json
from unittest.mock import patch

import httpx
import pytest

from app.services.search.brave import BraveSearch

QUERY = '"frozen bibliographic control"'


def envelope(**extra):
    return {"type": "search", "query": {"original": QUERY, "more_results_available": False}, **extra}


def search(payload, status=200):
    response = httpx.Response(status, content=json.dumps(payload),
        request=httpx.Request("GET", "https://api.search.brave.com/res/v1/web/search"))
    provider = BraveSearch("test-key")
    with patch("app.services.search.brave.httpx.get", return_value=response) as get:
        results = provider.search(QUERY)
    assert get.call_count == 1
    return provider, results


@pytest.mark.parametrize("payload", [envelope(), envelope(web=None), envelope(web={"results": []})])
def test_documented_empty_web_is_a_completed_search(payload):
    provider, results = search(payload)
    assert results == [] and provider.last_status == "completed"
    assert provider.last_reason_code == "brave_web_v2_empty_web"


@pytest.mark.parametrize("description", [None, "", "synthetic snippet"])
def test_nullable_description_does_not_discard_valid_results(description):
    provider, results = search(envelope(web={"results":[{
        "title":"Synthetic source", "url":"https://example.org/source", "description":description}]}))
    assert provider.last_status == "completed" and len(results) == 1
    assert results[0].snippet == (description or "")


@pytest.mark.parametrize("payload", [
    {}, [], None, {"type":"search"}, {"type":"error","query":{"original":QUERY}},
    {"type":"search","query":{"original":"wrong query"}},
    envelope(error={"message":"PRIVATE_RESPONSE_SENTINEL"}),
    envelope(web={}), envelope(web=[]), envelope(web={"results":None}),
    envelope(web={"results":[None]}),
    envelope(web={"results":[{"title":"Source","url":" "}]}),
    envelope(web={"results":[{"url":"https://example.org"}]}),
    envelope(web={"results":[{"title":"Source","url":"https://example.org","description":123}]}),
    {"type":"search","query":{"original":QUERY,"more_results_available":True}},
    {"type":"search","query":{"original":QUERY,"more_results_available":True},"web":{"results":[]}},
])
def test_malformed_or_inconsistent_empty_response_fails_closed(payload, caplog):
    provider, results = search(payload)
    assert not results and provider.last_status == "response_invalid"
    assert provider.last_reason_code.startswith("brave_web_v2_")
    assert "PRIVATE_RESPONSE_SENTINEL" not in caplog.text
    assert "PRIVATE_RESPONSE_SENTINEL" not in json.dumps(vars(provider))


@pytest.mark.parametrize("status,outcome", [(401,"access_restricted"),(403,"access_restricted"),
    (429,"rate_limited"),(500,"operational_failure")])
def test_http_error_cannot_become_an_empty_success(status, outcome):
    provider, results = search(envelope(),status)
    assert not results and provider.last_status == outcome
    assert provider.last_reason_code == f"HTTP_{status}"


def test_reason_is_reset_between_requests():
    provider = BraveSearch("test-key")
    with patch("app.services.search.brave.httpx.get",side_effect=httpx.ReadTimeout("PRIVATE_RESPONSE_SENTINEL")):
        assert provider.search(QUERY) == []
    assert provider.last_status == "timeout" and provider.last_reason_code == "ReadTimeout"
    with patch("app.services.search.brave.httpx.get",return_value=httpx.Response(200,json=envelope(),request=httpx.Request("GET","https://example.org"))):
        assert provider.search(QUERY) == []
    assert provider.last_status == "completed" and provider.last_reason_code == "brave_web_v2_empty_web"


def test_response_contract_change_invalidates_lookup_freshness(monkeypatch):
    from app.services.source_resolver import SourceResolver
    resolver = SourceResolver.__new__(SourceResolver)
    current = resolver._lookup_policy_signature()
    monkeypatch.setattr("app.services.search.brave.BRAVE_RESPONSE_CONTRACT", "brave-web-response-v1")
    assert resolver._lookup_policy_signature() != current


def test_exact_search_settings_are_sent():
    """Spellcheck would search a corrected phrase instead of the title we sent."""
    response = httpx.Response(200, json=envelope(), request=httpx.Request("GET", "https://example.org"))
    with patch("app.services.search.brave.httpx.get", return_value=response) as get:
        BraveSearch("test-key").search(QUERY)
    params = get.call_args.kwargs["params"]
    assert params["spellcheck"] == "false" and params["text_decorations"] == "false"
    assert params["result_filter"] == "web,news"


def test_news_results_follow_web_results_without_duplicates():
    provider, results = search(envelope(
        web={"results": [{"title": "Paper", "url": "https://example.org/paper"}]},
        news={"type": "news", "results": [{"title": "Story", "url": "https://news.example/story"},
                                          {"title": "Paper again", "url": "https://example.org/paper"}]}))
    assert [r.url for r in results] == ["https://example.org/paper", "https://news.example/story"]
    assert provider.last_status == "completed"


def test_news_only_answer_is_not_an_empty_search():
    provider, results = search(envelope(news={"results": [{"title": "Story", "url": "https://news.example/s"}]}))
    assert len(results) == 1 and provider.last_reason_code == "brave_web_v2_results"


def test_an_altered_query_is_never_a_certified_empty_search():
    altered = {"type": "search", "query": {"original": QUERY, "altered": "frozen bibliographic controls",
                                           "more_results_available": False}}
    provider, results = search(altered)
    assert not results and provider.last_status == "response_invalid"
    provider, results = search({**altered, "web": {"results": [{"title": "Lead", "url": "https://example.org/l"}]}})
    assert len(results) == 1 and provider.last_reason_code == "brave_web_v3_query_altered"


def test_case_and_spacing_of_the_query_echo_do_not_reject_a_valid_response():
    # Brave echoed a Cyrillic query lowercased (2026-09-25).
    payload = envelope(web={"results": [{"url": "https://a.example/x", "title": "X", "description": ""}]})
    payload["query"]["original"] = "  " + QUERY.upper() + " "
    provider, results = search(payload)
    assert len(results) == 1 and provider.last_status == "completed"


def test_a_different_query_echo_still_fails_closed():
    payload = envelope(web={"results": []})
    payload["query"]["original"] = '"another query"'
    provider, results = search(payload)
    assert results == [] and provider.last_status != "completed"


def test_a_served_request_records_the_list_price(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "BRAVE_USD_PER_REQUEST", 0.005)
    provider, _ = search(envelope(web={"results": []}))
    assert provider.last_cost_usd == pytest.approx(0.005)
    provider, _ = search({"type": "not-search"})            # served, then rejected by our contract
    assert provider.last_cost_usd == pytest.approx(0.005)


def test_an_error_status_or_no_price_records_no_cost(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "BRAVE_USD_PER_REQUEST", 0.005)
    provider, _ = search({}, status=429)
    assert provider.last_cost_usd is None
    monkeypatch.setattr(settings, "BRAVE_USD_PER_REQUEST", None)
    provider, _ = search(envelope(web={"results": []}))
    assert provider.last_cost_usd is None
