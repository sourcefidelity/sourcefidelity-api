"""Request shapes the Tavily and Exa adapters send (documentation review 2026-09-25)."""
from unittest.mock import Mock

import httpx
import pytest

from app.config import Settings, settings
from app.services.retrieval.web_search import WebSearchRetriever
from app.services.search.exa import ExaSearch
from app.services.search.tavily import TavilySearch


def _json(status_code: int, body: dict) -> httpx.Response:
    return httpx.Response(status_code, json=body, request=httpx.Request("POST", "https://provider.example/search"))


def _tavily(monkeypatch, body=None, status_code=200):
    post = Mock(return_value=_json(status_code, body if body is not None else {"results": []}))
    monkeypatch.setattr("app.services.search.tavily.httpx.post", post)
    return TavilySearch("test-key"), post


def test_tavily_authenticates_by_header_not_body(monkeypatch):
    provider, post = _tavily(monkeypatch)
    provider.search("frozen control")
    kwargs = post.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert "api_key" not in kwargs["json"]
    assert kwargs["json"]["include_usage"] is True
    assert kwargs["json"]["search_depth"] == "basic"


def test_tavily_sends_no_exact_match(monkeypatch):
    # exact_match lost 7 of 40 real references and gained 2 (2026-09-25).
    provider, post = _tavily(monkeypatch)
    provider.search('"Frozen source control" Writer 2020')
    assert "exact_match" not in post.call_args.kwargs["json"]


def test_tavily_433_opens_run_level_circuit(monkeypatch):
    provider, post = _tavily(monkeypatch, {}, status_code=433)
    assert provider.search("first") == [] and provider.last_status == "rate_limited"
    assert provider.search("second") == []
    post.assert_called_once()


def test_tavily_cost_needs_a_configured_credit_price(monkeypatch):
    body = {"results": [{"url": "https://a.example/x", "title": "X", "content": ""}], "usage": {"credits": 1}}
    monkeypatch.setattr(settings, "TAVILY_USD_PER_CREDIT", None)
    provider, _ = _tavily(monkeypatch, body)
    assert len(provider.search("q")) == 1
    assert provider.last_status == "completed" and provider.last_cost_usd is None   # unpriced, never 0
    monkeypatch.setattr(settings, "TAVILY_USD_PER_CREDIT", 0.008)
    provider.search("q")
    assert provider.last_cost_usd == pytest.approx(0.008)


def _exa(monkeypatch):
    post = Mock(return_value=_json(200, {"results": []}))
    monkeypatch.setattr("app.services.search.exa.httpx.post", post)
    return ExaSearch("test-key"), post


def test_exa_sends_no_category(monkeypatch):
    # The "publication" category lost 2 of 16 real articles and gained none
    # on the labelled corpus (2026-09-25).
    provider, post = _exa(monkeypatch)
    provider.search("frozen control")
    assert "category" not in post.call_args.kwargs["json"]


@pytest.mark.parametrize(("fallback", "searx_enabled", "expected"), [
    ("tavily", False, {"brave", "exa", "tavily"}),
    ("searxng", True, {"brave", "exa", "searxng"}),
    ("searxng", False, {"brave", "exa"}),
    ("none", True, {"brave", "exa"}),
])
def test_configured_web_fallback_provider(monkeypatch, fallback, searx_enabled, expected):
    from app.services.search.policy import API_FIRST_SEARCH_POLICY
    monkeypatch.setattr(settings, "SEARCH_POLICY_VERSION", API_FIRST_SEARCH_POLICY)
    monkeypatch.setattr(settings, "SEARCH_WEB_FALLBACK_PROVIDER", fallback)
    monkeypatch.setattr(settings, "SEARCH_SEARXNG_FALLBACK_ENABLED", searx_enabled)
    monkeypatch.setattr("app.services.retrieval.web_search.get_search_provider",
                        lambda name=None: Mock(name=name))
    search = WebSearchRetriever(health_store=Mock())
    assert set(search._policy_providers) == expected


def test_filetype_variant_is_sent_only_to_engines_with_the_operator(monkeypatch):
    from app.services.search.policy import API_FIRST_SEARCH_POLICY
    monkeypatch.setattr(settings, "SEARCH_POLICY_VERSION", API_FIRST_SEARCH_POLICY)
    monkeypatch.setattr(settings, "SEARCH_WEB_FALLBACK_PROVIDER", "tavily")
    monkeypatch.setattr(settings, "SEARCH_ESCALATION_MAX_CALLS", "brave:4,exa:4,tavily:4")
    monkeypatch.setattr(settings, "BRAVE_SEARCH_RETENTION_PERMITTED", True)
    monkeypatch.setattr(settings, "BRAVE_SEARCH_TRANSIENT_ENABLED", True)
    seen = []
    providers = {}
    for name in ("brave", "exa", "tavily"):
        provider = Mock(last_status="completed", last_cost_usd=None)
        provider.name = name
        provider.search.side_effect = lambda query, *a, n=name, **kw: seen.append((n, query)) or []
        providers[name] = provider
    monkeypatch.setattr("app.services.retrieval.web_search.get_search_provider",
                        lambda name=None: providers.get(name))
    WebSearchRetriever(health_store=Mock()).search_reference(
        doi=None, title="Frozen source control", author="Writer", year="2020")
    assert any("filetype:" in q for n, q in seen if n == "brave")
    assert all("filetype:" not in q for n, q in seen if n in {"exa", "tavily"})
    assert {n for n, _ in seen} >= {"brave", "exa"}


@pytest.mark.parametrize("status", [401, 402])
def test_exa_account_refusal_stops_calls_for_the_run(monkeypatch, status):
    post = Mock(return_value=_json(status, {"tag": "NO_MORE_CREDITS"}))
    monkeypatch.setattr("app.services.search.exa.httpx.post", post)
    provider = ExaSearch("test-key")
    assert provider.search("first") == [] and provider.last_status != "completed"
    assert provider.search("second") == [] and provider.last_status != "completed"
    post.assert_called_once()


def test_exa_content_refusal_does_not_stop_the_run(monkeypatch):
    post = Mock(return_value=_json(403, {"tag": "PROHIBITED_CONTENT"}))
    monkeypatch.setattr("app.services.search.exa.httpx.post", post)
    provider = ExaSearch("test-key")
    provider.search("first"); provider.search("second")
    assert post.call_count == 2


def test_exa_overload_is_retried_once(monkeypatch):
    post = Mock(side_effect=[_json(503, {"tag": "SERVICE_OVERLOADED"}), _json(200, {"results": []})])
    monkeypatch.setattr("app.services.search.exa.httpx.post", post)
    monkeypatch.setattr("app.services.search.exa.time.sleep", Mock())
    provider = ExaSearch("test-key")
    assert provider.search("q") == [] and provider.last_status == "completed"
    assert post.call_count == 2


@pytest.mark.parametrize("raw", ["$0.008", " 0.008 ", "0.008 USD", "$ 0.008"])
def test_tavily_price_accepts_a_currency_sign(raw):
    assert Settings(_env_file=None, TAVILY_USD_PER_CREDIT=raw).TAVILY_USD_PER_CREDIT == pytest.approx(0.008)


def test_blank_tavily_price_is_unpriced():
    assert Settings(_env_file=None, TAVILY_USD_PER_CREDIT="").TAVILY_USD_PER_CREDIT is None
