"""DeepSeek Search adapter and the configurable second required web provider (2026-10-18)."""
from unittest.mock import Mock

import httpx
import pytest
from pydantic import SecretStr

from app.config import settings
from app.services.search import get_search_provider
from app.services.search.deepseek import DeepSeekSearch, _bounded_query
from app.services.search.policy import is_required_web_provider_set


def _json(status_code: int, body: dict) -> httpx.Response:
    return httpx.Response(status_code, json=body, request=httpx.Request("POST", "https://api.deepseek.com/v1/tools/search"))


def _provider(monkeypatch, body=None, status_code=200):
    post = Mock(return_value=_json(status_code, body if body is not None else {"results": []}))
    monkeypatch.setattr("app.services.search.deepseek.httpx.post", post)
    return DeepSeekSearch("test-key"), post


def test_request_shape(monkeypatch):
    provider, post = _provider(monkeypatch)
    provider.search('"Frozen source control" Writer 2020', num_results=5)
    kwargs = post.call_args.kwargs
    assert post.call_args.args[0] == "https://api.deepseek.com/v1/tools/search"
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert kwargs["json"] == {"query": '"Frozen source control" Writer 2020', "count": 5}


def test_results_are_mapped_and_priced_per_successful_search(monkeypatch):
    body = {"results": [
        {"url": "https://a.example/paper.pdf?x=1", "title": "Paper", "snippet": "s" * 400, "summary": "long"},
        {"url": "https://b.example/page", "title": None, "snippet": None},
    ]}
    provider, _ = _provider(monkeypatch, body)
    results = provider.search("q", num_results=10)
    assert [r.url for r in results] == ["https://a.example/paper.pdf?x=1", "https://b.example/page"]
    assert results[0].is_pdf and not results[1].is_pdf
    assert len(results[0].snippet) == 300 and results[1].title == ""
    assert provider.last_status == "completed"
    assert provider.last_cost_usd == 0.00075


def test_empty_results_are_completed_and_unbilled(monkeypatch):
    provider, _ = _provider(monkeypatch, {"results": []})
    assert provider.search("q") == []
    assert provider.last_status == "completed"
    assert provider.last_cost_usd is None


def test_invalid_schema_is_not_an_empty_search(monkeypatch):
    provider, _ = _provider(monkeypatch, {"results": [{"title": "no url"}]})
    assert provider.search("q") == []
    assert provider.last_status == "response_invalid"


@pytest.mark.parametrize("status,expected", [(401, "access_restricted"), (402, "operational_failure")])
def test_account_refusal_opens_run_circuit(monkeypatch, status, expected):
    provider, post = _provider(monkeypatch, {"error": {}}, status_code=status)
    provider.search("q")
    assert provider.last_status == expected
    provider.search("q again")
    assert post.call_count == 1
    assert provider.last_status == expected


def test_rate_limit_defers_every_process(monkeypatch):
    deferred = []
    monkeypatch.setattr("app.services.retrieval.provider_gate.defer", lambda name, s: deferred.append(name))
    provider, _ = _provider(monkeypatch, {"error": {}}, status_code=429)
    provider.search("q")
    assert provider.last_status == "rate_limited" and deferred == ["deepseek"]


def test_long_query_keeps_its_phrase_closed():
    title = " ".join(["word"] * 40)
    query = _bounded_query(f'"{title}" Author 2020')
    assert len(query) <= 128
    assert query.startswith('"') and query.endswith('"') and query.count('"') == 2
    assert _bounded_query('"Short title" Author 2020') == '"Short title" Author 2020'


def test_factory_requires_deepseek_endpoint(monkeypatch):
    monkeypatch.setattr(settings, "LLM_API_KEY", SecretStr("test-key"))
    monkeypatch.setattr(settings, "LLM_BASE_URL", "https://other.example/v1")
    assert get_search_provider("deepseek") is None
    monkeypatch.setattr(settings, "LLM_BASE_URL", "https://api.deepseek.com/v1")
    assert isinstance(get_search_provider("deepseek"), DeepSeekSearch)


def test_records_from_either_second_provider_stay_valid():
    assert is_required_web_provider_set(["brave", "exa"])
    assert is_required_web_provider_set(["deepseek", "brave"])
    assert not is_required_web_provider_set(["brave"])
    assert not is_required_web_provider_set(["brave", "tavily"])
    assert not is_required_web_provider_set(["brave", "exa", "deepseek"])
    assert not is_required_web_provider_set(["exa", "deepseek"])


def test_retriever_runs_deepseek_as_the_second_required_provider(monkeypatch):
    from app.services.retrieval import web_search
    from app.services.search.policy import API_FIRST_SEARCH_POLICY
    monkeypatch.setattr(settings, "SEARCH_POLICY_VERSION", API_FIRST_SEARCH_POLICY)
    monkeypatch.setattr(settings, "SEARCH_WEB_FALLBACK_PROVIDER", "none")
    monkeypatch.setattr(web_search, "REQUIRED_API_PROVIDERS", ("brave", "deepseek"))
    built = []
    monkeypatch.setattr(web_search, "get_search_provider", lambda name=None: built.append(name) or None)
    web_search.WebSearchRetriever()
    assert built == ["brave", "deepseek"]
    assert "deepseek" not in web_search._FILETYPE_OPERATOR_PROVIDERS


def test_a_stored_record_is_judged_by_the_providers_it_names(monkeypatch):
    # 2026-10-19: after switching to DeepSeek, records searched with Exa read as
    # incomplete ("deepseek" blocking) because the current setting was used.
    from datetime import datetime, timezone
    from app.services import reference_discovery as rd
    from app.services.search.policy import API_FIRST_SEARCH_POLICY
    monkeypatch.setattr(rd, "REQUIRED_API_PROVIDERS", ("brave", "deepseek"))
    now = datetime.now(timezone.utc)
    import hashlib
    queries = [rd.ReferenceSearchQuery(query_id=f"q-{p}", route_category="bounded_web", provider=p,
                                       normalized_query='"A title"', query_sha256=hashlib.sha256('"A title"'.encode()).hexdigest(),
                                       execution_provider=p, required=True, execution_outcome="no_results",
                                       provider_calls=1, result_count=0)
               for p in ("brave", "exa")]
    attempts = [rd.ReferenceRouteAttempt(attempt_id="a-web", provider="brave", query_ids=[q.query_id for q in queries],
                                         route_category="bounded_web", required=True, permitted=True,
                                         outcome="no_match", started_at=now, completed_at=now)]
    args = (["bounded_web"], attempts, queries, API_FIRST_SEARCH_POLICY)
    assert rd._search_is_incomplete(*args, ["brave", "exa"]) is False
    assert rd.search_blocking_providers(*args, ["brave", "exa"]) == []
    # A new search that names no pair still uses the current setting.
    assert rd._search_is_incomplete(*args) is True
    assert rd.search_blocking_providers(*args) == ["deepseek"]
