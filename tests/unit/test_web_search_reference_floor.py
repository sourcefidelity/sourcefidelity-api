"""Each reference keeps a bounded floor of required API calls.

Diagnosed 2026-09-29: the job-wide ceiling was spent first come, first
served, so a later reference found Exa `budget_skipped` (reason
`configured_call_ceiling`) and its required search could never complete.
Provider doubles only; no live search.
"""
from unittest.mock import Mock

import pytest

from app.config import settings
from app.services.retrieval.web_search import WebSearchRetriever, reference_search_scope
from app.services.search.base import SearchResult
from app.services.search.policy import API_FIRST_SEARCH_POLICY
from app.services.source_resolver import SourceResolver

TITLE = "Frozen floor control"


@pytest.fixture
def cascade(monkeypatch):
    monkeypatch.setattr(settings, "SEARCH_POLICY_VERSION", API_FIRST_SEARCH_POLICY)
    monkeypatch.setattr(settings, "BRAVE_SEARCH_RETENTION_PERMITTED", True)
    monkeypatch.setattr(settings, "BRAVE_SEARCH_TRANSIENT_ENABLED", True)
    monkeypatch.setattr(settings, "SEARCH_SEARXNG_FALLBACK_ENABLED", False)
    monkeypatch.setattr(settings, "SEARCH_ESCALATION_MAX_CALLS", "brave:1,exa:1")
    monkeypatch.setattr(settings, "WEB_SEARCH_PER_REFERENCE_FLOOR", "brave:1,exa:1")
    monkeypatch.setattr(settings, "WEB_SEARCH_HARD_MAX_CALLS", "brave:3,exa:3")
    providers = {name: Mock(name=name, last_status="completed", last_cost_usd=None)
                 for name in ("brave", "exa", "searxng")}
    for name, provider in providers.items():
        provider.name = name
        provider.search.return_value = []
    monkeypatch.setattr("app.services.retrieval.web_search.get_search_provider",
                        lambda name=None: providers.get(name or "searxng"))
    return WebSearchRetriever(health_store=Mock()), providers


def _reference(search, title=TITLE):
    with reference_search_scope():
        return search.search_reference(doi=None, title=title, author="Writer", year="2020")


def _exa_outcomes(result):
    return [(a["outcome"], a["reason_code"]) for a in result.metadata["search_attempts"]
            if a["provider"] == "exa"]


def test_defaults_are_bounded():
    from app.config import Settings
    assert Settings.model_fields["WEB_SEARCH_PER_REFERENCE_FLOOR"].default == "brave:3,exa:2"
    assert Settings.model_fields["WEB_SEARCH_HARD_MAX_CALLS"].default == "brave:120,exa:80"


def test_later_references_keep_a_floor_after_the_job_ceiling_is_spent(cascade):
    search, providers = cascade
    first = _reference(search, "First frozen title")
    second = _reference(search, "Second frozen title")
    assert _exa_outcomes(first) == [("no_results", None)]
    # Job ceiling (1) spent by the first reference; the floor admits one more.
    assert _exa_outcomes(second) == [("no_results", None)]
    assert providers["exa"].search.call_count == 2
    assert search.search_metrics["floor_calls:exa"] == 1


def test_the_hard_cap_still_bounds_cost(cascade):
    search, providers = cascade
    outcomes = [_exa_outcomes(_reference(search, f"Frozen title {n}")) for n in range(5)]
    assert providers["exa"].search.call_count == 3                      # hard cap
    assert outcomes[3:] == [[("budget_skipped", "configured_call_ceiling")]] * 2


def test_no_floor_outside_a_reference_scope(cascade):
    search, providers = cascade
    search.search_reference(doi=None, title="One", author="Writer", year="2020")
    result = search.search_reference(doi=None, title="Two", author="Writer", year="2020")
    assert _exa_outcomes(result) == [("budget_skipped", "configured_call_ceiling")]
    assert providers["exa"].search.call_count == 1


def test_a_provider_the_operator_gave_no_ceiling_stays_off(cascade):
    search, providers = cascade
    search._escalation_limits = {"brave": 1, "exa": 0}
    assert _exa_outcomes(_reference(search)) == [("budget_skipped", "configured_call_ceiling")]
    providers["exa"].search.assert_not_called()


def test_the_floor_is_per_reference_not_per_call(cascade):
    search, providers = cascade
    _reference(search, "First frozen title")          # spends the job ceiling
    with reference_search_scope():
        search.search_reference(doi=None, title="Second", author="Writer", year="2020")
        again = search.search_reference(doi=None, title="Third", author="Writer", year="2020")
    assert _exa_outcomes(again) == [("budget_skipped", "configured_call_ceiling")]
    assert providers["exa"].search.call_count == 2


def test_rejected_brave_candidates_still_reach_exa_on_the_floor(cascade):
    """All Brave candidates rejected -> the resolver asks Exa, even after the ceiling."""
    search, providers = cascade
    _reference(search, "Earlier reference")            # spends both job ceilings
    for name in ("brave", "exa"):
        providers[name].search.reset_mock()
        providers[name].search.return_value = [SearchResult(f"https://{name}.example/source", TITLE, "")]
    resolver = SourceResolver.__new__(SourceResolver)
    seen = []

    def acquire(source, result, *args, **kwargs):
        name = result.locations[0].metadata["search_provider"]
        seen.append(name)
        result.metadata["location_attempts"] = [{"url": result.locations[0].url, "discovery_provider": name,
            "outcome": "identity_rejected" if name == "brave" else "acquired"}]
        if name == "exa":
            result.full_text = b"validated-test-representation"
        return result

    resolver._download_and_cache = acquire
    with reference_search_scope():
        result = resolver._try_source(search, None, TITLE, "Writer", "2020")
    assert seen == ["brave", "exa"]
    assert result.full_text
    providers["exa"].search.assert_called_once()
