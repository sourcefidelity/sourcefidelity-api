"""Brave and Exa report their own recovery, passively.

Owner-directed, 2026-09-24. Automatic re-runs now fire only for providers
that were blocking a search, and Brave and Exa are the required web APIs, so
they are the main blockers. But only SearXNG and Bright Data could report a
recovery: the API-first path kept no health record for Brave or Exa at all.
With the narrowed trigger, that meant nothing could ever trigger a re-run.

The signal is passive. It watches requests a paper was already making and
adds none of its own, so it costs nothing. It records the incident but does
not enforce a cooldown, so what gets searched does not change.
"""
from unittest.mock import Mock

import pytest

from app.config import settings
from app.services.retrieval.provider_runtime import ProviderHealthStore
from app.services.retrieval.web_search import WebSearchRetriever
from app.services.search.base import SearchResult
from app.services.search.policy import API_FIRST_SEARCH_POLICY


@pytest.fixture
def api_first(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SEARCH_POLICY_VERSION", API_FIRST_SEARCH_POLICY)
    monkeypatch.setattr(settings, "BRAVE_SEARCH_RETENTION_PERMITTED", True)
    monkeypatch.setattr(settings, "BRAVE_SEARCH_TRANSIENT_ENABLED", True)
    monkeypatch.setattr(settings, "SEARCH_SEARXNG_FALLBACK_ENABLED", False)
    monkeypatch.setattr(settings, "SEARCH_ESCALATION_MAX_CALLS", "brave:20,exa:20")
    providers = {name: Mock(name=name, last_status="completed", last_cost_usd=None,
                            last_reason_code=None)
                 for name in ("brave", "exa")}
    for name, provider in providers.items():
        provider.name = name
        provider.search.return_value = []
    monkeypatch.setattr("app.services.retrieval.web_search.get_search_provider",
                        lambda name=None: providers.get(name))
    store = ProviderHealthStore(str(tmp_path / "health.json"))
    recovered = []
    # These tests exercise the signal itself, so outages are treated as long
    # enough; the blip threshold has its own tests below.
    monkeypatch.setattr("app.services.retrieval.web_search._REQUIRED_API_MIN_OUTAGE_SECONDS", 0)

    def build():
        return WebSearchRetriever(health_store=store, on_provider_recovered=recovered.append)

    return build, providers, store, recovered


def _fail(provider, status="operational_failure"):
    provider.search.return_value = []
    provider.last_status = status


def _succeed(provider):
    provider.search.return_value = [SearchResult(
        url="https://example.org/a.pdf", title="T", snippet="", is_pdf=True)]
    provider.last_status = "completed"


def _query(retriever, name, providers):
    return retriever._run_policy_query(name, providers[name], '"a synthetic title" Writer 2020')


def test_a_real_brave_failure_opens_an_incident(api_first) -> None:
    build, providers, store, recovered = api_first
    _fail(providers["brave"])

    _query(build(), "brave", providers)

    assert store.incident("brave") is not None
    assert recovered == []


def test_a_later_success_reports_recovery_once(api_first) -> None:
    build, providers, store, recovered = api_first
    _fail(providers["exa"], "rate_limited")
    _query(build(), "exa", providers)

    _succeed(providers["exa"])
    retriever = build()
    _query(retriever, "exa", providers)
    _query(retriever, "exa", providers)

    assert recovered == ["exa"]
    assert store.incident("exa") is None


def test_an_empty_but_completed_search_also_counts_as_recovered(api_first) -> None:
    """The provider answered; finding nothing is a working search."""
    build, providers, store, recovered = api_first
    _fail(providers["brave"], "timeout")
    _query(build(), "brave", providers)

    providers["brave"].search.return_value = []
    providers["brave"].last_status = "completed"
    _query(build(), "brave", providers)

    assert recovered == ["brave"]


def test_success_without_a_prior_incident_reports_nothing(api_first) -> None:
    build, providers, store, recovered = api_first
    _succeed(providers["brave"])

    _query(build(), "brave", providers)

    assert recovered == []


def test_an_incident_does_not_stop_brave_being_searched(api_first) -> None:
    """Recording only: the API-first path never consults this cooldown."""
    build, providers, store, recovered = api_first
    _fail(providers["brave"])
    _query(build(), "brave", providers)
    calls_before = providers["brave"].search.call_count

    _query(build(), "brave", providers)

    assert providers["brave"].search.call_count == calls_before + 1


def test_our_own_call_ceiling_opens_no_incident(api_first, monkeypatch) -> None:
    build, providers, store, recovered = api_first
    monkeypatch.setattr(settings, "SEARCH_ESCALATION_MAX_CALLS", "brave:0,exa:0")

    _, attempts = _query(build(), "brave", providers)

    assert attempts[0]["outcome"] == "budget_skipped"
    assert providers["brave"].search.call_count == 0
    assert store.incident("brave") is None


def test_unconfirmed_retention_permission_opens_no_incident(api_first, monkeypatch) -> None:
    build, providers, store, recovered = api_first
    monkeypatch.setattr(settings, "BRAVE_SEARCH_RETENTION_PERMITTED", False)
    monkeypatch.setattr(settings, "BRAVE_SEARCH_TRANSIENT_ENABLED", False)

    _, attempts = _query(build(), "brave", providers)

    assert attempts[0]["outcome"] == "access_restricted"
    assert store.incident("brave") is None


def test_an_expired_reference_budget_is_not_a_provider_timeout(api_first, monkeypatch) -> None:
    """The provider was never called, so it must not be named as the cause."""
    build, providers, store, recovered = api_first
    monkeypatch.setattr("app.services.retrieval_deadline.expired", lambda: True)

    _, attempts = _query(build(), "brave", providers)

    assert attempts[0]["outcome"] == "budget_skipped"
    assert attempts[0]["reason_code"] == "reference_elapsed_budget_timeout"
    assert attempts[0]["provider_calls"] == 0
    assert store.incident("brave") is None


def test_the_incident_outcomes_match_what_the_trigger_can_cure() -> None:
    """An incident opens exactly when the failure could hold a search incomplete."""
    from app.services.reference_discovery import _PROVIDER_CURABLE_EXECUTION_OUTCOMES
    from app.services.retrieval.web_search import _REQUIRED_API_INCIDENT_OUTCOMES

    assert _REQUIRED_API_INCIDENT_OUTCOMES <= _PROVIDER_CURABLE_EXECUTION_OUTCOMES


def test_a_stub_store_cannot_fake_a_recovery(api_first, monkeypatch) -> None:
    """A Mock store's truthy answer must not schedule re-runs of real papers."""
    build, providers, store, recovered = api_first
    fake = Mock()
    fake.record_success.return_value = Mock()   # truthy, but not True
    _succeed(providers["brave"])

    retriever = build()
    retriever._health_store = fake
    _query(retriever, "brave", providers)

    assert recovered == []



def _backdate(store, name, seconds):
    """Make an open incident look `seconds` old."""
    import json
    from datetime import datetime, timedelta, timezone
    path = store.path if hasattr(store, "path") else store._path
    data = json.loads(open(path).read())
    data[name]["incident_opened_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    open(path, "w").write(json.dumps(data))


def test_a_blip_closes_the_incident_without_reporting_recovery(api_first, monkeypatch) -> None:
    """A route that fails and answers moments later is flapping, not recovering.

    Measured 2026-09-23: an intermittently reachable Brave route reported a
    recovery on every flip -- 95 requeues in 40 minutes and 14 jobs re-run in
    the first minute, each eligible to be re-run again at the next flip.
    """
    build, providers, store, recovered = api_first
    monkeypatch.setattr("app.services.retrieval.web_search._REQUIRED_API_MIN_OUTAGE_SECONDS", 600)
    for _ in range(5):
        _fail(providers["brave"])
        _query(build(), "brave", providers)
        _succeed(providers["brave"])
        _query(build(), "brave", providers)

    assert recovered == []
    assert store.incident("brave") is None


def test_an_outage_that_lasted_reports_recovery(api_first, monkeypatch) -> None:
    build, providers, store, recovered = api_first
    monkeypatch.setattr("app.services.retrieval.web_search._REQUIRED_API_MIN_OUTAGE_SECONDS", 600)
    _fail(providers["exa"], "timeout")
    _query(build(), "exa", providers)
    _backdate(store, "exa", 900)

    _succeed(providers["exa"])
    _query(build(), "exa", providers)

    assert recovered == ["exa"]


def test_an_unreadable_incident_age_is_treated_as_a_blip() -> None:
    from app.services.retrieval.web_search import _incident_age_seconds

    assert _incident_age_seconds({}) == 0.0
    assert _incident_age_seconds({"incident_opened_at": "not a date"}) == 0.0
