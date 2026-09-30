"""Development-only candidate-link audit. Provider doubles, not live searches."""

from dataclasses import asdict
import json
from unittest.mock import Mock

import pytest

from app.config import settings
from app.services.reference_discovery import ExpectedBibliographicFields
from app.services.retrieval.web_search import WebSearchRetriever
from app.services.search.base import SearchResult
from app.services.search.candidate_audit import AUDIT_POLICY, build_candidate_audit
from app.services.search.policy import API_FIRST_SEARCH_POLICY
from app.services.source_resolver import SourceResolver, _ACTIVE_DISCOVERY_TRACE

TITLE = "Frozen source control"


@pytest.fixture
def cascade(monkeypatch):
    monkeypatch.setattr(settings, "SEARCH_POLICY_VERSION", API_FIRST_SEARCH_POLICY)
    monkeypatch.setattr(settings, "BRAVE_SEARCH_RETENTION_PERMITTED", False)
    monkeypatch.setattr(settings, "BRAVE_SEARCH_TRANSIENT_ENABLED", True)
    monkeypatch.setattr(settings, "SEARCH_SEARXNG_FALLBACK_ENABLED", False)
    monkeypatch.setattr(settings, "SEARCH_ESCALATION_MAX_CALLS", "brave:4,exa:4,tavily:4")
    providers = {name: Mock(name=name, last_status="completed", last_cost_usd=None)
                 for name in ("brave", "exa", "searxng", "tavily")}
    for name, provider in providers.items():
        provider.name = name
        provider.search.return_value = []
    monkeypatch.setattr("app.services.retrieval.web_search.get_search_provider",
                        lambda name=None: providers.get(name or "searxng"))
    return WebSearchRetriever(health_store=Mock()), providers


def _resolve_and_record(search, dispositions):
    """Run the shared resolver tier loop with a stubbed acquisition step."""
    resolver = SourceResolver.__new__(SourceResolver)

    def acquisition(source, result, *args, **kwargs):
        attempts = []
        for location in result.locations:
            if location.url in dispositions:
                outcome, reason = dispositions[location.url]
                attempts.append({"url": location.url, "outcome": outcome, "reason_code": reason,
                                 "discovery_provider": location.metadata.get("search_provider"),
                                 "rank": 1})
        result.metadata["location_attempts"] = attempts
        return result

    resolver._download_and_cache = acquisition
    result = resolver._try_source(search, None, TITLE, "Writer", "2020")
    expected = ExpectedBibliographicFields(title=TITLE, authors=["Writer"], year="2020")
    token = _ACTIVE_DISCOVERY_TRACE.set({
        "reference_id": "audit-control", "expected": expected,
        "search_policy_version": API_FIRST_SEARCH_POLICY,
        "required": {"bounded_web"}, "queries": [], "attempts": [], "candidates": []})
    try:
        resolver._record_discovery_attempt(category="bounded_web", provider="web_search",
                                           result=result, required=True)
        trace, record = resolver._discovery_artifacts()
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)
    return result, trace, record


def _exa_hits(count=7):
    return [SearchResult(f"https://exa.example/AUDIT_URL_{i}.pdf", TITLE, "") for i in range(count)]


def test_setting_off_stores_no_candidate_audit_or_new_fields(cascade, monkeypatch):
    monkeypatch.setattr(settings, "SEARCH_CANDIDATE_AUDIT_URLS", False)
    search, providers = cascade
    providers["exa"].search.return_value = _exa_hits()
    result, trace, record = _resolve_and_record(
        search, {"https://exa.example/AUDIT_URL_0.pdf": ("identity_rejected", "identity_rejected")})
    serialized = json.dumps({"result": asdict(result), "trace": trace, "record": record}, default=str)
    assert "candidate_audit" not in serialized
    assert AUDIT_POLICY not in serialized
    assert "search_rank" not in serialized
    web = [a for a in trace["attempts"] if a["route_category"] == "bounded_web"]
    assert web and all("candidate_audit" not in a for a in web)


def test_setting_on_records_exa_candidates_with_dispositions(cascade, monkeypatch):
    monkeypatch.setattr(settings, "SEARCH_CANDIDATE_AUDIT_URLS", True)
    search, providers = cascade
    providers["exa"].search.return_value = _exa_hits()
    _, trace, _ = _resolve_and_record(search, {
        "https://exa.example/AUDIT_URL_0.pdf": ("identity_rejected", "identity_rejected"),
        "https://exa.example/AUDIT_URL_1.pdf": ("access_restricted", "http_403"),
        "https://exa.example/AUDIT_URL_2.pdf": ("completeness_rejected", "completeness_rejected"),
    })
    web = [a for a in trace["attempts"] if a["route_category"] == "bounded_web"]
    assert len(web) == 1
    audit = web[0]["candidate_audit"]
    assert audit["audit_policy"] == "dev-candidate-audit-v1"
    by_url = {entry["url"]: entry for entry in audit["candidates"]}
    assert set(by_url) == {f"https://exa.example/AUDIT_URL_{i}.pdf" for i in range(7)}
    assert {entry["provider"] for entry in by_url.values()} == {"exa"}
    exa_query_ids = {q["query_id"] for q in trace["queries"] if q["execution_provider"] == "exa"}
    assert all(entry["query_id"] in exa_query_ids for entry in by_url.values())
    assert sorted(entry["rank"] for entry in by_url.values()) == list(range(1, 8))
    got = {url.rsplit("_", 1)[1]: (e["disposition"], e["reason_code"]) for url, e in by_url.items()}
    assert got["0.pdf"] == ("identity_rejected", "identity_rejected")
    assert got["1.pdf"] == ("access_restricted", "http_403")
    assert got["2.pdf"] == ("completeness_rejected", "completeness_rejected")
    # Returned to acquisition but never reached by the stubbed step.
    assert got["3.pdf"] == ("not_attempted", "acquisition_not_run")
    # Ranked outside the bounded location set.
    assert got["5.pdf"] == ("not_attempted", "bounded_location_limit")
    assert got["6.pdf"] == ("not_attempted", "bounded_location_limit")


def test_brave_candidates_are_never_recorded_even_when_on(cascade, monkeypatch):
    monkeypatch.setattr(settings, "SEARCH_CANDIDATE_AUDIT_URLS", True)
    search, providers = cascade
    brave_url = "https://brave.example/BRAVE_AUDIT_SENTINEL"
    providers["brave"].search.return_value = [SearchResult(brave_url, TITLE, "")]
    providers["exa"].search.return_value = _exa_hits(1)
    result, trace, record = _resolve_and_record(search, {
        brave_url: ("identity_rejected", "identity_rejected"),
        "https://exa.example/AUDIT_URL_0.pdf": ("identity_rejected", "identity_rejected"),
    })
    serialized = json.dumps({"result": asdict(result), "trace": trace, "record": record}, default=str)
    assert "BRAVE_AUDIT_SENTINEL" not in serialized
    audits = [a["candidate_audit"] for a in trace["attempts"] if "candidate_audit" in a]
    assert [e["provider"] for audit in audits for e in audit["candidates"]] == ["exa"]
    assert [a["provider"] for a in trace["attempts"][0]["transient_search_audits"]] == ["brave"]


def test_builder_is_inert_when_off_and_bounds_long_fields(monkeypatch):
    phase = {"candidate_locations": [
        {"url": "https://tavily.example/" + "x" * 3000, "discovery_provider": "tavily",
         "search_query": "q", "search_rank": 2, "outcome": "metadata_only"},
        {"url": "https://brave.example/lead", "discovery_provider": "brave", "outcome": "metadata_only"},
        {"url": "https://unknown.example/lead", "outcome": "metadata_only"},
    ], "location_attempts": [
        {"url": "https://tavily.example/" + "x" * 3000, "outcome": "unexpected_value", "reason_code": "r" * 300},
    ]}
    monkeypatch.setattr(settings, "SEARCH_CANDIDATE_AUDIT_URLS", False)
    assert build_candidate_audit([phase], {("q", "tavily"): "query-1"}) is None
    monkeypatch.setattr(settings, "SEARCH_CANDIDATE_AUDIT_URLS", True)
    audit = build_candidate_audit([phase], {("q", "tavily"): "query-1"})
    [entry] = audit.candidates
    assert entry.provider == "tavily" and entry.query_id == "query-1" and entry.rank == 2
    assert entry.url_truncated and len(entry.url) == 2000
    assert entry.disposition == "unknown" and len(entry.reason_code) == 100


def test_searxng_is_not_recorded():
    # It relays engines whose terms are unchecked (owner review 2026-09-29).
    from app.services.search.candidate_audit import AUDITABLE_PROVIDERS
    assert AUDITABLE_PROVIDERS == frozenset({"exa", "tavily"})
