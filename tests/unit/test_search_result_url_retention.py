"""The deployed application keeps no search-result links (owner decision 2026-09-29).

With the development audit (SEARCH_CANDIDATE_AUDIT_URLS) off, no Exa, Tavily or
Brave result address reaches the discovery trace, the completed record, the
admitted source's validation evidence or the logs. The one address that stays
is the admitted source's own: its provenance, not a search listing. Provider
doubles only.
"""
from contextlib import nullcontext
import json
import logging
from unittest.mock import Mock

import pytest

from app.config import settings
from app.services.retrieval.base import (
    AcquisitionLocation, RepresentationKind, RetrievalResult, SourceRepresentation,
)
from app.services.retrieval.web_search import WebSearchRetriever
from app.services.search.base import SearchResult
from app.services.search.policy import API_FIRST_SEARCH_POLICY
from app.services.source_resolver import SourceResolver, _withhold_search_result_urls
from tests.unit.test_search_candidate_audit import TITLE, _resolve_and_record

EXA_URL = "https://exa.example/EXA_SENTINEL.pdf"
TAVILY_URL = "https://tavily.example/TAVILY_SENTINEL.pdf"
TAVILY_TITLE_URL = "https://tavily.example/TITLE_SENTINEL"


@pytest.fixture
def cascade(monkeypatch):
    monkeypatch.setattr(settings, "SEARCH_POLICY_VERSION", API_FIRST_SEARCH_POLICY)
    monkeypatch.setattr(settings, "BRAVE_SEARCH_RETENTION_PERMITTED", False)
    monkeypatch.setattr(settings, "BRAVE_SEARCH_TRANSIENT_ENABLED", True)
    monkeypatch.setattr(settings, "SEARCH_SEARXNG_FALLBACK_ENABLED", True)
    monkeypatch.setattr(settings, "SEARCH_WEB_FALLBACK_PROVIDER", "tavily")
    monkeypatch.setattr(settings, "SEARCH_ESCALATION_MAX_CALLS", "brave:4,exa:4,tavily:4")
    monkeypatch.setattr(settings, "SEARCH_CANDIDATE_AUDIT_URLS", False)
    # The retired flag is ignored even when an old deployment still sets it.
    monkeypatch.setattr(settings, "DEVELOPMENT_RETAIN_DISCOVERY_LEAD_URLS", True)
    providers = {name: Mock(name=name, last_status="completed", last_cost_usd=None)
                 for name in ("brave", "exa", "searxng", "tavily")}
    for name, provider in providers.items():
        provider.name = name
        provider.search.return_value = []
    providers["exa"].search.return_value = [SearchResult(EXA_URL, TITLE, "")]
    providers["tavily"].search.return_value = [
        SearchResult(TAVILY_URL, TAVILY_TITLE_URL, "snippet"),
    ]
    monkeypatch.setattr("app.services.retrieval.web_search.get_search_provider",
                        lambda name=None: providers.get(name or "searxng"))
    return WebSearchRetriever(health_store=Mock()), providers


def test_no_search_result_address_reaches_the_trace_or_record(cascade, caplog):
    search, providers = cascade
    caplog.set_level(logging.DEBUG)
    _result, trace, record = _resolve_and_record(search, {
        EXA_URL: ("identity_rejected", "identity_rejected"),
        TAVILY_URL: ("identity_unconfirmed", "identity_unconfirmed"),
    })
    assert providers["exa"].search.called
    stored = json.dumps({"trace": trace, "record": record}, default=str)
    for sentinel in ("EXA_SENTINEL", "TAVILY_SENTINEL", "TITLE_SENTINEL"):
        assert sentinel not in stored
        assert sentinel not in caplog.text
    web_candidates = [c for c in trace["candidates"] if c.get("discovery_provider") in {"exa", "tavily"}]
    assert web_candidates and all(c["development_location_url"] is None for c in web_candidates)
    assert all(c["location_sha256"] for c in web_candidates)


def test_stored_location_attempts_keep_only_the_admitted_address():
    attempts = [
        {"url": EXA_URL, "provider": "web_search", "discovery_provider": "exa",
         "candidate_title": "A search title", "outcome": "identity_rejected", "reason_code": "identity_rejected"},
        {"url": "https://brave.example/landing", "discovery_provider": "brave", "outcome": "unavailable",
         "independent_source_url": "https://brave.example/landing",
         "landing_metadata_observation": {"source_url": "https://brave.example/landing", "content_sha256": "a" * 64}},
        {"url": "https://archive.example/item", "provider": "web_search", "outcome": "unavailable",
         "discovered_urls": ["https://archive.example/file.pdf"]},
        {"url": TAVILY_URL, "provider": "web_search", "discovery_provider": "tavily", "outcome": "acquired",
         "independent_source_url": TAVILY_URL},
        {"url": "https://oa.example/paper.pdf", "provider": "openalex", "outcome": "access_restricted"},
    ]
    kept = _withhold_search_result_urls(attempts)
    serialized = json.dumps(kept)
    assert "exa.example" not in serialized and "brave.example" not in serialized
    assert "archive.example" not in serialized and "A search title" not in serialized
    assert kept[0] == {"provider": "web_search", "discovery_provider": "exa", "outcome": "identity_rejected",
                       "reason_code": "identity_rejected", "url_withheld": "search_result"}
    assert kept[1]["landing_metadata_observation"] == {"content_sha256": "a" * 64}
    # The acquired, admitted location is the source's own address.
    assert kept[3]["url"] == TAVILY_URL and kept[3]["independent_source_url"] == TAVILY_URL
    # A structured provider's open-access link is not a search result.
    assert kept[4]["url"] == "https://oa.example/paper.pdf"


def test_admission_evidence_withholds_search_result_addresses(monkeypatch):
    captured = {}

    def admit(session, backend, request):
        captured["evidence"] = request.validation_evidence
        return Mock(admission_state="accepted", id="rep-1")

    monkeypatch.setattr("app.services.source_resolver.admit_representation", admit)
    monkeypatch.setattr("app.services.source_resolver.commit_source_admissions", lambda session: None)
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._backend = None
    resolver._repository_session_factory = lambda: nullcontext(Mock())
    admitted_url = "https://repository.example/admitted.pdf"
    result = RetrievalResult(
        source_name="web_search", success=True, full_text_url=admitted_url,
        locations=[AcquisitionLocation(url=admitted_url, provider="web_search")],
        metadata={"license_class": "open_access", "location_attempts": [
            {"url": EXA_URL, "provider": "web_search", "discovery_provider": "exa", "outcome": "identity_rejected"},
            {"url": admitted_url, "provider": "web_search", "discovery_provider": "exa", "outcome": "acquired"},
        ]},
    )
    result.set_representation(SourceRepresentation(
        kind=RepresentationKind.PLAIN_TEXT, media_type="text/plain", content=b"Admitted text " * 50,
        source_url=admitted_url, original_kind=RepresentationKind.HTML))
    resolver._persist_retrieved_representation(
        result, ref_doi=None, ref_title=TITLE, ref_author="Writer", ref_year="2020",
        identity_confidence="high", identity_reason="test", downloaded_via_publisher=False,
        safety_report=None)
    evidence = json.dumps(captured["evidence"], default=str)
    assert "EXA_SENTINEL" not in evidence
    assert admitted_url in evidence
