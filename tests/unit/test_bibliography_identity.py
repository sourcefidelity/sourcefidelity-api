"""Identity-only search must never enter source acquisition or admission."""
import json
from unittest.mock import Mock

import pytest

from app.services.bibliography_identity import POLICY, eligible_identity_only
from app.services.retrieval.base import AcquisitionLocation, RetrievalResult
from app.services.retrieval.crossref import CrossrefRetriever
from app.services.schemas import ParsedReference
from app.services.source_resolver import SourceResolver


def reference(**changes):
    values = dict(reference_id="uncited", title="Youth culture and cinematic violence",
                  author="River, A.", year="2002", source_kind="journal_article",
                  raw_ref="River, A. (2002). Youth culture and cinematic violence.")
    return ParsedReference(**(values | changes))


def resolver(monkeypatch, sources):
    obj = SourceResolver.__new__(SourceResolver)
    obj._retrieval_sources = sources
    obj._acquisition_capabilities = None
    monkeypatch.setattr(obj, "_enrich_journal_registration", lambda _: None)
    monkeypatch.setattr(obj, "_enrich_book_editions", lambda: None)
    for name in ("resolve", "_check_local_cache", "_try_student_url", "_try_web_fetch",
                 "_download_and_cache", "_acquire_from_locations", "_persist_retrieved_representation"):
        monkeypatch.setattr(obj, name, Mock(side_effect=AssertionError("Acquisition forbidden")))
    return obj


@pytest.mark.parametrize("changes", [{"source_kind": "film"}, {"source_kind": "webpage"},
    {"source_kind": "television_series"}, {"source_kind": "social_media_post"},
    {"source_kind": "archival_source"}, {"needs_review": True},
    {"author": ""}, {"title": ""}])
def test_ineligible_never_searches(monkeypatch, changes):
    ref = reference(**changes)
    assert not eligible_identity_only(ref)
    with pytest.raises(ValueError):
        resolver(monkeypatch, []).resolve_reference(ref, identity_only=True)


@pytest.mark.parametrize("changes", [{"source_kind": "unknown"}, {"source_kind": "chapter"},
    {"source_kind": "thesis"}, {"source_kind": "report"}])
def test_searchable_kinds_include_the_unclassified(changes):
    """An absent or non-article kind is a classifier gap, not an unsearchable
    work: author, title and year settle identity for all of these.  Excluding
    them let an uncited fabricated reference leave the run unsearched."""
    assert eligible_identity_only(reference(**changes))


def test_confirmed_metadata_stops_without_web_or_content(monkeypatch):
    adapter = CrossrefRetriever()
    result = adapter._parse_message({"title": [reference().title],
        "author": [{"family": "River", "given": "A"}],
        "issued": {"date-parts": [[2002]]}, "DOI": "10.1234/example",
        "type": "journal-article", "abstract": "Incidental abstract is not evidence."})
    monkeypatch.setattr(adapter, "search_by_title_author", lambda *_: result)
    web = Mock(name="web")
    web.capabilities = {"web_discovery"}
    obj = resolver(monkeypatch, [adapter, web])
    output = obj.resolve_reference(reference(), identity_only=True)
    assert output.metadata["reference_discovery"]["outcome"] == "confirmed"
    assert output.metadata["identity_only_policy"] == POLICY
    assert not output.full_text and not output.abstract and not output.representation
    web.search_reference.assert_not_called()


def test_acquiring_metadata_adapter_is_not_invoked(monkeypatch):
    from app.services.retrieval.elsevier import ElsevierRetriever
    adapter = ElsevierRetriever()
    monkeypatch.setattr(adapter, "search_by_doi", Mock(side_effect=AssertionError("Can acquire XML")))
    output = resolver(monkeypatch, [adapter]).resolve_reference(
        reference(doi="10.1016/example"), identity_only=True)
    assert output.metadata["reference_discovery"]["outcome"] == "search_incomplete"
    adapter.search_by_doi.assert_not_called()


@pytest.mark.parametrize("transient", [False, True])
def test_web_leads_not_downloaded_and_brave_details_discarded(monkeypatch, transient):
    from app.services.search.transient import BRAVE_TRANSIENT_POLICY
    class Web:
        name = "web_search"
        capabilities = {"web_discovery"}
        _policy_providers = {"brave": object(), "exa": object()}
        _policy_query_cache = {}
        calls = 0

        def search_reference(self, **kwargs):
            self.calls += 1
            return RetrievalResult(source_name=self.name, success=True,
                locations=[AcquisitionLocation(url="https://example.org/private-lead.pdf", provider="brave" if transient else "exa",
                    metadata={"search_title": "Unexamined candidate title",
                              "search_provider": "brave" if transient else "exa"})],
                metadata={"search_retention_policy": BRAVE_TRANSIENT_POLICY if transient else None,
                    "search_attempts": [{"provider": "brave" if transient else "exa",
                        "query": reference().title, "outcome": "results", "result_count": 1}]})

        def search_after_failed_candidates(self, **kwargs):
            self.calls += 1
            return RetrievalResult(source_name=self.name, success=False, error="No further providers")

    web = Web()
    output = resolver(monkeypatch, [web]).resolve_reference(reference(), identity_only=True)
    assert web.calls == 2
    assert not output.full_text and not output.locations
    assert output.metadata["reference_discovery"]["outcome"] == "search_incomplete"
    encoded = json.dumps(output.metadata)
    if transient:
        assert "private-lead" not in encoded and "Unexamined candidate" not in encoded
        audits = output.metadata["reference_discovery_trace"]["attempts"][0]["transient_search_audits"]
        assert audits[0]["not_attempted"] == 1
    else:
        candidates = output.metadata["reference_discovery_trace"]["candidates"]
        assert candidates[0]["acquisition_outcome"] == "not_attempted"


def test_required_provider_failure_stays_incomplete(monkeypatch):
    adapter = CrossrefRetriever()
    monkeypatch.setattr(adapter, "search_by_title_author", Mock(side_effect=TimeoutError()))
    output = resolver(monkeypatch, [adapter]).resolve_reference(reference(), identity_only=True)
    assert output.metadata["reference_discovery"]["outcome"] == "search_incomplete"
    assert output.metadata["reference_discovery_trace"]["queries"][0]["execution_outcome"] != "no_results"


@pytest.mark.parametrize("failure", [None, "timeout", "budget_skipped", "cooldown_skipped", "rate_limited"])
def test_empty_cascade_requires_successful_routes_before_review(monkeypatch, failure):
    from app.config import settings
    from app.services.reference_credibility import assess_reference_credibility
    monkeypatch.setattr(settings, "SEARCH_POLICY_VERSION", "api-first-search-v2")
    class Metadata:
        capabilities = {"metadata_only_search", "title_author"}
        def __init__(self, name): self.name = name
        def search_by_title_author(self, *args):
            return RetrievalResult(source_name=self.name, success=False, error="No results",
                metadata={"identity_search_result_count": 0})
    class Web:
        name = "web_search"
        capabilities = {"web_discovery"}
        _policy_providers = {"brave": object(), "exa": object()}
        _policy_query_cache = {}
        def search_reference(self, **kwargs):
            return RetrievalResult(source_name=self.name, success=False, error="No results",
                metadata={"search_attempts": [dict(provider="brave", query=reference().title,
                    outcome="no_results", result_count=0, required=True)]})
        def search_after_failed_candidates(self, **kwargs):
            return RetrievalResult(source_name=self.name, success=False, error=failure or "No results",
                metadata={"search_attempts": [dict(provider="exa", query=reference().title,
                    outcome=failure or "no_results", result_count=None if failure else 0, required=True)]})
    ref = reference()
    output = resolver(monkeypatch, [Metadata("crossref"), Metadata("openalex"), Web()]).resolve_reference(ref, identity_only=True)
    findings = assess_reference_credibility(ref, output.metadata.get("reference_discovery"),
        output.metadata["reference_discovery_trace"])["findings"]
    assert bool(findings) is (failure is None)
    assert (output.metadata.get("reference_discovery") or {}).get("outcome") != "unlocated_after_search"


def test_budget_exhaustion_cannot_promote_metadata(monkeypatch):
    from app.services.candidate_budget import candidate_budget_scope
    adapter = CrossrefRetriever()
    result = adapter._parse_message({"title": [reference().title],
        "author": [{"family": "River", "given": "A"}],
        "issued": {"date-parts": [[2002]]}, "DOI": "10.1234/example"})
    monkeypatch.setattr(adapter, "search_by_title_author", lambda *_: result)
    with candidate_budget_scope(0):
        output = resolver(monkeypatch, [adapter]).resolve_reference(reference(), identity_only=True)
    assert output.metadata["reference_discovery"]["outcome"] == "search_incomplete"
    assert result.success  # adapter-owned cache is not poisoned
