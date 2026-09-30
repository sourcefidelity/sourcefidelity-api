"""Purpose-aware discovery ordering cannot substitute for independent identity."""
import json
from unittest.mock import Mock

import httpx
import pytest

from app.services.search.base import SearchResult
from app.services.search.candidate_ranking import candidate_score, identity_candidate_ranking, PURPOSE
from app.services.retrieval.web_search import WebSearchRetriever
from app.services.retrieval.base import AcquisitionLocation, RepresentationKind, RetrievalResult
from app.services.source_resolver import _location_rank
from app.services.candidate_budget import candidate_budget_scope

TITLE = "The separation of platforms and commerce"

BORK_TITLE = "The Antitrust Paradox: A Policy at War with Itself"
BORK_CATALOG = "The antitrust paradox : a policy at war with itself : Bork, Robert H : Free Download, Borrow, and Streaming : Internet Archive"


def test_exact_catalog_prefix_is_inspectable_without_rewriting_historical_scope():
    from app.services.reference_review_scope import scope, scope_for
    assert scope_for('bounded-reference-review-v5')(BORK_TITLE, BORK_CATALOG) == 'outside_bound'
    assert scope(BORK_TITLE, BORK_CATALOG) == 'material'
    exact = SearchResult('https://archive.org/details/example', BORK_CATALOG, '')
    topical = SearchResult('https://example.org/review.pdf', 'Antitrust policy and paradox', 'Bork 1978', True)
    assert candidate_score(exact, None, BORK_TITLE, 'Bork, R. H.', '1978') > candidate_score(
        topical, None, BORK_TITLE, 'Bork, R. H.', '1978')


@pytest.mark.parametrize('title,candidate', [
    (BORK_TITLE, BORK_CATALOG.replace('with itself :', 'with itself reconsidered :')),
    (BORK_TITLE, BORK_CATALOG.replace('Internet Archive', 'Unidentified Journal')),
    ('War', 'War : Free Download, Borrow, and Streaming : Internet Archive'),
])
def test_catalog_hint_requires_exact_substantial_title_and_recognizable_suffix(title, candidate):
    from app.services.reference_review_scope import catalog_title_match
    assert not catalog_title_match(title, candidate)


def test_catalog_discovery_hint_reaches_independent_inspection_not_identity_confirmation(monkeypatch):
    from tests.unit.test_bibliography_identity import resolver, reference
    from tests.unit.test_identity_landing import response
    from app.services import identity_landing
    fetch = Mock(return_value=response('Uncertain catalog', ''))
    monkeypatch.setattr(identity_landing, 'safe_request', fetch)
    class Web:
        name = 'web_search'
        capabilities = {'web_discovery'}
        _policy_providers = {'exa': object()}
        _policy_query_cache = {}
        def search_after_failed_candidates(self, **kwargs):
            return RetrievalResult(source_name=self.name, success=False)
        def search_reference(self, **kwargs):
            return RetrievalResult(source_name=self.name, success=True, locations=[AcquisitionLocation(
                url='https://archive.org/details/example', provider=self.name,
                representation_kind=RepresentationKind.HTML,
                metadata={'search_provider': 'exa', 'search_title': BORK_CATALOG})],
                metadata={'search_attempts': [{'provider': 'exa', 'query': BORK_TITLE,
                    'outcome': 'results', 'result_count': 1}]})
    result = resolver(monkeypatch, [Web()]).resolve_reference(
        reference(title=BORK_TITLE, author='Bork, R. H.', year='1978'), identity_only=True)
    assert fetch.call_count == 1
    assert not result.full_text and not result.representation
    assert not any(a.get('landing_page_identity') for a in result.metadata.get('location_attempts', []))


def test_uninspected_catalog_work_blocks_negative_review_without_establishing_identity():
    from tests.unit.test_bounded_reference_review import review_fixture, web_candidate
    from app.services.reference_credibility import assess_reference_credibility
    ref, trace = review_fixture()
    web_candidate(trace, ref.title.replace(':', ' : ') +
        ' : Example, A : Free Download, Borrow, and Streaming : Internet Archive')
    result = assess_reference_credibility(ref, None, trace)
    assert not result['findings']


def score(candidate, **kwargs):
    return candidate_score(candidate, kwargs.pop("doi", None), TITLE, "Khan, L.", "2019", **kwargs)


def test_exact_html_title_beats_pdf_with_all_words_in_snippet():
    html = SearchResult("https://journal.example/article", TITLE, "Lina Khan 2019")
    pdf = SearchResult("https://repository.example/unrelated.pdf", "Platform competition review",
        TITLE + " Khan 2019", True)
    assert score(html) > score(pdf)


def test_format_alone_does_not_change_score():
    assert score(SearchResult("https://example.org/article", TITLE, "")) == score(
        SearchResult("https://example.org/article.pdf", TITLE, "", True))


def test_catalog_priority_depends_on_need_but_never_overrides_work_match():
    catalog = SearchResult("https://library.example/catalog/1", TITLE, "Khan 2019 catalog record")
    html = SearchResult("https://journal.example/article", TITLE, "Khan 2019")
    unrelated = SearchResult("https://library.example/catalog/2", "Gardening methods", "Khan 2019 catalog record")
    assert score(catalog, purpose="identity") > score(html, purpose="identity")
    assert score(html, purpose="text") > score(catalog, purpose="text")
    assert score(html, purpose="identity") > score(unrelated, purpose="identity")


def test_doi_prefix_not_exact_identifier_hint():
    exact = SearchResult("https://doi.org/10.1234/item", "Unknown", "")
    prefix = SearchResult("https://doi.org/10.1234/item-other", "Unknown", "")
    assert score(exact, doi="10.1234/item") > score(prefix, doi="10.1234/item")


def test_purpose_is_reference_local_even_on_error():
    @identity_candidate_ranking
    def run():
        assert PURPOSE.get() == "identity"
        raise ValueError
    with pytest.raises(ValueError): run()
    assert PURPOSE.get() == "text"


def test_shared_acquisition_keeps_full_web_order_without_format_resort():
    locations = [AcquisitionLocation(url="https://example.org/" + str(i), provider="web_search",
        representation_kind=kind) for i, kind in enumerate((RepresentationKind.HTML, RepresentationKind.PDF, RepresentationKind.HTML))]
    assert sorted(locations, key=_location_rank) == locations


def test_zero_capacity_retains_unattempted_candidates():
    web = WebSearchRetriever.__new__(WebSearchRetriever)
    hit = SearchResult("https://example.org/article", TITLE, "")
    with candidate_budget_scope(0):
        result = web._locations_result({hit.url: (hit, TITLE, "exa")}, queries_run=[TITLE],
            doi=None, title=TITLE, author="Khan", year="2019", search_attempts=[])
    assert not result.success and not result.locations
    assert result.metadata["candidate_dispositions"][0]["outcome"] == "not_attempted"


@pytest.mark.parametrize("transient", [False, True])
def test_identity_confirmation_stops_paid_escalation_without_text(monkeypatch, transient):
    from tests.unit.test_bibliography_identity import resolver, reference
    from tests.unit.test_identity_landing import response
    from app.services import identity_landing
    from app.services.search.transient import BRAVE_TRANSIENT_POLICY
    class Web:
        name = "web_search"
        capabilities = {"web_discovery"}
        _policy_providers = {"brave": object(), "exa": object()}
        _policy_query_cache = {}
        search_after_failed_candidates = Mock(side_effect=AssertionError("No next tier needed"))
        def search_reference(self, **kwargs):
            provider = "brave" if transient else "exa"
            return RetrievalResult(source_name=self.name, success=True, locations=[AcquisitionLocation(
                url="https://library.example/catalog/item", provider=self.name,
                representation_kind=RepresentationKind.HTML, metadata={"search_provider": provider})],
                metadata={"search_attempts": [{"provider": provider, "query": kwargs["title"], "outcome": "results", "result_count": 1}],
                    **({"search_retention_policy": BRAVE_TRANSIENT_POLICY} if transient else {})})
    monkeypatch.setattr(identity_landing, "safe_request", lambda *a, **k: response(TITLE, "A. River"))
    ref = reference(title=TITLE, author="River, A.", year="2020")
    result = resolver(monkeypatch, [Web()]).resolve_reference(ref, identity_only=True)
    assert not result.full_text and not result.representation
    assert not Web.search_after_failed_candidates.called


@pytest.mark.parametrize("failure", [True, False])
def test_duplicate_provider_location_reuses_failure_or_observation(monkeypatch, failure):
    from tests.unit.test_bibliography_identity import resolver, reference
    from tests.unit.test_identity_landing import response
    from app.services import identity_landing
    from app.services.search.transient import BRAVE_TRANSIENT_POLICY
    class Web:
        name = "web_search"
        capabilities = {"web_discovery"}
        _policy_providers = {"brave": object(), "exa": object()}
        _policy_query_cache = {}
        def tier(self, provider):
            return RetrievalResult(source_name=self.name, success=True, locations=[AcquisitionLocation(
                url="https://library.example/catalog/item", provider=self.name,
                representation_kind=RepresentationKind.HTML, metadata={"search_provider": provider})],
                metadata={"search_attempts": [{"provider": provider, "query": TITLE, "outcome": "results", "result_count": 1}],
                    **({"search_retention_policy": BRAVE_TRANSIENT_POLICY} if provider == "brave" else {})})
        def search_reference(self, **kwargs): return self.tier("brave")
        def search_after_failed_candidates(self, **kwargs):
            return self.tier("exa") if "exa" not in kwargs["tried_providers"] else RetrievalResult(source_name=self.name, success=False)
    fetch = Mock(side_effect=TimeoutError if failure else lambda *a, **k: response(TITLE, "Another Author"))
    monkeypatch.setattr(identity_landing, "safe_request", fetch)
    output = resolver(monkeypatch, [Web()]).resolve_reference(reference(title=TITLE), identity_only=True)
    assert fetch.call_count == 1
    trace = output.metadata["reference_discovery_trace"]
    exa = [c for c in trace["candidates"] if c["discovery_provider"] == "exa"]
    assert exa and exa[0]["acquisition_outcome"] == ("transport_failure" if failure else "identity_unconfirmed")
    assert "_reused_location_attempts" not in json.dumps(output.metadata)


def test_complete_html_uses_existing_checks_before_pdf_link(monkeypatch):
    from tests.unit.test_retrieval_locations import _resolver
    from app.services.source_type import SourceKindAssessment
    import app.services.source_resolver as module
    words = "This substantive article discusses the interaction between platform services and competitive markets. " * 30
    html = f'''<html><head><title>{TITLE}</title><meta name="citation_title" content="{TITLE}">
    <meta name="citation_author" content="Lina Khan"><meta name="citation_publication_date" content="2019">
    <meta name="citation_pdf_url" content="https://journal.example/article.pdf"></head>
    <body><h1>{TITLE}</h1><div class="article-body"><p>{words}</p></div></body></html>'''
    fetch = Mock(return_value=httpx.Response(200, text=html, request=httpx.Request("GET", "https://journal.example/article")))
    monkeypatch.setattr(module, "safe_request", fetch)
    resolver = _resolver()
    resolver._safe_download = Mock(side_effect=AssertionError("Complete HTML should stop before PDF"))
    result = RetrievalResult(source_name="web_search", success=True, title=TITLE,
        locations=[AcquisitionLocation(url="https://journal.example/article", provider="web_search", representation_kind=RepresentationKind.HTML)])
    accepted = resolver._acquire_from_locations(result, expected_title=TITLE, expected_author="Khan, L.",
        expected_year="2019", expected_source_kind=SourceKindAssessment("webpage", "high", ()))
    assert accepted and result.representation.original_kind == RepresentationKind.HTML
    assert result.representation.completeness == "complete"
    assert result.metadata["location_attempts"][0]["reason_code"] == "accepted_complete_html"
    assert fetch.call_count == 1


def test_identity_query_capacity_tracks_html_and_pdf_allowances(monkeypatch):
    from app.services import identity_landing, identity_pdf
    from app.services.retrieval.web_search import _query_inspection_capacity
    monkeypatch.setattr(identity_landing, "remaining_inspections", lambda: 0)
    monkeypatch.setattr(identity_pdf, "remaining_inspections", lambda: 2)
    @identity_candidate_ranking
    def check():
        assert _query_inspection_capacity(TITLE) == 0
        assert _query_inspection_capacity(TITLE + " filetype:pdf") == 0
    check()
    assert _query_inspection_capacity(TITLE) == 5  # Ordinary acquisition unaffected.


def test_first_tier_reserves_html_capacity_for_second_required_provider(monkeypatch):
    from app.services import identity_landing, identity_pdf
    from app.services.retrieval.web_search import _query_inspection_capacity
    monkeypatch.setattr(identity_landing, "remaining_inspections", lambda: 3)
    monkeypatch.setattr(identity_pdf, "remaining_inspections", lambda: 2)
    @identity_candidate_ranking
    def check():
        assert _query_inspection_capacity(TITLE, "brave") == 2
        assert _query_inspection_capacity(TITLE, "exa") == 3
    check()


def test_exhausted_identity_inspection_does_not_dispatch_paid_query(monkeypatch):
    from app.config import settings
    from app.services import identity_landing, identity_pdf
    monkeypatch.setattr(identity_landing, "remaining_inspections", lambda: 0)
    monkeypatch.setattr(identity_pdf, "remaining_inspections", lambda: 0)
    monkeypatch.setattr(settings, "SEARCH_POLICY_VERSION", "api-first-search-v2")
    monkeypatch.setattr(settings, "SEARCH_SEARXNG_FALLBACK_ENABLED", False)
    provider = Mock()
    monkeypatch.setattr("app.services.retrieval.web_search.get_search_provider", lambda *a: provider)
    web = WebSearchRetriever(health_store=Mock())
    @identity_candidate_ranking
    def run():
        return web.search_reference(doi=None, title=TITLE, author="Khan", year="2019")
    result = run()
    provider.search.assert_not_called()
    assert len(result.metadata["search_attempts"]) == 2
    assert all(a["outcome"] == "budget_skipped" for a in result.metadata["search_attempts"])


def test_uninspectable_format_keeps_unattempted_disposition(monkeypatch):
    from app.services import identity_landing, identity_pdf
    monkeypatch.setattr(identity_landing, "remaining_inspections", lambda: 1)
    monkeypatch.setattr(identity_pdf, "remaining_inspections", lambda: 0)
    hits = [SearchResult("https://example.org/a.pdf", TITLE, "Khan 2019", True),
            SearchResult("https://example.org/a", TITLE, "Khan 2019")]
    web = WebSearchRetriever.__new__(WebSearchRetriever)
    @identity_candidate_ranking
    def run():
        return web._locations_result({c.url: (c, TITLE, "exa") for c in hits}, queries_run=[TITLE],
            doi=None, title=TITLE, author="Khan", year="2019", search_attempts=[])
    result = run()
    assert [l.url for l in result.locations] == [hits[1].url]
    assert result.metadata["candidate_dispositions"][0]["url"] == hits[0].url
    assert result.metadata["candidate_dispositions"][0]["outcome"] == "not_attempted"


def test_ranking_versions_do_not_rewrite_legacy_trace():
    from app.services.reference_discovery import ReferenceDiscoveryTrace, ExpectedBibliographicFields
    trace = ReferenceDiscoveryTrace(reference_id="x", expected=ExpectedBibliographicFields(title=TITLE), required_route_categories=[])
    assert "candidate_ranking_policy_version" not in trace.model_dump()
    assert "search_capacity_policy_version" not in trace.model_dump()
