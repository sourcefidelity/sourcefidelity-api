import hashlib
from unittest.mock import Mock

import httpx
import pytest

from app.services.reference_discovery import assess_reference_discovery_trace
from app.services.schemas import ParsedReference
from app.services.source_resolver import SourceResolver
from app.services.source_resolver import _ACTIVE_DISCOVERY_TRACE
from app.services.reference_discovery import ExpectedBibliographicFields
from app.services.retrieval.base import RetrievalResult
from app.services.web_source_metadata import extract_web_source_metadata


def page(author="Alex Rivera and Morgan Chen", date="2024-06-08"):
    return (
        '<html><head><meta property="og:title" content="Advances in language research">'
        f'<meta property="article:published_time" content="{date}"></head>'
        '<body><main><h1>Advances in language research</h1>'
        f'<p>Posted by {author}, Research Scientists</p><article>'
        + '<p>Researchers compared translated documents across several languages and recorded the results.</p>' * 12
        + '</article></main></body></html>'
    )


def resolver_for(monkeypatch, html):
    response = httpx.Response(200, text=html, headers={"content-type": "text/html"},
                              request=httpx.Request("GET", "https://publisher.example/new/article"))
    monkeypatch.setattr("app.services.source_resolver.safe_request", lambda *a, **k: response)
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._check_local_cache = Mock(return_value=RetrievalResult(source_name="local_cache", success=False))
    resolver._delete_lookup_cache = Mock()
    resolver._lookup_cache = None
    resolver._retrieval_sources = []
    resolver._backend = None
    resolver._repository_session_factory = None
    return resolver


def test_observed_header_fields_without_citation_input():
    html = page()
    observed = extract_web_source_metadata(html, "https://publisher.example/article")
    assert observed["title"] == "Advances in language research"
    assert observed["authors"] == ["Alex Rivera", "Morgan Chen"]
    assert observed["year"] == "2024"
    assert observed["publication_date"] == "2024-06-08"
    assert observed["html_sha256"] == hashlib.sha256(html.encode()).hexdigest()


@pytest.mark.parametrize('publication', ['<dt>Publication date</dt><dd><a>1991</a></dd>', ''])
def test_archive_upload_date_cannot_become_book_publication_year(publication):
    html=page(date='2023-05-01').replace('</main>',f'<dl>{publication}<dt>Addeddate</dt><dd>2023-05-04</dd></dl></main>')
    observed=extract_web_source_metadata(html,'https://archive.org/details/catalog-book')
    assert observed['year']==('1991' if publication else None)
    assert observed['date_method']==('catalog_publication_date' if publication else 'catalog_publication_date_unavailable')


def test_direct_web_result_carries_identity_and_final_location(monkeypatch):
    resolver = resolver_for(monkeypatch, page())
    result = resolver.resolve_reference(ParsedReference(
        reference_id="web-control", title="Advances in language research",
        author="Rivera, A., & Chen, M.", year="2024",
        url="https://publisher.example/old/article", source_kind="webpage",
    ))
    assert result.metadata["identity_confidence"] == "high"
    assert result.metadata["accepted_representation_sha256"] == hashlib.sha256(result.full_text).hexdigest()
    assert result.representation.source_url == "https://publisher.example/new/article"
    assert result.representation.completeness == "not_assessed"
    assert result.metadata["reference_discovery"]["outcome"] == "confirmed"
    trace = result.metadata["reference_discovery_trace"]
    assert not trace["required_route_categories"]
    assert trace["candidates"][0]["acquisition_outcome"] == "acquired"
    assert trace["candidates"][0]["observed"]["authors"] == ["Alex Rivera", "Morgan Chen"]
    # Removing affirmative identity cannot enable a negative without policy.
    trace["candidates"] = []
    trace["attempts"][-1]["outcome"] = "no_match"
    completion = assess_reference_discovery_trace(trace)
    assert not completion.ready
    assert "no_route_policy" in completion.blocker_codes


@pytest.mark.parametrize("author,year", [("Different, A.", "2024"), ("Rivera, A., & Chen, M.", "2023")])
def test_conflicting_author_or_year_is_not_admitted(monkeypatch, author, year):
    resolver = resolver_for(monkeypatch, page())
    result = resolver._try_web_fetch("https://publisher.example/article", "Advances in language research",
                                     expected_author=author, expected_year=year)
    assert not result.success
    assert result.metadata["identity_confidence"] == "rejected"
    assert "accepted_representation_sha256" not in result.metadata
    assert result.authors == ["Alex Rivera", "Morgan Chen"]


def test_missing_byline_does_not_borrow_expected_author(monkeypatch):
    resolver = resolver_for(monkeypatch, page().replace('<p>Posted by Alex Rivera and Morgan Chen, Research Scientists</p>', ''))
    result = resolver._try_web_fetch("https://publisher.example/article", "Advances in language research",
                                     expected_author="Rivera, A.", expected_year="2024")
    assert result.authors == []
    assert result.metadata["identity_confidence"] != "high"


def test_body_author_mentions_are_not_a_byline():
    html = page().replace('Posted by Alex Rivera and Morgan Chen, Research Scientists', 'The experiment cites Alex Rivera and Morgan Chen.')
    assert extract_web_source_metadata(html, "https://publisher.example/article")["authors"] == []


def test_structured_authors_are_preserved():
    html = page().replace('</head>', '<meta name="citation_author" content="Rivera, Alex"><meta name="citation_author" content="Chen, Morgan"></head>')
    fields = extract_web_source_metadata(html, "https://publisher.example/article")
    assert fields["authors"] == ["Rivera, Alex", "Chen, Morgan"]
    assert fields["author_method"] == "html_meta"


def test_visible_header_date_without_date_metadata():
    html = page().replace('<meta property="article:published_time" content="2024-06-08">', '')
    html = html.replace('</h1>', '</h1><p>June 8, 2024</p>')
    fields = extract_web_source_metadata(html, "https://publisher.example/article")
    assert fields["year"] == "2024"
    assert fields["publication_date"] == "2024-06-08"


@pytest.mark.parametrize("error", ["http_500", "http_429", "connect_error", "response_invalid", "unexpected_error"])
def test_redacted_provider_failures_never_become_no_match(error):
    trace = dict(reference_id="failure-control", expected=ExpectedBibliographicFields(title="A specific scholarly work"),
                 required={"academic_adapter"}, queries=[], attempts=[], candidates=[])
    token = _ACTIVE_DISCOVERY_TRACE.set(trace)
    try:
        SourceResolver._record_discovery_attempt(category="academic_adapter", provider="control",
            result=RetrievalResult(source_name="control", success=False, error=error), required=True)
        assert trace["attempts"][0].outcome == "operational_failure"
        assert trace["queries"][0].execution_outcome == "operational_failure"
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)


def test_failed_identifier_lookup_survives_empty_title_fallback():
    source = Mock(name="source")
    source.name = "control"
    source.capabilities = frozenset({"doi", "title_author"})
    source.search_by_doi.return_value = RetrievalResult(source_name="control", success=False, error="http_500")
    source.search_by_title_author.return_value = RetrievalResult(source_name="control", success=False, error="No results")
    resolver = SourceResolver.__new__(SourceResolver)
    result = resolver._lookup_source(source, "10.1234/control", "A specific scholarly work", None)
    assert [a["outcome"] for a in result.metadata["structured_search_attempts"]] == ["operational_failure", "no_results"]
    trace = dict(reference_id="failure-control", expected=ExpectedBibliographicFields(title="A specific scholarly work"),
                 required={"academic_adapter"}, queries=[], attempts=[], candidates=[])
    token = _ACTIVE_DISCOVERY_TRACE.set(trace)
    try:
        resolver._record_discovery_attempt(category="academic_adapter", provider="control", result=result, required=True)
        completion = assess_reference_discovery_trace(dict(
            reference_id=trace["reference_id"], expected=trace["expected"], required_route_categories=list(trace["required"]),
            queries=trace["queries"], attempts=trace["attempts"], candidates=trace["candidates"],
        ))
        assert completion.record.outcome == "search_incomplete"
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)


def test_doi_only_adapter_is_not_a_required_failed_title_search():
    source = Mock()
    source.name = "control"
    source.capabilities = frozenset({"doi", "metadata"})
    resolver = SourceResolver.__new__(SourceResolver)
    result = resolver._lookup_source(source, None, "A specific scholarly work", None)
    source.search_by_title_author.assert_not_called()
    assert result.metadata["lookup_applicable"] is False
    trace = dict(reference_id="capability-control", expected=ExpectedBibliographicFields(title="A specific scholarly work"),
                 required={"academic_adapter"}, queries=[], attempts=[], candidates=[])
    token = _ACTIVE_DISCOVERY_TRACE.set(trace)
    try:
        resolver._record_discovery_attempt(category="academic_adapter", provider="control", result=result, required=True)
        assert trace["attempts"][0].required is False
        assert trace["attempts"][0].reason_code == "route_not_applicable"
        # The declared academic-search requirement is not removed.
        assert trace["required"] == {"academic_adapter"}
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)
