"""Landing bibliographic identity is not acquired source evidence."""
import copy
from dataclasses import asdict
import hashlib
import json
from unittest.mock import patch

import httpx
import pytest

from app.services.reference_discovery import ExpectedBibliographicFields, assess_reference_discovery_trace
from app.services.retrieval.base import AcquisitionLocation, RepresentationKind, RetrievalResult
from app.services.source_resolver import SourceResolver, _ACTIVE_DISCOVERY_TRACE
from app.services.search.transient import BRAVE_TRANSIENT_POLICY, finalize_transient_brave

TITLE = "Synthetic bounded landing metadata control"
DOI = "10.1234/synthetic-metadata-control"
URL = "https://publisher.example/record"
HTML = f'''<html><head><title>{TITLE}</title>
<meta name="citation_title" content="{TITLE}">
<meta name="citation_doi" content="{DOI}">
<meta name="citation_author" content="Example, Alice">
<meta name="citation_publication_date" content="2020-01-01">
</head><body><h1>{TITLE}</h1><p>Metadata only. Full text unavailable.</p></body></html>'''


def acquire(provider, html=HTML, status=200, expected_year="2020"):
    response = httpx.Response(status, text=html, headers={"content-type":"text/html"},
        request=httpx.Request("GET",URL))
    resolver = SourceResolver.__new__(SourceResolver)
    result = RetrievalResult(source_name="web_search", success=True, title=TITLE,
        locations=[AcquisitionLocation(url=URL,provider="web_search",representation_kind=RepresentationKind.HTML,
            metadata={"search_provider":provider,"search_title":"SEARCH_TITLE_SENTINEL"})],
        metadata={"search_retention_policy":BRAVE_TRANSIENT_POLICY} if provider == "brave" else {})
    def fetch(*a, **kw):
        response.raise_for_status()
        return response
    with patch("app.services.source_resolver.safe_request",side_effect=fetch):
        accepted = resolver._acquire_from_locations(result,expected_title=TITLE,expected_doi=DOI,
            expected_author="Example, Alice",expected_year=expected_year)
    assert not accepted and not result.full_text and not result.representation
    finalize_transient_brave(result)
    result.metadata["retrieval_trace"]=[{"search_attempts":[{"provider":provider,"query":TITLE,
        "outcome":"results","result_count":1,"provider_calls":1}],
        "location_attempts":result.metadata["location_attempts"],
        "transient_search_audits":result.metadata.get("transient_search_audits",[])}]
    token = _ACTIVE_DISCOVERY_TRACE.set({"reference_id":"synthetic-reference",
        "expected":ExpectedBibliographicFields(title=TITLE,doi=DOI,authors=["Example, Alice"],year=expected_year),
        "search_policy_version":"api-first-search-v2","search_retention_policy":BRAVE_TRANSIENT_POLICY,
        "required_web_providers":["brave","exa"],"required":{"bounded_web"},
        "queries":[],"attempts":[],"candidates":[],"limitations":[]})
    try:
        resolver._record_discovery_attempt(category="bounded_web",provider="web_search",result=result,required=True)
        trace, record = resolver._discovery_artifacts()
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)
    return result, trace, record


@pytest.mark.parametrize("provider",["brave","exa"])
def test_landing_identity_survives_without_becoming_full_text(provider):
    result, trace, record = acquire(provider)
    assert record["outcome"] == "confirmed"
    candidate = trace["candidates"][0]
    assert candidate["identity_evidence_kind"] == "landing_page_metadata"
    assert candidate["location_provenance"] == "independently_acquired_content"
    assert candidate["validated_identity_content_sha256"] == hashlib.sha256(HTML.encode()).hexdigest()
    assert candidate["observed"]["title"] == TITLE
    assert candidate["acquisition_outcome"] == "unavailable"
    assert not result.full_text and not result.abstract
    assert "SEARCH_TITLE_SENTINEL" not in json.dumps(trace)
    if provider == "brave":
        assert not result.locations and not result.full_text_url
        assert result.metadata["transient_search_audits"][0]["identity_established"] == 1


@pytest.mark.parametrize("html",[
    HTML.replace(f'<meta name="citation_doi" content="{DOI}">', ''),
    HTML.replace("Example, Alice", "Different, Bob"),
    HTML.replace("2020-01-01", "2022-01-01"),
])
def test_incomplete_or_conflicting_metadata_cannot_confirm_identity(html):
    result, trace, record = acquire("brave",html=html)
    assert not trace["candidates"]
    assert record["outcome"] == "search_incomplete"
    assert result.metadata["transient_search_audits"][0]["identity_established"] == 0


@pytest.mark.parametrize("field,value", [
    ("validated_identity_content_sha256", None),
    ("location_sha256", None),
    ("location_provenance", "discovery_result"),
])
def test_source_page_binding_is_required_for_landing_identity(field, value):
    _, trace, _ = acquire("brave")
    damaged = copy.deepcopy(trace)
    damaged["candidates"][0][field] = value
    with pytest.raises(ValueError, match="Landing-page identity"):
        assess_reference_discovery_trace(damaged)


def test_access_denial_retains_no_landing_metadata_identity():
    result, trace, record = acquire("brave",status=403)
    assert not trace["candidates"]
    assert record["outcome"] == "search_incomplete"
    assert result.metadata["transient_search_audits"][0]["unresolved"] == 1


def test_discovered_pdf_preserves_parent_response_without_claiming_acquisition():
    resolver = SourceResolver.__new__(SourceResolver)
    response = httpx.Response(200, text=HTML, request=httpx.Request('GET', URL))
    pdf_url = 'https://publisher.example/full.pdf'
    result = RetrievalResult(source_name='fixture', success=True, title=TITLE,
        locations=[AcquisitionLocation(url=URL, provider='fixture', representation_kind=RepresentationKind.HTML)])
    with patch('app.services.source_resolver.safe_request', return_value=response), \
         patch('app.services.source_resolver.discover_scholarly_locations', return_value=[
             AcquisitionLocation(url=pdf_url, provider='fixture', representation_kind=RepresentationKind.PDF)]), \
         patch.object(resolver, '_safe_download', side_effect=ValueError('download unavailable')):
        assert not resolver._acquire_from_locations(result, expected_title=TITLE,
            expected_doi=DOI, expected_author='Example, Alice', expected_year='2020')
    parent, child = result.metadata['location_attempts']
    assert parent['reason_code'] == 'landing_page_discovered_full_text_candidates'
    assert parent['outcome'] == 'unavailable'
    assert parent['discovered_location_sha256'] == [hashlib.sha256(pdf_url.encode()).hexdigest()]
    assert child['discovered_from_location_sha256'] == hashlib.sha256(URL.encode()).hexdigest()
    assert child['discovered_from_response_sha256'] == parent['discovery_response_sha256'] == hashlib.sha256(response.content).hexdigest()
    assert not result.representation


def test_extra_observed_fields_do_not_count_as_missing_supplied_fields():
    response = httpx.Response(200,text=HTML,request=httpx.Request("GET",URL))
    identity = SourceResolver._landing_metadata_identity(response,expected_doi=DOI,
        expected_title=TITLE,expected_author=None,expected_year=None)
    assert identity and identity["observed"]["authors"] == ["Example, Alice"]


def test_title_alone_cannot_manufacture_confirmed_metadata_identity():
    response = httpx.Response(200,text=HTML,request=httpx.Request("GET",URL))
    assert SourceResolver._landing_metadata_identity(response,expected_doi=None,
        expected_title=TITLE,expected_author=None,expected_year=None) is None


@pytest.mark.parametrize("extra", [
    {"source_kind": "journal_article"},
    {"isbn": "9781234567897"},
])
def test_landing_identity_checks_complete_active_reference_before_promotion(extra):
    response = httpx.Response(200, text=HTML, request=httpx.Request("GET", URL))
    expected = ExpectedBibliographicFields(
        title=TITLE, doi=DOI, authors=["Example, Alice"], year="2020", **extra,
    )
    token = _ACTIVE_DISCOVERY_TRACE.set({"expected": expected})
    try:
        assert SourceResolver._landing_metadata_identity(
            response, expected_doi=DOI, expected_title=TITLE,
            expected_author="Example, Alice", expected_year="2020",
        ) is None
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)


def test_landing_identity_with_complete_matching_active_reference_is_preserved():
    response = httpx.Response(200, text=HTML, request=httpx.Request("GET", URL))
    token = _ACTIVE_DISCOVERY_TRACE.set({"expected": ExpectedBibliographicFields(
        title=TITLE, doi=DOI, authors=["Example, Alice"], year="2020",
    )})
    try:
        assert SourceResolver._landing_metadata_identity(
            response, expected_doi=DOI, expected_title=TITLE,
            expected_author="Example, Alice", expected_year="2020",
        ) is not None
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)


def test_parse_review_blocks_landing_identity_even_with_exact_fields():
    response = httpx.Response(200, text=HTML, request=httpx.Request("GET", URL))
    token = _ACTIVE_DISCOVERY_TRACE.set({"expected": ExpectedBibliographicFields(
        title=TITLE, doi=DOI, authors=["Example, Alice"], year="2020",
        reference_parse_review=True,
    )})
    try:
        assert SourceResolver._landing_metadata_identity(
            response, expected_doi=DOI, expected_title=TITLE,
            expected_author="Example, Alice", expected_year="2020",
        ) is None
    finally:
        _ACTIVE_DISCOVERY_TRACE.reset(token)


def test_landing_component_fields_survive_discovery_round_trip():
    html = HTML.replace('</head>', '<meta name="citation_inbook_title" content="A collection">'
                        '<meta name="citation_firstpage" content="10">'
                        '<meta name="citation_lastpage" content="20"></head>')
    _, trace, record = acquire('exa', html=html)
    observed = trace['candidates'][0]['observed']
    assert observed['container_title'] == 'A collection'
    assert observed['pages'] == '10-20'
    assert record['outcome'] == 'confirmed'


@pytest.mark.parametrize("provider", ["brave", "exa"])
def test_generic_results_shell_is_unresolved_not_a_rejected_work(provider):
    # Structure observed in a real catalog shell, without its content or URL.
    html = '''<html><head><title>Results - Catalog Connection</title>
    <meta name="viewport" content="width=device-width"></head>
    <body><div id="app"></div><script src="/app.js"></script></body></html>'''
    result, trace, record = acquire(provider, html=html)
    assert record["outcome"] == "search_incomplete"
    if provider == "brave":
        audit = result.metadata["transient_search_audits"][0]
        assert audit["unresolved"] == 1 and audit["identity_rejected"] == 0
        assert not trace["candidates"] and not result.locations
        assert URL not in json.dumps(asdict(result))
    else:
        candidate = trace["candidates"][0]
        assert candidate["acquisition_outcome"] == "identity_unconfirmed"
        assert candidate["disposition_reason_code"] == "landing_page_bibliography_unavailable"


@pytest.mark.parametrize("provider", ["brave", "exa"])
def test_independent_bibliographic_title_conflict_still_rejects(provider):
    result, trace, record = acquire(provider, html=HTML.replace(TITLE, "Unrelated marine biology study"))
    assert not result.full_text
    if provider == "brave":
        audit = result.metadata["transient_search_audits"][0]
        assert audit["identity_rejected"] == 1 and audit["unresolved"] == 0
    else:
        assert trace["candidates"][0]["acquisition_outcome"] == "identity_rejected"


@pytest.mark.parametrize("provider", ["brave", "exa"])
@pytest.mark.parametrize("year", ["n.d.", "N.D."])
def test_unknown_date_does_not_block_independently_matching_identity(provider, year):
    _, trace, record = acquire(provider, expected_year=year)
    assert record["outcome"] == "confirmed"
    assert trace["expected"]["year"] == year
    comparison = next(c for c in trace["candidates"][0]["comparisons"] if c["field_name"] == "year")
    assert comparison["outcome"] == "unknown"  # Never manufacture a date agreement.
    assert assess_reference_discovery_trace(json.loads(json.dumps(trace))).record.outcome == "confirmed"


def test_a_supplied_year_still_requires_observed_year():
    missing = HTML.replace('<meta name="citation_publication_date" content="2020-01-01">', '')
    _, trace, record = acquire("brave", html=missing)
    assert not trace["candidates"] and record["outcome"] == "search_incomplete"


def test_a_same_titled_page_by_someone_else_keeps_its_byline_without_its_address():
    # Kozlovic, 2026-09-30: the page's own title and byline are kept as an
    # observation (never identity), so the author difference can be shown.
    # Its first version read the absent address and failed the paper run.
    html = HTML.replace("Example, Alice", "Different, Bob").replace(
        f'<meta name="citation_doi" content="{DOI}">', '')
    result, trace, record = acquire("exa", html=html)
    assert record["outcome"] != "confirmed"
    [candidate] = trace["candidates"]
    assert candidate["observed"]["title"] == TITLE and candidate["observed"]["authors"] == ["Different, Bob"]
    assert candidate["identity_evidence_kind"] is None and not candidate["validated_identity_content_sha256"]
    attempt = result.metadata["location_attempts"][0]
    assert "source_url" not in attempt["landing_page_observation"]


def test_a_reused_page_inspection_keeps_the_page_observation():
    # The same page returned by two search engines is inspected once and its
    # fields are copied to the second; the byline observation must be among them.
    import inspect
    from app.services import source_resolver
    from app.services.search import transient
    for module in (source_resolver, transient):
        assert '"landing_metadata_observation", "landing_page_observation",' in inspect.getsource(module)


def test_an_old_page_that_states_the_cited_article_is_kept_as_its_text():
    # Hess, Jump Cut archive, 2026-09-30: no metadata or marked article body, a
    # web-page type, but its title element, byline and source statement name
    # the cited article. Its text is kept, with completeness not established.
    from app.services.source_type import SourceKindAssessment
    body = ' '.join(['The essay argues about rivers and plains in considerable detail.'] * 60)
    html = ('<html><head><title>Rivers of the Northern Plains by Mary Stone</title></head><body>'
            '<h1>RIVER REVIEW</h1><p>Rivers of the Northern Plains by Mary Stone from River Review, '
            f'no. 3, 1976, pp. 4-9</p><p>{body}</p></body></html>')
    response = httpx.Response(200, text=html, headers={"content-type": "text/html"},
                              request=httpx.Request("GET", URL))
    resolver = SourceResolver.__new__(SourceResolver)
    result = RetrievalResult(source_name="web_search", success=True, title="Rivers of the Northern Plains",
        locations=[AcquisitionLocation(url=URL, provider="web_search", representation_kind=RepresentationKind.HTML,
                                       metadata={"search_provider": "exa"})], metadata={})
    with patch("app.services.source_resolver.safe_request", return_value=response):
        accepted = resolver._acquire_from_locations(
            result, expected_title="Rivers of the Northern Plains", expected_doi=None,
            expected_author="Stone, M", expected_year="1976",
            expected_source_kind=SourceKindAssessment("journal_article", "high", ("reference structure",)))
    # About 660 words: kept as limited text, too short to be judged a whole article.
    assert accepted and result.representation.completeness == "not_assessed"
    assert "rivers and plains" in result.representation.content.decode()
    # The same page naming another author is not kept.
    other = SourceResolver.__new__(SourceResolver)
    result = RetrievalResult(source_name="web_search", success=True, title="Rivers of the Northern Plains",
        locations=[AcquisitionLocation(url=URL, provider="web_search", representation_kind=RepresentationKind.HTML,
                                       metadata={"search_provider": "exa"})], metadata={})
    with patch("app.services.source_resolver.safe_request", return_value=response):
        assert not other._acquire_from_locations(
            result, expected_title="Rivers of the Northern Plains", expected_doi=None,
            expected_author="Lake, T", expected_year="1976",
            expected_source_kind=SourceKindAssessment("journal_article", "high", ("reference structure",)))


@pytest.mark.parametrize("kind,complete", [("journal_article", True), ("monograph", False)])
def test_a_long_page_stating_the_cited_article_is_complete_only_for_a_fitting_kind(kind, complete):
    # Rule A with kind lengths, owner decision 2026-09-30 (Hess).
    from app.services.source_type import SourceKindAssessment
    body = " ".join(["The essay argues about rivers and plains in considerable detail."] * 200)
    html = ('<html><head><title>Rivers of the Northern Plains by Mary Stone</title></head><body>'
            '<h1>RIVER REVIEW</h1><p>Rivers of the Northern Plains by Mary Stone from River Review, '
            f'no. 3, 1976, pp. 4-9</p><p>{body}</p></body></html>')
    response = httpx.Response(200, text=html, headers={"content-type": "text/html"},
                              request=httpx.Request("GET", URL))
    resolver = SourceResolver.__new__(SourceResolver)
    result = RetrievalResult(source_name="web_search", success=True, title="Rivers of the Northern Plains",
        locations=[AcquisitionLocation(url=URL, provider="web_search", representation_kind=RepresentationKind.HTML,
                                       metadata={"search_provider": "exa"})], metadata={})
    with patch("app.services.source_resolver.safe_request", return_value=response):
        accepted = resolver._acquire_from_locations(
            result, expected_title="Rivers of the Northern Plains", expected_doi=None,
            expected_author="Stone, M", expected_year="1976",
            expected_source_kind=SourceKindAssessment(kind, "high", ("reference structure",)))
    if complete:
        assert accepted and result.representation.completeness == "complete"
        marker = result.representation.metadata["stated_page_completeness"]
        assert marker["rule"] == "stated-page-coverage-v1" and marker["cited_kind"] == kind and marker["words"] > 1000
    else:
        assert not accepted or result.representation.completeness != "complete"
