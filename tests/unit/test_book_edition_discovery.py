"""Metadata enrichment is edition-specific and never substitutes for acquisition."""
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from app.services.retrieval.base import RetrievalResult
from app.services.retrieval.google_books import GoogleBooksRetriever
from app.services.schemas import ParsedReference
from app.services.source_resolver import SourceResolver, SourceResolutionError
from app.services.bibliographic_scripts import cross_script_comparison_unresolved


def _volume(year="1983", identifiers=True):
    info = {
        "title": "Cinema", "subtitle": "A history of the screen",
        "authors": ["Alex Morgan"], "publishedDate": year,
        "publisher": "Example Press", "pageCount": 240,
        "description": "Catalog description, not source evidence.",
    }
    if identifiers:
        info["industryIdentifiers"] = [{"identifier": "0306406152", "type": "ISBN_10"}]
    return {"id": "volume-" + year, "volumeInfo": info}


@pytest.mark.parametrize("left,right,unresolved", [
    ("Economic history", "经济史", True),
    ("Economic history 2013", "经济史 2013", True),
    ("Istoriya", "История", True),
    ("Logos", "λόγος", True),
    ("Ｈｉｓｔｏｒｙ", "History", False),
    ("History", "Unrelated chemistry", False),
    ("经济史", "经济史", False),
    ("", "经济史", False),
    ("123", "经济史", False),
    ("经济史 Economic history", "Economic history", False),
])
def test_script_comparability_is_not_translation(left, right, unresolved):
    assert cross_script_comparison_unresolved(left, right) is unresolved


@pytest.mark.parametrize("author,year", [("Alex Morgan", "1984"), ("Different Writer", "2020")])
def test_cross_script_metadata_is_unknown_not_confirmed_or_exhausted(monkeypatch, author, year):
    volume = _volume(year)
    volume['volumeInfo'].update(title="电影史", subtitle=None, authors=[author])
    resolver, reference, _, _ = _setup(monkeypatch, {"items": [volume]})
    reference.source_kind = 'monograph'
    result = resolver.resolve_reference(reference)
    candidate = result.metadata['reference_discovery']['candidates'][0]
    title = next(c for c in candidate['comparisons'] if c['field_name'] == 'title')
    assert title['outcome'] == 'unknown'
    assert title['reason_code'] == 'cross_script_title_unresolved'
    assert candidate['acquisition_outcome'] == 'metadata_only'
    assert candidate['disposition_reason_code'] == 'cross_script_title_unresolved'
    assert not candidate['plausible_identity_match']
    assert result.metadata['reference_discovery']['outcome'] == 'search_incomplete'
    assert not result.success and result.full_text is None


def test_same_script_title_mismatch_still_rejected(monkeypatch):
    volume = _volume()
    volume['volumeInfo'].update(title="Quantum chemistry methods", subtitle=None)
    resolver, reference, _, _ = _setup(monkeypatch, {"items": [volume]})
    reference.source_kind = 'monograph'
    result = resolver.resolve_reference(reference)
    candidate = result.metadata['reference_discovery']['candidates'][0]
    assert candidate['acquisition_outcome'] == 'identity_rejected'


def test_cross_script_does_not_override_conflicting_isbn():
    from app.services.retrieval.google_books import _metadata_match
    confidence, _ = _metadata_match(identifiers=('9787508077550',),
        candidate_title='电影史', candidate_authors=[], candidate_publisher=None,
        candidate_date=None, expected_isbn='9780306406157', expected_title='Film history',
        expected_author=None, expected_publisher=None, expected_year=None)
    assert confidence == 'none'


def _setup(monkeypatch, payload=None):
    response = httpx.Response(200, json=payload or {"items": [_volume()]},
                              request=httpx.Request("GET", "https://www.googleapis.com/books/v1/volumes"))
    transport = Mock(return_value=response)
    monkeypatch.setattr("app.services.retrieval.google_books.httpx.get", transport)
    resolver = SourceResolver.__new__(SourceResolver)
    resolver._acquisition_capabilities = None
    result = RetrievalResult(source_name="test", success=False)
    resolver.resolve = Mock(return_value=result)
    reference = ParsedReference(
        reference_id="ref-book", title="Cinema: A history of the screen",
        author="Morgan, A.", year="1984", source_kind="webpage",
        raw_ref="Morgan, A. (1984). Cinema: A history of the screen. New York: Example Press. https://example.org/book",
    )
    return resolver, reference, result, transport


def test_subtitle_identity_and_exact_query_trace(monkeypatch):
    resolver, reference, result, transport = _setup(monkeypatch)
    output = resolver.resolve_reference(reference)
    assert output is result and not output.success
    assert output.full_text is None and output.abstract is None
    transport.assert_called_once()
    assert "key" not in output.metadata["reference_discovery_trace"]["queries"][0]["normalized_query"]
    record = output.metadata["reference_discovery"]
    candidate = record["candidates"][0]
    assert candidate["observed"]["title"] == reference.title
    assert candidate["edition_metadata"]["publisher"] == "Example Press"
    assert candidate["edition_metadata"]["published_date"] == "1983"
    assert len(candidate["edition_metadata"]["record_sha256"]) == 64
    assert candidate["edition_metadata"]["edition_binding"] == "unresolved"
    assert not candidate["location_available"]
    assert candidate["acquisition_outcome"] == "metadata_only"
    assert record["outcome"] == "possible_match"
    assert not record["contributes_to_neutral_pattern"]
    year = next(c for c in candidate["comparisons"] if c["field_name"] == "year")
    assert year["outcome"] == "unknown"
    assert record["limitations"]
    assert record["expected"]["source_kind"] == "webpage"


def test_alternate_publication_years_remain_separate_candidates(monkeypatch):
    resolver, reference, _, _ = _setup(monkeypatch, {"items": [_volume("1983"), _volume("1984")]})
    record = resolver.resolve_reference(reference).metadata["reference_discovery"]
    assert len(record["attempts"]) == 1
    assert not record["attempts"][0]["required"]
    assert len(record["candidates"]) == 2
    assert len({c["candidate_id"] for c in record["candidates"]}) == 2
    assert {c["observed"]["year"] for c in record["candidates"]} == {"1983", "1984"}
    assert not record["contributes_to_neutral_pattern"]


@pytest.mark.parametrize('mode', ['positive', 'single', 'duplicate', 'rejected', 'matching', 'multiple', 'stale', 'hash', 'unpermitted'])
def test_catalog_year_discrepancy_is_separate_from_edition_identity(monkeypatch, mode):
    from copy import deepcopy
    from app.services.reference_formatting import book_publication_year_discrepancy
    years = ['1983','1982'] if mode == 'multiple' else ['1984'] if mode == 'matching' else ['1983']
    volumes = [_volume(y) for y in years]
    if mode != 'single':
        second = deepcopy(volumes[0])
        second['id'] += '-independent-volume'
        volumes.append(second)
    resolver, reference, _, _ = _setup(monkeypatch, {'items':volumes})
    reference.source_kind = 'monograph'
    record = resolver.resolve_reference(reference).metadata['reference_discovery']
    if mode == 'duplicate': record['candidates'][1]['edition_metadata'] = deepcopy(record['candidates'][0]['edition_metadata'])
    if mode == 'stale': reference.year = '2000'
    if mode == 'hash': record['candidates'][0]['comparisons'][1]['expected_sha256'] = '0'*64
    if mode == 'unpermitted': record['attempts'][0]['permitted'] = False
    if mode == 'rejected': record['candidates'][0]['acquisition_outcome'] = 'identity_rejected'
    before = deepcopy(record)
    finding = book_publication_year_discrepancy(reference, record)
    assert bool(finding) == (mode == 'positive')
    assert record == before
    if finding:
        assert finding['field_difference'] == {'field_name':'year', 'submitted_value':'1984', 'located_value':'1983'}
        assert finding['exact_edition_established'] is False
        assert record['outcome'] == 'possible_match'


@pytest.mark.parametrize('damage', [None, 'rejected', 'stale', 'unpermitted'])
def test_only_credible_matching_year_neutralizes_other_edition_notice(monkeypatch, damage):
    from app.services.evidence_report import _identity_view
    resolver, reference, _, _ = _setup(monkeypatch, {'items':[_volume('1983'),_volume('1984')]})
    record=resolver.resolve_reference(reference).metadata['reference_discovery']
    candidate=next(c for c in record['candidates'] if c['observed']['year']=='1984')
    if damage=='rejected': candidate['acquisition_outcome']='identity_rejected'
    if damage=='stale': next(c for c in candidate['comparisons'] if c['field_name']=='title')['expected_sha256']='0'*64
    if damage=='unpermitted': record['attempts'][0]['permitted']=False
    assert _identity_view(record)['edition_year_unresolved'] == bool(damage)


@pytest.mark.parametrize('damage',[None,'author','hash','unrelated','not_catalog'])
def test_catalog_short_main_title_with_submitted_year_prevents_false_year_flag(monkeypatch,damage):
    from copy import deepcopy
    from app.services.reference_formatting import book_publication_year_discrepancy
    old=_volume('1983');old['volumeInfo']['title']='The Cinema History'
    second=deepcopy(old);second['id']='second-1983'
    matching=_volume('1984');matching['volumeInfo'].update(title='Cinema History',subtitle=None)
    if damage=='author':matching['volumeInfo']['authors']=['Different Writer']
    if damage=='unrelated':matching['volumeInfo']['title']='Another Cinema Book'
    resolver,reference,_,_=_setup(monkeypatch,{'items':[old,second,matching]})
    reference.source_kind='monograph';reference.title='The Cinema History: A history of the screen'
    record=resolver.resolve_reference(reference).metadata['reference_discovery']
    c=next(c for c in record['candidates'] if c['observed']['year']=='1984')
    if damage=='hash':next(x for x in c['comparisons'] if x['field_name']=='title')['expected_sha256']='0'*64
    if damage=='not_catalog':c['disposition_reason_code']='source_identity_rejected'
    original=deepcopy(record)
    assert bool(book_publication_year_discrepancy(reference,record))==bool(damage)
    assert record==original


def test_same_observed_isbn_can_bind_year_conflict(monkeypatch):
    resolver, reference, _, _ = _setup(monkeypatch)
    record = resolver.resolve_reference(reference, isbn="9780306406157").metadata["reference_discovery"]
    candidate = record["candidates"][0]
    assert candidate["edition_metadata"]["edition_binding"] == "same_isbn"
    assert record["outcome"] == "bibliographic_conflict"


def test_query_isbn_missing_from_response_cannot_bind_edition(monkeypatch):
    resolver, reference, _, _ = _setup(monkeypatch, {"items": [_volume(identifiers=False)]})
    record = resolver.resolve_reference(reference, isbn="9780306406157").metadata["reference_discovery"]
    assert record["outcome"] == "possible_match"
    assert record["candidates"][0]["observed"]["isbn"] == ""
    assert record["candidates"][0]["edition_metadata"]["edition_binding"] == "unresolved"


def test_missing_observed_isbn_cannot_confirm_even_when_year_agrees(monkeypatch):
    resolver, reference, _, _ = _setup(monkeypatch, {"items": [_volume("1984", identifiers=False)]})
    record = resolver.resolve_reference(reference, isbn="9780306406157").metadata["reference_discovery"]
    assert record["outcome"] == "possible_match"


def test_optional_empty_lookup_does_not_complete_required_academic_route(monkeypatch):
    resolver, reference, _, _ = _setup(monkeypatch, {"totalItems": 0})
    reference.source_kind = "monograph"
    result = resolver.resolve_reference(reference)
    record = result.metadata["reference_discovery"]
    assert record["attempts"][0]["outcome"] == "no_match"
    assert not record["attempts"][0]["required"]
    assert record["outcome"] == "search_incomplete"


def test_timeout_remains_operational_and_does_not_interrupt_resolution(monkeypatch):
    resolver, reference, result, transport = _setup(monkeypatch)
    transport.side_effect = httpx.ReadTimeout("private request context")
    output = resolver.resolve_reference(reference)
    assert output is result
    assert output.metadata["reference_discovery_trace"]["queries"][0]["execution_outcome"] == "timeout"
    assert "private request context" not in str(output.metadata)


@pytest.mark.parametrize("status,outcome", [(429,"rate_limited"),(403,"access_restricted"),(500,"operational_failure")])
def test_failed_lookup_is_not_completed_no_match(monkeypatch, status, outcome):
    resolver, reference, _, transport = _setup(monkeypatch)
    reference.source_kind = "monograph"
    transport.return_value = httpx.Response(status, request=httpx.Request("GET", "https://www.googleapis.com/books/v1/volumes?key=private-test"))
    result = resolver.resolve_reference(reference)
    trace = result.metadata["reference_discovery_trace"]
    assert trace["queries"][0]["execution_outcome"] == outcome
    assert trace["attempts"][0]["outcome"] != "no_match"
    assert not trace["candidates"]
    assert result.metadata["reference_discovery"]["outcome"] == "search_incomplete"
    assert "private-test" not in str(result.metadata)


@pytest.mark.parametrize("payload", [{}, [], {"items": "invalid"}, {"items": [None]}, {"items": [{"id":"v", "volumeInfo":{"authors":None}}]}])
def test_malformed_provider_payload_is_typed(monkeypatch, payload):
    _setup(monkeypatch)
    monkeypatch.setattr("app.services.retrieval.google_books.httpx.get", lambda *a, **kw: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: payload,
    ))
    result = GoogleBooksRetriever().search_metadata_result(title="Cinema")
    assert result.outcome == "response_invalid"
    assert not result.candidates


def test_failed_source_acquisition_retains_edition_discovery(monkeypatch):
    resolver, reference, _, _ = _setup(monkeypatch)
    resolver.resolve.side_effect = SourceResolutionError("not retrieved")
    with pytest.raises(SourceResolutionError) as caught:
        resolver.resolve_reference(reference)
    assert caught.value.reference_discovery["candidates"][0]["edition_metadata"]


def test_unrelated_google_volume_cannot_generate_reference_error(monkeypatch):
    item = _volume()
    item["volumeInfo"].update(title="Another book", subtitle="unrelated", authors=["Someone Else"])
    resolver, reference, _, _ = _setup(monkeypatch, {"items": [item]})
    result = resolver.resolve_reference(reference)
    trace = result.metadata["reference_discovery_trace"]
    assert trace["candidates"][0]["acquisition_outcome"] == "identity_rejected"
    assert "reference_discovery" not in result.metadata  # no policy for this legacy kind


def test_restricted_acquisition_profile_does_not_add_metadata_calls(monkeypatch):
    resolver, reference, _, transport = _setup(monkeypatch)
    resolver._acquisition_capabilities = {"student_url"}
    resolver.resolve_reference(reference)
    transport.assert_not_called()


def test_journal_reference_does_not_trigger_book_lookup(monkeypatch):
    resolver, reference, _, transport = _setup(monkeypatch)
    reference.source_kind = "journal_article"
    reference.raw_ref = "Morgan (1984). Cinema: A history of the screen. Journal, 10, 12-20."
    resolver.resolve_reference(reference)
    transport.assert_not_called()
