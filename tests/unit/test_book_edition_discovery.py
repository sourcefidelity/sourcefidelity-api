"""Metadata enrichment is edition-specific and never substitutes for acquisition."""
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from app.services.retrieval.base import RetrievalResult
from app.services.retrieval.google_books import GoogleBooksRetriever
from app.services.schemas import ParsedReference
from app.services.source_resolver import SourceResolver, SourceResolutionError


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
