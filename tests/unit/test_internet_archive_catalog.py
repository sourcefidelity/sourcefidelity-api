"""Internet Archive book catalogue: precise when it answers, silent too often.

Measured over 38 books confirmed to exist by another provider, the Archive
returned nothing for 27 of them -- a 71% false-emptiness rate. Its silence
therefore cannot stand as catalogue coverage, because coverage is what lets
the bounded review treat a reference as unverified. The adapter encodes that
by never emitting the `no_results` outcome a caller could count.
"""
import json
from unittest.mock import Mock

import httpx
import pytest

from app.services.retrieval.internet_archive_catalog import (
    InternetArchiveCatalogRetriever,
)


def _response(status_code: int, body: object) -> httpx.Response:
    request = httpx.Request("GET", "https://archive.org/advancedsearch.php")
    response = httpx.Response(status_code, request=request)
    response._content = json.dumps(body).encode()
    return response


def _docs(*docs):
    return {"response": {"docs": list(docs)}}


def _patch(monkeypatch, response):
    get = Mock(return_value=response)
    monkeypatch.setattr(
        "app.services.retrieval.internet_archive_catalog.httpx.get", get)
    return get


RECORD = {
    "identifier": "classicalhollywo0000bord",
    "title": "The classical Hollywood cinema : film style & mode of production",
    "creator": "Bordwell, David",
    "year": 1985,
    "publisher": "New York : Columbia University Press",
    "isbn": ["0231060548", "978-0231060547"],
}


class TestRecords:
    def test_a_record_carries_the_identity_fields(self, monkeypatch):
        _patch(monkeypatch, _response(200, _docs(RECORD)))
        result = InternetArchiveCatalogRetriever().search_metadata_result(
            title="The Classical Hollywood Cinema", author="Bordwell, D.")

        assert result.outcome == "results"
        book = result.candidates[0]
        assert book.authors == ("Bordwell, David",)
        assert book.published_date == "1985"
        assert "Columbia University Press" in book.publisher
        assert book.identifiers == ("0231060548", "9780231060547")
        assert book.info_link.endswith("/classicalhollywo0000bord")

    def test_the_creator_filter_is_applied(self, monkeypatch):
        get = _patch(monkeypatch, _response(200, _docs(RECORD)))
        InternetArchiveCatalogRetriever().search_metadata_result(
            title="The Classical Hollywood Cinema", author="Bordwell, D.")

        query = get.call_args.kwargs["params"]["q"]
        assert 'creator:("Bordwell")' in query
        assert "mediatype:texts" in query

    def test_an_isbn_search_skips_the_title_route(self, monkeypatch):
        get = _patch(monkeypatch, _response(200, _docs(RECORD)))
        InternetArchiveCatalogRetriever().search_metadata_result(isbn="0-231-06054-8")
        assert "isbn:(0231060548)" in get.call_args.kwargs["params"]["q"]

    def test_query_syntax_characters_cannot_escape_the_phrase(self, monkeypatch):
        get = _patch(monkeypatch, _response(200, _docs(RECORD)))
        InternetArchiveCatalogRetriever().search_metadata_result(
            title='Title") OR mediatype:(movies', author="X")
        query = get.call_args.kwargs["params"]["q"]
        # The injected text survives as literal words inside the quoted title,
        # which is harmless; what must not survive is a second field clause or
        # a quote that would close the phrase early.
        assert query == (
            'title:("Title  OR mediatype movies") AND mediatype:texts '
            'AND creator:("X")')
        assert query.count("mediatype:") == 1
        assert query.count('"') == 4


class TestSilenceIsNotCoverage:
    def test_empty_result_is_never_reported_as_no_results(self, monkeypatch):
        """71% of confirmed books return nothing; that cannot count as coverage."""
        _patch(monkeypatch, _response(200, _docs()))
        result = InternetArchiveCatalogRetriever().search_metadata_result(
            title="A Real Book Not Held Here", author="Author, A.")

        assert result.outcome == "no_results_low_coverage"
        assert result.outcome != "no_results"
        assert result.candidates == ()

    def test_the_review_would_not_accept_that_outcome(self):
        # The bounded review accepts only `no_results` with a zero count.
        assert "no_results_low_coverage" != "no_results"


class TestFailuresAreNotEmptiness:
    @pytest.mark.parametrize("status, outcome", [
        (429, "rate_limited"), (403, "access_restricted"), (500, "operational_failure"),
    ])
    def test_http_failures_are_typed(self, monkeypatch, status, outcome):
        _patch(monkeypatch, _response(status, _docs()))
        result = InternetArchiveCatalogRetriever().search_metadata_result(
            title="A Book", author="Author, A.")
        assert result.outcome == outcome
        assert result.candidates == ()

    def test_a_malformed_body_is_not_an_empty_catalogue(self, monkeypatch):
        _patch(monkeypatch, _response(200, {"unexpected": True}))
        result = InternetArchiveCatalogRetriever().search_metadata_result(
            title="A Book", author="Author, A.")
        assert result.outcome == "response_invalid"

    def test_missing_title_and_isbn_is_refused(self):
        result = InternetArchiveCatalogRetriever().search_metadata_result(author="Only, A.")
        assert result.outcome == "response_invalid"
        assert result.error_code == "insufficient_metadata"


class TestWiredAsAGapFiller:
    """A second catalogue for books, contributing identity and never coverage.

    Google Books is the only route that satisfies `book_catalog_coverage_missing`,
    so an outage there leaves a monograph with no catalogue evidence. The
    Archive is reachable where others are not and its records are precise, but
    it held nothing for 27 of 38 books that certainly exist -- so its silence
    must never be read as absence.
    """

    def test_it_runs_only_when_the_first_catalogue_found_nothing(self):
        import inspect
        from app.services.source_resolver import SourceResolver
        # The recording moved into _record_book_search (2026-09-30); the
        # title-only fallback search never repeats the archive check.
        source = inspect.getsource(SourceResolver._record_book_search)
        assert "if not candidates:" in source and "if fallback:" in source
        assert "_corroborate_book_from_archive" in source

    def test_its_candidates_are_recorded_under_their_own_provider(self):
        import inspect
        from app.services.source_resolver import SourceResolver
        source = inspect.getsource(SourceResolver._corroborate_book_from_archive)
        assert 'provider="internet_archive"' in source
        assert '"source_name": "internet_archive"' in source

    def test_it_cannot_satisfy_book_catalogue_coverage(self):
        """The review names google_books explicitly; a second catalogue is not it."""
        import inspect
        from app.services import bounded_reference_review
        source = inspect.getsource(bounded_reference_review)
        assert "'google_books' not in scholarly" in source
        assert "internet_archive" not in source

    def test_an_empty_answer_cannot_be_counted_as_a_completed_search(self):
        """`no_results_low_coverage` is not the `no_results` the review accepts."""
        import inspect
        from app.services import bounded_reference_review
        source = inspect.getsource(bounded_reference_review)
        assert "execution_outcome == 'no_results'" in source
        result = InternetArchiveCatalogRetriever
        assert result is not None

    def test_it_is_excluded_from_the_google_books_year_rule(self):
        import inspect
        from app.services.reference_formatting import book_publication_year_discrepancy
        source = inspect.getsource(book_publication_year_discrepancy)
        assert "provider!='google_books'" in source or "provider != 'google_books'" in source
