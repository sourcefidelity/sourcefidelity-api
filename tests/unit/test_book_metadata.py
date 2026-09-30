"""Book identifier, source-kind, and Google Books metadata regressions."""

from types import SimpleNamespace

import fitz

from app.services.book_metadata import (
    content_sha256,
    document_kind_for_source_kind,
    extract_isbn_candidates,
    isbn10_to_isbn13,
    is_valid_isbn,
    normalize_source_kind,
)
from app.services.retrieval.google_books import GoogleBooksRetriever


def _book_pdf(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 550, 740), text, fontsize=11)
    payload = document.tobytes()
    document.close()
    return payload


def test_isbn_extraction_validates_and_collapses_equivalent_forms():
    payload = _book_pdf(
        "Copyright page\nISBN 0-306-40615-2\nISBN-13: 978-0-306-40615-7"
    )

    report = extract_isbn_candidates(payload)

    assert len(report.candidates) == 1
    assert report.candidates[0].canonical_isbn13 == "9780306406157"
    assert report.candidates[0].labelled_occurrences == 1
    assert report.content_sha256 == content_sha256(payload)


def test_unlabelled_isbn10_like_number_is_ignored():
    report = extract_isbn_candidates(_book_pdf("Account number 0306406152"))

    assert report.candidates == ()


def test_isbn_checksums_and_conversion():
    assert is_valid_isbn("0-306-40615-2")
    assert is_valid_isbn("978-0-306-40615-7")
    assert not is_valid_isbn("978-0-306-40615-8")
    assert isbn10_to_isbn13("0-306-40615-2") == "9780306406157"


def test_zotero_like_source_kinds_map_to_completeness_rules():
    assert normalize_source_kind("edited collection") == "edited_collection"
    assert document_kind_for_source_kind("edited_collection") == "book"
    assert document_kind_for_source_kind("book_section") == "chapter"
    assert document_kind_for_source_kind("conference_paper") == "article"
    assert document_kind_for_source_kind("report") == "unknown"


def test_google_books_exact_isbn_returns_full_metadata(monkeypatch):
    payload = {
        "items": [
            {
                "id": "volume-1",
                "volumeInfo": {
                    "title": "Example Book",
                    "subtitle": "A Study",
                    "authors": ["Alex Rivera"],
                    "publisher": "Example Press",
                    "publishedDate": "2024",
                    "description": "A bibliographic description.",
                    "industryIdentifiers": [
                        {"type": "ISBN_13", "identifier": "9780306406157"}
                    ],
                    "pageCount": 240,
                    "printType": "BOOK",
                    "previewLink": "https://books.example/preview",
                    "infoLink": "https://books.example/info",
                },
            }
        ]
    }
    calls = []

    def fake_get(_url, *, params, timeout, headers=None):
        calls.append((params, timeout))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)

    monkeypatch.setattr("app.services.retrieval.google_books.httpx.get", fake_get)

    result = GoogleBooksRetriever().lookup_page_count(
        isbn="9780306406157",
        title="Example Book",
        author="Rivera, Alex",
    )

    assert result.success is True
    assert result.expected_pages == 240
    assert result.match_confidence == "high"
    assert calls[0][0]["q"] == "isbn:9780306406157"


def test_google_books_title_search_is_discovery_not_exact_edition(monkeypatch):
    payload = {
        "items": [
            {
                "id": "volume-1",
                "volumeInfo": {
                    "title": "Example Book",
                    "authors": ["Alex Rivera"],
                    "pageCount": 240,
                    "printType": "BOOK",
                },
            }
        ]
    }

    monkeypatch.setattr(
        "app.services.retrieval.google_books.httpx.get",
        lambda _url, **_kwargs: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: payload,
        ),
    )
    retriever = GoogleBooksRetriever()

    candidates = retriever.search_metadata(
        title="Example Book",
        author="Alex Rivera",
    )
    page_count = retriever.lookup_page_count(
        title="Example Book",
        author="Alex Rivera",
    )

    assert candidates[0].match_confidence == "medium"
    assert candidates[0].page_count == 240
    assert page_count.success is False


def test_google_books_rejects_queried_isbn_record_with_wrong_title(monkeypatch):
    payload = {
        "items": [
            {
                "id": "wrong",
                "volumeInfo": {
                    "title": "Unrelated Work",
                    "authors": ["Someone Else"],
                    "industryIdentifiers": [
                        {"type": "ISBN_13", "identifier": "9780306406157"}
                    ],
                    "pageCount": 500,
                },
            }
        ]
    }
    monkeypatch.setattr(
        "app.services.retrieval.google_books.httpx.get",
        lambda _url, **_kwargs: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: payload,
        ),
    )

    result = GoogleBooksRetriever().lookup_page_count(
        isbn="9780306406157",
        title="Example Book",
    )

    assert result.success is False


def test_google_books_isbn_query_without_observed_identifiers_cannot_bind_extent(monkeypatch):
    payload = {
        "items": [
            {
                "id": "volume-with-omitted-identifiers",
                "volumeInfo": {
                    "title": "Example Book",
                    "authors": ["Alex Rivera"],
                    "pageCount": 240,
                },
            }
        ]
    }
    monkeypatch.setattr(
        "app.services.retrieval.google_books.httpx.get",
        lambda _url, **_kwargs: SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: payload,
        ),
    )

    result = GoogleBooksRetriever().lookup_page_count(
        isbn="9780306406157",
        title="Example Book",
        author="Rivera, Alex",
    )

    assert result.success is False
    assert result.expected_pages is None
