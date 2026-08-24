"""Google Books bibliographic metadata and exact-edition page lookup.

Google Books is metadata/candidate evidence, not a trusted full-text source.
Title/author search is useful for discovery; only an exact ISBN match may vote
on PDF completeness.
"""

import logging
from dataclasses import dataclass
import re

import httpx

from app.config import settings
from app.services.book_metadata import isbn10_to_isbn13, normalize_isbn

logger = logging.getLogger(__name__)

GOOGLE_BOOKS_BASE = "https://www.googleapis.com/books/v1/volumes"


@dataclass
class PageCountResult:
    """Result of a page-count lookup."""

    success: bool
    expected_pages: int | None = None
    source: str = "Google Books"
    title: str | None = None
    isbn: str | None = None
    match_confidence: str = "none"
    match_reason: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class BookMetadata:
    volume_id: str
    title: str | None
    subtitle: str | None
    authors: tuple[str, ...]
    publisher: str | None
    published_date: str | None
    description: str | None
    identifiers: tuple[str, ...]
    page_count: int | None
    print_type: str | None
    preview_link: str | None
    info_link: str | None
    match_confidence: str
    match_reason: str


class GoogleBooksRetriever:
    """Search Google Books while keeping discovery separate from admission."""

    def search_metadata(
        self,
        *,
        isbn: str | None = None,
        title: str | None = None,
        author: str | None = None,
        publisher: str | None = None,
        year: str | None = None,
        max_results: int = 10,
    ) -> list[BookMetadata]:
        normalized_isbn = normalize_isbn(isbn) if isbn else None
        if isbn and normalized_isbn is None:
            return []
        if normalized_isbn:
            query = f"isbn:{normalized_isbn}"
        else:
            terms = []
            if title:
                terms.append(f"intitle:{title.strip()}")
            if author:
                terms.append(f"inauthor:{author.strip()}")
            if publisher:
                terms.append(f"inpublisher:{publisher.strip()}")
            if not terms:
                return []
            query = " ".join(terms)

        params: dict[str, str | int] = {
            "q": query,
            "maxResults": max(1, min(max_results, 20)),
            "printType": "books",
        }
        if settings.GOOGLE_BOOKS_API_KEY:
            params["key"] = settings.GOOGLE_BOOKS_API_KEY
        try:
            response = httpx.get(GOOGLE_BOOKS_BASE, params=params, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.debug("Google Books metadata search failed: %s", exc)
            return []

        results: list[BookMetadata] = []
        for item in payload.get("items") or []:
            info = item.get("volumeInfo") or {}
            identifiers = tuple(
                normalized
                for record in (info.get("industryIdentifiers") or [])
                if (normalized := normalize_isbn(str(record.get("identifier") or "")))
            )
            confidence, reason = _metadata_match(
                identifiers=identifiers,
                candidate_title=info.get("title"),
                candidate_authors=info.get("authors") or [],
                candidate_publisher=info.get("publisher"),
                candidate_date=info.get("publishedDate"),
                expected_isbn=normalized_isbn,
                expected_title=title,
                expected_author=author,
                expected_publisher=publisher,
                expected_year=year,
            )
            page_count = info.get("pageCount")
            results.append(
                BookMetadata(
                    volume_id=str(item.get("id") or ""),
                    title=info.get("title"),
                    subtitle=info.get("subtitle"),
                    authors=tuple(str(value) for value in (info.get("authors") or [])),
                    publisher=info.get("publisher"),
                    published_date=info.get("publishedDate"),
                    description=info.get("description"),
                    identifiers=identifiers,
                    page_count=(
                        int(page_count)
                        if isinstance(page_count, int) and page_count > 0
                        else None
                    ),
                    print_type=info.get("printType"),
                    preview_link=info.get("previewLink"),
                    info_link=info.get("infoLink"),
                    match_confidence=confidence,
                    match_reason=reason,
                )
            )
        rank = {"high": 3, "medium": 2, "low": 1, "none": 0}
        return sorted(
            results,
            key=lambda result: (
                rank[result.match_confidence],
                result.page_count is not None,
            ),
            reverse=True,
        )

    def lookup_page_count(
        self,
        *,
        isbn: str | None = None,
        title: str | None = None,
        author: str | None = None,
    ) -> PageCountResult:
        # Title/author search is supported by ``search_metadata`` but cannot
        # establish the edition-specific extent needed for rejection.
        if not isbn:
            return PageCountResult(
                success=False,
                error="Exact ISBN required for completeness page count",
            )
        normalized = normalize_isbn(isbn)
        if normalized is None:
            return PageCountResult(success=False, error="Invalid ISBN")
        for result in self.search_metadata(
            isbn=normalized,
            title=title,
            author=author,
        ):
            if result.match_confidence == "high" and result.page_count:
                return PageCountResult(
                    success=True,
                    expected_pages=result.page_count,
                    title=result.title,
                    isbn=normalized,
                    match_confidence="high",
                    match_reason=result.match_reason,
                )
        return PageCountResult(
            success=False,
            isbn=normalized,
            error="No identity-compatible exact-edition page count",
        )


def _tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) >= 3
    }


def _metadata_match(
    *,
    identifiers: tuple[str, ...],
    candidate_title: str | None,
    candidate_authors: list,
    candidate_publisher: str | None,
    candidate_date: str | None,
    expected_isbn: str | None,
    expected_title: str | None,
    expected_author: str | None,
    expected_publisher: str | None,
    expected_year: str | None,
) -> tuple[str, str]:
    if expected_isbn:
        canonical = isbn10_to_isbn13(expected_isbn)
        candidate_canonical = {
            isbn10_to_isbn13(identifier) for identifier in identifiers
        }
        if candidate_canonical and canonical not in candidate_canonical:
            return "none", "Google Books record did not return the queried ISBN"

    title_expected = _tokens(expected_title)
    title_candidate = _tokens(candidate_title)
    title_overlap = (
        len(title_expected & title_candidate) / len(title_expected)
        if title_expected
        else None
    )
    if title_overlap is not None and title_overlap < 0.60:
        return "none", f"title overlap too low ({title_overlap:.2f})"

    support: list[str] = []
    if expected_author:
        surname = expected_author.split(",", 1)[0].split()[-1].casefold()
        if any(surname in str(author).casefold() for author in candidate_authors):
            support.append("author")
    if expected_publisher and _tokens(expected_publisher) & _tokens(candidate_publisher):
        support.append("publisher")
    if expected_year and candidate_date and expected_year[:4] == candidate_date[:4]:
        support.append("year")

    if expected_isbn:
        # Google Books occasionally omits ``industryIdentifiers`` from a
        # result returned by an ``isbn:`` query.  In that case the query itself
        # is identifier evidence, but cited title agreement is mandatory.
        if not identifiers and (title_overlap is None or title_overlap < 0.80):
            return "none", "ISBN-query result omitted identifiers and lacked strong title evidence"
        detail = "exact ISBN" if identifiers else "exact ISBN query + title confirmation"
        if title_overlap is not None:
            detail += f" + title overlap {title_overlap:.2f}"
        if support:
            detail += " + " + "/".join(support)
        return "high", detail
    if title_overlap is not None and title_overlap >= 0.80 and support:
        return "medium", f"title overlap {title_overlap:.2f} + {'/'.join(support)}"
    if title_overlap is not None and title_overlap >= 0.75:
        return "low", f"title-only candidate overlap {title_overlap:.2f}"
    return "none", "insufficient bibliographic identity evidence"
