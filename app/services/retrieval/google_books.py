"""Google Books bibliographic metadata and exact-edition page lookup.

Google Books is metadata/candidate evidence, not a trusted full-text source.
Title/author search is useful for discovery; only an exact ISBN match may vote
on PDF completeness.
"""

import logging
import hashlib
import json
from dataclasses import dataclass
import re

import httpx

from app.config import secret_value, settings
from app.services.book_metadata import isbn10_to_isbn13, normalize_isbn
from app.services.bibliographic_scripts import cross_script_comparison_unresolved
from app.services.processing_metrics import record_provider_request

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
    record_sha256: str = ""


@dataclass(frozen=True)
class BookMetadataSearch:
    query: str
    outcome: str
    candidates: tuple[BookMetadata, ...] = ()
    error_code: str | None = None


# Exactly the fields consumed below. Keep this in step with _record_digest.
_VOLUME_FIELDS = (
    "totalItems,items(id,volumeInfo(title,subtitle,authors,publisher,"
    "publishedDate,description,industryIdentifiers,pageCount,printType,"
    "previewLink,infoLink))"
)
_HEADERS = {
    # The catalog enables gzip only when the agent string advertises it.
    "Accept-Encoding": "gzip",
    "User-Agent": "SourceFidelity/0.1 (academic source verification) (gzip)",
}
_DIGEST_VOLUME_KEYS = (
    "title", "subtitle", "authors", "publisher", "publishedDate",
    "description", "industryIdentifiers", "pageCount", "printType",
    "previewLink", "infoLink",
)


def _record_digest(item: dict, info: dict) -> str:
    """Identify a catalog record by the fields this adapter reads."""
    projection = {
        "id": item.get("id"),
        "volumeInfo": {key: info.get(key) for key in _DIGEST_VOLUME_KEYS},
    }
    return hashlib.sha256(
        json.dumps(projection, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class GoogleBooksRetriever:
    """Search Google Books while keeping discovery separate from admission."""

    def search_metadata(self, **kwargs) -> list[BookMetadata]:
        """Compatibility interface for existing page-count consumers."""
        return list(self.search_metadata_result(**kwargs).candidates)

    def search_metadata_result(
        self,
        *,
        isbn: str | None = None,
        title: str | None = None,
        author: str | None = None,
        publisher: str | None = None,
        year: str | None = None,
        max_results: int = 10,
    ) -> BookMetadataSearch:
        normalized_isbn = normalize_isbn(isbn) if isbn else None
        if isbn and normalized_isbn is None:
            return BookMetadataSearch("", "response_invalid", error_code="invalid_isbn")
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
                return BookMetadataSearch("", "response_invalid", error_code="insufficient_metadata")
            query = " ".join(terms)

        params: dict[str, str | int] = {
            "q": query,
            "maxResults": max(1, min(max_results, 20)),
            "printType": "books",
            # Ask for only the fields this adapter reads. The volume resource
            # otherwise carries sale info, access flags, thumbnail URLs and
            # search snippets that we never use but did hash, so an unrelated
            # change on the catalog's side moved our evidence hash.
            "fields": _VOLUME_FIELDS,
        }
        if settings.GOOGLE_BOOKS_API_KEY:
            params["key"] = secret_value(settings.GOOGLE_BOOKS_API_KEY)
        try:
            record_provider_request("google_books")
            response = httpx.get(
                GOOGLE_BOOKS_BASE, params=params, timeout=15, headers=_HEADERS
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.debug(
                "Google Books metadata search failed (type=%s)",
                type(exc).__name__,
            )
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            outcome = (
                # A caller error is not a catalog outcome. One such TypeError
                # surfaced as "no identity-compatible exact-edition page count",
                # which reads exactly like a genuine empty answer.
                "internal_error"
                if isinstance(exc, (TypeError, AttributeError, NameError, ImportError)) else
                "rate_limited" if status == 429 else
                "access_restricted" if status in {401, 403, 451} else
                "timeout" if isinstance(exc, httpx.TimeoutException) else
                "response_invalid" if isinstance(exc, ValueError) else
                "operational_failure"
            )
            return BookMetadataSearch(query, outcome, error_code=outcome)

        if (not isinstance(payload, dict) or "error" in payload
                or not isinstance(payload.get("items", []), list)
                or (not payload.get("items") and (type(payload.get("totalItems")) is not int or payload["totalItems"] != 0))):
            return BookMetadataSearch(query, "response_invalid", error_code="invalid_response")

        results: list[BookMetadata] = []
        for item in (payload.get("items") or [])[:int(params["maxResults"])]:
            if (not isinstance(item, dict) or not isinstance(item.get("volumeInfo"), dict)
                    or not isinstance(item.get("id"), str) or not item["id"]):
                return BookMetadataSearch(query, "response_invalid", error_code="invalid_volume")
            info = item.get("volumeInfo") or {}
            if (not all(info.get(key) is None or isinstance(info[key], str)
                        for key in ("title", "subtitle", "publisher", "publishedDate", "description", "printType", "previewLink", "infoLink"))
                    or not isinstance(info.get("authors", []), list)
                    or not all(isinstance(value, str) for value in info.get("authors", []))
                    or not isinstance(info.get("industryIdentifiers", []), list)
                    or not all(isinstance(value, dict) for value in info.get("industryIdentifiers", []))):
                return BookMetadataSearch(query, "response_invalid", error_code="invalid_volume_fields")
            identifiers = tuple(
                normalized
                for record in (info.get("industryIdentifiers") or [])
                if (normalized := normalize_isbn(str(record.get("identifier") or "")))
            )
            confidence, reason = _metadata_match(
                identifiers=identifiers,
                candidate_title=": ".join(
                    value for value in (info.get("title"), info.get("subtitle")) if value
                ),
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
                    # Hash a canonical projection of the fields this adapter
                    # actually reads, not the raw payload. The hash identifies a
                    # catalog record in findings, so it must move only when
                    # something we rely on moves — and must not depend on the
                    # server having honoured the `fields` request.
                    record_sha256=_record_digest(item, info),
                )
            )
        # The catalog can return one volume more than once in a single response.
        # Downstream, a repeated volume collides on its derived candidate id and
        # invalidates the whole discovery trace, costing the reference its entire
        # assessment. Drop the repeat here so the reported count and the reviewed
        # records describe the same set of distinct volumes.
        distinct: dict[str, BookMetadata] = {}
        for result in results:
            distinct.setdefault(result.volume_id or result.record_sha256, result)
        results = list(distinct.values())
        rank = {"high": 3, "medium": 2, "low": 1, "unresolved": 0, "none": 0}
        ranked = sorted(
            results,
            key=lambda result: (
                rank[result.match_confidence],
                result.page_count is not None,
            ),
            reverse=True,
        )
        return BookMetadataSearch(query, "results" if ranked else "no_results", tuple(ranked))

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

    if cross_script_comparison_unresolved(expected_title, candidate_title):
        return "unresolved", "cross_script_title_unresolved"

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
        # A query is not an observed identifier. Missing ISBNs can establish a
        # possible work match but never bind an edition or its page count.
        if not identifiers and (title_overlap is None or title_overlap < 0.80):
            return "none", "ISBN-query result omitted identifiers and lacked strong title evidence"
        if not identifiers:
            return "medium", "ISBN-query result omitted identifiers; edition unresolved"
        detail = "exact ISBN"
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
