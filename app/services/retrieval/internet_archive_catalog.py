"""Internet Archive book-catalogue search.

A second catalogue beside Google Books, reachable on networks where other
catalogue hosts are not: measured 2026-09-22, `openlibrary.org` was refused at
the TLS handshake all day and Google Books answered 429, while `archive.org`
served metadata throughout.

The Archive's search returns title, creator, year, publisher and ISBN -- the
same combination the identity rule scores. Its answers are precise: over the
stored corpus every reference it returned a record for was accepted by
`assess_work_identity`. Its coverage is narrower than Google Books, which is
why an EMPTY Archive result is reported as `no_results_low_coverage` rather
than `no_results`: this catalogue is good evidence that a book exists and poor
evidence that one does not, and the two must not be confused by a caller that
turns emptiness into a finding about a student's reference.

Search is anonymous and bounded; no credentials and no item files are fetched.
"""

import hashlib
import logging
import re

import httpx

from app.services.retrieval.google_books import BookMetadata, BookMetadataSearch
from app.services.processing_metrics import record_provider_request

logger = logging.getLogger(__name__)

_ARCHIVE_SEARCH = "https://archive.org/advancedsearch.php"
_USER_AGENT = "SourceFidelity/0.1 (book catalogue search)"
CATALOG_VERSION = "archive-book-catalog-v1"

# Lucene-significant characters the Archive's query parser treats specially.
_QUERY_UNSAFE = re.compile(r'["():\[\]{}~^?*\\/]+')


def _escaped(value: str) -> str:
    return _QUERY_UNSAFE.sub(" ", str(value or "")).strip()


def _surname(author: str | None) -> str:
    if not author:
        return ""
    return _escaped(str(author).split(",", 1)[0].split()[-1] if str(author).strip() else "")


class InternetArchiveCatalogRetriever:
    """Search the Internet Archive's text collection for a cited book."""

    name = "internet_archive"

    def search_metadata_result(
        self,
        *,
        isbn: str | None = None,
        title: str | None = None,
        author: str | None = None,
        publisher: str | None = None,
        year: str | None = None,
        max_results: int = 10,
        timeout: float = 30.0,
    ) -> BookMetadataSearch:
        clean_title = _escaped(title)
        clean_isbn = re.sub(r"[^0-9Xx]", "", str(isbn or ""))
        if clean_isbn:
            query = f"isbn:({clean_isbn}) AND mediatype:texts"
        elif clean_title:
            query = f'title:("{clean_title[:200]}") AND mediatype:texts'
            surname = _surname(author)
            if surname:
                # The creator filter is what keeps precision high: without it
                # a title phrase returns adjacent works, which the identity
                # rule then has to reject one by one.
                query += f' AND creator:("{surname}")'
        else:
            return BookMetadataSearch("", "response_invalid",
                                      error_code="insufficient_metadata")

        try:
            record_provider_request("internet_archive")
            response = httpx.get(
                _ARCHIVE_SEARCH,
                params={
                    "q": query,
                    "fl[]": ["identifier", "title", "creator", "year",
                             "publisher", "isbn", "date"],
                    "rows": max(1, min(max_results, 20)),
                    "output": "json",
                },
                headers={"User-Agent": _USER_AGENT},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.debug("Archive catalogue search failed (type=%s)", type(exc).__name__)
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            outcome = (
                "rate_limited" if status == 429 else
                "access_restricted" if status in {401, 403, 451} else
                "timeout" if isinstance(exc, httpx.TimeoutException) else
                "response_invalid" if isinstance(exc, ValueError) else
                "operational_failure"
            )
            return BookMetadataSearch(query, outcome, error_code=outcome)

        if not isinstance(payload, dict) or not isinstance(payload.get("response"), dict):
            return BookMetadataSearch(query, "response_invalid", error_code="invalid_response")
        docs = payload["response"].get("docs")
        if not isinstance(docs, list):
            return BookMetadataSearch(query, "response_invalid", error_code="invalid_response")

        candidates: list[BookMetadata] = []
        for doc in docs[: max(1, min(max_results, 20))]:
            if not isinstance(doc, dict) or not isinstance(doc.get("identifier"), str):
                return BookMetadataSearch(query, "response_invalid", error_code="invalid_record")
            record_title = doc.get("title")
            if isinstance(record_title, list):
                record_title = record_title[0] if record_title else None
            if record_title is not None and not isinstance(record_title, str):
                return BookMetadataSearch(query, "response_invalid", error_code="invalid_record")
            creator = doc.get("creator")
            authors = tuple(
                str(value) for value in
                (creator if isinstance(creator, list) else [creator] if creator else [])
            )
            identifiers = tuple(
                re.sub(r"[^0-9Xx]", "", str(value))
                for value in (doc.get("isbn") if isinstance(doc.get("isbn"), list)
                              else [doc["isbn"]] if doc.get("isbn") else [])
                if re.sub(r"[^0-9Xx]", "", str(value))
            )
            published = doc.get("year") or doc.get("date")
            candidates.append(BookMetadata(
                volume_id=str(doc["identifier"]),
                title=record_title,
                subtitle=None,
                authors=authors,
                publisher=str(doc.get("publisher") or "") or None,
                published_date=str(published) if published is not None else None,
                description=None,
                identifiers=identifiers,
                page_count=None,
                print_type="BOOK",
                preview_link=None,
                info_link=f"https://archive.org/details/{doc['identifier']}",
                match_confidence="unscored",
                match_reason="archive_catalog_record",
                record_sha256=hashlib.sha256(
                    f"{CATALOG_VERSION}\x1f{doc['identifier']}".encode()).hexdigest(),
            ))

        if not candidates:
            # Deliberately NOT "no_results". This catalogue's silence is weak
            # evidence of absence, and the distinction has to survive into the
            # caller rather than being recovered by guesswork there.
            return BookMetadataSearch(query, "no_results_low_coverage")
        return BookMetadataSearch(query, "results", candidates=tuple(candidates))
