"""Open Library book metadata — a second, independent book catalog.

Google Books was the only catalog route, so every monograph's review rested on
one provider: when it failed or returned an unusable response, the reference
could not complete its book-catalog coverage at all. Open Library is operated by
the Internet Archive, needs no key or registration, and indexes editions
independently, which is what makes it useful here rather than merely additional.

This adapter observes metadata only. It never acquires or admits source text.
"""

import logging
import re

import httpx

from app.log_safety import safe_exception_code
from app.services.retrieval.base import OBSERVED_AUTHOR_LIMIT, BOOK_SOURCE_KINDS, RetrievalResult, RetrievalSource
from app.services.retrieval.provider_runtime import ProviderPolicy, provider_policy
from app.services.processing_metrics import record_provider_request

logger = logging.getLogger(__name__)

OPEN_LIBRARY_SEARCH = "https://openlibrary.org/search.json"
_HEADERS = {
    "User-Agent": "SourceFidelity/0.1 (academic source verification; contact via repository)"
}
# Enough rows to see whether the cited edition is present without pulling a page
# of loosely related titles into candidate review.
_MAX_ROWS = 5
_FIELDS = "key,title,subtitle,author_name,first_publish_year,publish_year,publisher,isbn,number_of_pages_median"
# Characters the catalog's query parser reads as syntax rather than as text.
# A slash is the costly one: "American cinema/American culture" returns an
# unrelated book, while the same words without it match at rank one.
_QUERY_SYNTAX = re.compile(r'[+\-&|!(){}\[\]^"~*?:\\/]')


def _sanitize_query(value: str) -> str:
    return " ".join(_QUERY_SYNTAX.sub(" ", value).split())


class OpenLibraryRetriever(RetrievalSource):
    """Search Open Library for an edition matching a cited book."""
    # A second independent book catalogue; monograph coverage should not
    # rest on Google Books alone.
    required_for_search_completion = True
    # A book catalogue. Measured 2026-09-23: 9 candidates on journal-article
    # references, none plausible.
    supported_source_kinds = BOOK_SOURCE_KINDS

    name = "open_library"
    capabilities = frozenset({"title_author", "metadata", "metadata_only_search"})
    documentation_url = "https://openlibrary.org/dev/docs/api/search"

    def __init__(self) -> None:
        self.policy = provider_policy(
            self.name,
            ProviderPolicy(timeout_seconds=20.0, min_interval_seconds=1.0),
        )
        self.provider_metrics: dict[str, int] = {"calls": 0, "timeouts": 0, "errors": 0}

    def search_by_doi(self, doi: str) -> RetrievalResult:
        return RetrievalResult(
            source_name=self.name,
            success=False,
            error="Open Library indexes editions, not DOIs",
        )

    def search_by_title_author(
        self, title: str, author: str | None = None
    ) -> RetrievalResult:
        cleaned = " ".join(str(title or "").split())
        if not cleaned:
            return RetrievalResult(
                source_name=self.name, success=False, error="No title supplied"
            )
        # The general `q` search, not the `title` field. Measured 2026-09-21:
        # a field search demands the record's title match the cited one, so
        # Bork's "The Antitrust Paradox: A Policy at War with Itself" returns
        # nothing because the catalog holds it as "The antitrust paradox",
        # while `q` with the same string finds it. The caller compares the
        # returned title itself, so a looser search costs nothing.
        surname = _sanitize_query(str(author or "").split(",", 1)[0])
        terms = _sanitize_query(cleaned)[:300]
        query = f"{terms} {surname[:100]}".strip() if len(surname) >= 2 else terms
        if not query:
            return RetrievalResult(
                source_name=self.name, success=False, error="No usable title terms"
            )
        params: dict[str, str | int] = {
            "q": query,
            "limit": _MAX_ROWS,
            "fields": _FIELDS,
        }
        self.provider_metrics["calls"] += 1
        record_provider_request("open_library")
        try:
            response = httpx.get(
                OPEN_LIBRARY_SEARCH,
                params=params,
                headers=_HEADERS,
                timeout=self.policy.timeout_seconds,
            )
        except httpx.TimeoutException:
            self.provider_metrics["timeouts"] += 1
            return RetrievalResult(
                source_name=self.name, success=False, error="read_timeout"
            )
        except httpx.HTTPError as exc:
            self.provider_metrics["errors"] += 1
            return RetrievalResult(
                source_name=self.name,
                success=False,
                error=f"open_library:{safe_exception_code(exc)}",
            )
        if response.status_code != 200:
            return RetrievalResult(
                source_name=self.name,
                success=False,
                error=f"Open Library HTTP {response.status_code}",
            )
        try:
            payload = response.json()
        except ValueError:
            return RetrievalResult(
                source_name=self.name, success=False, error="response_invalid"
            )
        rows = payload.get("docs") or []
        if not rows:
            return RetrievalResult(
                source_name=self.name,
                success=False,
                error="No results",
                metadata={"identity_search_result_count": 0},
            )
        top = rows[0]
        full_title = ": ".join(
            value for value in (top.get("title"), top.get("subtitle")) if value
        )
        year = top.get("first_publish_year") or next(
            iter(top.get("publish_year") or []), None
        )
        return RetrievalResult(
            source_name=self.name,
            success=True,
            title=full_title or None,
            authors=[str(value) for value in (top.get("author_name") or [])][:OBSERVED_AUTHOR_LIMIT],
            year=str(year) if year else None,
            metadata={
                "identity_search_result_count": len(rows),
                "open_library_key": top.get("key"),
                "publisher": next(iter(top.get("publisher") or []), None),
                "isbns": [str(value) for value in (top.get("isbn") or [])][:8],
            },
        )

    def statements_of_responsibility(self, title: str, author: str) -> dict:
        """The title-page "by" statements of the work matching a cited book.

        Author lists cannot tell a single-authored book from an edited one:
        Open Library lists David Neumeyer as the author of *The Oxford Handbook
        of Film Music Studies*, which he edited. The edition's statement of
        responsibility transcribes the title page, and there an edited volume
        reads "edited by ..." (checked 2026-09-27 on four edited volumes and
        two monographs, including Belton's). Metadata only; no text acquired.
        """
        surname = str(author or "").split(",", 1)[0].strip()
        found = self.search_by_title_author(title, author)
        key = str((found.metadata or {}).get("open_library_key") or "")
        if not found.success or not re.fullmatch(r"/works/OL\d+W", key):
            return {"classification": "unresolved", "reason": "work_not_found"}
        if not _title_words(found.title or "")[:len(_title_words(title))] == _title_words(title):
            return {"classification": "unresolved", "reason": "title_differs"}
        if not any(surname.casefold() in name.casefold() for name in found.authors):
            return {"classification": "unresolved", "reason": "author_differs"}
        self.provider_metrics["calls"] += 1
        record_provider_request("open_library")
        try:
            response = httpx.get(
                f"{OPEN_LIBRARY_BASE}{key}/editions.json",
                params={"limit": _MAX_EDITIONS},
                headers=_HEADERS,
                timeout=self.policy.timeout_seconds,
            )
            entries = response.json().get("entries") or [] if response.status_code == 200 else None
        except (httpx.HTTPError, ValueError, AttributeError):
            entries = None
        if entries is None:
            return {"classification": "unresolved", "reason": "editions_unavailable", "work_key": key}
        statements = [" ".join(entry["by_statement"].split())[:300] for entry in entries[:_MAX_EDITIONS]
                      if isinstance(entry, dict) and isinstance(entry.get("by_statement"), str)
                      and entry["by_statement"].strip()]
        return {"classification": classify_responsibility(statements, surname), "work_key": key,
                "statements": statements[:8], "provider": self.name}


OPEN_LIBRARY_BASE = "https://openlibrary.org"
_MAX_EDITIONS = 20
_EDITORIAL = re.compile(r"\b(?:edited|editors?|eds?|compiled|compiler|herausgegeben|hrsg)\b", re.IGNORECASE)
# A second name, or a contributor clause, leaves the statement short of
# showing a single author.
_SEVERAL_NAMES = re.compile(r"\b(?:and|with)\b|[&;,]", re.IGNORECASE)


def _title_words(value: str) -> list[str]:
    return re.findall(r"\w+", value.casefold())


def classify_responsibility(statements: list[str], surname: str) -> str:
    """single_authored, edited or unresolved, from title-page statements.

    Single-authored needs every statement to name the cited author alone; any
    editorial statement on any edition makes the work edited. Anything else,
    including no statement at all, is unresolved.
    """
    statements = [value for value in statements if value.strip()]
    if any(_EDITORIAL.search(value) for value in statements):
        return "edited"
    key = surname.strip().casefold()
    if key and statements and all(
            key in value.casefold() and not _SEVERAL_NAMES.search(value.strip(" .")) for value in statements):
        return "single_authored"
    return "unresolved"
