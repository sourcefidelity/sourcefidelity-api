"""Open Library page-count lookup.

Open Library (openlibrary.org) is a free, keyless Internet Archive project.
Used as a fallback behind Google Books for page-count lookup.

NOTE: Like Google Books, openlibrary.org is regionally unavailable in
Chinese-web-optimized deployments (shares archive.org's regional block).
It's a fallback for users in regions where it is reachable; the completeness
checker works without it via local signals.

API: https://openlibrary.org/dev/docs/api/read
  Read API:  GET /api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data
  Page count at: response["ISBN:{isbn}"]["number_of_pages"]
"""

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

OPEN_LIBRARY_READ = "https://openlibrary.org/api/books"


@dataclass
class PageCountResult:
    """Result of a page-count lookup."""

    success: bool
    expected_pages: int | None = None
    source: str = "Open Library"
    title: str | None = None
    error: str | None = None


class OpenLibraryRetriever:
    """Looks up expected page counts from Open Library (keyless)."""

    def _headers(self) -> dict:
        # Courtesy identification (no auth required).
        return {"User-Agent": "SourceFidelity/0.1.0 (open-source source-verification tool)"}

    def lookup_page_count(
        self,
        *,
        isbn: str | None = None,
        title: str | None = None,
        author: str | None = None,
    ) -> PageCountResult:
        # Only use the ISBN Read API for page counts. The title-based Search API
        # returns number_of_pages_median across possibly-unrelated works and is
        # unreliable for completeness checking (a generic title can match the
        # wrong book entirely). When there's no ISBN, or the ISBN doesn't match,
        # return no result rather than guessing.
        if not isbn:
            return PageCountResult(success=False, error="No ISBN provided (title search disabled for page counts)")
        return self._lookup_by_isbn(isbn)

    def _lookup_by_isbn(self, isbn: str) -> PageCountResult:
        """Use the Read API for an exact edition match."""
        bibkey = f"ISBN:{isbn}"
        params = {"bibkeys": bibkey, "format": "json", "jscmd": "data"}
        try:
            resp = httpx.get(OPEN_LIBRARY_READ, params=params, headers=self._headers(), timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.debug("Open Library Read API failed: %s", e)
            return PageCountResult(success=False, error=str(e))

        record = data.get(bibkey)
        if not record:
            return PageCountResult(success=False, error="No edition match")

        pages = record.get("number_of_pages")
        if pages:
            return PageCountResult(
                success=True,
                expected_pages=int(pages),
                title=record.get("title"),
            )
        return PageCountResult(success=False, error="No pageCount", title=record.get("title"))
