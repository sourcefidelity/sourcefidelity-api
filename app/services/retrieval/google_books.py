"""Google Books page-count lookup.

Not a full RetrievalSource (Google Books doesn't provide full-text retrieval
for citation verification). This adapter exists solely to look up the
expected page count of a book for completeness checking.

Requires a free API key (GOOGLE_BOOKS_API_KEY). Anonymous access works but
is rate-limited to ~100 requests/day; with a key, ~1,000/day.
NOTE: Regionally unavailable in Chinese-web-optimized deployments — the
completeness checker works without it via local signals.
"""

import logging
from dataclasses import dataclass

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

GOOGLE_BOOKS_BASE = "https://www.googleapis.com/books/v1/volumes"


@dataclass
class PageCountResult:
    """Result of a page-count lookup."""

    success: bool
    expected_pages: int | None = None
    source: str = "Google Books"
    title: str | None = None
    error: str | None = None


class GoogleBooksRetriever:
    """Looks up expected page counts from Google Books."""

    def lookup_page_count(
        self,
        *,
        isbn: str | None = None,
        title: str | None = None,
        author: str | None = None,
    ) -> PageCountResult:
        # Prefer ISBN (edition-specific). Title search is disabled for page
        # counts because a generic title can match an unrelated book and return
        # that book's pageCount — a misleading signal for completeness.
        if not isbn:
            return PageCountResult(success=False, error="No ISBN provided (title search disabled for page counts)")
        params = {"q": f"isbn:{isbn}"}
        if settings.GOOGLE_BOOKS_API_KEY:
            params["key"] = settings.GOOGLE_BOOKS_API_KEY

        if settings.GOOGLE_BOOKS_API_KEY:
            params["key"] = settings.GOOGLE_BOOKS_API_KEY

        try:
            resp = httpx.get(GOOGLE_BOOKS_BASE, params=params, timeout=15)
            if resp.status_code == 429:
                return PageCountResult(success=False, error="Rate limited (429)")
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.debug("Google Books lookup failed: %s", e)
            return PageCountResult(success=False, error=str(e))

        items = data.get("items") or []
        if not items:
            return PageCountResult(success=False, error="No results")

        # Take the first item with a populated pageCount.
        for item in items:
            info = item.get("volumeInfo", {}) or {}
            pages = info.get("pageCount")
            if pages:
                return PageCountResult(
                    success=True,
                    expected_pages=int(pages),
                    title=info.get("title"),
                )

        return PageCountResult(
            success=False,
            error="No pageCount in results",
            title=(items[0].get("volumeInfo", {}) or {}).get("title"),
        )
