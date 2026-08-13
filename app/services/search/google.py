"""Google Custom Search API provider.

Uses the Custom Search JSON API (https://developers.google.com/custom-search/v1/overview).
Requires GOOGLE_SEARCH_API_KEY + GOOGLE_SEARCH_CSE_ID in .env.

Free tier: 100 queries/day. Paid: $5 per 1000 queries.
For a batch of 30 papers with ~40 refs each (~1200 lookups, but most hit the
academic-DB chain first so only misses trigger search): typically 200-400
search queries per batch = $1-2 in search costs.
"""

import logging
from typing import Optional

import httpx

from app.services.search.base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)

_GOOGLE_CSE_URL = "https://www.googleapis.com/customsearch/v1"


class GoogleCustomSearch(SearchProvider):
    """Google Custom Search API implementation."""

    def __init__(self, api_key: str, cse_id: str):
        self._api_key = api_key
        self._cse_id = cse_id

    @property
    def name(self) -> str:
        return "Google Custom Search"

    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Search via Google Custom Search API."""
        # Google caps at 10 results per request
        num = min(num_results, 10)
        params = {
            "key": self._api_key,
            "cx": self._cse_id,
            "q": query,
            "num": num,
        }
        try:
            resp = httpx.get(_GOOGLE_CSE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Google search failed for query '%s': %s", query[:60], e)
            return []

        results = []
        for item in data.get("items", []):
            url = item.get("link", "")
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            # Detect PDF: Google's fileFormat field, or URL ending in .pdf
            is_pdf = bool(item.get("fileFormat")) or url.lower().endswith(".pdf")
            if url:
                results.append(SearchResult(
                    url=url, title=title, snippet=snippet, is_pdf=is_pdf,
                ))
        return results
