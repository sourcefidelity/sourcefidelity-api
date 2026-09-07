"""Google Custom Search API provider.

Uses the Custom Search JSON API (https://developers.google.com/custom-search/v1/overview).
Requires GOOGLE_SEARCH_API_KEY + GOOGLE_SEARCH_CSE_ID in .env.

The API is closed to new customers and scheduled for discontinuation for
existing customers on January 1, 2027. It remains only for existing configured
deployments during migration; do not adopt it as a new production dependency.
"""

import logging
from typing import Optional

import httpx

from app.services.search.base import (
    SearchProvider,
    SearchResult,
    classify_search_failure,
    safe_search_failure_log,
)

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
        self.last_status = "started"
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
            self.last_status = classify_search_failure(e)
            query_sha256, failure = safe_search_failure_log(query, e)
            logger.warning(
                "Google search failed query_sha256=%s failure=%s",
                query_sha256,
                failure,
            )
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
        self.last_status = "completed"
        return results
