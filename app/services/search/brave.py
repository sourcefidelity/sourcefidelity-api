"""Brave Search API provider.

Brave Search is an independent search engine (not a Google/Bing proxy) with
its own index. It uses an official metered API and is available where
consumer-search scraping is unstable or unsupported.

API: https://api.search.brave.com/res/v1/web/search
Pricing and monthly credits change independently of the application; consult
Brave's current primary pricing documentation before deployment.

Good for personal users (Path A/C) who don't want to self-host SearXNG.
"""

import logging

import httpx

from app.services.search.base import (
    SearchProvider,
    SearchResult,
    classify_search_failure,
    safe_search_failure_log,
)

logger = logging.getLogger(__name__)

_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


class BraveSearch(SearchProvider):
    """Brave Search API implementation."""

    def __init__(self, api_key: str):
        self._api_key = api_key

    @property
    def name(self) -> str:
        # The lower-cased provider name is the configured budget/health key.
        return "Brave"

    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Search via Brave Search API."""
        self.last_status = "started"
        headers = {
            "X-Subscription-Token": self._api_key,
            "Accept": "application/json",
        }
        params = {
            "q": query,
            "count": min(num_results, 20),
        }
        try:
            resp = httpx.get(_BRAVE_URL, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            self.last_status = classify_search_failure(e)
            query_sha256, failure = safe_search_failure_log(query, e)
            logger.warning(
                "Brave search failed query_sha256=%s failure=%s",
                query_sha256,
                failure,
            )
            return []

        results = []
        # Brave returns results in web.results
        web_results = data.get("web", {}).get("results", [])
        for item in web_results[:num_results]:
            url = item.get("url", "")
            title = item.get("title", "")
            snippet = item.get("description", "")
            is_pdf = url.lower().endswith(".pdf")
            if url:
                results.append(SearchResult(
                    url=url, title=title, snippet=snippet, is_pdf=is_pdf,
                ))
        self.last_status = "completed"
        return results
