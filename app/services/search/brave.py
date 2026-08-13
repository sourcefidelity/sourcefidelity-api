"""Brave Search API provider.

Brave Search is an independent search engine (not a Google/Bing proxy) with
its own index. Has a free tier (2,000 queries/month) and is available where
Google/Bing APIs are deprecated or unavailable.

API: https://api.search.brave.com/res/v1/web/search
Free tier: 2,000 queries/month (sufficient for ~1-2 class batches)
Paid: ~$3/1000 queries after free tier

Good for personal users (Path A/C) who don't want to self-host SearXNG.
"""

import logging

import httpx

from app.services.search.base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)

_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


class BraveSearch(SearchProvider):
    """Brave Search API implementation."""

    def __init__(self, api_key: str):
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "Brave Search"

    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Search via Brave Search API."""
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
            logger.warning("Brave search failed for '%s': %s", query[:60], e)
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
        return results
