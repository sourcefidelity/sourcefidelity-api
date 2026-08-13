"""Bing Web Search API provider.

Uses the Bing Web Search API v7 (https://www.microsoft.com/en-us/bing/apis/bing-web-search-api).
Requires BING_SEARCH_API_KEY in .env.

Pricing: S1 tier ~$3/1000 transactions (similar to Google). May work better
on networks where Google is blocked (a constrained regional network) — Bing has better
regional availability in some markets.
"""

import logging

import httpx

from app.services.search.base import SearchProvider, SearchResult

logger = logging.getLogger(__name__)

_BING_URL = "https://api.bing.microsoft.com/v7.0/search"


class BingWebSearch(SearchProvider):
    """Bing Web Search API v7 implementation."""

    def __init__(self, api_key: str):
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "Bing Web Search"

    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Search via Bing Web Search API."""
        headers = {"Ocp-Apim-Subscription-Key": self._api_key}
        params = {
            "q": query,
            "count": min(num_results, 50),  # Bing allows up to 50
            "responseFilter": "Web",
        }
        try:
            resp = httpx.get(_BING_URL, headers=headers, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Bing search failed for query '%s': %s", query[:60], e)
            return []

        results = []
        web_pages = data.get("webPages", {}).get("value", [])
        for item in web_pages:
            url = item.get("url", "")
            title = item.get("name", "")
            snippet = item.get("snippet", "")
            is_pdf = url.lower().endswith(".pdf")
            if url:
                results.append(SearchResult(
                    url=url, title=title, snippet=snippet, is_pdf=is_pdf,
                ))
        return results
