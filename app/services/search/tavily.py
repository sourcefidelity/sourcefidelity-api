"""Tavily search provider — AI-focused search API.

Tavily is designed for AI/agent applications. Returns clean, relevant results
with content snippets. Has a free tier (1,000 queries/month).

API: POST https://api.tavily.com/search
Body: {"api_key": "...", "query": "...", "max_results": 10}
Response: {"results": [{"url", "title", "content"}]}

Good for academic source search — designed to find relevant content, not
just keyword matches. Free tier covers ~1-2 class batches.
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

_TAVILY_URL = "https://api.tavily.com/search"


class TavilySearch(SearchProvider):
    """Tavily AI search API implementation."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._circuit_status: str | None = None

    @property
    def name(self) -> str:
        return "Tavily"

    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Search via Tavily API."""
        if self._circuit_status is not None:
            self.last_status = self._circuit_status
            return []
        self.last_status = "started"
        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "max_results": min(num_results, 10),
        }
        try:
            resp = httpx.post(_TAVILY_URL, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            self.last_status = classify_search_failure(e)
            if (
                isinstance(e, httpx.HTTPStatusError)
                and e.response.status_code in {401, 403, 429, 432}
            ):
                self._circuit_status = self.last_status
            query_sha256, failure = safe_search_failure_log(query, e)
            logger.warning(
                "Tavily search failed query_sha256=%s failure=%s",
                query_sha256,
                failure,
            )
            return []

        results = []
        for item in data.get("results", []):
            url = item.get("url", "")
            title = item.get("title", "")
            snippet = item.get("content", "")
            is_pdf = url.lower().endswith(".pdf")
            if url:
                results.append(SearchResult(
                    url=url, title=title, snippet=snippet, is_pdf=is_pdf,
                ))
        self.last_status = "completed"
        return results
