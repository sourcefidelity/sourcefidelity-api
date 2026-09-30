"""Tavily search provider — AI-focused search API.

Tavily is designed for AI/agent applications. Returns clean, relevant results
with content snippets. Has a free tier (1,000 queries/month).

API: POST https://api.tavily.com/search
Headers: Authorization: Bearer <key> (the only documented method, 2026-09-25)
Body: {"query": "...", "max_results": 10, "search_depth": "basic", ...}
Response: {"results": [{"url", "title", "content"}], "usage": {"credits": 1}}

Good for academic source search — designed to find relevant content, not
just keyword matches. Free tier covers ~1-2 class batches.
"""

import logging

import httpx

from app.config import settings
from app.services.retrieval_deadline import remaining
from app.services.search.base import (
    SearchProvider,
    SearchResult,
    classify_search_failure,
    safe_search_failure_log,
)

logger = logging.getLogger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"
# Documented account-level refusals: 432 and 433 are plan and pay-as-you-go
# limits; none of them clears within a run, so the adapter stops calling.
_CIRCUIT_STATUSES = frozenset({401, 403, 429, 432, 433})


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
        self.last_cost_usd = None
        payload = {
            "query": query,
            "search_depth": "basic",   # 1 credit; "advanced" costs 2
            "include_answer": False,
            "max_results": min(num_results, 10),
            "include_usage": True,
            # No `exact_match`: measured 2026-09-25 on the labelled corpus it
            # found 28 of 40 real references against 33 without it (7 lost,
            # 2 gained); the quoted title already steers the ranking.
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        try:
            resp = httpx.post(_TAVILY_URL, json=payload, headers=headers, timeout=remaining(15))
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            self.last_status = classify_search_failure(e)
            if (
                isinstance(e, httpx.HTTPStatusError)
                and e.response.status_code in _CIRCUIT_STATUSES
            ):
                self._circuit_status = self.last_status
            query_sha256, failure = safe_search_failure_log(query, e)
            logger.warning(
                "Tavily search failed query_sha256=%s failure=%s",
                query_sha256,
                failure,
            )
            return []

        credits = (data.get("usage") or {}).get("credits") if isinstance(data, dict) else None
        rate = getattr(settings, "TAVILY_USD_PER_CREDIT", None)
        if type(credits) in (int, float) and credits >= 0 and rate is not None:
            self.last_cost_usd = credits * rate   # unpriced stays None, never 0
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
