"""DeepSeek Search API provider.

API (api-docs.deepseek.com/guides/search, checked 2026-10-18):
POST https://api.deepseek.com/v1/tools/search
Headers: Authorization: Bearer <DeepSeek API key>
Body: {"query": "...", "count": 1-50}; the query is 1-128 characters and keeps
its operators ("...", site:, -term, lang:, after:/before:); no filetype:.
Response: {"results": [{"title", "url", "snippet", "summary", ...}]}

US$0.75 per 1,000 successful searches; failed requests, empty results and
timeouts are not billed. 5 requests per second per account, whatever the key.
Measured 2026-10-18 on the labelled corpus (STATE §6): 36/40 real references,
6/54 fabricated titles, median 0.32 s.
"""

import logging

import httpx
from app.services.retrieval_deadline import remaining

from app.services.search.base import (
    SearchProvider,
    SearchResult,
    classify_search_failure,
    safe_search_failure_log,
)

logger = logging.getLogger(__name__)

_DEEPSEEK_SEARCH_URL = "https://api.deepseek.com/v1/tools/search"
_MAX_QUERY_CHARS = 128
_USD_PER_SUCCESSFUL_SEARCH = 0.00075
# 401 invalid key and 402 insufficient balance do not clear within a run.
_CIRCUIT_STATUSES = frozenset({401, 402})


def _bounded_query(query: str) -> str:
    """Fit the documented 128-character limit by dropping trailing words,
    closing a cut quoted phrase so it never turns into loose words."""
    words = " ".join(query.split()).split(" ")
    while words:
        text = " ".join(words)
        if text.count('"') % 2:
            text += '"'
        if len(text) <= _MAX_QUERY_CHARS:
            return text
        words.pop()
    return query[:_MAX_QUERY_CHARS]


def _defer_after_refusal() -> None:
    """Hold every process back after a 429 (the limit is per account)."""
    from app.services.retrieval import provider_gate
    try:
        provider_gate.defer("deepseek", 1.0)
    except (TypeError, ValueError, AttributeError):
        pass


class DeepSeekSearch(SearchProvider):
    """DeepSeek's standalone Search API."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._circuit_status: str | None = None

    @property
    def name(self) -> str:
        return "DeepSeek Search"

    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        if self._circuit_status is not None:
            self.last_status = self._circuit_status
            self.last_cost_usd = None
            return []
        self.last_status = "started"
        self.last_cost_usd = None
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        # The price is per request, whatever the count.
        payload = {"query": _bounded_query(query), "count": max(1, min(num_results, 50))}
        try:
            from app.services.retrieval.provider_gate import retry_unopened_connection
            resp = retry_unopened_connection(
                lambda: httpx.post(_DEEPSEEK_SEARCH_URL, headers=headers, json=payload, timeout=remaining(15)))
            if resp.status_code == 429:
                _defer_after_refusal()
            resp.raise_for_status()
            items = resp.json()["results"]
            if not isinstance(items, list) or any(
                not isinstance(item, dict) or not isinstance(item.get("url"), str)
                or not item["url"] or not isinstance(item.get("title") or "", str)
                or not isinstance(item.get("snippet") or "", str)
                for item in items
            ):
                raise ValueError("invalid search result schema")
        except Exception as e:
            self.last_status = classify_search_failure(e)
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in _CIRCUIT_STATUSES:
                self._circuit_status = self.last_status
            query_sha256, failure = safe_search_failure_log(query, e)
            logger.warning("DeepSeek search failed query_sha256=%s failure=%s", query_sha256, failure)
            return []

        if items:
            self.last_cost_usd = _USD_PER_SUCCESSFUL_SEARCH
        results = []
        for item in items[:num_results]:
            url = item["url"]
            results.append(SearchResult(
                url=url,
                title=item.get("title") or "",
                snippet=(item.get("snippet") or "")[:300],
                is_pdf=url.lower().split("?")[0].endswith(".pdf"),
            ))
        self.last_status = "completed"
        return results
