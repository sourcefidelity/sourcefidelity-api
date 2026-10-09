"""Mojeek search provider — independent crawler-based web index.

Evaluation only (owner-authorized comparison, 2026-10-01): not registered in
``get_search_provider`` and not part of any search policy.

API: GET https://api.mojeek.com/search?q=...&api_key=...&fmt=json&t=N
Response: {"response": {"status": "OK", "results": [{"url", "title", "desc"}]}}

The key is a query parameter, so request URLs must never reach logs or
exception text: httpx request logging is raised to WARNING for this module's
calls and failures are recorded by class name or status code only.

Terms (mojeek.com/services/search/web-search-api, checked 2026-10-01): results
may be cached for one hour below the Business plan; results may be used with AI.
"""

import logging
import time

import httpx

from app.services.retrieval_deadline import remaining
from app.services.search.base import (
    SearchProvider,
    SearchResult,
    classify_search_failure,
    safe_search_failure_log,
)

logger = logging.getLogger(__name__)

_MOJEEK_URL = "https://api.mojeek.com/search"
_CIRCUIT_STATUSES = frozenset({401, 402, 403})


class MojeekSearch(SearchProvider):
    """Mojeek web search API implementation."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._circuit_status: str | None = None
        for name in ("httpx", "httpcore"):
            logging.getLogger(name).setLevel(logging.WARNING)

    @classmethod
    def from_settings(cls) -> "MojeekSearch | None":
        from app.config import settings
        key = settings.MOJEEK_API_KEY
        return cls(key.get_secret_value()) if key else None

    @property
    def name(self) -> str:
        return "Mojeek"

    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        if self._circuit_status is not None:
            self.last_status = self._circuit_status
            self.last_cost_usd = None
            return []
        self.last_status = "started"
        self.last_cost_usd = None
        params = {"q": query, "api_key": self._api_key, "fmt": "json",
                  "t": min(num_results, 10), "lb": "EN", "lbb": 100}
        try:
            resp = httpx.get(_MOJEEK_URL, params=params, timeout=remaining(15))
            resp.raise_for_status()
            body = resp.json()["response"]
            if body.get("status") != "OK":
                raise ValueError("provider status not OK")
            items = body.get("results") or []
            if not isinstance(items, list) or any(
                not isinstance(item, dict) or not isinstance(item.get("url"), str)
                for item in items
            ):
                raise ValueError("invalid search result schema")
        except Exception as e:
            self.last_status = classify_search_failure(e)
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in _CIRCUIT_STATUSES:
                self._circuit_status = self.last_status
            query_sha256, failure = safe_search_failure_log(query, e)
            logger.warning("Mojeek search failed query_sha256=%s failure=%s", query_sha256, failure)
            return []
        finally:
            time.sleep(0.5)

        results = []
        for item in items[:num_results]:
            url = item["url"]
            if url:
                results.append(SearchResult(
                    url=url, title=str(item.get("title") or ""),
                    snippet=str(item.get("desc") or "")[:300],
                    is_pdf=url.lower().split("?")[0].endswith(".pdf"),
                ))
        self.last_status = "completed"
        return results
